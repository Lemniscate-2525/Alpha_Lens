"""
AlphaLens - Hierarchical & Section-Aware Chunker
Implements Parent-Child chunking and financial table preservation.
"""
import hashlib
from typing import List, Tuple
from src.core.models import Chunk, Document, DocumentType, SpeakerRole, TranscriptSection
from src.config import settings
from src.observability.logging import logger

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return len(text.split())


class HierarchicalFinancialChunker:
    """
    Splits documents into a two-level hierarchy:
    1. Parent Chunks (broad context: ~800-1200 tokens, e.g. full conversational exchange or section)
    2. Child Chunks (retrieval precision: ~150-250 tokens with overlap)
    """

    @classmethod
    def chunk_document(cls, doc: Document) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Returns (parent_chunks, child_chunks).
        Child chunks reference their enclosing parent_chunk_id.
        """
        if doc.document_type == DocumentType.EARNINGS_CALL_TRANSCRIPT:
            return cls._chunk_transcript(doc)
        else:
            return cls._chunk_general_financial_doc(doc)

    @classmethod
    def _chunk_transcript(cls, doc: Document) -> Tuple[List[Chunk], List[Chunk]]:
        from src.ingestion.parsers import FinancialDocumentParser
        turns = FinancialDocumentParser.extract_transcript_turns(doc.raw_text)

        parent_chunks: List[Chunk] = []
        child_chunks: List[Chunk] = []
        chunk_idx = 0

        for turn_idx, turn in enumerate(turns):
            turn_text = turn["text"]
            speaker_name = turn["speaker_name"]
            speaker_role = turn["speaker_role"]
            section = turn["section"]

            # Parent Chunk represents the complete speaker turn
            parent_id = f"{doc.document_id}_parent_{turn_idx}"
            parent_chunk = Chunk(
                chunk_id=parent_id,
                document_id=doc.document_id,
                parent_chunk_id=None,
                chunk_index=chunk_idx,
                content=f"[{speaker_name} ({speaker_role.value}) - {section.value}]\n{turn_text}",
                content_hash=hashlib.sha256(turn_text.encode("utf-8")).hexdigest(),
                token_count=count_tokens(turn_text),
                is_parent=True,
                ticker=doc.ticker,
                fiscal_quarter=doc.fiscal_quarter,
                fiscal_year=doc.fiscal_year,
                document_type=doc.document_type,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                section=section,
            )
            parent_chunks.append(parent_chunk)
            chunk_idx += 1

            # Generate Child Chunks for this turn
            words = turn_text.split()
            child_window_size = 80  # ~120-160 tokens
            child_overlap = 20

            if len(words) <= child_window_size:
                child_id = f"{parent_id}_child_0"
                child_content = f"[{speaker_name} - {section.value}]: {turn_text}"
                child_chunk = Chunk(
                    chunk_id=child_id,
                    document_id=doc.document_id,
                    parent_chunk_id=parent_id,
                    chunk_index=chunk_idx,
                    content=child_content,
                    content_hash=hashlib.sha256(child_content.encode("utf-8")).hexdigest(),
                    token_count=count_tokens(child_content),
                    is_parent=False,
                    ticker=doc.ticker,
                    fiscal_quarter=doc.fiscal_quarter,
                    fiscal_year=doc.fiscal_year,
                    document_type=doc.document_type,
                    speaker_name=speaker_name,
                    speaker_role=speaker_role,
                    section=section,
                )
                child_chunks.append(child_chunk)
                chunk_idx += 1
            else:
                step = child_window_size - child_overlap
                sub_idx = 0
                for start_i in range(0, len(words), step):
                    window = words[start_i: start_i + child_window_size]
                    if not window:
                        continue
                    window_text = " ".join(window)
                    child_id = f"{parent_id}_child_{sub_idx}"
                    child_content = f"[{speaker_name} - {section.value}]: {window_text}"
                    child_chunk = Chunk(
                        chunk_id=child_id,
                        document_id=doc.document_id,
                        parent_chunk_id=parent_id,
                        chunk_index=chunk_idx,
                        content=child_content,
                        content_hash=hashlib.sha256(child_content.encode("utf-8")).hexdigest(),
                        token_count=count_tokens(child_content),
                        is_parent=False,
                        ticker=doc.ticker,
                        fiscal_quarter=doc.fiscal_quarter,
                        fiscal_year=doc.fiscal_year,
                        document_type=doc.document_type,
                        speaker_name=speaker_name,
                        speaker_role=speaker_role,
                        section=section,
                    )
                    child_chunks.append(child_chunk)
                    chunk_idx += 1
                    sub_idx += 1

        return parent_chunks, child_chunks

    @classmethod
    def _chunk_general_financial_doc(cls, doc: Document) -> Tuple[List[Chunk], List[Chunk]]:
        """Handles SEC 10-Q/10-K filings with tables and disclosure paragraphs."""
        paragraphs = [p.strip() for p in doc.raw_text.split("\n\n") if p.strip()]
        parent_chunks: List[Chunk] = []
        child_chunks: List[Chunk] = []
        chunk_idx = 0

        # Check for markdown table headers
        table_headers = ""

        for p_idx, para in enumerate(paragraphs):
            if "|" in para:
                # Capture table header line if present
                lines = para.split("\n")
                if len(lines) >= 2 and "|" in lines[0] and "---" in lines[1]:
                    table_headers = f"{lines[0]}\n{lines[1]}"

            parent_id = f"{doc.document_id}_parent_{p_idx}"
            parent_chunk = Chunk(
                chunk_id=parent_id,
                document_id=doc.document_id,
                parent_chunk_id=None,
                chunk_index=chunk_idx,
                content=para,
                content_hash=hashlib.sha256(para.encode("utf-8")).hexdigest(),
                token_count=count_tokens(para),
                is_parent=True,
                ticker=doc.ticker,
                fiscal_quarter=doc.fiscal_quarter,
                fiscal_year=doc.fiscal_year,
                document_type=doc.document_type,
                section=TranscriptSection.DISCLOSURE,
            )
            parent_chunks.append(parent_chunk)
            chunk_idx += 1

            # Child chunking with table header preservation
            words = para.split()
            child_window = 75
            child_overlap = 20

            if len(words) <= child_window:
                child_id = f"{parent_id}_child_0"
                content = para
                if "|" in para and table_headers and not para.startswith("| Financial Metric"):
                    content = f"{table_headers}\n{para}"

                child_chunks.append(Chunk(
                    chunk_id=child_id,
                    document_id=doc.document_id,
                    parent_chunk_id=parent_id,
                    chunk_index=chunk_idx,
                    content=content,
                    content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    token_count=count_tokens(content),
                    is_parent=False,
                    ticker=doc.ticker,
                    fiscal_quarter=doc.fiscal_quarter,
                    fiscal_year=doc.fiscal_year,
                    document_type=doc.document_type,
                    section=TranscriptSection.DISCLOSURE,
                ))
                chunk_idx += 1
            else:
                sub_i = 0
                step = child_window - child_overlap
                for start_w in range(0, len(words), step):
                    w_slice = words[start_w: start_w + child_window]
                    slice_text = " ".join(w_slice)
                    child_id = f"{parent_id}_child_{sub_i}"
                    child_chunks.append(Chunk(
                        chunk_id=child_id,
                        document_id=doc.document_id,
                        parent_chunk_id=parent_id,
                        chunk_index=chunk_idx,
                        content=slice_text,
                        content_hash=hashlib.sha256(slice_text.encode("utf-8")).hexdigest(),
                        token_count=count_tokens(slice_text),
                        is_parent=False,
                        ticker=doc.ticker,
                        fiscal_quarter=doc.fiscal_quarter,
                        fiscal_year=doc.fiscal_year,
                        document_type=doc.document_type,
                        section=TranscriptSection.DISCLOSURE,
                    ))
                    chunk_idx += 1
                    sub_i += 1

        return parent_chunks, child_chunks

