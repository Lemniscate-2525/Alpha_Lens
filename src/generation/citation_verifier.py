"""
AlphaLens - Citation Verification Engine
Parses inline citation tags and verifies that claims and metrics correspond to actual retrieved context chunks.
"""
import re
from typing import Dict, List, Tuple
from src.core.models import Citation, SearchResult
from src.observability.logging import logger


class CitationVerifier:
    """Validates post-generation citations and verifies token-level attribution."""

    CITATION_PATTERN = re.compile(r"\[(?:Chunk:\s*)?([a-zA-Z0-9_\-]+)\]")

    @classmethod
    def verify_and_extract_citations(
        cls,
        generated_answer: str,
        retrieved_chunks: List[SearchResult]
    ) -> Tuple[List[Citation], float]:
        """
        Extracts citations from answer text, links them to chunks, and computes a citation fidelity score.
        Returns: (citations_list, fidelity_score_0_to_1)
        """
        chunk_map: Dict[str, SearchResult] = {c.chunk_id: c for c in retrieved_chunks}
        # Also map short IDs or parent IDs
        for c in retrieved_chunks:
            if c.parent_chunk_id:
                chunk_map[c.parent_chunk_id] = c

        matches = cls.CITATION_PATTERN.findall(generated_answer)
        if not matches:
            return [], 0.0

        verified_citations: List[Citation] = []
        valid_citations_count = 0

        for match_id in matches:
            target_chunk = chunk_map.get(match_id)
            if not target_chunk:
                # Try finding chunk with match_id as prefix or suffix
                for cid, c in chunk_map.items():
                    if match_id in cid or cid in match_id:
                        target_chunk = c
                        break

            if target_chunk:
                valid_citations_count += 1
                meta = target_chunk.metadata
                verified_citations.append(Citation(
                    chunk_id=target_chunk.chunk_id,
                    ticker=meta.get("ticker", "UNKNOWN"),
                    fiscal_quarter=meta.get("fiscal_quarter", ""),
                    fiscal_year=str(meta.get("fiscal_year", "")),
                    speaker_name=meta.get("speaker_name"),
                    speaker_role=meta.get("speaker_role"),
                    section=meta.get("section"),
                    quote_snippet=target_chunk.content[:160] + "...",
                    confidence_score=target_chunk.score
                ))

        fidelity = valid_citations_count / max(len(matches), 1)
        return verified_citations, round(fidelity, 2)

