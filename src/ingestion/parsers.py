"""
AlphaLens - Document Parsers
Section-aware financial transcript parser and SEC tabular disclosure parser.
"""
import hashlib
import json
import re
from typing import Any, Dict, List, Tuple
from src.core.models import Document, DocumentType, SpeakerRole, TranscriptSection
from src.core.exceptions import DocumentParsingError
from src.observability.logging import logger


class FinancialDocumentParser:
    """Parses raw SEC filings and earnings call transcripts into structured document models."""

    @staticmethod
    def compute_sha256(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @classmethod
    def parse_json_document(cls, file_content: str) -> Document:
        """Parses a structured JSON document payload."""
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            raise DocumentParsingError(f"Invalid JSON payload: {str(e)}")

        required_fields = ["ticker", "company_name", "fiscal_quarter", "fiscal_year", "report_date", "document_type", "raw_text"]
        for field in required_fields:
            if field not in data:
                raise DocumentParsingError(f"Missing required field in document: '{field}'")

        content_hash = cls.compute_sha256(data["raw_text"])
        doc_id = f"{data['ticker']}_{data['fiscal_year']}_{data['fiscal_quarter']}_{data['document_type']}"

        return Document(
            document_id=doc_id,
            ticker=data["ticker"].upper(),
            company_name=data["company_name"],
            fiscal_quarter=data["fiscal_quarter"].upper(),
            fiscal_year=str(data["fiscal_year"]),
            report_date=data["report_date"],
            document_type=DocumentType(data["document_type"]),
            raw_text=data["raw_text"],
            content_hash=content_hash,
            metadata=data.get("metadata", {})
        )

    @classmethod
    def extract_transcript_turns(
        cls, raw_text: str
    ) -> List[Dict[str, Any]]:
        """
        Parses earnings call transcripts into structured conversational speaker turns.
        Detects sections: PREPARED_REMARKS vs ANALYST_QA.
        Detects speaker name and role: CEO, CFO, Analyst, Operator.
        """
        turns: List[Dict[str, Any]] = []
        current_section = TranscriptSection.PREPARED_REMARKS

        lines = raw_text.split("\n")
        current_speaker: str = "Unknown"
        current_role: SpeakerRole = SpeakerRole.UNKNOWN
        current_text_buffer: List[str] = []

        # Regex to detect speaker headers like: "Jensen Huang: Thanks, Colette." or "Colette Kress - CFO"
        speaker_pattern = re.compile(r"^([A-Z][a-zA-Z\.\s\-\']+)(?:\s*\((?:[^\)]+)\))?\s*:\s*(.*)$")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check section header tag
            if "[SECTION: PREPARED_REMARKS]" in line_str:
                current_section = TranscriptSection.PREPARED_REMARKS
                continue
            elif "[SECTION: ANALYST_QA]" in line_str:
                current_section = TranscriptSection.ANALYST_QA
                continue

            match = speaker_pattern.match(line_str)
            if match:
                # Flush previous speaker block if any
                if current_text_buffer:
                    speech_text = " ".join(current_text_buffer).strip()
                    if speech_text:
                        turns.append({
                            "speaker_name": current_speaker,
                            "speaker_role": current_role,
                            "section": current_section,
                            "text": speech_text
                        })
                    current_text_buffer = []

                speaker_candidate = match.group(1).strip()
                speech_start = match.group(2).strip()

                current_speaker = speaker_candidate
                current_role = cls._infer_speaker_role(speaker_candidate)
                if speech_start:
                    current_text_buffer.append(speech_start)
            else:
                current_text_buffer.append(line_str)

        # Flush final turn
        if current_text_buffer:
            speech_text = " ".join(current_text_buffer).strip()
            if speech_text:
                turns.append({
                    "speaker_name": current_speaker,
                    "speaker_role": current_role,
                    "section": current_section,
                    "text": speech_text
                })

        return turns

    @staticmethod
    def _infer_speaker_role(speaker_name: str) -> SpeakerRole:
        name_lower = speaker_name.lower()
        if "operator" in name_lower:
            return SpeakerRole.OPERATOR
        elif "jensen huang" in name_lower or "satya nadella" in name_lower or "tim cook" in name_lower or "ceo" in name_lower:
            return SpeakerRole.CEO
        elif "colette kress" in name_lower or "amy hood" in name_lower or "luca maestri" in name_lower or "cfo" in name_lower:
            return SpeakerRole.CFO
        elif "goldman" in name_lower or "morgan stanley" in name_lower or "ubs" in name_lower or "analyst" in name_lower:
            return SpeakerRole.ANALYST
        return SpeakerRole.EXECUTIVE

