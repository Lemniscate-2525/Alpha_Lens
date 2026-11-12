"""
Unit tests for Financial Document & Transcript Parsers.
"""
import pytest
from src.ingestion.parsers import FinancialDocumentParser
from src.core.models import SpeakerRole, TranscriptSection


SAMPLE_TRANSCRIPT_RAW = """[SECTION: PREPARED_REMARKS]
Jensen Huang: Welcome everyone. We achieved record revenue of $30 billion.
Colette Kress: Q2 was a record quarter. Gross margin was 75.1%.

[SECTION: ANALYST_QA]
Operator: First question from Toshiya Hari with Goldman Sachs.
Toshiya Hari: Thanks Jensen. Can you explain the Blackwell mask change?
Jensen Huang: Functionality was complete; mask was modified to improve yields.
"""


def test_infer_speaker_role():
    assert FinancialDocumentParser._infer_speaker_role("Jensen Huang") == SpeakerRole.CEO
    assert FinancialDocumentParser._infer_speaker_role("Colette Kress") == SpeakerRole.CFO
    assert FinancialDocumentParser._infer_speaker_role("Amy Hood - CFO") == SpeakerRole.CFO
    assert FinancialDocumentParser._infer_speaker_role("Operator") == SpeakerRole.OPERATOR
    assert FinancialDocumentParser._infer_speaker_role("Toshiya Hari - Goldman Sachs") == SpeakerRole.ANALYST


def test_extract_transcript_turns():
    turns = FinancialDocumentParser.extract_transcript_turns(SAMPLE_TRANSCRIPT_RAW)
    assert len(turns) >= 4

    # Verify sections
    assert turns[0]["section"] == TranscriptSection.PREPARED_REMARKS
    assert turns[0]["speaker_role"] == SpeakerRole.CEO
    assert "30 billion" in turns[0]["text"]

    qa_turns = [t for t in turns if t["section"] == TranscriptSection.ANALYST_QA]
    assert len(qa_turns) >= 2
