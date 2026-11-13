"""
Unit tests for Hierarchical Parent-Child Financial Chunker.
"""
from src.core.models import Document, DocumentType
from src.ingestion.chunker import HierarchicalFinancialChunker, count_tokens


def test_token_counter():
    tokens = count_tokens("NVIDIA reported record Data Center revenue of $26.3 billion.")
    assert tokens > 5


def test_hierarchical_transcript_chunking():
    doc = Document(
        document_id="NVDA_2025_Q2_TEST",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        fiscal_quarter="Q2",
        fiscal_year="2025",
        report_date="2024-08-28",
        document_type=DocumentType.EARNINGS_CALL_TRANSCRIPT,
        raw_text="""[SECTION: PREPARED_REMARKS]
Jensen Huang: Welcome to our Q2 earnings call. Our computing platform is in high demand globally.
Colette Kress: Data Center revenue was $26.3 billion, up 154% year-on-year. Gross margin was 75.7% non-GAAP.""",
        content_hash="dummy_hash"
    )

    parent_chunks, child_chunks = HierarchicalFinancialChunker.chunk_document(doc)

    assert len(parent_chunks) == 2
    assert len(child_chunks) >= 2

    # Verify that child chunks point to their parent
    for child in child_chunks:
        assert child.parent_chunk_id is not None
        assert child.is_parent is False
        assert any(p.chunk_id == child.parent_chunk_id for p in parent_chunks)

    # Verify that parent chunks have is_parent = True
    for parent in parent_chunks:
        assert parent.is_parent is True
        assert parent.parent_chunk_id is None

