"""
AlphaLens - Golden Evaluation Dataset
Curated ground-truth financial evaluation dataset across multiple tickers, quarters, and query categories.
"""
from typing import List
from src.core.models import EvaluationQuery

GOLDEN_BENCHMARK_QUERIES: List[EvaluationQuery] = [
    # 1. Exact Numeric & Metric Lookups
    EvaluationQuery(
        query_id="eval-001",
        query="What was NVIDIA's total revenue and Data Center revenue in Q2 fiscal 2025?",
        ticker="NVDA",
        expected_answer="Total revenue was $30.04 billion (up 15% sequentially and 122% year-on-year); Data Center revenue was $26.3 billion (up 16% sequentially and 154% year-on-year).",
        ground_truth_chunk_ids=["NVDA_2025_Q2_earnings_call_transcript"],
        is_unanswerable=False,
        category="metric_lookup"
    ),
    EvaluationQuery(
        query_id="eval-002",
        query="What was NVIDIA's gross margin GAAP and non-GAAP in Q2 fiscal 2025?",
        ticker="NVDA",
        expected_answer="GAAP gross margin was 75.1% and non-GAAP gross margin was 75.7%.",
        ground_truth_chunk_ids=["NVDA_2025_Q2_earnings_call_transcript"],
        is_unanswerable=False,
        category="metric_lookup"
    ),
    EvaluationQuery(
        query_id="eval-003",
        query="What was NVIDIA's Data Center revenue in Q3 fiscal 2025?",
        ticker="NVDA",
        expected_answer="Data Center revenue reached $30.8 billion in Q3 FY25, up 17% sequentially and 112% year-on-year.",
        ground_truth_chunk_ids=["NVDA_2025_Q3_earnings_call_transcript"],
        is_unanswerable=False,
        category="metric_lookup"
    ),
    EvaluationQuery(
        query_id="eval-004",
        query="What was Microsoft Cloud revenue and Azure growth rate in Q4 fiscal 2024?",
        ticker="MSFT",
        expected_answer="Microsoft Cloud revenue was $36.8 billion (up 21% YoY). Azure and other cloud services revenue grew 29% (30% in constant currency), with 8 points from AI services.",
        ground_truth_chunk_ids=["MSFT_2024_Q4_earnings_call_transcript"],
        is_unanswerable=False,
        category="metric_lookup"
    ),
    EvaluationQuery(
        query_id="eval-005",
        query="What was NVIDIA's GAAP diluted earnings per share (EPS) reported in the Q3 FY25 10-Q filing?",
        ticker="NVDA",
        expected_answer="Diluted earnings per share (GAAP) was $0.78 for the three months ended October 27, 2024.",
        ground_truth_chunk_ids=["NVDA_2025_Q3_sec_10q"],
        is_unanswerable=False,
        category="metric_lookup"
    ),

    # 2. Executive Attribution & Technical Explanations
    EvaluationQuery(
        query_id="eval-006",
        query="What did Jensen Huang explain regarding the Blackwell mask change to improve production yield?",
        ticker="NVDA",
        expected_answer="Jensen Huang stated that Blackwell functionality was complete from the start, but a mask modification was made to the Blackwell silicon to improve yield, with no functional changes necessary.",
        ground_truth_chunk_ids=["NVDA_2025_Q2_earnings_call_transcript"],
        is_unanswerable=False,
        category="executive_attribution"
    ),
    EvaluationQuery(
        query_id="eval-007",
        query="What did CFO Colette Kress say about Blackwell gross margin dilution during the initial ramp in Q4?",
        ticker="NVDA",
        expected_answer="Colette Kress noted that due to initial manufacturing yields and new packaging like CoWoS-L and liquid cooling, Q4 gross margins would face a headwind into the low-70% range before rebounding to mid-70% in FY26.",
        ground_truth_chunk_ids=["NVDA_2025_Q2_earnings_call_transcript", "NVDA_2025_Q3_earnings_call_transcript"],
        is_unanswerable=False,
        category="executive_attribution"
    ),
    EvaluationQuery(
        query_id="eval-008",
        query="How did Jensen Huang respond to reports about thermal or liquid cooling overheating issues in Blackwell server racks?",
        ticker="NVDA",
        expected_answer="Jensen Huang stated production and delivery schedules are on track with no delay, explaining that engineering 120kW liquid-cooled racks requires precision plumbing and multiple qualified cold-plate solutions.",
        ground_truth_chunk_ids=["NVDA_2025_Q3_earnings_call_transcript"],
        is_unanswerable=False,
        category="executive_attribution"
    ),
    EvaluationQuery(
        query_id="eval-009",
        query="What did Amy Hood identify as the primary constraint on Azure growth in Q4 FY24?",
        ticker="MSFT",
        expected_answer="Amy Hood identified server availability and power infrastructure timing as the primary constraints, noting demand was ahead of capacity.",
        ground_truth_chunk_ids=["MSFT_2024_Q4_earnings_call_transcript"],
        is_unanswerable=False,
        category="executive_attribution"
    ),

    # 3. Multi-Quarter / Cross-Sectional Synthesis
    EvaluationQuery(
        query_id="eval-010",
        query="How did NVIDIA's non-GAAP gross margin guidance change from Q3 FY25 to Q4 FY25?",
        ticker="NVDA",
        expected_answer="Q3 gross margin was guided to 75.0% (and delivered 75.1%), while Q4 gross margin was guided to 73.5% reflecting initial Blackwell ramp costs.",
        ground_truth_chunk_ids=["NVDA_2025_Q2_earnings_call_transcript", "NVDA_2025_Q3_earnings_call_transcript"],
        is_unanswerable=False,
        category="comparative_synthesis"
    ),

    # 4. Strict Unanswerable Queries (Hallucination Defense)
    EvaluationQuery(
        query_id="eval-011",
        query="What was NVIDIA's official gross margin guidance for fiscal year 2029?",
        ticker="NVDA",
        expected_answer="INFORMATION_NOT_DISCLOSED_IN_FILINGS",
        ground_truth_chunk_ids=[],
        is_unanswerable=True,
        category="unanswerable_rejection"
    ),
    EvaluationQuery(
        query_id="eval-012",
        query="Did Apple announce an acquisition of a semiconductor foundry in their Q4 conference call?",
        ticker="AAPL",
        expected_answer="INFORMATION_NOT_DISCLOSED_IN_FILINGS",
        ground_truth_chunk_ids=[],
        is_unanswerable=True,
        category="unanswerable_rejection"
    )
]
