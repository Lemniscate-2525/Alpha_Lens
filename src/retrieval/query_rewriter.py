"""
AlphaLens - Query Rewriter & Metadata Filter Extractor
Expands financial acronyms and automatically infers ticker, temporal, and role metadata filters.
"""
import re
from typing import Dict, Optional, Tuple
from src.core.models import SpeakerRole


class FinancialQueryRewriter:
    """Pre-processes raw financial user queries for enhanced retrieval recall."""

    ACRONYM_MAP = {
        "eps": "earnings per share diluted EPS",
        "capex": "capital expenditures",
        "csp": "cloud service providers hyperscalers",
        "csp's": "cloud service providers hyperscalers",
        "tco": "total cost of ownership",
        "gm": "gross margin percentage",
        "opex": "operating expenses",
        "rca": "root cause analysis",
        "fcf": "free cash flow",
    }

    TICKER_SET = {"NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA"}

    @classmethod
    def expand_query(cls, query: str) -> str:
        """Expands common financial abbreviations without distorting semantic meaning."""
        tokens = query.split()
        expanded_tokens = []
        for token in tokens:
            cleaned = token.lower().strip("?,.!")
            if cleaned in cls.ACRONYM_MAP:
                expanded_tokens.append(f"{token} ({cls.ACRONYM_MAP[cleaned]})")
            else:
                expanded_tokens.append(token)
        return " ".join(expanded_tokens)

    @classmethod
    def extract_metadata_filters(cls, query: str) -> Dict[str, Optional[str]]:
        """
        Extracts implicit metadata filters from natural language queries.
        E.g., "What did NVDA's CFO say about Q3 gross margin in 2025?"
        Returns: {'ticker': 'NVDA', 'fiscal_quarter': 'Q3', 'fiscal_year': '2025', 'speaker_role': 'CFO'}
        """
        filters: Dict[str, Optional[str]] = {
            "ticker": None,
            "fiscal_quarter": None,
            "fiscal_year": None,
            "speaker_role": None,
        }

        # Ticker detection
        words = re.findall(r"\b[A-Za-z]{2,5}\b", query)
        for w in words:
            upper_w = w.upper()
            if upper_w in cls.TICKER_SET:
                filters["ticker"] = upper_w
                break

        # Also detect company names
        query_lower = query.lower()
        if not filters["ticker"]:
            if "nvidia" in query_lower:
                filters["ticker"] = "NVDA"
            elif "microsoft" in query_lower:
                filters["ticker"] = "MSFT"
            elif "apple" in query_lower:
                filters["ticker"] = "AAPL"

        # Fiscal Quarter detection
        q_match = re.search(r"\b(Q[1-4])\b", query, re.IGNORECASE)
        if q_match:
            filters["fiscal_quarter"] = q_match.group(1).upper()

        # Fiscal Year detection (e.g. 2024, 2025, FY24, FY25)
        fy_match = re.search(r"\b(202[3-6]|FY2[3-6])\b", query, re.IGNORECASE)
        if fy_match:
            val = fy_match.group(1).upper()
            if val.startswith("FY"):
                val = f"20{val[2:]}"
            filters["fiscal_year"] = val

        # Speaker Role detection
        if "cfo" in query_lower or "colette" in query_lower or "amy hood" in query_lower:
            filters["speaker_role"] = SpeakerRole.CFO.value
        elif "ceo" in query_lower or "jensen" in query_lower or "satya" in query_lower:
            filters["speaker_role"] = SpeakerRole.CEO.value

        return filters

