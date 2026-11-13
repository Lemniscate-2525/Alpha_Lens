"""
AlphaLens Main Streamlit Application Entrypoint.
Production-grade financial RAG dashboard for earnings calls and SEC Form 10-Q filings.
Rule: Strict compliance with no long hyphens rule (no markdown hyphens, no em-dashes).
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.styles import CUSTOM_CSS, divider_html
from frontend.components.query_interface import render_query_interface
from frontend.components.filings_browser import render_filings_browser
from frontend.components.system_telemetry import render_system_telemetry
from frontend.components.benchmark_view import render_benchmark_view
from src.config import settings


st.set_page_config(
    page_title="AlphaLens: Financial RAG System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing AlphaLens Production RAG Engine...")
def get_rag_service():
    """Initializes and caches the singleton RAG service instance."""
    from src.services.rag_service import rag_service
    from src.ingestion.pipeline import ingestion_pipeline
    
    # Pre-index raw data if not already indexed
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        ingestion_pipeline.ingest_directory(raw_dir)
        
    return rag_service


def main():
    # Brand Header Banner
    st.markdown(
        f"""<div class="brand-header">
            <div>
                <h1 class="brand-title">ALPHALENS</h1>
                <div class="brand-subtitle">Production Financial RAG Engine for SEC Filings & Earnings Call Transcripts</div>
            </div>
            <div style="text-align: right;">
                <span class="badge-green">ENGINE ONLINE</span>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Provider: {settings.LLM_PROVIDER.upper()} ({settings.OLLAMA_MODEL_NAME if settings.LLM_PROVIDER == 'ollama' else settings.GROQ_MODEL_NAME})</div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    # Sidebar Navigation and Configuration
    with st.sidebar:
        st.markdown("### Architecture Stack")
        st.markdown(
            f"""<div style="background:#1e293b; border:1px solid #334155; border-radius:6px; padding:12px; margin-bottom:14px; font-size:12px;">
                <div style="margin-bottom:6px;"><strong>Inference:</strong> <span class="badge-blue">{settings.LLM_PROVIDER.upper()}</span></div>
                <div style="margin-bottom:6px;"><strong>Embeddings:</strong> {settings.EMBEDDING_MODEL_NAME}</div>
                <div style="margin-bottom:6px;"><strong>Reranker:</strong> {settings.RERANKER_MODEL_NAME}</div>
                <div style="margin-bottom:6px;"><strong>Vector DB:</strong> {settings.VECTOR_STORE_PROVIDER.upper()}</div>
                <div><strong>Cache:</strong> L1 Exact + L2 Semantic (0.94)</div>
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("### System SLA Thresholds")
        st.markdown(
            """<div style="background:#1e293b; border:1px solid #334155; border-radius:6px; padding:12px; margin-bottom:14px; font-size:12px;">
                <div style="margin-bottom:4px;">Retrieval P95: <strong>&lt; 150ms</strong></div>
                <div style="margin-bottom:4px;">Rerank P95: <strong>&lt; 80ms</strong></div>
                <div style="margin-bottom:4px;">End-to-End P95: <strong>&lt; 500ms</strong></div>
                <div>Error Budget: <strong>&lt; 0.1%</strong></div>
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown(divider_html(), unsafe_allow_html=True)
        st.caption("AlphaLens AI Systems Engineering | Flagship Portfolio")

    # Load Core Service
    rag_service = get_rag_service()

    # Main Dashboard Tabs
    tab_query, tab_filings, tab_telemetry, tab_benchmarks = st.tabs([
        "Terminal Query",
        "Filings Browser",
        "SLA Telemetry",
        "Golden Benchmarks"
    ])

    with tab_query:
        render_query_interface(rag_service)

    with tab_filings:
        render_filings_browser(rag_service)

    with tab_telemetry:
        render_system_telemetry(rag_service)

    with tab_benchmarks:
        render_benchmark_view(rag_service)


if __name__ == "__main__":
    main()
