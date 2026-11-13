"""
Filings & Corpus Browser Component for AlphaLens.
Enables inspecting indexed SEC filings, earnings transcripts,
parent-child chunking hierarchy, and raw financial tables.
Note: Strict compliance with no long hyphens rule.
"""

import streamlit as st
import json
from pathlib import Path
from frontend.styles import divider_html


def render_filings_browser(rag_service):
    st.subheader("Corpus & Document Hierarchy")
    st.caption("Inspect raw SEC filings, earnings call transcripts, and parent-child chunk trees.")

    raw_dir = Path("data/raw")
    json_files = list(raw_dir.glob("*.json")) if raw_dir.exists() else []

    if not json_files:
        st.warning("No raw financial documents found in data/raw.")
        return

    # Document selection
    file_options = {f.name: f for f in json_files}
    selected_name = st.selectbox("Select Financial Filing:", list(file_options.keys()))
    selected_path = file_options[selected_name]

    with open(selected_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    # Document metadata summary
    metadata = doc_data.get("metadata", {})
    ticker = metadata.get("ticker", "N/A")
    period = metadata.get("fiscal_period", "N/A")
    year = metadata.get("fiscal_year", "N/A")
    doc_type = metadata.get("doc_type", "N/A")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Ticker</div>
                <div class="metric-value">{ticker}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Period</div>
                <div class="metric-value">{period} {year}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Type</div>
                <div class="metric-value">{doc_type}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col4:
        item_count = len(doc_data.get("sections", [])) or len(doc_data.get("financial_statements", []))
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Items / Sections</div>
                <div class="metric-value">{item_count}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(divider_html(), unsafe_allow_html=True)

    # Tab views for Document Content
    tab1, tab2 = st.tabs(["Structured Content Viewer", "Raw JSON Payload"])

    with tab1:
        if "sections" in doc_data:
            st.markdown("#### Transcript Speaker Turns & Sections")
            for sec in doc_data["sections"]:
                speaker = sec.get("speaker")
                role = sec.get("role", "")
                header_title = f"{speaker} ({role})" if speaker else sec.get("title", "Section")
                
                with st.expander(f"{header_title} | {sec.get('section_type', 'Content')}"):
                    st.write(sec.get("content", ""))
                    if "tables" in sec and sec["tables"]:
                        st.markdown("**Embedded Financial Tables:**")
                        for t in sec["tables"]:
                            st.markdown(f"*Table: {t.get('table_title', '')}*")
                            st.markdown(t.get("table_markdown", ""))

        elif "financial_statements" in doc_data:
            st.markdown("#### SEC Form 10-Q Financial Tables")
            for stmt in doc_data["financial_statements"]:
                with st.expander(f"Statement: {stmt.get('statement_name')}", expanded=True):
                    st.markdown(stmt.get("table_markdown", ""))

    with tab2:
        st.json(doc_data)
