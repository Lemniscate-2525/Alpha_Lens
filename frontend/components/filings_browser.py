"""
Filings & Corpus Browser Component for AlphaLens.
Enables inspecting indexed SEC filings, earnings transcripts,
uploading new filings via backend API, and monitoring ingestion pipeline jobs.
Rule: Zero local data/raw access, pure API communication, no long hyphens.
"""

import streamlit as st
import json
from frontend.styles import divider_html
from frontend.api_client import (
    api_client as default_api_client,
    AlphaLensAPIError,
    APIConnectionError,
    APIAuthenticationError
)


def render_filings_browser(client=None):
    api = client or default_api_client
    st.subheader("Corpus & Document Ingestion Management")
    st.caption("Inspect indexed SEC filings, submit new documents, and monitor distributed ingestion jobs.")

    # Top Action Buttons
    tcol1, tcol2 = st.columns([4, 1])
    with tcol2:
        if st.button("Refresh Corpus", use_container_width=True):
            st.rerun()

    # Fetch Documents and Jobs from Backend
    try:
        documents = api.list_documents()
        jobs_data = api.list_jobs(offset=0, limit=20)
        jobs = jobs_data.get("jobs", [])
    except APIConnectionError as e:
        st.error(f"Backend Unavailable: {e.message}")
        st.info("The backend service on Render may be sleeping or connecting to Redis.")
        return
    except APIAuthenticationError as e:
        st.error(f"Authentication Failed: {e.message}")
        return
    except Exception as e:
        st.error(f"Failed to fetch documents from backend: {str(e)}")
        return

    # Corpus Metrics Row
    unique_tickers = list(set(d.get("ticker", "N/A") for d in documents))
    total_child_chunks = sum(int(d.get("total_child_chunks", 0)) for d in documents)
    active_jobs = [j for j in jobs if j.get("status") in ["queued", "processing", "retrying"]]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Indexed Documents</div>
                <div class="metric-value">{len(documents)}</div>
                <div class="metric-delta">From Render Vector Store</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Covered Tickers</div>
                <div class="metric-value">{', '.join(unique_tickers) if unique_tickers else 'None'}</div>
                <div class="metric-delta">SEC & Transcript Records</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Index Chunks</div>
                <div class="metric-value">{total_child_chunks}</div>
                <div class="metric-delta">Parent-Child Hierarchy</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Active Queue Tasks</div>
                <div class="metric-value">{len(active_jobs)}</div>
                <div class="metric-delta">Redis Stream Workers</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(divider_html(), unsafe_allow_html=True)

    # Document Management Tabs
    tab1, tab2, tab3 = st.tabs(["Indexed Filings Directory", "Upload New Document", "Ingestion Queue History"])

    with tab1:
        st.markdown("#### Indexed Documents in Production Corpus")
        if not documents:
            st.info("No documents currently indexed in backend vector store. Upload a document below.")
        else:
            for doc in documents:
                doc_id = doc.get("document_id", "Unknown")
                ticker = doc.get("ticker", "N/A")
                status_str = doc.get("status", "unknown").upper()
                badge_class = "badge-green" if status_str == "COMPLETED" else "badge-blue"

                with st.expander(f"{ticker} | {doc_id} [{status_str}]", expanded=False):
                    dcol1, dcol2, dcol3, dcol4 = st.columns([2, 2, 2, 1.5])
                    with dcol1:
                        st.write(f"**Document ID:** `{doc_id}`")
                        st.write(f"**Ticker:** {ticker}")
                    with dcol2:
                        st.write(f"**Parent Chunks:** {doc.get('total_parent_chunks', 0)}")
                        st.write(f"**Child Chunks:** {doc.get('total_child_chunks', 0)}")
                    with dcol3:
                        st.write(f"**Index Status:** <span class='{badge_class}'>{status_str}</span>", unsafe_allow_html=True)
                        st.write(f"**Created:** {doc.get('created_at', 'N/A')[:19] if doc.get('created_at') else 'N/A'}")
                    with dcol4:
                        if st.button("Delete Doc", key=f"del_{doc_id}", type="secondary"):
                            try:
                                res = api.delete_document(doc_id)
                                st.success(f"Deleted {doc_id} ({res.get('chunks_deleted', 0)} chunks removed).")
                                st.rerun()
                            except Exception as del_err:
                                st.error(f"Deletion failed: {del_err}")

    with tab2:
        st.markdown("#### Submit Document for Ingestion")
        st.caption("Upload raw JSON filings (e.g. 10-Q filing or earnings call transcript) to enqueue onto the Redis background stream.")
        
        uploaded_file = st.file_uploader("Select Financial Filing JSON File:", type=["json", "txt"])
        raw_text_input = st.text_area("Or Paste Raw Document JSON Payload:", height=150, placeholder='{"ticker": "NVDA", "company_name": "NVIDIA", "fiscal_quarter": "Q3", ...}')

        if st.button("Submit to Ingestion Pipeline", type="primary", use_container_width=True):
            payload_str = ""
            if uploaded_file is not None:
                payload_str = uploaded_file.getvalue().decode("utf-8")
            elif raw_text_input.strip():
                payload_str = raw_text_input.strip()

            if not payload_str:
                st.warning("Please select a file or paste JSON document text before submitting.")
            else:
                try:
                    with st.spinner("Submitting document to Redis queue on Render..."):
                        job_res = api.upload_document_json(payload_str)
                        st.success(f"Document accepted! Job ID: {job_res.get('job_id')} (Status: {job_res.get('status')})")
                        st.info("The background worker is chunking, embedding, and updating shared indexes asynchronously.")
                except AlphaLensAPIError as e:
                    st.error(f"Ingestion submission error: {e.message}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    with tab3:
        st.markdown("#### Redis Stream Background Job Registry")
        if not jobs:
            st.info("No recent ingestion jobs found in registry.")
        else:
            table_rows = []
            for j in jobs:
                table_rows.append({
                    "Job ID": j.get("job_id", ""),
                    "Document": j.get("document_id", ""),
                    "Ticker": j.get("ticker", ""),
                    "Status": j.get("status", "").upper(),
                    "Stage": j.get("stage", ""),
                    "Progress": f"{j.get('progress', 0)}%",
                    "Child Chunks": j.get("total_child_chunks", 0),
                    "Duration (ms)": f"{j.get('processing_time_ms', 0):.1f}"
                })
            st.table(table_rows)
