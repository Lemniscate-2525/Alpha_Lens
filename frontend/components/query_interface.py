"""
Query Interface Component for AlphaLens.
Provides interactive financial question answering, pipeline mode toggles,
citation badges, and source chunk inspection.
Note: Strict compliance with no long hyphens rule.
"""

import streamlit as st
import time
from typing import Dict, Any, List
from frontend.styles import divider_html
from frontend.api_client import (
    api_client as default_api_client,
    APIConnectionError,
    APIAuthenticationError,
    APIRateLimitError,
    APITimeoutError,
    AlphaLensAPIError
)


def render_query_interface(client=None):
    api = client or default_api_client
    st.subheader("Financial Intelligence Terminal")
    st.caption("Ask grounded questions over 10-Q SEC filings and earnings call transcripts.")

    # Top Control Bar
    col1, col2, col3 = st.columns([1.5, 2, 2.5])

    with col1:
        ticker = st.selectbox(
            "Company Filter",
            options=["ALL", "NVDA", "MSFT"],
            index=0,
            help="Filter search scope to a specific company ticker"
        )

    with col2:
        doc_type = st.selectbox(
            "Document Scope",
            options=["All Filings", "10-Q Reports", "Earnings Transcripts"],
            index=0
        )

    with col3:
        retrieval_strategy = st.selectbox(
            "Retrieval Strategy",
            options=[
                "Hybrid (Dense + BM25) + Reranker",
                "Dense Vector Only",
                "BM25 Sparse Only"
            ],
            index=0,
            help="Select the retrieval and ranking algorithm"
        )

    # Preset Financial Queries
    preset_col1, preset_col2 = st.columns([1, 1])
    sample_queries = [
        "What was NVIDIA Q2 FY25 total revenue and GAAP gross margin?",
        "What did Colette Kress say about Blackwell mask changes and production ramp?",
        "What Azure AI capacity constraints did Amy Hood mention for Microsoft Q4 FY24?",
        "What were Tesla vehicle deliveries in Q3? (Adversarial Test)"
    ]

    selected_sample = st.selectbox(
        "Or select a sample audit question:",
        options=["(Type custom question below)"] + sample_queries,
        index=0
    )

    default_text = "" if selected_sample == "(Type custom question below)" else selected_sample
    user_query = st.text_area(
        "Enter Financial Query:",
        value=default_text,
        height=90,
        placeholder="e.g. Compare NVIDIA Data Center growth with Microsoft Azure cloud metrics..."
    )

    run_button = st.button("Execute Financial Query", type="primary", use_container_width=True)

    st.markdown(divider_html(), unsafe_allow_html=True)

    if run_button and user_query.strip():
        with st.spinner("Executing retrieval, cross-attention reranking, and grounded generation..."):
            start_time = time.perf_counter()
            
            # Map ticker filter
            ticker_param = None if ticker == "ALL" else ticker
            use_reranker = "Reranker" in retrieval_strategy

            # Execute query via remote backend API
            try:
                response = api.execute_query(
                    query=user_query.strip(),
                    ticker=ticker_param,
                    use_reranker=use_reranker,
                    use_semantic_cache=True
                )
                exec_time = (time.perf_counter() - start_time) * 1000.0

                # Render Telemetry Summary
                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                
                with mcol1:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Total Latency</div>
                            <div class="metric-value">{response.total_latency_ms:.1f}ms</div>
                            <div class="metric-delta">P95 Target: &lt;500ms</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mcol2:
                    cache_status = "L1 Exact Hit" if response.cache_hit_type == "exact" else ("L2 Semantic Hit" if response.cache_hit_type == "semantic" else "Cache Miss")
                    badge_class = "badge-green" if response.is_cache_hit else "badge-purple"
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Cache Status</div>
                            <div class="metric-value"><span class="{badge_class}">{cache_status}</span></div>
                            <div class="metric-delta">Two-Tier Redis State</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mcol3:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Retrieval Latency</div>
                            <div class="metric-value">{response.retrieval_latency_ms:.1f}ms</div>
                            <div class="metric-delta">Hybrid RRF k=60</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mcol4:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Rerank Latency</div>
                            <div class="metric-value">{response.rerank_latency_ms:.1f}ms</div>
                            <div class="metric-delta">CrossEncoder 6-layer</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                with mcol5:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Query Cost</div>
                            <div class="metric-value">${response.estimated_cost_usd:.5f}</div>
                            <div class="metric-delta">Tokens: {response.tokens_used}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                # Render Answer
                st.markdown("### Synthesized Financial Answer")
                st.markdown(
                    f"""<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-top: 8px; font-size: 15px; line-height: 1.6;">
                        {response.answer}
                    </div>""",
                    unsafe_allow_html=True
                )

                # Citations
                if response.citations:
                    st.markdown("#### Citations & Audit Trail")
                    for cit in response.citations:
                        st.markdown(
                            f"""<div style="margin-bottom: 6px;">
                                <span class="citation-pill">[{cit.citation_id}]</span>
                                <strong>{cit.document_title}</strong> (Score: {cit.confidence_score:.3f})
                                <div style="font-size: 12px; color: #94a3b8; margin-left: 28px;">Quote: "{cit.snippet_text[:160]}..."</div>
                            </div>""",
                            unsafe_allow_html=True
                        )

                # Source Chunks Inspector
                st.markdown(divider_html(), unsafe_allow_html=True)
                with st.expander(f"View Retrieved Context Chunks ({len(response.context_chunks)} chunks retrieved)", expanded=False):
                    for i, chunk in enumerate(response.context_chunks, 1):
                        speaker_str = f"Speaker: {chunk.speaker} ({chunk.role})" if chunk.speaker else f"Section: {chunk.section}"
                        st.markdown(
                            f"""<div class="chunk-box">
                                <div class="chunk-header">
                                    <span><strong>Chunk #{i}</strong> | {chunk.ticker} {chunk.fiscal_period or ''} | {chunk.doc_type}</span>
                                    <span>Score: {chunk.score:.4f}</span>
                                </div>
                                <div style="font-size: 11px; color: #38bdf8; margin-bottom: 6px;">{speaker_str}</div>
                                <div class="chunk-text">{chunk.content}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )

            except APIConnectionError as e:
                st.error(f"Backend Offline or Cold-Starting: {e.message}")
                st.info("If deploying on Render free tier, the web service spins down after inactivity and may take ~30-50s to wake up.")
            except APIAuthenticationError as e:
                st.error(f"Authentication Error: {e.message}")
            except APIRateLimitError as e:
                st.warning(f"Rate Limit Exceeded: {e.message}")
            except APITimeoutError as e:
                st.warning(f"Request Timeout: {e.message}")
            except AlphaLensAPIError as e:
                st.error(f"API Error: {e.message}")
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

