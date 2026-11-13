"""
System Telemetry & Reliability Dashboard Component for AlphaLens.
Visualizes latency percentiles (P50, P90, P95, P99), cache hit distributions,
circuit breaker state, and rate limit token buckets.
Note: Strict compliance with no long hyphens rule.
"""

import streamlit as st
from frontend.styles import divider_html
from src.observability.metrics import metrics_collector
from src.services.circuit_breaker import llm_circuit_breaker


def render_system_telemetry(rag_service):
    st.subheader("Production Observability & SLA Monitor")
    st.caption("Real-time distributed systems health, latency percentiles, and fault-tolerance status.")

    stats = metrics_collector.get_summary()

    # Latency Percentiles Row
    st.markdown("#### Latency Percentiles (SLA Targets)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P50 Median Latency</div>
                <div class="metric-value">{stats['p50_latency_ms']:.1f}ms</div>
                <div class="metric-delta">Target: &lt;150ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P90 Latency</div>
                <div class="metric-value">{stats['p90_latency_ms']:.1f}ms</div>
                <div class="metric-delta">Target: &lt;350ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P95 Latency</div>
                <div class="metric-value">{stats['p95_latency_ms']:.1f}ms</div>
                <div class="metric-delta">Target: &lt;500ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P99 Tail Latency</div>
                <div class="metric-value">{stats['p99_latency_ms']:.1f}ms</div>
                <div class="metric-delta">Target: &lt;1200ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(divider_html(), unsafe_allow_html=True)

    # Cache and Reliability Row
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### Two-Tier Cache Performance")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Cache Hits</div>
                <div class="metric-value">{stats['cache_hits']}</div>
                <div class="metric-delta">L1 Exact + L2 Semantic (sim &gt;= 0.94)</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Cache Misses</div>
                <div class="metric-value">{stats['cache_misses']}</div>
                <div class="metric-delta">Full pipeline evaluation triggered</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown("#### LLM Circuit Breaker Status")
        cb_state = llm_circuit_breaker.state.value.upper()
        badge_style = "badge-green" if cb_state == "CLOSED" else ("badge-amber" if cb_state == "HALF-OPEN" else "badge-purple")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Circuit State</div>
                <div class="metric-value"><span class="{badge_style}">{cb_state}</span></div>
                <div class="metric-delta">Failures: {llm_circuit_breaker.failure_count} / Threshold: {llm_circuit_breaker.failure_threshold}</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Recovery Timeout</div>
                <div class="metric-value">{llm_circuit_breaker.recovery_timeout}s</div>
                <div class="metric-delta">Automatic half-open state transition</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col_c:
        st.markdown("#### Cost & Financial Governance")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Cost Incurred</div>
                <div class="metric-value">${stats['total_cost_usd']:.5f}</div>
                <div class="metric-delta">Local Ollama inference: $0.00</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Ingestion Jobs</div>
                <div class="metric-value">{stats['ingestion_jobs_completed']}</div>
                <div class="metric-delta">Errors: {stats['ingestion_jobs_failed']}</div>
            </div>""",
            unsafe_allow_html=True
        )
