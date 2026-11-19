"""
System Telemetry & Reliability Dashboard Component for AlphaLens.
Visualizes real-time latency percentiles (P50, P90, P95, P99), cache hit distributions,
backend cluster health, and Redis state retrieved directly from the Render API.
Rule: Zero backend model or module imports, no simulated data, strict compliance with no long hyphens rule.
"""

import streamlit as st
from frontend.styles import divider_html
from frontend.api_client import (
    api_client as default_api_client,
    AlphaLensAPIError,
    APIConnectionError,
    APIAuthenticationError
)


def render_system_telemetry(client=None):
    api = client or default_api_client
    st.subheader("Production Observability & SLA Monitor")
    st.caption("Live systems telemetry, latency percentiles, and SLA compliance retrieved from the deployed backend.")

    # Refresh Button
    rcol1, rcol2 = st.columns([5, 1])
    with rcol2:
        if st.button("Refresh Telemetry", use_container_width=True):
            st.rerun()

    # Query Backend Telemetry Endpoints
    try:
        stats = api.get_metrics()
        health = api.health_check()
    except APIConnectionError as e:
        st.error(f"Backend Offline: {e.message}")
        st.info("Metrics are unavailable while the backend container is offline.")
        return
    except APIAuthenticationError as e:
        st.error(f"Authentication Error: {e.message}")
        return
    except Exception as e:
        st.error(f"Failed to retrieve telemetry: {str(e)}")
        return

    # Latency Percentiles Row
    st.markdown("#### Latency Percentiles (SLA Targets)")
    col1, col2, col3, col4 = st.columns(4)

    p50 = stats.get("p50_latency_ms", 0.0)
    p90 = stats.get("p90_latency_ms", 0.0)
    p95 = stats.get("p95_latency_ms", 0.0)
    p99 = stats.get("p99_latency_ms", 0.0)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P50 Median Latency</div>
                <div class="metric-value">{p50:.1f}ms</div>
                <div class="metric-delta">Target: &lt;150ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P90 Latency</div>
                <div class="metric-value">{p90:.1f}ms</div>
                <div class="metric-delta">Target: &lt;350ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P95 Latency</div>
                <div class="metric-value">{p95:.1f}ms</div>
                <div class="metric-delta">Target: &lt;500ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">P99 Tail Latency</div>
                <div class="metric-value">{p99:.1f}ms</div>
                <div class="metric-delta">Target: &lt;1200ms</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(divider_html(), unsafe_allow_html=True)

    # Cache and Reliability Row
    col_a, col_b, col_c = st.columns(3)

    cache_hits = stats.get("cache_hits", 0)
    cache_misses = stats.get("cache_misses", 0)
    total_reqs = stats.get("total_requests", 0)
    current_rps = stats.get("current_rps", 0.0)
    total_cost = stats.get("total_cost_usd", 0.0)
    jobs_comp = stats.get("ingestion_jobs_completed", 0)
    jobs_fail = stats.get("ingestion_jobs_failed", 0)

    with col_a:
        st.markdown("#### Two-Tier Cache Performance")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Cache Hits</div>
                <div class="metric-value">{cache_hits}</div>
                <div class="metric-delta">L1 Exact + L2 Semantic (sim &gt;= 0.94)</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Cache Misses</div>
                <div class="metric-value">{cache_misses}</div>
                <div class="metric-delta">Total API Requests: {total_reqs}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown("#### Cloud Backend Cluster Status")
        status_val = health.get("status", "unknown").upper()
        redis_ok = health.get("redis_connected", False)
        status_badge = "badge-green" if status_val == "HEALTHY" else "badge-amber"
        redis_badge = "badge-green" if redis_ok else "badge-purple"

        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">API Health</div>
                <div class="metric-value"><span class="{status_badge}">{status_val}</span></div>
                <div class="metric-delta">Environment: {health.get('environment', 'production').upper()}</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Redis Connection</div>
                <div class="metric-value"><span class="{redis_badge}">{'CONNECTED' if redis_ok else 'DISCONNECTED'}</span></div>
                <div class="metric-delta">Rate Limiting & Singleflight Mutex Active</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col_c:
        st.markdown("#### Cost & Ingestion Governance")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Cost Incurred</div>
                <div class="metric-value">${total_cost:.5f}</div>
                <div class="metric-delta">Current Throughput: {current_rps:.2f} RPS</div>
            </div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Ingestion Jobs Processed</div>
                <div class="metric-value">{jobs_comp}</div>
                <div class="metric-delta">Failed Jobs: {jobs_fail} | Indexed Chunks: {health.get('indexed_chunks', 0)}</div>
            </div>""",
            unsafe_allow_html=True
        )
