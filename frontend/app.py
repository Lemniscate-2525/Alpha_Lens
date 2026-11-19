"""
AlphaLens Main Streamlit Application Entrypoint.
Production-grade financial RAG dashboard for earnings calls and SEC Form 10-Q filings.
Operates exclusively as a client consuming the deployed FastAPI backend on Render.
Rule: Zero backend model imports, no local RAG initialization, no long hyphens.
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
from frontend.api_client import api_client, APIConnectionError, APIAuthenticationError


st.set_page_config(
    page_title="AlphaLens: Financial RAG System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=20, show_spinner=False)
def check_backend_status():
    """Lightweight availability probe with 20-second cache to prevent repeat pings."""
    try:
        health = api_client.health_check()
        return {"online": True, "health": health, "error": None}
    except APIConnectionError as e:
        return {"online": False, "health": None, "error": f"Backend offline: {e.message}"}
    except APIAuthenticationError as e:
        return {"online": False, "health": None, "error": f"Authentication failed: {e.message}"}
    except Exception as e:
        return {"online": False, "health": None, "error": str(e)}


def main():
    # Probe backend health
    status_info = check_backend_status()
    is_online = status_info["online"]
    health_data = status_info.get("health") or {}

    badge_html = (
        '<span class="badge-green">ENGINE ONLINE</span>'
        if is_online
        else '<span class="badge-amber">BACKEND SLEEPING / COLD START</span>'
    )
    status_sub = (
        f"Backend: Render ({api_client.base_url}) | Redis: {'Connected' if health_data.get('redis_connected') else 'Pending'}"
        if is_online
        else "Render free-tier service is spinning up. The UI is fully functional; queries will connect once live."
    )

    # Brand Header Banner
    st.markdown(
        f"""<div class="brand-header">
            <div>
                <h1 class="brand-title">ALPHALENS</h1>
                <div class="brand-subtitle">Production Financial RAG Engine for SEC Filings & Earnings Call Transcripts</div>
            </div>
            <div style="text-align: right;">
                {badge_html}
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{status_sub}</div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    if not is_online:
        st.warning(
            f"**Backend Connection Notice:** Could not reach the API at `{api_client.base_url}`. "
            "Render web services spin down during inactivity and take ~30-50 seconds to boot on first request. "
            "Please wait a few moments and click **Refresh Status** in the sidebar."
        )

    # Sidebar Navigation and Configuration
    with st.sidebar:
        st.markdown("### Cloud Architecture")
        backend_env = health_data.get("environment", "production").upper()
        chunks_count = health_data.get("indexed_chunks", "N/A")
        st.markdown(
            f"""<div style="background:#1e293b; border:1px solid #334155; border-radius:6px; padding:12px; margin-bottom:14px; font-size:12px;">
                <div style="margin-bottom:6px;"><strong>API Host:</strong> <span class="badge-blue">{api_client.base_url.split('://')[-1]}</span></div>
                <div style="margin-bottom:6px;"><strong>Status:</strong> {'Healthy' if is_online else 'Connecting'}</div>
                <div style="margin-bottom:6px;"><strong>Environment:</strong> {backend_env}</div>
                <div style="margin-bottom:6px;"><strong>Indexed Chunks:</strong> {chunks_count}</div>
                <div><strong>Inference Mode:</strong> Server-Side Remote RAG</div>
            </div>""",
            unsafe_allow_html=True
        )

        if st.button("Refresh Backend Status", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("### Target SLA Thresholds")
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

    # Main Dashboard Tabs
    tab_query, tab_filings, tab_telemetry, tab_benchmarks = st.tabs([
        "Terminal Query",
        "Filings Browser",
        "SLA Telemetry",
        "Golden Benchmarks"
    ])

    with tab_query:
        render_query_interface(client=api_client)

    with tab_filings:
        render_filings_browser(client=api_client)

    with tab_telemetry:
        render_system_telemetry(client=api_client)

    with tab_benchmarks:
        render_benchmark_view(client=api_client)


if __name__ == "__main__":
    main()
