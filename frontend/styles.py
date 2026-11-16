"""
Custom CSS styles and UI helper components for AlphaLens Streamlit frontend.
Rules: No long hyphens (no em-dash, no triple hyphens). Clean modern financial terminal styling.
"""

CUSTOM_CSS = """
<style>
/* AlphaLens Financial Terminal Theme */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
}

/* Header Banner */
.brand-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.brand-title {
    font-size: 26px;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: -0.5px;
    margin: 0;
}

.brand-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Metric Cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    font-weight: 600;
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}

.metric-delta {
    font-size: 12px;
    color: #10b981;
    margin-top: 2px;
}

/* Badges */
.badge-green {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
}

.badge-blue {
    background-color: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
}

.badge-purple {
    background-color: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
}

.badge-amber {
    background-color: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
}

/* Custom Divider (No long hyphens) */
.custom-divider {
    height: 1px;
    background: #334155;
    margin: 20px 0;
    border: none;
}

/* Citation Pill */
.citation-pill {
    background-color: #0369a1;
    color: #e0f2fe;
    font-size: 11px;
    padding: 2px 7px;
    border-radius: 10px;
    font-weight: 700;
    display: inline-block;
    margin-right: 6px;
}

/* Context Chunk Card */
.chunk-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 12px;
    color: #94a3b8;
}

.chunk-text {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    line-height: 1.5;
    color: #cbd5e1;
}

/* Terminal code style */
.code-block {
    background-color: #030712;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #38bdf8;
    overflow-x: auto;
}
</style>
"""

def divider_html() -> str:
    """Returns a clean HTML divider without using markdown hyphens."""
    return '<div class="custom-divider"></div>'

