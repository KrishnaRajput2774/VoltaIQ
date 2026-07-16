"""
VoltIQ Design System - Global Theme & CSS
All pages import get_base_css() to apply the premium dark enterprise theme.
"""

FONT_IMPORT = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
"""

BASE_CSS = """
<style>
/* ============================================================
   VOLTIQ DESIGN SYSTEM - DARK ENTERPRISE THEME
   ============================================================ */

/* ---------- RESET & BASE ---------- */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[class*="css"],
.stApp,
.stApp > header,
section[data-testid="stSidebar"],
.block-container {
    font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #F8FAFC;
}

/* ---------- BACKGROUND ---------- */
.stApp {
    background: #0B1120 !important;
}

.block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: #0D1526 !important;
    border-right: 1px solid #1E2D45 !important;
    min-width: 260px !important;
    width: 260px !important;
    /* Lock sidebar open — prevent Streamlit from collapsing it */
    transform: none !important;
    margin-left: 0 !important;
    left: 0 !important;
    visibility: visible !important;
    display: flex !important;
    position: relative !important;
}

section[data-testid="stSidebar"] > div {
    padding: 0 !important;
}

/* ---------- SIDEBAR NAVIGATION LINKS ---------- */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 24px 20px 20px 20px;
    border-bottom: 1px solid #1E2D45;
    margin-bottom: 8px;
}

.sidebar-logo-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #2563EB, #10B981);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.sidebar-brand-name {
    font-size: 20px;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
}

.sidebar-brand-tag {
    font-size: 10px;
    color: #475569;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.nav-section-label {
    font-size: 10px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 16px 20px 6px 20px;
}

/* Override default streamlit nav button styles */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] button[kind="secondary"] {
    width: 100%;
    text-align: left;
    background: transparent !important;
    border: none !important;
    color: #94A3B8 !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.15s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(37, 99, 235, 0.08) !important;
    color: #F8FAFC !important;
}

/* Streamlit radio used for navigation */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0 !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    color: #94A3B8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(37, 99, 235, 0.08) !important;
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
    background: rgba(37, 99, 235, 0.12) !important;
    color: #60A5FA !important;
    border-right: 2px solid #2563EB !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    flex-direction: column !important;
}

/* Hide default radio circle */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ---------- TOP HEADER BAR ---------- */
.voltiq-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 20px 0;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 24px;
}

.voltiq-page-title {
    font-size: 26px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}

.voltiq-page-subtitle {
    font-size: 14px;
    color: #64748B;
    font-weight: 400;
    margin-top: 3px;
    line-height: 1.4;
}

.topbar-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid;
}

.status-online {
    background: rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
    color: #10B981;
}

.status-offline {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.3);
    color: #EF4444;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---------- KPI CARDS ---------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--kpi-accent, #2563EB);
    opacity: 0.8;
}

.kpi-card:hover {
    border-color: #334155;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}

.kpi-card:hover::before {
    opacity: 1;
}

.kpi-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin-bottom: 14px;
    background: var(--kpi-icon-bg, rgba(37,99,235,0.15));
}

.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 6px;
}

.kpi-trend {
    font-size: 12px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 4px;
}

.trend-up { color: #10B981; }
.trend-down { color: #EF4444; }
.trend-neutral { color: #94A3B8; }

/* ---------- GLASS CARDS ---------- */
.glass-card {
    background: rgba(17, 24, 39, 0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: border-color 0.2s ease;
}

.glass-card:hover {
    border-color: #334155;
}

.glass-card-title {
    font-size: 15px;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.glass-card-title .card-icon {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}

/* ---------- STATUS BADGES ---------- */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.badge-success {
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.12);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

.badge-danger {
    background: rgba(239, 68, 68, 0.12);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.25);
}

.badge-info {
    background: rgba(37, 99, 235, 0.12);
    color: #60A5FA;
    border: 1px solid rgba(37, 99, 235, 0.25);
}

.badge-neutral {
    background: rgba(148, 163, 184, 0.08);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* ---------- ZONE CARDS ---------- */
.zone-card {
    border-radius: 12px;
    padding: 16px 18px;
    margin: 12px 0;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}

.zone-healthy {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.zone-attention {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
}

.zone-critical {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
}

.zone-icon {
    font-size: 22px;
    flex-shrink: 0;
    margin-top: 1px;
}

.zone-content-title {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 2px;
}

.zone-healthy .zone-content-title { color: #10B981; }
.zone-attention .zone-content-title { color: #F59E0B; }
.zone-critical .zone-content-title { color: #EF4444; }

.zone-content-desc {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 400;
}

/* ---------- INSIGHT / RECOMMENDATION CARDS ---------- */
.insight-card {
    background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(16,185,129,0.05));
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
}

.insight-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.insight-card-title {
    font-size: 14px;
    font-weight: 600;
    color: #60A5FA;
}

.insight-card-body {
    font-size: 13px;
    color: #94A3B8;
    line-height: 1.6;
}

/* ---------- SECTION HEADERS ---------- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}

.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1E293B, transparent);
}

.section-header-title {
    font-size: 13px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
}

/* ---------- DATA TABLE OVERRIDE ---------- */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}

[data-testid="stDataFrameResizable"] {
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
}

/* ---------- INPUT OVERRIDES ---------- */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0F172A !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
    font-size: 14px !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding: 0 4px !important;
}

/* ---------- BUTTON OVERRIDES ---------- */
.stButton > button {
    background: #1E293B !important;
    color: #94A3B8 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #334155 !important;
    color: #F8FAFC !important;
    border-color: #475569 !important;
}

.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}

.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #1E40AF, #1D4ED8) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ---------- METRIC OVERRIDES ---------- */
[data-testid="stMetric"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
}

[data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* ---------- TAB OVERRIDES ---------- */
[data-testid="stTabs"] [role="tablist"] {
    background: #111827 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #1E293B !important;
    gap: 2px !important;
}

[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1E293B !important;
    color: #F8FAFC !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}

/* ---------- EXPANDER OVERRIDES ---------- */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    color: #E2E8F0 !important;
    font-weight: 500 !important;
}

/* ---------- SELECT BOX OVERRIDES ---------- */
[data-testid="stSelectbox"] > div > div {
    background: #0F172A !important;
    border-color: #1E293B !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
}

/* ---------- SPINNER ---------- */
[data-testid="stSpinner"] > div {
    border-top-color: #2563EB !important;
}

/* ---------- INFO / SUCCESS / ERROR OVERRIDES ---------- */
[data-testid="stNotification"] {
    border-radius: 10px !important;
}

.stAlert {
    border-radius: 10px !important;
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0B1120; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ---------- DIVIDER ---------- */
hr {
    border: none !important;
    border-top: 1px solid #1E293B !important;
    margin: 24px 0 !important;
}

/* ---------- SIDEBAR STATUS FOOTER ---------- */
.sidebar-footer {
    padding: 16px 20px;
    border-top: 1px solid #1E2D45;
    margin-top: auto;
}

.sidebar-footer-text {
    font-size: 11px;
    color: #334155;
    text-align: center;
}

/* ---------- CHAT INTERFACE ---------- */
.chat-wrapper {
    background: #0D1526;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 24px;
    min-height: 420px;
    max-height: 520px;
    overflow-y: auto;
    margin-bottom: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.chat-msg-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
}

.chat-msg-row.user-row {
    flex-direction: row-reverse;
}

.chat-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    font-weight: 700;
}

.avatar-user {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    color: #fff;
    font-size: 13px;
}

.avatar-bot {
    background: linear-gradient(135deg, #047857, #10B981);
    color: #fff;
    font-size: 16px;
}

.chat-bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 14px;
    line-height: 1.6;
}

.bubble-user {
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(37, 99, 235, 0.25);
    color: #E2E8F0;
    border-radius: 14px 4px 14px 14px;
}

.bubble-bot {
    background: #111827;
    border: 1px solid #1E293B;
    color: #CBD5E1;
    border-radius: 4px 14px 14px 14px;
}

.chat-source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 8px;
}

.source-chip {
    background: rgba(37, 99, 235, 0.1);
    border: 1px solid rgba(37, 99, 235, 0.2);
    color: #60A5FA;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
}

.chat-input-area {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 4px 4px 4px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: border-color 0.2s;
}

.chat-input-area:focus-within {
    border-color: #2563EB;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
}

.suggested-questions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
}

.suggestion-chip {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid #1E293B;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #94A3B8;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
}

.suggestion-chip:hover {
    border-color: #2563EB;
    color: #60A5FA;
    background: rgba(37,99,235,0.08);
}

/* ---------- MODULE CARDS (HOME) ---------- */
.module-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.module-card::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    opacity: 0;
    transition: opacity 0.3s ease;
    background: radial-gradient(circle at top left, rgba(37,99,235,0.05), transparent 60%);
}

.module-card:hover {
    border-color: #334155;
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.5);
}

.module-card:hover::after {
    opacity: 1;
}

.module-icon-box {
    width: 50px;
    height: 50px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 16px;
}

.module-card-title {
    font-size: 17px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 8px;
    letter-spacing: -0.2px;
}

.module-card-desc {
    font-size: 13px;
    color: #64748B;
    line-height: 1.65;
    margin-bottom: 14px;
}

/* ---------- SCORE COLORS ---------- */
.score-high { color: #10B981; font-weight: 700; }
.score-med  { color: #F59E0B; font-weight: 700; }
.score-low  { color: #EF4444; font-weight: 700; }

/* ---------- EMPTY STATE ---------- */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    text-align: center;
    border: 1px dashed #1E293B;
    border-radius: 14px;
    background: rgba(17,24,39,0.4);
}

.empty-state-icon {
    font-size: 40px;
    margin-bottom: 14px;
    opacity: 0.4;
}

.empty-state-title {
    font-size: 15px;
    font-weight: 600;
    color: #475569;
    margin-bottom: 6px;
}

.empty-state-desc {
    font-size: 13px;
    color: #334155;
    max-width: 300px;
    line-height: 1.5;
}

/* ---------- ERROR CARD ---------- */
.error-card {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 12px;
    padding: 18px 20px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin: 12px 0;
}

.error-card-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.error-card-title { font-size: 14px; font-weight: 600; color: #EF4444; margin-bottom: 4px; }
.error-card-desc { font-size: 12px; color: #94A3B8; line-height: 1.5; }

/* ---------- LABEL OVERRIDES ---------- */
label, .stLabel, [data-testid="stWidgetLabel"] {
    color: #94A3B8 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 4px !important;
}

/* ---------- HIDE STREAMLIT BRANDING ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ---------- SKELETON LOADER ---------- */
@keyframes skeleton-glow {
    0% { background-position: 100% 50%; }
    100% { background-position: 0 50%; }
}
.skeleton-loader {
    background: linear-gradient(90deg, #111827 25%, #1E293B 50%, #111827 75%);
    background-size: 200% 100%;
    animation: skeleton-glow 1.5s infinite;
    border-radius: 12px;
    margin-bottom: 16px;
    border: 1px solid #1E293B;
}

</style>
"""


def get_base_css() -> str:
    """Return full CSS injection string for premium dark theme."""
    return FONT_IMPORT + BASE_CSS


def get_plotly_theme() -> dict:
    """Return Plotly layout defaults for consistent dark chart styling.

    The returned dict contains ``xaxis`` and ``yaxis`` sub-dicts with the
    base dark-theme grid/tick colours.  Use :func:`apply_plotly_theme` when
    you also need to set axis *titles* or other per-chart axis overrides -
    that helper merges the extras into the base defaults so you never get a
    duplicate-keyword-argument error from ``figure.update_layout(**theme,
    xaxis=...``.
    """
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, Plus Jakarta Sans, sans-serif", "color": "#94A3B8", "size": 12},
        "xaxis": {
            "gridcolor": "#1E293B",
            "linecolor": "#1E293B",
            "tickcolor": "#475569",
            "tickfont": {"color": "#64748B"},
            "showgrid": True,
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": "#1E293B",
            "linecolor": "#1E293B",
            "tickcolor": "#475569",
            "tickfont": {"color": "#64748B"},
            "showgrid": True,
            "zeroline": False,
        },
        "legend": {
            "bgcolor": "rgba(17,24,39,0.8)",
            "bordercolor": "#1E293B",
            "borderwidth": 1,
            "font": {"color": "#94A3B8"},
        },
        "hoverlabel": {
            "bgcolor": "#1E293B",
            "bordercolor": "#334155",
            "font": {"color": "#F8FAFC", "size": 13},
        },
        "margin": {"t": 40, "b": 20, "l": 20, "r": 20},
    }


def apply_plotly_theme(
    fig,
    height: int = 300,
    xaxis_title: str = "",
    yaxis_title: str = "",
    xaxis_extra: dict = None,
    yaxis_extra: dict = None,
    **extra_layout,
) -> None:
    """Apply the VoltIQ dark theme to a Plotly figure safely.

    This is the **recommended** way to theme charts because it merges
    per-chart axis overrides into the base theme dict before calling
    ``update_layout``, so you never get a duplicate-kwarg TypeError.

    Args:
        fig:            A Plotly Figure object.
        height:         Chart height in pixels.
        xaxis_title:    Optional x-axis label.
        yaxis_title:    Optional y-axis label.
        xaxis_extra:    Additional x-axis layout keys to merge.
        yaxis_extra:    Additional y-axis layout keys to merge.
        **extra_layout: Any additional top-level layout kwargs.
    """
    theme = get_plotly_theme()

    # Initialize a base layout configuration dict
    layout_config = {
        "paper_bgcolor": theme["paper_bgcolor"],
        "plot_bgcolor": theme["plot_bgcolor"],
        "height": height,
    }

    # Helper function to merge or replace dict settings safely
    def merge_or_replace(key, default_val):
        if key in extra_layout:
            override = extra_layout.pop(key)
            if isinstance(override, dict) and isinstance(default_val, dict):
                return {**default_val, **override}
            return override
        return default_val

    # Safely merge dict sub-objects
    layout_config["font"] = merge_or_replace("font", theme["font"])
    layout_config["legend"] = merge_or_replace("legend", theme["legend"])
    layout_config["hoverlabel"] = merge_or_replace("hoverlabel", theme["hoverlabel"])
    layout_config["margin"] = merge_or_replace("margin", theme["margin"])

    # Handle title (which can be a string or a dict)
    if "title" in extra_layout:
        layout_config["title"] = extra_layout.pop("title")

    # Merge x-axis settings
    xaxis_base = dict(theme["xaxis"])
    if xaxis_title:
        xaxis_base["title"] = xaxis_title
    if xaxis_extra:
        xaxis_base.update(xaxis_extra)
    layout_config["xaxis"] = merge_or_replace("xaxis", xaxis_base)

    # Merge y-axis settings
    yaxis_base = dict(theme["yaxis"])
    if yaxis_title:
        yaxis_base["title"] = yaxis_title
    if yaxis_extra:
        yaxis_base.update(yaxis_extra)
    layout_config["yaxis"] = merge_or_replace("yaxis", yaxis_base)

    # Apply any other miscellaneous layout parameters remaining in extra_layout
    layout_config.update(extra_layout)

    # Single call to update_layout with no duplicate argument collisions
    fig.update_layout(**layout_config)
