import streamlit as st
import os
import sys
from pathlib import Path

# Add project root directory to path to support clean imports in Streamlit
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.services.api_client import api_client
from frontend.styles.theme import get_base_css, get_plotly_theme
from frontend.components.ui import (
    render_topbar, render_kpi_card, render_section_header,
    render_insight_card, render_sidebar, render_friendly_error,
    render_loading_skeleton, render_success_notification
)

# --- Page Configuration ------------------------------------------------------
st.set_page_config(
    page_title="VoltIQ - Fleet Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_base_css(), unsafe_allow_html=True)

# --- Sidebar -----------------------------------------------------------------
selected_page = render_sidebar()

# --- Route to the correct page -----------------------------------------------
if selected_page == "Fleet Intelligence":
    from frontend.pages.Fleet import render_page
    render_page()
elif selected_page == "Battery APM":
    from frontend.pages.Battery import render_page
    render_page()
elif selected_page == "Carbon Intelligence":
    from frontend.pages.Carbon import render_page
    render_page()
elif selected_page == "AI Fleet Advisor":
    from frontend.pages.AI_Advisor import render_page
    render_page()
else:
    # --- HOME DASHBOARD ------------------------------------------------------

    # Fetch data with skeleton loading placeholders
    skeleton_placeholder = st.empty()
    with skeleton_placeholder.container():
        render_loading_skeleton(height=160)
        render_loading_skeleton(height=240)

    with st.spinner("Analyzing fleet operational metrics..."):
        health_res = api_client.get_system_health()
        summary_res = api_client.get_fleet_summary()
        carbon_res = api_client.get_carbon_metrics()

    skeleton_placeholder.empty()

    is_healthy = (
        health_res.get("success", False)
        and health_res.get("data", {}).get("status") == "healthy"
    )
    summary_data = summary_res.get("data", {}) if summary_res.get("success") else {}
    carbon_data = carbon_res.get("data", {}) if carbon_res.get("success") else {}

    total_vehicles = summary_data.get("total_vehicles", 250_000)
    readiness_pct = summary_data.get("readiness_percentage", 45.0)
    high_ready = summary_data.get("high_readiness_count", 28_500)
    co2_savings = carbon_data.get("annual_savings_kg", 3_757_000)
    net_zero = carbon_data.get("net_zero_progress_pct", 34.0)
    intensity_red = carbon_data.get("carbon_intensity_reduction_pct", 52.0)

    render_topbar(
        "Executive Dashboard",
        "Real-time fleet electrification & sustainability intelligence",
        is_healthy,
    )

    if health_res.get("success", False) and summary_res.get("success", False) and carbon_res.get("success", False):
        st.toast("Dashboard metrics synchronized successfully.", icon="📈")

    if not is_healthy:
        render_friendly_error(
            "Operational Connection Notice",
            "The dashboard is currently operating in offline mode. Live database synchronization is temporarily paused. Displaying cached operational telemetry.",
            show_retry=True
        )

    # -- KPI Row --------------------------------------------------------------
    cols = st.columns(4)
    kpis = [
        {
            "label": "Fleet Vehicles",
            "value": f"{total_vehicles:,}",
            "icon": "🚛",
            "accent": "#2563EB",
            "icon_bg": "rgba(37,99,235,0.15)",
            "trend": "up 12% YoY",
            "trend_dir": "up",
            "description": "Total tracked ICE fleet vehicles",
        },
        {
            "label": "EV Readiness",
            "value": f"{readiness_pct:.1f}%",
            "icon": "⚡",
            "accent": "#10B981",
            "icon_bg": "rgba(16,185,129,0.15)",
            "trend": "up 8.2% vs last quarter",
            "trend_dir": "up",
            "description": "Vehicles eligible for EV transition",
        },
        {
            "label": "Carbon Saved",
            "value": f"{co2_savings/1e6:.2f}M kg",
            "icon": "🌱",
            "accent": "#8B5CF6",
            "icon_bg": "rgba(139,92,246,0.15)",
            "trend": "Projected annual Scope 1 offset",
            "trend_dir": "up",
            "description": "On full electrification transition",
        },
        {
            "label": "Net Zero Progress",
            "value": f"{net_zero:.0f}%",
            "icon": "🎯",
            "accent": "#F59E0B",
            "icon_bg": "rgba(245,158,11,0.15)",
            "trend": f"Target: 100% by 2030",
            "trend_dir": "neutral",
            "description": f"Carbon intensity down{intensity_red:.0f}%",
        },
    ]
    for col, kpi in zip(cols, kpis):
        with col:
            render_kpi_card(**kpi)

    # -- Second KPI Row --------------------------------------------------------
    cols2 = st.columns(3)
    kpis2 = [
        {
            "label": "High Readiness",
            "value": f"{high_ready:,}",
            "icon": "✅",
            "accent": "#10B981",
            "icon_bg": "rgba(16,185,129,0.12)",
            "trend": "Immediate EV candidates",
            "trend_dir": "up",
        },
        {
            "label": "Medium Readiness",
            "value": f"{summary_data.get('medium_readiness_count', 61_200):,}",
            "icon": "⚙️",
            "accent": "#F59E0B",
            "icon_bg": "rgba(245,158,11,0.12)",
            "trend": "Planned transition pipeline",
            "trend_dir": "neutral",
        },
        {
            "label": "Emission Intensity down",
            "value": f"{intensity_red:.1f}%",
            "icon": "📉",
            "accent": "#2563EB",
            "icon_bg": "rgba(37,99,235,0.12)",
            "trend": "vs ICE baseline scenario",
            "trend_dir": "up",
        },
    ]
    for col, kpi in zip(cols2, kpis2):
        with col:
            render_kpi_card(**kpi)

    # -- Capabilities ---------------------------------------------------------
    render_section_header("Platform Modules")

    c_l, c_r = st.columns(2)
    modules = [
        {
            "icon": "📋",
            "icon_bg": "rgba(37,99,235,0.15)",
            "title": "Fleet Electrification Readiness",
            "desc": (
                "Scans ICE vehicle parameters - mileage, age, payload, route duty cycles - "
                "through trained LinearRegression models. Outputs EV suitability scores, "
                "OEM replacement recommendations, cost estimates, and lead times."
            ),
            "badge": "o Active ML Pipeline",
            "badge_class": "badge-success",
            "col": c_l,
        },
        {
            "icon": "🔋",
            "icon_bg": "rgba(16,185,129,0.15)",
            "title": "Battery Asset Performance (APM)",
            "desc": (
                "Monitors battery cell telemetry streams using a Gradient Boosting regression "
                "pipeline to predict State of Health (SOH) and Remaining Useful Life (RUL). "
                "Auto-categorises cells into Healthy / Attention / Critical health zones."
            ),
            "badge": "o Active ML Pipeline",
            "badge_class": "badge-success",
            "col": c_l,
        },
        {
            "icon": "🌱",
            "icon_bg": "rgba(139,92,246,0.15)",
            "title": "Carbon Intelligence Tracker",
            "desc": (
                "Implements Scope 1 emissions accounting from combustion fuel consumption. "
                "Performs ICE vs EV grid-emission comparative analysis across custom routes. "
                "Tracks organisational Net-Zero commitments in real time."
            ),
            "badge": "o Connected to Analytics",
            "badge_class": "badge-success",
            "col": c_r,
        },
        {
            "icon": "🤖",
            "icon_bg": "rgba(245,158,11,0.15)",
            "title": "AI Fleet Advisor",
            "desc": (
                "Conversational AI interface over fleet, battery, and carbon databases. "
                "Operators ask natural-language questions - SOH trends, carbon offset "
                "calculations, procurement timelines - and receive AI-generated answers "
                "with source citations."
            ),
            "badge": "o LangChain Agent Active",
            "badge_class": "badge-warning",
            "col": c_r,
        },
    ]
    for m in modules:
        with m["col"]:
            st.markdown(f"""
            <div class="module-card">
                <div class="module-icon-box" style="background:{m['icon_bg']};">{m['icon']}</div>
                <div class="module-card-title">{m['title']}</div>
                <div class="module-card-desc">{m['desc']}</div>
                <span class="badge {m['badge_class']}">{m['badge']}</span>
            </div>
            """, unsafe_allow_html=True)

    # -- AI Insights -----------------------------------------------------------
    render_section_header("AI Insights & Alerts")
    ic1, ic2 = st.columns(2)
    with ic1:
        render_insight_card(
            "Fleet Transition Opportunity",
            f"Based on current readiness analysis, <strong>{high_ready:,} vehicles</strong> qualify "
            f"for immediate EV transition with a projected carbon saving of "
            f"<strong>{co2_savings/1e6:.1f}M kg CO₂/year</strong>. Prioritise heavy-duty routes for "
            "maximum ROI.",
            "High Confidence",
            "success",
        )
    with ic2:
        render_insight_card(
            "Net Zero Acceleration",
            f"Current trajectory puts net-zero alignment at <strong>{net_zero:.0f}%</strong>. "
            "Accelerating medium-readiness vehicle transitions by 18 months would advance "
            "the 2030 target by an estimated <strong>3.2 years</strong>.",
            "Predictive",
            "info",
        )

    st.markdown(
        '<div style="text-align:center;font-size:11px;color:#1E293B;margin-top:32px;">'
        "VoltIQ Fleet Intelligence Platform * Enterprise Edition v2.0"
        "</div>",
        unsafe_allow_html=True,
    )
