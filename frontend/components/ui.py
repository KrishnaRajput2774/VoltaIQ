"""
VoltIQ Reusable UI Components
All pages import from here for consistent rendering.
"""
import streamlit as st
from typing import Optional


def render_topbar(title: str, subtitle: str, is_healthy: bool = True):
    """Render the premium page topbar with title and system status."""
    status_class = "status-online" if is_healthy else "status-offline"
    status_text = "All Systems Operational" if is_healthy else "Offline Mode"
    status_icon = "o" if is_healthy else "o"

    st.markdown(f"""
    <div class="voltiq-topbar">
        <div>
            <div class="voltiq-page-title">{title}</div>
            <div class="voltiq-page-subtitle">{subtitle}</div>
        </div>
        <div class="topbar-status-pill {status_class}">
            <span class="status-dot"></span>
            {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    icon: str,
    accent: str = "#2563EB",
    accent_color: Optional[str] = None,   # legacy alias - prefer 'accent'
    icon_bg: str = "rgba(37,99,235,0.15)",
    trend: Optional[str] = None,
    trend_dir: str = "up",
    description: Optional[str] = None,
):
    """Render a single premium KPI card.

    Args:
        label:        Short uppercase metric label.
        value:        Formatted metric value string.
        icon:         Emoji or symbol shown in the icon box.
        accent:       CSS colour for the top accent bar and icon bg tint.
        accent_color: Deprecated alias for ``accent`` - still accepted.
        icon_bg:      Background colour for the icon box.
        trend:        Optional trend text (e.g. "up 12% YoY").
        trend_dir:    "up" | "down" | "neutral" controls trend colour.
        description:  Optional small descriptive line below the trend.
    """
    # Honour legacy accent_color kwarg
    resolved_accent = accent_color if accent_color is not None else accent
    trend_class = f"trend-{trend_dir}"
    trend_html = f'<div class="kpi-trend {trend_class}">{trend}</div>' if trend else ""
    desc_html = f'<div style="font-size:11px;color:#475569;margin-top:4px;">{description}</div>' if description else ""
    st.markdown(f"""
    <div class="kpi-card" style="--kpi-accent:{resolved_accent}; --kpi-icon-bg:{icon_bg};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {trend_html}
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str):
    """Render a styled section separator with label."""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-header-title">{title}</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)


def render_glass_card_open(title: str, icon: str = "", icon_bg: str = "rgba(37,99,235,0.15)"):
    """Open a glass card container."""
    icon_html = f'<div class="card-icon" style="background:{icon_bg};">{icon}</div>' if icon else ""
    st.markdown(f"""
    <div class="glass-card">
        <div class="glass-card-title">
            {icon_html}
            {title}
        </div>
    """, unsafe_allow_html=True)


def render_glass_card_close():
    """Close a glass card container."""
    st.markdown("</div>", unsafe_allow_html=True)


def render_error_card(error_message: str):
    """Render a premium error card when backend fails."""
    render_friendly_error("Service Temporarily Offline", "VoltIQ was unable to retrieve intelligence metrics. Please verify system status.", show_retry=True)


def render_friendly_error(title: str, description: str, show_retry: bool = True):
    """Render a clean, non-technical error alert with an optional Retry button."""
    st.markdown(f"""
    <div class="error-card">
        <div class="error-card-icon">⚠️</div>
        <div>
            <div class="error-card-title">{title}</div>
            <div class="error-card-desc">{description}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if show_retry:
        if st.button("Retry", key=f"retry_btn_{title.lower().replace(' ', '_').replace(':', '_')}"):
            st.rerun()


def render_loading_skeleton(height: int = 150):
    """Render a pulsing premium loader skeleton block."""
    st.markdown(f"""
    <div class="skeleton-loader" style="height:{height}px; width:100%;"></div>
    """, unsafe_allow_html=True)


def render_success_notification(message: str):
    """Render an elegant, non-intrusive success notification banner."""
    st.markdown(f"""
    <div style="background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:14px 18px; margin-bottom:20px; display:flex; align-items:center; gap:12px;">
        <div style="font-size:18px;">✅</div>
        <div>
            <div style="font-size:13px; font-weight:600; color:#10B981;">Execution Success</div>
            <div style="font-size:12px; color:#94A3B8;">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(title: str = "No Data", description: str = "", icon: str = "🔍"):
    """Render a premium empty state placeholder.

    Backward-compat note: old callers pass a single prompt string as the
    first positional argument (used as the description).  When ``title``
    looks like a sentence (contains spaces and no short label pattern) we
    treat it as the description so legacy call sites keep working.
    """
    # Legacy single-message call: render_empty_state("Enter a vehicle ID...")
    if description == "" and (len(title) > 40 or title[0].isupper() and " " in title and not title.isupper()):
        display_title = "No Data Available"
        display_desc = title
    else:
        display_title = title
        display_desc = description
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{display_title}</div>
        <div class="empty-state-desc">{display_desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, variant: str = "info"):
    """Render an inline status badge."""
    st.markdown(f'<span class="badge badge-{variant}">{text}</span>', unsafe_allow_html=True)


def render_zone_card(zone: str, description: str = ""):
    """Render a health zone card for battery status."""
    z = zone.lower()
    if "healthy" in z:
        variant = "healthy"
        icon = "✅"
        title = zone
        desc = description or "Operating within nominal parameters. No action required."
    elif "attention" in z:
        variant = "attention"
        icon = "⚠️"
        title = zone
        desc = description or "Schedule maintenance - cell balancing recommended soon."
    else:
        variant = "critical"
        icon = "🔴"
        title = zone
        desc = description or "Critical degradation detected. Replacement recommended immediately."

    st.markdown(f"""
    <div class="zone-card zone-{variant}">
        <div class="zone-icon">{icon}</div>
        <div>
            <div class="zone-content-title">{title}</div>
            <div class="zone-content-desc">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_insight_card(title: str, body: str, badge_text: str = "", badge_variant: str = "info"):
    """Render an AI insight/recommendation card."""
    badge_html = f'<span class="badge badge-{badge_variant}" style="font-size:10px;">{badge_text}</span>' if badge_text else ""
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-card-header">
            <div class="insight-card-title">💡 {title}</div>
            {badge_html}
        </div>
        <div class="insight-card-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the premium VoltIQ sidebar navigation."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">⚡</div>
            <div>
                <div class="sidebar-brand-name">VoltIQ</div>
                <div class="sidebar-brand-tag">Fleet Intelligence Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-section-label">Main Navigation</div>', unsafe_allow_html=True)

        pages = {
            "🏠  Dashboard": "Dashboard",
            "📋  Fleet Intelligence": "Fleet Intelligence",
            "🔋  Battery APM": "Battery APM",
            "🌱  Carbon Intelligence": "Carbon Intelligence",
            "🤖  AI Fleet Advisor": "AI Fleet Advisor",
        }

        selection = st.radio(
            "navigation",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.markdown('<div class="nav-section-label" style="margin-top:16px;">Resources</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:0 20px;">
            <div style="font-size:13px;color:#475569;padding:8px 0;border-bottom:1px solid #1E2D45;cursor:pointer;">📊  Reports</div>
            <div style="font-size:13px;color:#475569;padding:8px 0;border-bottom:1px solid #1E2D45;cursor:pointer;">⚙️  Settings</div>
            <div style="font-size:13px;color:#475569;padding:8px 0;cursor:pointer;">📖  Documentation</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-footer" style="margin-top:32px;">
            <div style="font-size:11px;color:#1E293B;text-align:center;">
                VoltIQ Platform v2.0<br>
                <span style="color:#10B981;">o </span>
                <span style="color:#334155;">Enterprise Intelligence Engine Connected</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        return pages[selection]
