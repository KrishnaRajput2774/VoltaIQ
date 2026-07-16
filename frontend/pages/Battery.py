import streamlit as st
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.services.api_client import api_client
from frontend.styles.theme import get_base_css, apply_plotly_theme
from frontend.components.ui import (
    render_topbar, render_section_header,
    render_glass_card_open, render_glass_card_close,
    render_zone_card, render_insight_card, render_friendly_error,
    render_loading_skeleton, render_success_notification,
    render_empty_state
)
import plotly.graph_objects as go
import numpy as np

try:
    st.set_page_config(page_title="VoltIQ - Battery APM", layout="wide")
except Exception:
    pass


# -- Cached dataset loader (avoids reload on every interaction) ----------------
@st.cache_data(show_spinner=False)
def _load_battery_data():
    """Load battery dataset once and cache for the session."""
    try:
        from app.utils.data_loader import data_loader
        df = data_loader.load("battery")
        ids = sorted(df["Battery_ID"].unique().tolist())
        return df, ids
    except Exception:
        return None, ["B0005", "B0006", "B0007", "B0018"]


def render_page():
    st.markdown(get_base_css(), unsafe_allow_html=True)

    render_topbar(
        "Battery Asset Performance Management",
        "Predictive SOH * Remaining Useful Life * Health Zone Classification * Degradation Analytics",
    )

    # Load (cached) dataset with skeleton loading
    skeleton_placeholder = st.empty()
    with skeleton_placeholder.container():
        render_loading_skeleton(height=100)

    with st.spinner("Synchronizing battery asset registry..."):
        df_batt, unique_ids = _load_battery_data()

    skeleton_placeholder.empty()

    # -- Asset selector --------------------------------------------------------
    render_section_header("Asset Selection & Live Diagnostics")
    col_search, col_diag, col_chart = st.columns([1, 1.2, 1.8])

    with col_search:
        render_glass_card_open("Inspect Battery Asset", "🔋", "rgba(16,185,129,0.15)")
        selected_battery = st.selectbox("Battery Asset ID", unique_ids, index=0)
        inspect_btn = st.button("Inspect Asset", type="primary")
        st.markdown("""
        <div style="margin-top:14px;padding:10px;background:#0B1120;border:1px solid #1E293B;border-radius:10px;">
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">Fleet Summary</div>
            <div style="font-size:12px;color:#64748B;line-height:2.2;">
                Total Assets: <span style="color:#F8FAFC;font-weight:600;">4</span><br>
                Healthy: <span style="color:#10B981;font-weight:600;">2</span><br>
                Attention: <span style="color:#F59E0B;font-weight:600;">1</span><br>
                Critical: <span style="color:#EF4444;font-weight:600;">1</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_glass_card_close()

    with col_diag:
        if inspect_btn and selected_battery:
            with st.spinner("Analysing battery logs..."):
                health_res = api_client.get_battery_health_by_id(selected_battery)

            if health_res.get("success", False):
                st.toast(f"Asset telemetry for {selected_battery} updated.", icon="🔋")
                h = health_res["data"]
                soh = h["state_of_health"]
                rul = h["remaining_useful_life_cycles"]
                zone = h["health_zone"]

                soh_color = (
                    "#10B981" if "healthy" in zone.lower()
                    else "#F59E0B" if "attention" in zone.lower()
                    else "#EF4444"
                )

                render_success_notification(f"Asset diagnostics loaded successfully for battery asset {selected_battery}.")
                render_glass_card_open(f"Diagnostics: {selected_battery}", "📊", "rgba(16,185,129,0.12)")

                # SOH Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=soh * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    number={"suffix": "%", "font": {"size": 32, "color": "#F8FAFC", "family": "Inter"}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#334155",
                            "tickfont": {"color": "#64748B"},
                        },
                        "bar": {"color": soh_color, "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "rgba(239,68,68,0.12)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.10)"},
                            {"range": [70, 100], "color": "rgba(16,185,129,0.10)"},
                        ],
                        "threshold": {
                            "line": {"color": soh_color, "width": 3},
                            "thickness": 0.8,
                            "value": soh * 100,
                        },
                    },
                    title={
                        "text": "State of Health (SOH)",
                        "font": {"color": "#94A3B8", "size": 12, "family": "Inter"},
                    },
                ))
                # Gauge has no x/y axes - use update_layout directly with explicit params
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter, Plus Jakarta Sans, sans-serif", "color": "#94A3B8", "size": 12},
                    height=200,
                    margin=dict(t=20, b=10, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, width="stretch")

                # RUL + SOH metric tiles
                st.markdown(f"""
                <div style="display:flex;gap:12px;margin-bottom:12px;">
                    <div style="flex:1;background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:14px;">
                        <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Remaining Useful Life</div>
                        <div style="font-size:26px;font-weight:700;color:#60A5FA;">{rul} cycles</div>
                    </div>
                    <div style="flex:1;background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:14px;">
                        <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">SOH Reading</div>
                        <div style="font-size:26px;font-weight:700;color:{soh_color};">{soh * 100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                render_zone_card(zone)
                render_glass_card_close()

                # AI Insight card based on zone
                if "critical" in zone.lower():
                    render_insight_card(
                        "Immediate Action Required",
                        f"Battery <strong>{selected_battery}</strong> has degraded beyond safe thresholds "
                        f"(SOH: {soh * 100:.1f}%). Schedule immediate cell replacement to prevent failure.",
                        "Critical", "danger",
                    )
                elif "attention" in zone.lower():
                    render_insight_card(
                        "Preventive Maintenance Due",
                        f"Battery <strong>{selected_battery}</strong> shows early degradation "
                        f"(SOH: {soh * 100:.1f}%). With {rul} cycles remaining, cell balancing is "
                        "recommended within the next 2 maintenance windows.",
                        "Action Recommended", "warning",
                    )
                else:
                    render_insight_card(
                        "Asset Operating Nominally",
                        f"Battery <strong>{selected_battery}</strong> is in excellent condition "
                        f"(SOH: {soh * 100:.1f}%). {rul} cycles remaining. Continue standard monitoring.",
                        "Healthy", "success",
                    )
            else:
                render_friendly_error(
                    "Telemetry Diagnostics Unavailable",
                    f"VoltIQ was unable to retrieve live telemetry diagnostics for battery asset '{selected_battery}'. Please check network connections.",
                    show_retry=True
                )
        else:
            render_empty_state(
                "Select an Asset",
                "Choose a Battery ID and click Inspect to load live diagnostic data and health predictions.",
            )

    with col_chart:
        render_glass_card_open("Capacity Fade Curve", "📉", "rgba(37,99,235,0.12)")

        if df_batt is not None and selected_battery:
            sub_df = df_batt[df_batt["Battery_ID"] == selected_battery].sort_values("Cycle_Number")

            if not sub_df.empty:
                fig_decay = go.Figure()
                fig_decay.add_trace(go.Scatter(
                    x=sub_df["Cycle_Number"],
                    y=sub_df["Capacity_Ah"],
                    mode="lines",
                    name="Capacity (Ah)",
                    line=dict(color="#10B981", width=2.5, shape="spline"),
                    fill="tozeroy",
                    fillcolor="rgba(16,185,129,0.08)",
                    hovertemplate="Cycle %{x}<br>Capacity: %{y:.3f} Ah<extra></extra>",
                ))

                # Trend line
                if len(sub_df) > 5:
                    z = np.polyfit(sub_df["Cycle_Number"], sub_df["Capacity_Ah"], 1)
                    p = np.poly1d(z)
                    fig_decay.add_trace(go.Scatter(
                        x=sub_df["Cycle_Number"],
                        y=p(sub_df["Cycle_Number"]),
                        mode="lines",
                        name="Trend",
                        line=dict(color="#F59E0B", width=1.5, dash="dot"),
                        hovertemplate="Trend: %{y:.3f} Ah<extra></extra>",
                    ))

                apply_plotly_theme(
                    fig_decay,
                    height=310,
                    xaxis_title="Cycle Number",
                    yaxis_title="Capacity (Ah)",
                    showlegend=True,
                )
                st.plotly_chart(fig_decay, width="stretch")

                # Temperature chart
                if "Temperature_C" in sub_df.columns:
                    fig_temp = go.Figure(go.Scatter(
                        x=sub_df["Cycle_Number"],
                        y=sub_df["Temperature_C"],
                        mode="lines",
                        name="Temperature",
                        line=dict(color="#F59E0B", width=2, shape="spline"),
                        fill="tozeroy",
                        fillcolor="rgba(245,158,11,0.06)",
                        hovertemplate="Cycle %{x}<br>Temp: %{y:.1f} C<extra></extra>",
                    ))
                    apply_plotly_theme(
                        fig_temp,
                        height=160,
                        xaxis_title="Cycle",
                        yaxis_title="Temp (C)",
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.markdown(
                        '<div style="font-size:11px;color:#475569;text-transform:uppercase;'
                        'letter-spacing:0.8px;margin-bottom:6px;">Temperature Profile</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(fig_temp, width="stretch")
            else:
                st.info("No cycle data found for this battery.")
        else:
            st.info("Select a battery ID above to display historical degradation data.")
        render_glass_card_close()

    # -- Manual Telemetry Predictor --------------------------------------------
    render_section_header("Manual Telemetry Predictor")

    with st.expander("🧠  Run Manual SOH & RUL Telemetry Analysis - Trained ML Pipeline"):
        st.markdown(
            '<div style="font-size:13px;color:#64748B;margin-bottom:18px;">'
            "Submit point-in-time battery parameters to predict State of Health and Remaining Useful Life."
            "</div>",
            unsafe_allow_html=True,
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            cycle_number = st.number_input("Telemetry Cycle Number", min_value=1, max_value=1000, value=50)
            voltage_v = st.number_input("Terminal Voltage (V)", min_value=0.0, max_value=5.0, value=3.65)
            temperature_c = st.number_input("Core Temperature (C)", min_value=-20.0, max_value=80.0, value=32.5)

        with col_t2:
            capacity_ah = st.number_input("Capacity (Ah)", min_value=0.0, max_value=3.0, value=1.75)
            voltage_sag = st.number_input("Voltage Sag under Load (V)", min_value=0.0, max_value=1.0, value=0.03)
            degrad_rate = st.number_input("Degradation Rate (Ah/cycle)", value=-0.002, format="%.5f")
            cycle_normalized = st.number_input("Normalized Cycle Index (0-1)", value=0.0)

        manual_btn = st.button("Evaluate Telemetry Parameters", type="primary")

        if manual_btn:
            payload = {
                "cycle_number": int(cycle_number),
                "voltage_v": float(voltage_v),
                "temperature_c": float(temperature_c),
                "capacity_ah": float(capacity_ah),
                "voltage_sag_v": float(voltage_sag),
                "degradation_rate": float(degrad_rate),
                "cycle_normalized": float(cycle_normalized) if cycle_normalized > 0 else None,
            }
            with st.spinner("Invoking battery ML predictor..."):
                pred_res = api_client.predict_battery(payload)

            if pred_res.get("success", False):
                pred = pred_res["data"]
                soh = pred["state_of_health"]
                rul = pred["remaining_useful_life_cycles"]
                zone = pred["health_zone"]
                soh_color = (
                    "#10B981" if "healthy" in zone.lower()
                    else "#F59E0B" if "attention" in zone.lower()
                    else "#EF4444"
                )
                render_success_notification("Telemetry point parameters evaluated successfully.")
                st.markdown(f"""
                <div class="glass-card" style="margin-top:16px;">
                    <div class="glass-card-title">
                        <div class="card-icon" style="background:rgba(16,185,129,0.15);">⚡</div>
                        Telemetry Evaluation Results
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
                        <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:16px;">
                            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">Predicted SOH</div>
                            <div style="font-size:32px;font-weight:800;color:{soh_color};">{soh * 100:.2f}%</div>
                            <div style="font-size:11px;color:#334155;margin-top:4px;">SOH Predictive Engine</div>
                        </div>
                        <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:16px;">
                            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">Remaining Useful Life</div>
                            <div style="font-size:32px;font-weight:800;color:#60A5FA;">{rul}</div>
                            <div style="font-size:11px;color:#334155;margin-top:4px;">cycles &middot; RUL Predictive Engine</div>
                        </div>
                        <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:16px;">
                            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">Health Zone</div>
                            <div style="font-size:18px;font-weight:700;color:{soh_color};margin-top:8px;">{zone}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                render_zone_card(zone)
            else:
                render_friendly_error(
                    "Telemetry Evaluation Failed",
                    "The battery analytics engine was unable to analyze telemetry inputs. Please check parameter settings.",
                    show_retry=False
                )


if __name__ == "__main__":
    render_page()
