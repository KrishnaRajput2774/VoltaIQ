import streamlit as st
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.services.api_client import api_client
from frontend.styles.theme import get_base_css, apply_plotly_theme, get_plotly_theme
from frontend.components.ui import (
    render_topbar, render_kpi_card, render_section_header,
    render_glass_card_open, render_glass_card_close,
    render_insight_card, render_friendly_error,
    render_loading_skeleton, render_success_notification,
    render_empty_state
)
import plotly.graph_objects as go
import pandas as pd

try:
    st.set_page_config(page_title="VoltIQ - Fleet Intelligence", layout="wide")
except Exception:
    pass


def render_page():
    st.markdown(get_base_css(), unsafe_allow_html=True)

    render_topbar(
        "Fleet Electrification Readiness",
        "Transition analysis * ML procurement recommendations * Readiness scoring",
    )

    # -- Fetch Data ------------------------------------------------------------
    skeleton_placeholder = st.empty()
    with skeleton_placeholder.container():
        render_loading_skeleton(height=180)
        render_loading_skeleton(height=240)

    with st.spinner("Analyzing fleet operational metrics..."):
        res = api_client.get_fleet_summary()

    skeleton_placeholder.empty()

    if res.get("success", False):
        st.toast("Fleet readiness summary synchronized successfully.", icon="📋")
        summary = res["data"]

        # -- KPI Cards ---------------------------------------------------------
        cols = st.columns(4)
        kpis = [
            {
                "label": "Total Fleet Size",
                "value": f"{summary['total_vehicles']:,}",
                "icon": "🚛",
                "accent": "#2563EB",
                "icon_bg": "rgba(37,99,235,0.15)",
                "trend": "ICE vehicles tracked",
                "trend_dir": "neutral",
            },
            {
                "label": "Transition Ready",
                "value": f"{summary['readiness_percentage']:.1f}%",
                "icon": "⚡",
                "accent": "#10B981",
                "icon_bg": "rgba(16,185,129,0.15)",
                "trend": "Eligible for EV switch",
                "trend_dir": "up",
            },
            {
                "label": "High Readiness",
                "value": f"{summary['high_readiness_count']:,}",
                "icon": "✅",
                "accent": "#10B981",
                "icon_bg": "rgba(16,185,129,0.12)",
                "trend": "Score >= 0.60",
                "trend_dir": "up",
            },
            {
                "label": "Transition Pipeline",
                "value": f"{summary['high_readiness_count'] + summary['medium_readiness_count']:,}",
                "icon": "📈",
                "accent": "#F59E0B",
                "icon_bg": "rgba(245,158,11,0.15)",
                "trend": "High + Medium combined",
                "trend_dir": "neutral",
            },
        ]
        for col, kpi in zip(cols, kpis):
            with col:
                render_kpi_card(**kpi)

        # -- Charts Row --------------------------------------------------------
        render_section_header("Readiness Distribution & Recommendations")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            render_glass_card_open("Transition Feasibility Segmentation", "🍩", "rgba(37,99,235,0.12)")

            categories = [
                "High Readiness (>= 0.6)",
                "Moderate Readiness (0.4-0.6)",
                "Low Readiness (< 0.4)",
            ]
            values = [
                summary["high_readiness_count"],
                summary["medium_readiness_count"],
                summary["low_readiness_count"],
            ]
            total = sum(values)

            fig_pie = go.Figure(data=[go.Pie(
                labels=categories,
                values=values,
                hole=0.55,
                marker=dict(
                    colors=["#10B981", "#F59E0B", "#EF4444"],
                    line=dict(color="#0B1120", width=2),
                ),
                textinfo="percent",
                textfont=dict(size=13, color="#F8FAFC"),
                hovertemplate="<b>%{label}</b><br>%{value:,} vehicles<br>%{percent}<extra></extra>",
            )])
            fig_pie.add_annotation(
                text=f"<b>{total:,}</b><br><span style='font-size:10px'>Total</span>",
                x=0.5, y=0.5,
                font=dict(size=14, color="#F8FAFC", family="Inter"),
                showarrow=False,
            )
            apply_plotly_theme(
                fig_pie,
                height=300,
                showlegend=True,
                legend=dict(
                    orientation="v", x=1.0, y=0.5,
                    bgcolor="rgba(17,24,39,0.8)", bordercolor="#1E293B",
                    borderwidth=1, font=dict(color="#94A3B8"),
                ),
                margin=dict(t=10, b=10, l=10, r=120),
            )
            st.plotly_chart(fig_pie, width="stretch")
            render_glass_card_close()

        with c_right:
            render_glass_card_open("Top Replacement Candidates", "🏆", "rgba(16,185,129,0.12)")
            recs = summary.get("recommendations", [])
            if recs:
                recs_df = pd.DataFrame([
                    {
                        "Vehicle ID": r["vehicle_id"],
                        "Class": r["vehicle_type"],
                        "Readiness Score": f"{r['ev_readiness_score']:.3f}",
                        "Recommended EV": r["recommended_ev_replacement"],
                        "Lead Time": f"{r['lead_time_months']} mo",
                        "Est. Cost": f"${r['estimated_cost_usd']:,.0f}",
                    }
                    for r in recs
                ])
                st.dataframe(recs_df, width="stretch", hide_index=True, height=270)
            else:
                st.info("No recommendation data available.")
            render_glass_card_close()

        # -- Readiness Bar Chart -----------------------------------------------
        render_section_header("Fleet Readiness Overview")
        render_glass_card_open("Readiness Tier Distribution", "📊", "rgba(37,99,235,0.12)")

        fig_bar = go.Figure(data=[
            go.Bar(
                x=["High Readiness", "Moderate Readiness", "Low Readiness"],
                y=[summary["high_readiness_count"], summary["medium_readiness_count"], summary["low_readiness_count"]],
                marker=dict(
                    color=["#10B981", "#F59E0B", "#EF4444"],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                text=[f"{summary['high_readiness_count']:,}", f"{summary['medium_readiness_count']:,}", f"{summary['low_readiness_count']:,}"],
                textposition="outside",
                textfont=dict(color="#94A3B8", size=12),
                hovertemplate="<b>%{x}</b><br>%{y:,} vehicles<extra></extra>",
            )
        ])
        apply_plotly_theme(fig_bar, height=280, yaxis_title="Vehicle Count", bargap=0.35)
        st.plotly_chart(fig_bar, width="stretch")
        render_glass_card_close()

    else:
        render_friendly_error(
            "Fleet Analytics Unavailable",
            "VoltIQ was unable to retrieve the fleet readiness summary. Please check your network connection.",
            show_retry=True
        )

    # -- Interactive Analysis --------------------------------------------------
    render_section_header("Interactive Vehicle Analysis")
    tab1, tab2 = st.tabs(["🔍  Vehicle Registry Lookup", "🧠  ML Readiness Estimator"])

    with tab1:
        col_look1, col_look2 = st.columns([1, 2])
        with col_look1:
            render_glass_card_open("Search Vehicle Registry", "🔍", "rgba(37,99,235,0.12)")
            v_id_input = st.text_input("Vehicle ID", "1", placeholder="e.g. 1, 42, VH-0013")
            search_btn = st.button("Lookup Suitability", type="primary")
            render_glass_card_close()

        with col_look2:
            if search_btn and v_id_input:
                with st.spinner("Fetching vehicle telemetry..."):
                    v_res = api_client.get_vehicle_readiness(v_id_input)
                if v_res.get("success", False):
                    st.toast("Vehicle record loaded successfully.", icon="🚛")
                    v_data = v_res["data"]
                    score = v_data["ev_readiness_score"]
                    if score >= 0.6:
                        accent = "#10B981"
                        badge_v = "success"
                    elif score >= 0.4:
                        accent = "#F59E0B"
                        badge_v = "warning"
                    else:
                        accent = "#EF4444"
                        badge_v = "danger"

                    render_success_notification(f"Vehicle registry lookup completed successfully for Vehicle {v_data['vehicle_id']}.")
                    st.markdown(f"""
                    <div class="glass-card" style="border-color:{accent}33;">
                        <div class="glass-card-title">
                            <div class="card-icon" style="background:rgba(37,99,235,0.15);">🚛</div>
                            Vehicle {v_data['vehicle_id']} - Readiness Report
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
                            <div>
                                <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">EV Readiness Score</div>
                                <div style="font-size:36px;font-weight:800;color:{accent};">{score:.4f}</div>
                            </div>
                            <div>
                                <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Category</div>
                                <div style="margin-top:8px;">
                                    <span class="badge badge-{badge_v}">{v_data['readiness_category']}</span>
                                </div>
                            </div>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                            <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:12px;">
                                <div style="font-size:11px;color:#475569;margin-bottom:4px;">RECOMMENDED EV</div>
                                <div style="font-size:13px;color:#60A5FA;font-weight:600;">{v_data['recommended_ev_replacement']}</div>
                            </div>
                            <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:12px;">
                                <div style="font-size:11px;color:#475569;margin-bottom:4px;">ESTIMATED COST</div>
                                <div style="font-size:13px;color:#F8FAFC;font-weight:600;">${v_data['estimated_cost_usd']:,.0f}</div>
                            </div>
                            <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:12px;">
                                <div style="font-size:11px;color:#475569;margin-bottom:4px;">LEAD TIME</div>
                                <div style="font-size:13px;color:#F8FAFC;font-weight:600;">{v_data['lead_time_months']} months</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    render_friendly_error(
                        "Vehicle Registry Lookup Failed",
                        "The requested vehicle ID could not be located in the registry. Please check the ID and try again.",
                        show_retry=False
                    )
            else:
                render_empty_state(
                    "No Vehicle Selected",
                    "Enter a Vehicle ID in the search panel and click Lookup to load readiness diagnostics.",
                )

    with tab2:
        render_glass_card_open("ML Electrification Readiness Estimator", "🧠", "rgba(139,92,246,0.15)")
        st.markdown(
            '<div style="font-size:13px;color:#64748B;margin-bottom:18px;">'
            "Submit operational parameters to predict EV transition readiness via the trained LinearRegression pipeline."
            "</div>",
            unsafe_allow_html=True,
        )

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=8)
            usage = st.number_input("Usage Hours", min_value=0.0, max_value=100000.0, value=8000.0)
            fuel = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, max_value=50.0, value=10.5)
            health = st.number_input("Vehicle Health Score (0-100)", min_value=0.0, max_value=100.0, value=80.0)
            maint_cost = st.number_input("Annual Maintenance Cost (USD)", min_value=0.0, max_value=50000.0, value=600.0)
            p_score = st.slider("Predictive Maintenance Score", 0.0, 1.0, 0.75)

        with col_in2:
            load_cap = st.number_input("Load Capacity (kg)", min_value=0.0, max_value=30000.0, value=5000.0)
            act_load = st.number_input("Actual Load (kg)", min_value=0.0, max_value=30000.0, value=3000.0)
            load_util = st.slider("Load Utilization (%)", 0.0, 100.0, 60.0)
            fuel_hour = st.number_input("Fuel per Hour (L)", min_value=0.0, max_value=10.0, value=1.25)
            maint_days = st.number_input("Days Since Last Maintenance", min_value=0, max_value=365, value=45)
            failures = st.number_input("Failure History (Count)", min_value=0, max_value=20, value=2)

        with col_in3:
            anomalies = st.number_input("Anomalies Detected", min_value=0, max_value=100, value=1)
            dtc = st.number_input("Diagnostic Trouble Codes (OBD DTCs)", min_value=0, max_value=50, value=0)
            pcr = st.number_input("PCR (Performance-Cost Ratio)", value=0.55)
            uir = st.number_input("UIR (Utilisation-Idle Ratio)", value=0.60)
            tpi = st.number_input("TPI (Total Performance Index)", value=0.70)
            mbf = st.number_input("MBF (Mean Between Failures)", value=250.0)

        col_in4, col_in5, col_in6 = st.columns(3)
        with col_in4:
            v_type = st.selectbox("Vehicle Type", ["Light Truck", "Heavy Truck", "Van", "Passenger Car", "Delivery Truck", "Refrigerated Truck"])
            route = st.selectbox("Route Type", ["Urban Delivery", "Highway Freight", "Regional Distribution", "Mixed Routing"])
        with col_in5:
            road = st.selectbox("Road Conditions", ["Smooth", "Rough", "Paved", "Off-Road"])
            weather = st.selectbox("Weather Conditions", ["Clear", "Rain", "Snow", "Fog"])
        with col_in6:
            brake = st.selectbox("Brake Condition", ["Good", "Worn", "New", "Critical"])
            ads = st.number_input("ADS (Avg Downtime Score)", value=3.5)

        ohi = st.number_input("OHI (Overall Health Index)", value=85.0)
        cmes = st.number_input("CMES (Cumulative Maintenance Effort)", value=1500.0)
        uer = st.number_input("UER (Utilisation Efficiency Ratio)", value=0.75)

        predict_btn = st.button("Calculate EV Readiness Score", type="primary")
        render_glass_card_close()

        if predict_btn:
            payload = {
                "vehicle_age_years": int(age), "usage_hours": float(usage),
                "fuel_consumption": float(fuel), "health_score": float(health),
                "load_capacity": float(load_cap), "actual_load": float(act_load),
                "load_utilization_pct": float(load_util), "fuel_per_hour": float(fuel_hour),
                "maintenance_cost": float(maint_cost), "days_since_last_maintenance": int(maint_days),
                "failure_history": int(failures), "anomalies_detected": int(anomalies),
                "diagnostic_trouble_code_count": int(dtc), "predictive_score": float(p_score),
                "pcr": float(pcr), "uir": float(uir), "tpi": float(tpi), "mbf": float(mbf),
                "ads": float(ads), "ohi": float(ohi), "cmes": float(cmes), "uer": float(uer),
                "vehicle_type": v_type, "route_info": route,
                "road_conditions": road, "weather_conditions": weather, "brake_condition": brake,
            }
            with st.spinner("Executing ML pipeline..."):
                pred_res = api_client.predict_fleet(payload)

            if pred_res.get("success", False):
                data = pred_res["data"]
                score = data["ev_readiness_score"]
                cat = data["readiness_category"]
                accent = "#10B981" if score >= 0.6 else "#F59E0B" if score >= 0.4 else "#EF4444"
                badge_v = "success" if score >= 0.6 else "warning" if score >= 0.4 else "danger"

                render_success_notification("Predictive EV transition analysis calculated successfully.")
                st.markdown(f"""
                <div class="glass-card" style="border-color:{accent}44;margin-top:16px;">
                    <div class="glass-card-title">
                        <div class="card-icon" style="background:rgba(16,185,129,0.15);">⚡</div>
                        ML Prediction Result
                    </div>
                    <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
                        <div>
                            <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">EV Readiness Score</div>
                            <div style="font-size:48px;font-weight:800;color:{accent};">{score:.4f}</div>
                        </div>
                        <div>
                            <div style="font-size:11px;color:#475569;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.8px;">Feasibility</div>
                            <span class="badge badge-{badge_v}" style="font-size:14px;padding:6px 16px;">{cat}</span>
                        </div>
                        <div style="margin-left:auto;">
                            <div style="font-size:11px;color:#334155;">Analytics Pipeline: <code style="color:#60A5FA;">VoltIQ Transition Engine</code></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                render_friendly_error(
                    "Transition Calculation Failed",
                    "The predictive engine was unable to complete the EV feasibility calculations. Please verify inputs.",
                    show_retry=False
                )


if __name__ == "__main__":
    render_page()
