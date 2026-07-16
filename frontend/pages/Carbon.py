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

try:
    st.set_page_config(page_title="VoltIQ - Carbon Intelligence", layout="wide")
except Exception:
    pass


def render_page():
    st.markdown(get_base_css(), unsafe_allow_html=True)

    render_topbar(
        "Carbon Intelligence Tracker",
        "Scope 1 baseline accounting * ICE vs EV comparative offsets * Net-Zero trajectory monitoring",
    )

    # -- Fetch Metrics ---------------------------------------------------------
    skeleton_placeholder = st.empty()
    with skeleton_placeholder.container():
        render_loading_skeleton(height=180)
        render_loading_skeleton(height=240)

    with st.spinner("Analyzing carbon footprint metrics..."):
        res = api_client.get_carbon_metrics()

    skeleton_placeholder.empty()

    if res.get("success", False):
        st.toast("Carbon intelligence metrics synchronized.", icon="🌱")
        metrics = res["data"]
        baseline = metrics["baseline_co2_kg"]
        ev_co2 = metrics["ev_scenario_co2_kg"]
        savings = metrics["annual_savings_kg"]
        intensity_red = metrics["carbon_intensity_reduction_pct"]
        net_zero = metrics["net_zero_progress_pct"]

        # -- KPI Cards ---------------------------------------------------------
        cols = st.columns(4)
        kpi_data = [
            {
                "label": "Baseline ICE CO2",
                "value": f"{baseline / 1e6:.2f}M kg",
                "icon": "🏭",
                "accent": "#EF4444",
                "icon_bg": "rgba(239,68,68,0.15)",
                "trend": "Annual Scope 1 combustion",
                "trend_dir": "down",
                "description": "Current fleet carbon footprint",
            },
            {
                "label": "Projected EV CO2",
                "value": f"{ev_co2 / 1e6:.2f}M kg",
                "icon": "⚡",
                "accent": "#10B981",
                "icon_bg": "rgba(16,185,129,0.15)",
                "trend": "Grid emissions (EV scenario)",
                "trend_dir": "up",
                "description": "With full electrification",
            },
            {
                "label": "Avoided Emissions",
                "value": f"{savings / 1e6:.2f}M kg",
                "icon": "🌱",
                "accent": "#8B5CF6",
                "icon_bg": "rgba(139,92,246,0.15)",
                "trend": "Net annual savings",
                "trend_dir": "up",
                "description": "CO2 offset per year",
            },
            {
                "label": "Net Zero Progress",
                "value": f"{net_zero:.1f}%",
                "icon": "🎯",
                "accent": "#F59E0B",
                "icon_bg": "rgba(245,158,11,0.15)",
                "trend": f"Intensity down {intensity_red:.1f}%",
                "trend_dir": "up",
                "description": "2030 target alignment",
            },
        ]
        for col, kpi in zip(cols, kpi_data):
            with col:
                render_kpi_card(**kpi)

        # -- Charts Row --------------------------------------------------------
        render_section_header("Scenario Analysis & Net-Zero Progress")
        c_left, c_right = st.columns([1.2, 0.8])

        with c_left:
            render_glass_card_open("Comparative Emissions Scenario Analysis", "📊", "rgba(239,68,68,0.12)")
            fig_bar = go.Figure(data=[
                go.Bar(
                    name="ICE Baseline",
                    x=["Annual CO2 Emissions"],
                    y=[baseline],
                    marker=dict(color="#EF4444", line=dict(width=0)),
                    text=[f"{baseline / 1e6:.2f}M kg"],
                    textposition="outside",
                    textfont=dict(color="#94A3B8"),
                    width=0.3,
                    hovertemplate="<b>ICE Baseline</b><br>%{y:,.0f} kg CO2<extra></extra>",
                ),
                go.Bar(
                    name="EV Scenario",
                    x=["Annual CO2 Emissions"],
                    y=[ev_co2],
                    marker=dict(color="#10B981", line=dict(width=0)),
                    text=[f"{ev_co2 / 1e6:.2f}M kg"],
                    textposition="outside",
                    textfont=dict(color="#94A3B8"),
                    width=0.3,
                    hovertemplate="<b>EV Scenario</b><br>%{y:,.0f} kg CO2<extra></extra>",
                ),
            ])
            fig_bar.add_annotation(
                text=f"<b>Down {intensity_red:.1f}% Reduction</b>",
                x=0, y=max(baseline, ev_co2) * 1.18,
                font=dict(size=13, color="#10B981", family="Inter"),
                showarrow=False,
            )
            theme = get_plotly_theme()
            apply_plotly_theme(
                fig_bar,
                height=300,
                yaxis_title="Annual CO2 (kg)",
                barmode="group",
                bargap=0.5,
                legend=dict(
                    **theme["legend"],
                    orientation="h", x=0, y=1.1,
                ),
            )
            st.plotly_chart(fig_bar, width="stretch")
            render_glass_card_close()

        with c_right:
            render_glass_card_open("Net-Zero Progress Gauge", "🎯", "rgba(139,92,246,0.12)")
            gauge_color = "#10B981" if net_zero >= 70 else "#F59E0B" if net_zero >= 40 else "#EF4444"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=net_zero,
                delta={
                    "reference": 85,
                    "suffix": "% vs target",
                    "font": {"size": 11, "color": "#64748B"},
                },
                domain={"x": [0, 1], "y": [0, 1]},
                number={"suffix": "%", "font": {"size": 36, "color": "#F8FAFC", "family": "Inter"}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#334155",
                        "tickfont": {"color": "#64748B", "size": 10},
                    },
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(239,68,68,0.1)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.1)"},
                        {"range": [70, 100], "color": "rgba(16,185,129,0.1)"},
                    ],
                    "threshold": {
                        "line": {"color": "#60A5FA", "width": 3},
                        "thickness": 0.8,
                        "value": 85,
                    },
                },
                title={"text": "2030 Net-Zero Target Alignment", "font": {"color": "#64748B", "size": 11}},
            ))
            # Gauge has no standard x/y axes - avoid apply_plotly_theme to prevent conflict
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter, Plus Jakarta Sans, sans-serif", "color": "#94A3B8", "size": 12},
                height=280,
                margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, width="stretch")

            st.markdown(f"""
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;">
                <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);
                    border-radius:20px;padding:4px 12px;font-size:11px;color:#10B981;">
                    Scope 1 Accounting Active
                </div>
                <div style="background:rgba(37,99,235,0.1);border:1px solid rgba(37,99,235,0.2);
                    border-radius:20px;padding:4px 12px;font-size:11px;color:#60A5FA;">
                    {intensity_red:.1f}% Intensity Reduction
                </div>
                <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.2);
                    border-radius:20px;padding:4px 12px;font-size:11px;color:#F59E0B;">
                    Target: 85% by 2028
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_glass_card_close()

        # -- Waterfall Chart ---------------------------------------------------
        render_section_header("Carbon Offset Waterfall")
        render_glass_card_open("Emissions Reduction Breakdown", "💧", "rgba(16,185,129,0.12)")

        fig_waterfall = go.Figure(go.Waterfall(
            name="Carbon Breakdown",
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["ICE Baseline", "EV Transition Savings", "Net Emissions"],
            y=[baseline, -savings, None],
            connector={"line": {"color": "#334155", "width": 1}},
            decreasing={"marker": {"color": "#10B981"}},
            increasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#2563EB"}},
            text=[f"{baseline / 1e6:.2f}M", f"-{savings / 1e6:.2f}M", f"{ev_co2 / 1e6:.2f}M"],
            textfont={"color": "#94A3B8", "size": 12},
            hovertemplate="%{x}<br>%{y:,.0f} kg CO2<extra></extra>",
        ))
        apply_plotly_theme(fig_waterfall, height=260, yaxis_title="CO2 (kg)", showlegend=False)
        st.plotly_chart(fig_waterfall, width="stretch")
        render_glass_card_close()

        # -- Insights ----------------------------------------------------------
        render_section_header("Executive Insights")
        ic1, ic2 = st.columns(2)
        with ic1:
            render_insight_card(
                "Fleet Carbon Savings Opportunity",
                f"Full electrification would avoid <strong>{savings / 1e6:.2f}M kg CO2/year</strong>, "
                f"equivalent to removing ~<strong>{int(savings / 2300):,} passenger cars</strong> from the road.",
                "High Impact", "success",
            )
        with ic2:
            render_insight_card(
                "Carbon Intensity Progress",
                f"Carbon intensity has been reduced by <strong>{intensity_red:.1f}%</strong> from baseline. "
                f"Net-zero alignment at <strong>{net_zero:.1f}%</strong> - on track for the 2028 interim milestone.",
                "On Track", "info",
            )

    else:
        render_friendly_error(
            "Carbon Diagnostics Unavailable",
            "The system is currently unable to load carbon intelligence metrics. Please verify system status.",
            show_retry=True
        )

    # -- Comparative Vehicle Analysis ------------------------------------------
    render_section_header("Comparative Vehicle Carbon Analysis")
    col_in, col_out = st.columns([1, 1.5])

    with col_in:
        render_glass_card_open("Analysis Parameters", "🔬", "rgba(37,99,235,0.12)")
        vehicle_id = st.text_input("Vehicle ID", "1", placeholder="e.g. 1, 42")
        distance = st.number_input("Estimated Annual Distance (km)", min_value=0.0, max_value=200_000.0, value=20_000.0)
        fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol/Gasoline", "CNG"])
        calculate_btn = st.button("Generate Carbon Offsets", type="primary")
        render_glass_card_close()

    with col_out:
        if calculate_btn:
            with st.spinner("Analysing comparative emissions..."):
                payload = {
                    "vehicle_id": vehicle_id,
                    "annual_distance_km": float(distance),
                    "fuel_type": fuel_type,
                }
                res_a = api_client.analyze_carbon(payload)

            if res_a.get("success", False):
                a = res_a["data"]
                b_co2 = a["baseline_annual_co2_kg"]
                e_co2 = a["ev_projected_co2_kg"]
                sv = a["net_annual_savings_kg"]
                sv_pct = a["savings_percentage"]

                st.toast("Carbon analysis report generated.", icon="🌱")
                render_success_notification("Comparative carbon footprint analysis report generated successfully.")
                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-title">
                        <div class="card-icon" style="background:rgba(16,185,129,0.15);">🌱</div>
                        Vehicle {vehicle_id} - Carbon Analysis Report
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;">
                        <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:14px;">
                            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">ICE Annual CO2</div>
                            <div style="font-size:22px;font-weight:700;color:#EF4444;">{b_co2:,.0f} kg</div>
                        </div>
                        <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:14px;">
                            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">EV Projected CO2</div>
                            <div style="font-size:22px;font-weight:700;color:#10B981;">{e_co2:,.0f} kg</div>
                        </div>
                        <div style="background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:14px;">
                            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Net Savings</div>
                            <div style="font-size:22px;font-weight:700;color:#8B5CF6;">{sv:,.0f} kg</div>
                            <div style="font-size:11px;color:#10B981;margin-top:2px;">Down {sv_pct:.1f}% reduction</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig_comp = go.Figure(data=[
                    go.Bar(
                        name="ICE Baseline",
                        x=["Vehicle Carbon Footprint"],
                        y=[b_co2],
                        marker=dict(color="#EF4444", line=dict(width=0)),
                        text=[f"{b_co2:,.0f} kg"],
                        textposition="outside",
                        hovertemplate="ICE Baseline<br>%{y:,.0f} kg CO2<extra></extra>",
                    ),
                    go.Bar(
                        name="EV Scenario",
                        x=["Vehicle Carbon Footprint"],
                        y=[e_co2],
                        marker=dict(color="#10B981", line=dict(width=0)),
                        text=[f"{e_co2:,.0f} kg"],
                        textposition="outside",
                        hovertemplate="EV Scenario<br>%{y:,.0f} kg CO2<extra></extra>",
                    ),
                ])
                theme = get_plotly_theme()
                apply_plotly_theme(
                    fig_comp,
                    height=230,
                    yaxis_title="Annual CO2 (kg)",
                    barmode="group",
                    bargap=0.4,
                    legend=dict(
                        **theme["legend"],
                        orientation="h", x=0, y=1.15,
                    ),
                )
                st.plotly_chart(fig_comp, width="stretch")
            else:
                render_friendly_error(
                    "Emissions Analysis Failed",
                    "The carbon offset calculator was unable to process the vehicle parameters. Please check parameter values.",
                    show_retry=False
                )
        else:
            render_empty_state(
                "Enter Parameters",
                "Fill in vehicle details on the left and click Generate to produce a comparative carbon offset report.",
            )


if __name__ == "__main__":
    render_page()
