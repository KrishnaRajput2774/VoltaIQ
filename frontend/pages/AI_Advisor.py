import streamlit as st
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.services.api_client import api_client
from frontend.styles.theme import get_base_css
from frontend.components.ui import render_topbar, render_section_header

try:
    st.set_page_config(page_title="VoltIQ - AI Fleet Advisor", layout="wide")
except Exception:
    pass

SUGGESTED_QUESTIONS = [
    "Which vehicles should I electrify first?",
    "Show unhealthy batteries.",
    "Summarize fleet readiness.",
    "Compare carbon emissions.",
    "Predict battery health."
]


def render_page():
    st.markdown(get_base_css(), unsafe_allow_html=True)

    render_topbar(
        "AI Fleet Advisor",
        "Conversational fleet intelligence * Natural language queries * Source-cited AI responses",
    )

    # -- Chat History init -----------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = None

    # -- Landing Screen (Only shown when history is empty) ---------------------
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; margin-top:1.5rem; margin-bottom:2rem;">
            <div style="font-size:64px; margin-bottom:12px; display:inline-block;
                        background:linear-gradient(135deg, #2563EB, #10B981);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        filter: drop-shadow(0 0 10px rgba(37,99,235,0.3));">⚡</div>
            <h1 style="font-size:32px; font-weight:800; color:#F8FAFC; margin:0 0 8px 0; letter-spacing:-0.5px;">VoltIQ Assistant</h1>
            <p style="font-size:14px; color:#94A3B8; max-width:600px; margin:0 auto; line-height:1.5;">
                Ask anything about your fleet's electrification status, battery performance metrics, carbon offset projections, and operational targets.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        # Capabilities Showcase
        cap_cols = st.columns(3)
        capabilities = [
            {
                "icon": "📋",
                "title": "Transition Readiness",
                "desc": "Identify prime candidates for electrification based on route telemetry, mileage, and payload cycles.",
            },
            {
                "icon": "🔋",
                "title": "Battery Analytics",
                "desc": "Monitor battery cell State of Health (SOH), degradation, and predict remaining cycles.",
            },
            {
                "icon": "🌱",
                "title": "Carbon Intelligence",
                "desc": "Track Scope 1 emissions reduction progress and compare offset statistics against baseline data.",
            }
        ]
        for col, cap in zip(cap_cols, capabilities):
            with col:
                st.markdown(f"""
                <div style="background:#111827; border:1px solid #1E293B; border-radius:14px; padding:20px; height:100%;">
                    <div style="font-size:24px; margin-bottom:10px;">{cap['icon']}</div>
                    <div style="font-size:14px; font-weight:700; color:#F8FAFC; margin-bottom:6px;">{cap['title']}</div>
                    <div style="font-size:12px; color:#64748B; line-height:1.6;">{cap['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

        # Suggested Questions
        st.markdown('<div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;text-align:center;font-weight:700;">Suggested Queries</div>', unsafe_allow_html=True)

        # Draw suggestions in columns for clean, standard buttons
        col_s1, col_s2 = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            col_target = col_s1 if i % 2 == 0 else col_s2
            with col_target:
                if st.button(q, key=f"sug_btn_{i}", use_container_width=True):
                    user_query = q
    else:
        # -- Active Chat Interface ---------------------------------------------
        # Header with Clear Conversation
        col_title, col_clear = st.columns([5, 1])
        with col_title:
            st.markdown('<div style="font-size:12px;color:#64748B;font-weight:600;margin-top:6px;">ACTIVE CONVERSATION</div>', unsafe_allow_html=True)
        with col_clear:
            if st.button("🗑 Clear Conversation", key="clear_chat_btn", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Render Chat Thread using native Streamlit chat bubbles
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            avatar = "👤" if msg["role"] == "user" else "⚡"
            with st.chat_message(role, avatar=avatar):
                st.write(msg["message"])
                if msg.get("sources"):
                    source_chips = " ".join(f"`{s}`" for s in msg["sources"])
                    st.markdown(f"**Sources:** {source_chips}")

    # -- Chat Input Area -------------------------------------------------------
    chat_input_val = st.chat_input("Ask VoltIQ Assistant...")
    if chat_input_val:
        user_query = chat_input_val

    # -- Handle Submission -----------------------------------------------------
    if user_query:
        st.session_state.chat_history.append({"role": "user", "message": user_query, "sources": []})
        st.rerun()

    # -- Process Bot Response if user just queried -----------------------------
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        # Get query
        user_msg = st.session_state.chat_history[-1]["message"]

        # Show thinking state
        with st.spinner("Analyzing fleet operational logs..."):
            res = api_client.chat_query({
                "message": user_msg,
                "chat_history": st.session_state.chat_history[:-1],
            })

        if res.get("success", False):
            bot_data = res["data"]
            st.session_state.chat_history.append({
                "role": "bot",
                "message": bot_data["response"],
                "sources": bot_data.get("sources", []),
            })
            st.toast("Advisor response received.", icon="💡")
        else:
            st.session_state.chat_history.append({
                "role": "bot",
                "message": "⚠️ The VoltIQ Advisor Engine is currently offline or unable to process the request. Please try again shortly.",
                "sources": [],
            })
        st.rerun()

    # -- System Information Footer ---------------------------------------------
    st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)
    render_section_header("System Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1E293B;border-radius:12px;padding:16px;">
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">AI Engine</div>
            <div style="font-size:13px;color:#F8FAFC;font-weight:600;">VoltIQ Advisor Engine</div>
            <div style="font-size:11px;color:#64748B;margin-top:4px;">Context-Aware Conversation</div>
        </div>
        """, unsafe_allow_html=True)
    with info_col2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1E293B;border-radius:12px;padding:16px;">
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">Data Sources</div>
            <div style="font-size:13px;color:#F8FAFC;font-weight:600;">Fleet * Battery * Carbon</div>
            <div style="font-size:11px;color:#64748B;margin-top:4px;">Synchronized Fleet Analytics</div>
        </div>
        """, unsafe_allow_html=True)
    with info_col3:
        msg_count = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1E293B;border-radius:12px;padding:16px;">
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">Session</div>
            <div style="font-size:13px;color:#F8FAFC;font-weight:600;">{msg_count} queries sent</div>
            <div style="font-size:11px;color:#64748B;margin-top:4px;">{len(st.session_state.chat_history)} total messages</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_page()
