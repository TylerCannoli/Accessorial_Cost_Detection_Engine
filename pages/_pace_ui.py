"""Shared PACE mode utilities — imported by all dashboard pages."""
import streamlit as st


def get_pace_mode() -> str:
    """Return the active freight mode ('ltl' or 'ftl') from session state."""
    return st.session_state.get("pace_mode", "ltl")


def show_mode_badge() -> None:
    """Inject an LTL / FTL mode badge into the sidebar."""
    mode = get_pace_mode()
    label = "LTL Freight" if mode == "ltl" else "FTL Freight"
    color = "#1a6fd4" if mode == "ltl" else "#d47a1a"
    st.sidebar.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span style="background:{color};color:#fff;padding:4px 12px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:700;'
        f'letter-spacing:0.06em;text-transform:uppercase;">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
