import streamlit as st

def logout():
    """
    PB-4: Securely terminates the user session and redirects to login.
    Clears all session state and cached data.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    # Intentionally NOT clearing cache_resource so the ML model
    # stays loaded in memory between sessions for faster reloads.
    try:
        st.switch_page("pages/_Login.py")
    except Exception:
        # If multipage routing is unavailable in this runtime, refresh app state.
        st.rerun()

def check_auth():
    """
    Helper to verify if user is logged in.
    Returns True if authenticated, False otherwise.
    """
    return st.session_state.get('authenticated', False)


def pace_role_is_admin(role=None) -> bool:
    """
    True if role is the admin role (case-insensitive).
    If role is None, uses st.session_state["role"] (DB may return "Admin", "ADMIN", etc.).
    """
    r = role if role is not None else st.session_state.get("role")
    return str(r or "").strip().lower() == "admin"


def require_auth() -> None:
    """
    Auth guard for protected pages.
    Uses st.switch_page to redirect unauthenticated users to login.
    Only called when a page is accessed directly without a session.
    """
    if not check_auth():
        st.switch_page("pages/_Login.py")
        st.stop()
