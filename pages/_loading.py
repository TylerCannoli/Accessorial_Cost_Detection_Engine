"""
Loading screen — shown after login, pre-warms all data caches before
redirecting to the destination page stored in session state.
"""
import time
import base64
import os
import pandas as pd
import streamlit as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import check_auth
from pipeline.config import is_pace_model_ready
from utils.styling import remove_nav_toggle_fallback, inject_persistent_nav_hides, get_background_css

st.set_page_config(
    page_title="PACE — Loading",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="expanded",
)
remove_nav_toggle_fallback()
inject_persistent_nav_hides()

# Redirect if not authenticated
if not check_auth():
    st.switch_page("pages/_Login.py")
    st.stop()

# Preserve pace_mode across any session state resets that may follow
_pace_mode = st.session_state.get("pace_mode", "ltl")

dest = st.session_state.get("post_load_dest", "pages/0_Home.py")

# If already loaded this session, skip straight to destination
if st.session_state.get("_data_preloaded"):
    st.switch_page(dest)
    st.stop()

# Loading timeout — abort after 45 seconds and fall through
_LOAD_TIMEOUT_SECS = 45
_load_start = time.time()

# CRITICAL: Mark as preloaded IMMEDIATELY so that if the WebSocket drops
# during loading, the session is not invalidated on reconnect.
# Each page loads its own data lazily via load_shipments_with_fallback().
st.session_state["_data_preloaded"] = True

# ── Loading page CSS ───────────────────────────────────────────────────────────
_bg_props = get_background_css()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

@keyframes pace-in {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.stApp {{
    background: none;
    font-family: 'Inter', sans-serif;
    animation: pace-in 0.4s ease-out;
}}

/* Blurred background layer */
.stApp::before {{
    content: '';
    position: fixed;
    inset: -20px;
    z-index: -1;
    {_bg_props}
    filter: blur(2px);
}}

/* Dot grid overlay */
.stApp::after {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(180,80,220,0.06) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}}

#MainMenu, footer {{ visibility: hidden !important; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
#pace-nav-toggle-btn {{ display: none !important; }}
.block-container {{ position:relative; z-index:1; padding-top:0 !important; }}

[data-testid="stProgressBar"] > div > div {{
    background: linear-gradient(90deg, #9333EA, #E040FB) !important;
    border-radius: 4px !important;
    box-shadow: 0 0 12px rgba(147,51,234,0.6) !important;
}}
[data-testid="stProgressBar"] > div {{
    background: rgba(30,10,60,0.6) !important;
    border-radius: 4px !important;
    border: 1px solid rgba(180,80,220,0.25) !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Logo ───────────────────────────────────────────────────────────────────────
_logo = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")
if os.path.exists(_logo):
    with open(_logo, "rb") as f:
        _lb64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div style="text-align:center;padding:52px 0 28px;">'
        f'<img src="data:image/png;base64,{_lb64}" '
        f'style="width:180px;filter:drop-shadow(0 0 28px rgba(180,80,220,0.7));"/>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown("""
    <div style="text-align:center;padding:60px 0 28px;">
        <div style="font-size:56px;filter:drop-shadow(0 0 20px rgba(147,51,234,0.8));">⬡</div>
        <h1 style="font-size:38px;font-weight:800;letter-spacing:3px;margin:10px 0 4px;
                   background:linear-gradient(135deg,#9333EA,#E040FB);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">PACE</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:#CBD5E1;font-size:12px;margin:-12px 0 28px;'>"
    "Preparing your workspace…</p>",
    unsafe_allow_html=True,
)

# ── Progress bar + status ──────────────────────────────────────────────────────
progress_bar = st.progress(0)
status_slot  = st.empty()


def _step(msg: str, pct: int):
    progress_bar.progress(pct)
    status_slot.markdown(
        f"<p style='text-align:center;color:#E2E8F0;font-size:13px;margin:6px 0;'>{msg}…</p>",
        unsafe_allow_html=True,
    )


# ── Pre-warm all caches ────────────────────────────────────────────────────────
try:
    from utils.database import (
        get_connection_safe, verify_pace_user,
        get_shipments, get_accessorial_charges,
        get_carriers, get_facilities, get_shipments_with_charges,
    )
    from utils.mock_data import generate_mock_shipments


    _step("Connecting to database", 5)
    if time.time() - _load_start > _LOAD_TIMEOUT_SECS:
        raise TimeoutError("Loading timed out — falling back to demo data.")
    conn = get_connection_safe()

    # ── Verify non-fallback users against the DB ───────────────────────────────
    if st.session_state.get("role") == "pending":
        _u = st.session_state.get("username", "")
        _p = st.session_state.pop("_pending_password", "")
        verified_role = verify_pace_user(conn, _u, _p) if conn is not None else None
        if verified_role is None:
            _err = "Invalid credentials or database unavailable."
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["_login_error"] = _err
            st.session_state["pace_mode"] = _pace_mode  # restore mode after clear
            st.switch_page("pages/_Login.py")
            st.stop()
        norm_role = str(verified_role).strip().lower()
        st.session_state["role"] = norm_role
        dest = "pages/8_Admin.py" if norm_role == "admin" else "pages/0_Home.py"
        st.session_state["post_load_dest"] = dest

    _step("Loading shipment records", 20)
    if time.time() - _load_start > _LOAD_TIMEOUT_SECS:
        raise TimeoutError("Loading timed out — falling back to demo data.")
    df = get_shipments(conn) if conn is not None else pd.DataFrame()
    if df.empty:
        df = generate_mock_shipments(1000)

    _step("Loading accessorial data", 42)
    if time.time() - _load_start > _LOAD_TIMEOUT_SECS:
        raise TimeoutError("Loading timed out — falling back to demo data.")
    get_accessorial_charges(conn)
    get_shipments_with_charges(conn)

    _step("Loading carrier & facility data", 58)
    get_carriers(conn)
    get_facilities(conn)

    _step("Initializing models", 75)

    _step("Preparing dashboards", 90)
    _df = df.copy()
    _df["ship_date_dt"] = pd.to_datetime(_df["ship_date"])
    _df["week"] = _df["ship_date_dt"].dt.to_period("W").dt.start_time
    st.session_state["_preload_df"] = _df
    st.session_state["_preload_weekly"] = (
        _df.groupby("week")
           .agg(shipments=("shipment_id", "count"),
                revenue=("base_freight_usd", "sum"),
                total_cost=("total_cost_usd", "sum"))
           .reset_index()
    )
    st.session_state["_preload_carrier_cpm"] = (
        _df.groupby("carrier")["cost_per_mile"].mean()
           .reset_index().sort_values("cost_per_mile")
    )

    _step("Loading PACE model", 95)
    try:
        from pipeline.inference import get_inference_engine
        if not is_pace_model_ready(mode=_pace_mode):
            from scripts.download_weights import ensure_weights_ready
            ensure_weights_ready()
        if is_pace_model_ready(mode=_pace_mode):
            get_inference_engine(mode=_pace_mode)
    except Exception as e:
        import logging
        logging.warning("PACE: model pre-warm failed (model may not be trained yet): %s", e)

    progress_bar.progress(100)
    status_slot.markdown(
        "<p style='text-align:center;color:#34D399;font-size:14px;font-weight:600;"
        "margin:8px 0;'>✓ &nbsp;Everything ready</p>",
        unsafe_allow_html=True,
    )
    time.sleep(0.15)

except Exception as e:
    progress_bar.progress(100)
    status_slot.markdown(
        f"<p style='text-align:center;color:#FCD34D;font-size:13px;margin:6px 0;'>"
        f"⚠ Loaded with warnings — {e}</p>",
        unsafe_allow_html=True,
    )
    time.sleep(0.3)

st.switch_page(dest)
