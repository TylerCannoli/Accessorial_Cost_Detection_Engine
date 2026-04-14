# File: pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.database import load_shipments_with_fallback
from utils.styling import inject_css, sidebar_account, NAVY_900, NAVY_500, chart_theme, risk_badge_html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PACE — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Auth guard ────────────────────────────────────────────────────────────────
require_auth()
username = st.session_state.get("username", "User")
sidebar_account(username)

# ── Load data (live DB with mock fallback) ────────────────────────────────────
df_raw = load_shipments_with_fallback()
df_raw["ship_date_dt"] = pd.to_datetime(df_raw["ship_date"])
df_all = df_raw  # module-level alias used by dialogs

# ── Shared chart style constants ──────────────────────────────────────────────
_DARK = dict(plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
             font=dict(color="#A78BFA"))
_GRID = dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
             linecolor="rgba(150,50,200,0.2)")



def _sort_buttons(chart_key: str):
    """Handle sort buttons."""
    opts = ["Value ↑", "Value ↓", "A-Z"]
    skey = f"sort_{chart_key}"
    if skey not in st.session_state:
        st.session_state[skey] = "Value ↓"
    cols = st.columns(len(opts))
    for col, lbl in zip(cols, opts):
        with col:
            kind = "primary" if st.session_state[skey] == lbl else "secondary"
            if st.button(lbl, key=f"sb_{chart_key}_{lbl}", type=kind,
                         width="stretch"):
                st.session_state[skey] = lbl
    return st.session_state[skey]


# ── Chart-builder functions ───────────────────────────────────────────────────
def _build_risk_dist_fig(df: pd.DataFrame, height=260) -> go.Figure:
    """Handle build risk dist fig."""
    if len(df) == 0:
        return go.Figure()
    hist_df = df.copy()
    hist_df["bucket"] = pd.cut(
        hist_df["risk_score"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=["0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
                "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"],
        include_lowest=True,
    )
    bucket_counts = hist_df["bucket"].value_counts().sort_index().reset_index()
    bucket_counts.columns = ["Bracket", "Count"]

    def bucket_color(label):
        """Handle bucket color."""
        pct = int(label.split("–")[0])
        if pct >= 67: return "#EF4444"    # high risk — red
        if pct >= 34: return "#A855F7"    # medium risk — purple
        return "#34D399"                  # low risk — green

    bar_colors = [bucket_color(b) for b in bucket_counts["Bracket"].astype(str)]
    fig = go.Figure(go.Bar(
        x=bucket_counts["Bracket"].astype(str),
        y=bucket_counts["Count"],
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>%{y} shipments<extra></extra>",
    ))
    fig.update_layout(
        **_DARK, height=height,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(tickfont=dict(size=11), **_GRID),
        yaxis=dict(tickfont=dict(size=11), **_GRID),
    )
    return fig


def _build_carrier_risk_fig(df: pd.DataFrame, height=260, sort_by="Value ↓") -> go.Figure:
    """Handle build carrier risk fig."""
    if len(df) == 0:
        return go.Figure()
    carrier_risk = df.groupby("carrier")["risk_score"].mean().reset_index()
    carrier_risk["risk_pct"] = carrier_risk["risk_score"].round(1)  # already 0-100
    if sort_by == "Value ↑":
        carrier_risk = carrier_risk.sort_values("risk_score", ascending=False)
    elif sort_by == "Value ↓":
        carrier_risk = carrier_risk.sort_values("risk_score", ascending=True)
    else:
        carrier_risk = carrier_risk.sort_values("carrier", ascending=False)
    fig = go.Figure(go.Bar(
        x=carrier_risk["risk_pct"],
        y=carrier_risk["carrier"],
        orientation="h",
        marker_color="#9333EA",
        text=carrier_risk["risk_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_DARK, height=height,
        margin=dict(l=0, r=80, t=8, b=0),
        xaxis=dict(title="Avg Risk Score (%)", range=[0, 100],
                   tickfont=dict(size=11), **_GRID),
        yaxis=dict(tickfont=dict(size=11), color="#94A3B8"),
    )
    return fig


# ── Expand dialogs (module-level) ─────────────────────────────────────────────
@st.dialog("Risk Score Distribution", width="large")
def _popup_risk_dist():
    """Handle popup risk dist."""
    st.caption(f"{len(df_all):,} shipments · all time")
    st.plotly_chart(_build_risk_dist_fig(df_all, height=480),
                    width="stretch")


@st.dialog("Avg Risk Score by Carrier", width="large")
def _popup_carrier_risk():
    """Handle popup carrier risk."""
    sort_by = _sort_buttons("carrier_risk")
    st.caption(f"{len(df_all):,} shipments · all carriers")
    if len(df_all) == 0:
        st.info("No data available.")
    else:
        st.plotly_chart(_build_carrier_risk_fig(df_all, height=480, sort_by=sort_by),
                        width="stretch")


# ── Inline filters ────────────────────────────────────────────────────────────
min_date = df_all["ship_date_dt"].min().date()
max_date = df_all["ship_date_dt"].max().date()
carriers = sorted(df_all["carrier"].unique())

# Default to last 90 days so the delta KPIs compare that window to the prior period
_default_start = max(min_date, (pd.Timestamp.today() - pd.Timedelta(days=90)).date())

with st.expander("Filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        date_range = st.date_input(
            "Ship Date Range", value=(_default_start, max_date),
            min_value=min_date, max_value=max_date, key="dash_date"
        )
    with f2:
        sel_carriers = st.multiselect("Carrier", carriers, default=carriers, key="dash_carriers")
    with f3:
        sel_tiers = st.multiselect(
            "Risk Tier", ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"], key="dash_tiers"
        )

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_all.copy()
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["ship_date_dt"] >= start) & (df["ship_date_dt"] <= end)]
if sel_carriers:
    df = df[df["carrier"].isin(sel_carriers)]
if sel_tiers:
    df = df[df["risk_tier"].isin(sel_tiers)]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## Risk Dashboard")
st.caption(f"Showing {len(df):,} shipments matching current filters")
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
total        = len(df)
avg_risk     = df["risk_score"].mean() if total else 0          # already 0-100 after DB normalization
high_risk    = len(df[df["risk_tier"] == "High"])
est_cost     = df["accessorial_charge_usd"].sum()

# ── Period-over-period delta: compare current window to the same-length prior window ──
if len(date_range) == 2:
    _start_ts   = pd.Timestamp(date_range[0])
    _end_ts     = pd.Timestamp(date_range[1])
    _period_days = max(1, (_end_ts - _start_ts).days)
    _prev_end    = _start_ts - pd.Timedelta(days=1)
    _prev_start  = _prev_end - pd.Timedelta(days=_period_days)
    df_prev = df_all[(df_all["ship_date_dt"] >= _prev_start) & (df_all["ship_date_dt"] <= _prev_end)]
    if sel_carriers:
        df_prev = df_prev[df_prev["carrier"].isin(sel_carriers)]
    if sel_tiers:
        df_prev = df_prev[df_prev["risk_tier"].isin(sel_tiers)]
else:
    df_prev = df_all.iloc[0:0]  # empty frame — no comparison possible

_prev_total     = len(df_prev)
_prev_avg_risk  = df_prev["risk_score"].mean() if _prev_total else 0
_prev_high      = len(df_prev[df_prev["risk_tier"] == "High"])
_prev_cost      = df_prev["accessorial_charge_usd"].sum()

total_delta     = total - _prev_total
avg_risk_delta  = (avg_risk - _prev_avg_risk) if total else 0
high_risk_delta = high_risk - _prev_high
est_cost_delta  = est_cost - _prev_cost

def _fmt_usd(v: float) -> str:
    """Handle fmt usd."""
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Shipments",     f"{total:,}",            delta=f"{total_delta:+,} vs prev period")
with c2:
    st.metric("Avg Risk",      f"{avg_risk:.1f}%",      delta=f"{avg_risk_delta:+.1f}%")
with c3:
    st.metric("High Risk",     f"{high_risk:,}",        delta=f"{high_risk_delta:+,} vs prev period")
with c4:
    st.metric("Accessorial $", _fmt_usd(est_cost),      delta=_fmt_usd(est_cost_delta))

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="medium")

with col_left:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Risk Score Distribution")
            st.caption("Number of shipments by risk score bracket")
        with btn:
            if st.button("⤢", key="exp_risk_dist", help="Expand chart"):
                _popup_risk_dist()

        if total > 0:
            st.plotly_chart(_build_risk_dist_fig(df), width="stretch")
        else:
            st.info("No data matches the current filters.")

with col_right:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Avg Risk Score by Carrier")
            st.caption("Carriers ranked by average predicted risk")
        with btn:
            if st.button("⤢", key="exp_carrier_risk", help="Expand chart"):
                _popup_carrier_risk()

        if total > 0:
            st.plotly_chart(_build_carrier_risk_fig(df), width="stretch")
        else:
            st.info("No data matches the current filters.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Risk tier summary row ──────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### Risk Tier Breakdown")
    st.caption("Shipment counts and accessorial exposure by tier")
    t1, t2, t3 = st.columns(3)
    for col, tier, bg, fg in [
        (t1, "Low",    "rgba(5,150,105,0.85)",   "#34D399"),
        (t2, "Medium", "rgba(80,20,160,0.85)",    "#A78BFA"),
        (t3, "High",   "rgba(180,20,20,0.85)",    "#F87171"),
    ]:
        tier_df = df[df["risk_tier"] == tier]
        with col:
            st.markdown(
                f"<div style='background:{bg};border:1px solid {fg}33;border-radius:10px;"
                f"padding:16px 20px;text-align:center;'>"
                f"<div style='font-size:11px;font-weight:600;letter-spacing:0.8px;"
                f"color:{fg};text-transform:uppercase;margin-bottom:6px;'>{tier} Risk</div>"
                f"<div style='font-size:28px;font-weight:700;color:#FFFFFF;'>{len(tier_df):,}</div>"
                f"<div style='font-size:12px;color:{fg};margin-top:4px;'>"
                f"${tier_df['accessorial_charge_usd'].sum():,.0f} accessorial</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)

# ── Shipments table ───────────────────────────────────────────────────────────
with st.container(border=True):
    th_col, search_col = st.columns([3, 1])
    with th_col:
        st.markdown("#### Recent Shipments")
    with search_col:
        search = st.text_input("Search", placeholder="🔍 Search shipment ID…", label_visibility="collapsed")

    table_df = df.copy()
    if search:
        table_df = table_df[table_df["shipment_id"].astype(str).str.contains(search.upper(), na=False)]

    st.dataframe(
        table_df[[
            "shipment_id", "ship_date", "carrier", "facility",
            "risk_score", "risk_tier", "base_freight_usd", "accessorial_charge_usd",
        ]].rename(columns={
            "shipment_id":            "Shipment ID",
            "ship_date":              "Ship Date",
            "carrier":                "Carrier",
            "facility":               "Facility",
            "risk_score":             "Risk Score",
            "risk_tier":              "Risk Tier",
            "base_freight_usd":       "Base Freight ($)",
            "accessorial_charge_usd": "Est. Accessorial ($)",
        }),
        width="stretch",
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", format="%.1f", min_value=0, max_value=100,
            ),
            "Base Freight ($)":      st.column_config.NumberColumn(format="$%.2f"),
            "Est. Accessorial ($)":  st.column_config.NumberColumn(format="$%.2f"),
        },
        height=400,
    )
    st.caption(f"{len(table_df):,} shipments shown")
