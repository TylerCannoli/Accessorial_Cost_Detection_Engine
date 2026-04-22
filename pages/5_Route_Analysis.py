import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.database import load_shipments_with_fallback
from utils.styling import inject_css, sidebar_account
from pages._pace_ui import show_mode_badge

st.set_page_config(
    page_title="PACE — Route Analysis",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

require_auth()
username = st.session_state.get("username", "User")
sidebar_account(username)
show_mode_badge()

df_raw = load_shipments_with_fallback()
df_raw["ship_date_dt"] = pd.to_datetime(df_raw["ship_date"])
df_all = df_raw  # module-level alias used by dialogs


# ── Date-range filter helpers ─────────────────────────────────────────────────
def _filter_by_range(df: pd.DataFrame, sel: str) -> pd.DataFrame:
    days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    if sel not in days_map:
        return df
    cutoff = df["ship_date_dt"].max() - pd.Timedelta(days=days_map[sel])
    return df[df["ship_date_dt"] >= cutoff]


def _range_buttons(chart_key: str):
    opts = ["1M", "3M", "6M", "1Y", "All"]
    skey = f"popup_range_{chart_key}"
    if skey not in st.session_state:
        st.session_state[skey] = "All"
    cols = st.columns(len(opts))
    for col, lbl in zip(cols, opts):
        with col:
            kind = "primary" if st.session_state[skey] == lbl else "secondary"
            if st.button(lbl, key=f"rb_{chart_key}_{lbl}", type=kind,
                         width="stretch"):
                st.session_state[skey] = lbl
    return st.session_state[skey]


# ── Lane metrics builder (shared by page and dialogs) ─────────────────────────
def _build_lane_metrics(df: pd.DataFrame, group_col: str, label: str,
                        min_vol: int = 1) -> pd.DataFrame:
    lm = (
        df.groupby(group_col)
        .agg(
            shipments        =("shipment_id",           "count"),
            avg_cost         =("total_cost_usd",         "mean"),
            total_cost       =("total_cost_usd",         "sum"),
            avg_cpm          =("cost_per_mile",          "mean"),
            avg_miles        =("miles",                  "mean"),
            avg_risk         =("risk_score",             "mean"),
            accessorial_cost =("accessorial_charge_usd", "sum"),
            high_risk_count  =("risk_tier",
                               lambda x: (x == "High").sum()),
        )
        .reset_index()
        .rename(columns={group_col: label})
    )
    lm = lm[lm["shipments"] >= min_vol].copy()
    lm["accessorial_rate"] = (
        lm["accessorial_cost"] / lm["total_cost"] * 100
    ).round(1)
    lm["high_risk_pct"] = (
        lm["high_risk_count"] / lm["shipments"] * 100
    ).round(1)
    return lm


# ── Chart-builder functions ───────────────────────────────────────────────────
def _build_expensive_fig(lane_metrics: pd.DataFrame, label: str,
                         height=300) -> go.Figure:
    top_exp = lane_metrics.nlargest(8, "avg_cpm").sort_values("avg_cpm")
    fig = go.Figure(go.Bar(
        x=top_exp["avg_cpm"],
        y=top_exp[label].apply(lambda v: v if len(v) <= 28 else v[:25] + "…"),
        orientation="h",
        marker_color="#DC2626",
        text=top_exp["avg_cpm"].apply(lambda v: f"${v:.2f}/mi"),
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0, r=160, t=8, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        xaxis=dict(tickprefix="$", gridcolor="rgba(150,50,200,0.15)",
                   color="#94A3B8", linecolor="rgba(150,50,200,0.2)"),
        yaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
                   linecolor="rgba(150,50,200,0.2)"),
    )
    return fig


def _build_efficient_fig(lane_metrics: pd.DataFrame, label: str,
                         height=300) -> go.Figure:
    top_cheap = lane_metrics.nsmallest(8, "avg_cpm").sort_values("avg_cpm", ascending=False)
    fig = go.Figure(go.Bar(
        x=top_cheap["avg_cpm"],
        y=top_cheap[label].apply(lambda v: v if len(v) <= 28 else v[:25] + "…"),
        orientation="h",
        marker_color="#059669",
        text=top_cheap["avg_cpm"].apply(lambda v: f"${v:.2f}/mi"),
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0, r=160, t=8, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        xaxis=dict(tickprefix="$", gridcolor="rgba(150,50,200,0.15)",
                   color="#94A3B8", linecolor="rgba(150,50,200,0.2)"),
        yaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
                   linecolor="rgba(150,50,200,0.2)"),
    )
    return fig


def _build_scatter_fig(lane_metrics: pd.DataFrame, label: str,
                       height=320) -> go.Figure:
    scatter_fig = px.scatter(
        lane_metrics,
        x="shipments",
        y="avg_cost",
        size="total_cost",
        color="avg_risk",
        hover_name=label,
        color_continuous_scale=["#059669", "#D97706", "#DC2626"],
        labels={
            "shipments": "Shipment Volume",
            "avg_cost":  "Avg Total Cost ($)",
            "avg_risk":  "Avg Risk Score",
            "total_cost": "Total Spend",
        },
        hover_data={"avg_cpm": ":.2f", "accessorial_rate": ":.1f"},
    )
    scatter_fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        xaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
                   linecolor="rgba(150,50,200,0.2)"),
        yaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
                   linecolor="rgba(150,50,200,0.2)", tickprefix="$"),
    )
    return scatter_fig


# ── Helper: resolve group_col + label from session state ─────────────────────
def _resolve_group():
    view_by = st.session_state.get("route_view_by", "Lane (Origin → Dest)")
    if view_by == "Lane (Origin → Dest)":
        return "lane", "Lane"
    elif "Origin" in view_by:
        return "origin_city", "Origin"
    else:
        return "destination_city", "Destination"


# ── Expand dialogs (module-level) ─────────────────────────────────────────────
@st.dialog("Most Expensive Lanes", width="large")
def _popup_expensive():
    sel = _range_buttons("expensive")
    df_f = _filter_by_range(df_all, sel)
    group_col, label = _resolve_group()
    lm = _build_lane_metrics(df_f, group_col, label, min_vol=1)
    st.caption(f"{len(df_f):,} shipments · {sel} view")
    if lm.empty:
        st.info("No data for the selected range.")
    else:
        st.plotly_chart(_build_expensive_fig(lm, label, height=480),
                        width="stretch")


@st.dialog("Most Efficient Lanes", width="large")
def _popup_efficient():
    sel = _range_buttons("efficient")
    df_f = _filter_by_range(df_all, sel)
    group_col, label = _resolve_group()
    lm = _build_lane_metrics(df_f, group_col, label, min_vol=1)
    st.caption(f"{len(df_f):,} shipments · {sel} view")
    if lm.empty:
        st.info("No data for the selected range.")
    else:
        st.plotly_chart(_build_efficient_fig(lm, label, height=480),
                        width="stretch")


@st.dialog("Lane Volume vs Avg Cost", width="large")
def _popup_scatter():
    sel = _range_buttons("scatter")
    df_f = _filter_by_range(df_all, sel)
    group_col, label = _resolve_group()
    lm = _build_lane_metrics(df_f, group_col, label, min_vol=1)
    st.caption(f"{len(df_f):,} shipments · {sel} view")
    if lm.empty:
        st.info("No data for the selected range.")
    else:
        st.plotly_chart(_build_scatter_fig(lm, label, height=500),
                        width="stretch")


# ── Inline filters ────────────────────────────────────────────────────────────
with st.expander("Filters", expanded=False):
    f1, f2 = st.columns(2)
    with f1:
        min_vol = st.slider("Minimum shipments per lane", 1, 10, 2,
                            help="Filter out low-volume lanes")
    with f2:
        view_by = st.radio("Analyze by",
                           ["Lane (Origin → Dest)", "Origin City", "Destination City"],
                           horizontal=True, key="route_view_by")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## Route Analysis")
st.caption("Identify your most and least expensive shipping lanes and optimize route selection.")
st.divider()

# ── Build lane metrics ────────────────────────────────────────────────────────
if view_by == "Lane (Origin → Dest)":
    group_col = "lane"
    label     = "Lane"
else:
    group_col = "origin_city" if "Origin" in view_by else "destination_city"
    label     = "Origin" if "Origin" in view_by else "Destination"

df = df_raw.copy()
lane_metrics = _build_lane_metrics(df, group_col, label, min_vol=min_vol)

# ── KPI row ───────────────────────────────────────────────────────────────────
total_lanes   = len(lane_metrics)
busiest_lane  = lane_metrics.loc[lane_metrics["shipments"].idxmax(), label] if total_lanes else "—"
cheapest_lane = lane_metrics.loc[lane_metrics["avg_cpm"].idxmin(),  label] if total_lanes else "—"
most_exp_lane = lane_metrics.loc[lane_metrics["avg_cpm"].idxmax(),  label] if total_lanes else "—"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Active Lanes",    f"{total_lanes}")
with k2:
    st.metric("Busiest Lane",    busiest_lane if len(busiest_lane) < 30 else busiest_lane[:27] + "…")
with k3:
    st.metric("Cheapest $/Mile", cheapest_lane if len(cheapest_lane) < 30 else cheapest_lane[:27] + "…")
with k4:
    st.metric("Most Expensive",  most_exp_lane if len(most_exp_lane) < 30 else most_exp_lane[:27] + "…")

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
chart_l, chart_r = st.columns(2, gap="medium")

with chart_l:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Most Expensive Lanes")
            st.caption("Top 8 by average cost per mile")
        with btn:
            if st.button("⤢", key="exp_expensive", help="Expand chart"):
                _popup_expensive()
        st.plotly_chart(_build_expensive_fig(lane_metrics, label),
                        width="stretch")

with chart_r:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Most Efficient Lanes")
            st.caption("Top 8 by lowest average cost per mile")
        with btn:
            if st.button("⤢", key="exp_efficient", help="Expand chart"):
                _popup_efficient()
        st.plotly_chart(_build_efficient_fig(lane_metrics, label),
                        width="stretch")

st.markdown("<br>", unsafe_allow_html=True)

# ── Volume vs cost scatter ────────────────────────────────────────────────────
with st.container(border=True):
    hdr, btn = st.columns([9, 1])
    with hdr:
        st.markdown("#### Lane Volume vs Avg Cost")
        st.caption("Identify high-volume expensive lanes — biggest opportunity for savings")
    with btn:
        if st.button("⤢", key="exp_scatter", help="Expand chart"):
            _popup_scatter()
    st.plotly_chart(_build_scatter_fig(lane_metrics, label),
                    width="stretch")

st.markdown("<br>", unsafe_allow_html=True)

# ── Full lane table ───────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### All Lanes — Full Breakdown")
    display = lane_metrics.copy()
    display["avg_cost"]  = display["avg_cost"].round(2)
    display["avg_cpm"]   = display["avg_cpm"].round(3)
    display["avg_miles"] = display["avg_miles"].round(0).astype(int)
    display["avg_risk"]  = display["avg_risk"].round(3)

    st.dataframe(
        display[[label, "shipments", "avg_cost", "avg_cpm", "avg_miles",
                 "avg_risk", "accessorial_rate", "high_risk_pct", "total_cost"]]
        .rename(columns={
            "shipments":       "Shipments",
            "avg_cost":        "Avg Cost ($)",
            "avg_cpm":         "Avg $/Mile",
            "avg_miles":       "Avg Miles",
            "avg_risk":        "Avg Risk",
            "accessorial_rate":"Accessorial %",
            "high_risk_pct":   "High Risk %",
            "total_cost":      "Total Spend ($)",
        })
        .sort_values("Avg $/Mile", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "Avg Risk": st.column_config.ProgressColumn(
                "Avg Risk", format="%.1f", min_value=0, max_value=100),
            "Avg Cost ($)":    st.column_config.NumberColumn(format="$%.2f"),
            "Avg $/Mile":      st.column_config.NumberColumn(format="$%.3f"),
            "Total Spend ($)": st.column_config.NumberColumn(format="$%.0f"),
        },
        height=420,
    )
