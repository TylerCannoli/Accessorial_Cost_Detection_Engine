import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.database import load_shipments_with_fallback
from utils.styling import inject_css, sidebar_account, inject_metric_wrap_css
from pages._pace_ui import show_mode_badge

st.set_page_config(page_title="PACE — Home", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

require_auth()
username = st.session_state.get("username", "User")
display_name = str(username).split("@")[0].replace(".", " ").replace("_", " ").title()
sidebar_account(username)
show_mode_badge()

if st.session_state.get("upload_scored") is not None:
    df_all = st.session_state["upload_scored"].copy()
    st.success(
        f"End-to-end pipeline active ✅ (Scored rows: {len(df_all):,})",
        icon="✅",
    )
    st.info("Showing uploaded + scored data from Upload page.", icon="📄")
elif st.session_state.get("upload_df") is not None:
    df_all = st.session_state["upload_df"].copy()
    st.info("Showing uploaded data (not scored yet).", icon="📄")
else:
    df_all = load_shipments_with_fallback()

defaults = {
    "shipment_id": range(1, len(df_all) + 1),
    "ship_date":   pd.Timestamp.today(),
    "carrier":     "Unknown",
    "facility":    "Unknown",
    "risk_tier":   "Unknown",
    "base_freight_usd":       0.0,
    "accessorial_charge_usd": 0.0,
    "total_cost_usd":         0.0,
    "cost_per_mile":          0.0,
}
for col, default in defaults.items():
    if col not in df_all.columns:
        df_all[col] = default

if "total_cost_usd" not in df_all.columns or df_all["total_cost_usd"].isna().all():
    df_all["total_cost_usd"] = (
        pd.to_numeric(df_all.get("base_freight_usd", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df_all.get("accessorial_charge_usd", 0), errors="coerce").fillna(0)
    )

df_all["ship_date_dt"]            = pd.to_datetime(df_all.get("ship_date"), errors="coerce")
df_all["base_freight_usd"]        = pd.to_numeric(df_all["base_freight_usd"],        errors="coerce").fillna(0)
df_all["accessorial_charge_usd"]  = pd.to_numeric(df_all["accessorial_charge_usd"],  errors="coerce").fillna(0)
df_all["total_cost_usd"]          = pd.to_numeric(df_all["total_cost_usd"],          errors="coerce").fillna(0)
df_all["cost_per_mile"]           = pd.to_numeric(df_all["cost_per_mile"],           errors="coerce").fillna(0)

# ── Shared chart layout helper ─────────────────────────────────────────────────
_DARK   = dict(plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
               font=dict(color="#A78BFA"))
_GRID   = dict(gridcolor="rgba(150,50,200,0.18)", color="#A78BFA")
_LEGEND = dict(orientation="h", y=1.05, font=dict(color="#FFFFFF"))


def _build_carrier_list(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("carrier", dropna=True)
              .agg(shipments=("shipment_id", "count"))
              .reset_index()
              .rename(columns={"carrier": "Carrier", "shipments": "Shipments"})
              .sort_values(["Shipments", "Carrier"], ascending=[False, True]))


def _build_accessorial_by_carrier_fig(df: pd.DataFrame, height=280) -> go.Figure:
    acc = (df.groupby("carrier", dropna=True)["accessorial_charge_usd"]
             .sum()
             .reset_index()
             .rename(columns={"carrier": "Carrier", "accessorial_charge_usd": "Accessorial"})
             .sort_values("Accessorial", ascending=False)
             .head(12))
    fig = go.Figure(go.Bar(
        x=acc["Accessorial"],
        y=acc["Carrier"],
        orientation="h",
        marker=dict(
            color=acc["Accessorial"],
            colorscale="Reds",
            line=dict(color="#7F1D1D", width=1),
            showscale=False,
        ),
        text=acc["Accessorial"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Accessorial: $%{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_DARK,
        height=height,
        margin=dict(l=0, r=16, t=8, b=0),
        xaxis=dict(**_GRID, tickprefix="$"),
        yaxis=dict(**_GRID, automargin=True, categoryorder="total ascending"),
        showlegend=False,
    )
    return fig


@st.dialog("Carriers Used", width="large")
def _popup_cpm():
    carrier_df = _build_carrier_list(df_all)
    st.caption(f"{len(carrier_df):,} carriers across {len(df_all):,} shipments")
    st.dataframe(carrier_df, use_container_width=True, hide_index=True)


@st.dialog("Total Accessorial Charges by Carrier", width="large")
def _popup_breakdown():
    st.caption(f"{len(df_all):,} shipments · ranked by total accessorial charges")
    st.plotly_chart(_build_accessorial_by_carrier_fig(df_all, height=480), use_container_width=True)


valid_dates = df_all["ship_date_dt"].dropna()
min_d = valid_dates.min().date() if not valid_dates.empty else None
max_d = valid_dates.max().date() if not valid_dates.empty else None

carriers = ["All"]
if "carrier" in df_all.columns:
    carriers += sorted(df_all["carrier"].dropna().astype(str).unique().tolist())

facilities = ["All"]
if "facility" in df_all.columns:
    facilities += sorted(df_all["facility"].dropna().astype(str).unique().tolist())

tiers = ["All"]
if "risk_tier" in df_all.columns:
    existing_tiers = set(df_all["risk_tier"].dropna().astype(str).unique())
    tiers += [t for t in ["Low", "Medium", "High"] if t in existing_tiers]
    tiers += sorted([t for t in existing_tiers if t not in {"Low", "Medium", "High"}])

with st.expander("Filters", expanded=False):
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        if min_d and max_d:
            date_range = st.date_input(
                "Ship Date Range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                key="home_date",
            )
        else:
            date_range = None
            st.caption("No valid ship_date values")

    with f2:
        carrier_sel = st.selectbox("Carrier", carriers, index=0, key="home_carrier")

    with f3:
        facility_sel = st.selectbox("Facility", facilities, index=0, key="home_facility")

    with f4:
        tier_sel = st.selectbox("Risk Tier", tiers, index=0, key="home_tier")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_all.copy()

if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and "ship_date_dt" in df.columns:
    start_dt = pd.Timestamp(date_range[0])
    end_dt   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[
        (df["ship_date_dt"].notna()) &
        (df["ship_date_dt"] >= start_dt) &
        (df["ship_date_dt"] <= end_dt)
    ]

if carrier_sel != "All" and "carrier" in df.columns:
    df = df[df["carrier"] == carrier_sel]

if facility_sel != "All" and "facility" in df.columns:
    df = df[df["facility"] == facility_sel]

if tier_sel != "All" and "risk_tier" in df.columns:
    df = df[df["risk_tier"] == tier_sel]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"## Welcome back, {display_name}")
st.caption("Real-time visibility into your freight operations.")
st.divider()

# ── KPI calculations ──────────────────────────────────────────────────────────
total_shipments   = len(df)
total_revenue     = df["base_freight_usd"].sum()
total_accessorial = df["accessorial_charge_usd"].sum()
total_costs       = df["total_cost_usd"].sum()
avg_cpm           = df["cost_per_mile"].mean() if total_shipments else 0
accessorial_rate  = (
    len(df[df["accessorial_charge_usd"] > 0]) / total_shipments * 100
    if total_shipments else 0
)

# KPI text wrapping fix
inject_metric_wrap_css()


def _fmt_dollars(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


# ── Quick navigation guide ────────────────────────────────────────────────────
st.markdown("### Site Navigation")
st.caption("Explore PACE's website map. A complete guide to our pages, including key features and capabilities.")

nav_left, nav_right = st.columns(2, gap="medium")

with nav_left:
    st.markdown(
        """
        <div style="
            background:#262244;
            border-radius:12px;
            border:1px solid rgba(255,255,255,0.18);
            padding:16px 18px;
            min-height:260px;
            color:#FFFFFF;
        ">
            <div style="margin-bottom:14px;">
                <div style="font-size:1.15rem;font-weight:700;line-height:1.2;">Upload</div>
                <div style="margin-top:4px;opacity:0.95;">Upload data and run end-to-end accessorial cost risk scoring for new shipment files.</div>
                <hr style="border:none;border-top:1px solid rgba(255,255,255,0.22);margin:10px 0 0 0;">
            </div>
            <div style="margin-bottom:14px;">
                <div style="font-size:1.15rem;font-weight:700;line-height:1.2;">Shipments</div>
                <div style="margin-top:4px;opacity:0.95;">Explore shipment-level detail, risk factors, and recommended actions.</div>
                <hr style="border:none;border-top:1px solid rgba(255,255,255,0.22);margin:10px 0 0 0;">
            </div>
            <div>
                <div style="font-size:1.15rem;font-weight:700;line-height:1.2;">Route Analysis</div>
                <div style="margin-top:4px;opacity:0.95;">Compare lane performance across miles, cost, and risk metrics.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:
    st.markdown(
        """
        <div style="
            background:#262244;
            border-radius:12px;
            border:1px solid rgba(255,255,255,0.18);
            padding:16px 18px;
            min-height:260px;
            color:#FFFFFF;
        ">
            <div style="margin-bottom:14px;">
                <div style="font-size:1.15rem;font-weight:700;line-height:1.2;">Cost Estimate</div>
                <div style="margin-top:4px;opacity:0.95;">Predict shipment cost with confidence intervals and feature impact.</div>
                <hr style="border:none;border-top:1px solid rgba(255,255,255,0.22);margin:10px 0 0 0;">
            </div>
            <div style="margin-bottom:14px;">
                <div style="font-size:1.15rem;font-weight:700;line-height:1.2;">Carrier Benchmarking</div>
                <div style="margin-top:4px;opacity:0.95;">Compare carriers side-by-side; single carrier view shows deep DOT-style detail.</div>
                <hr style="border:none;border-top:1px solid rgba(255,255,255,0.22);margin:10px 0 0 0;">
            </div>
            <div>
                <div style="font-size:1.15rem;font-weight:700;line-height:1.2;">Accessorial Tracker</div>
                <div style="margin-top:4px;opacity:0.95;">Monitor charge trends by type, carrier, and facility.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── Layout: charts left, KPI cards right ──────────────────────────────────────
main_left, main_right = st.columns([3.2, 1.15], gap="medium")

with main_right:
    kpis = [
        ("Shipments",             f"{total_shipments:,}",
         "Total number of shipment records in the selected dataset."),
        ("Revenue",               _fmt_dollars(total_revenue),
         "Sum of all revenue billed to customers across every shipment."),
        ("TLC",                   _fmt_dollars(total_costs),
         "Total Logistics Costs, all expenditures incurred from moving products."),
        ("Accessorial",           _fmt_dollars(total_accessorial),
         "Extra charges beyond base freight."),
        ("CPM",                   f"${avg_cpm:.2f}",
         "Average Cost Per Mile across all shipments."),
        ("Accessorial Usage Rate", f"{accessorial_rate:.1f}%",
         "How often shipments incur extra charges beyond the base linehaul rate."),
    ]
    for label, value, help_text in kpis:
        with st.container(border=True):
            st.metric(label, value, help=help_text)

with main_left:
    # ── Carrier charts ─────────────────────────────────────────────────────────
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Carrier Breakdown")
        with btn:
            if st.button("⤢", key="exp_cpm", help="Expand list"):
                _popup_cpm()
        carriers_used_df = _build_carrier_list(df)
        st.caption(f"{len(carriers_used_df):,} Carriers in current filtered range.")
        st.dataframe(carriers_used_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Total Accessorial Charges by Carrier")
        with btn:
            if st.button("⤢", key="exp_bd", help="Expand chart"):
                _popup_breakdown()
        st.plotly_chart(_build_accessorial_by_carrier_fig(df), use_container_width=True)