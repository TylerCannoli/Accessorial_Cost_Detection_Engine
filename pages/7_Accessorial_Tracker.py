# File: pages/7_Accessorial_Tracker.py
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.database import get_connection_safe, get_shipments_with_charges
from utils.mock_data import generate_mock_shipments
from utils.styling import inject_css, sidebar_account, NAVY_500, TIER_COLORS, CHARGE_COLORS, RISK_HIGH_FG
from pipeline.config import CHARGE_TYPE_LABELS, is_pace_model_ready

st.set_page_config(
    page_title="PACE — Accessorial Tracker",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

require_auth()
username = st.session_state.get("username", "User")
sidebar_account(username)

# ── Model availability ────────────────────────────────────────────
MODEL_READY = is_pace_model_ready()

# ── Load data ─────────────────────────────────────────────────────
conn   = get_connection_safe()
df_raw = get_shipments_with_charges(conn) if conn is not None else pd.DataFrame()

if df_raw.empty:
    _mock  = generate_mock_shipments(300)
    df_raw = _mock[_mock["accessorial_charge_usd"] > 0].copy()
    st.info("Live database unavailable — showing demo data.", icon="ℹ️")

# ── Normalize columns ─────────────────────────────────────────────
# Support both old schema and new PACE schema
if "accessorial_type" not in df_raw.columns:
    df_raw["accessorial_type"] = "Unknown"
if "charge_type" not in df_raw.columns:
    df_raw["charge_type"] = df_raw.get("accessorial_type", "Unknown")
if "risk_score_pct" not in df_raw.columns:
    if "risk_score" in df_raw.columns:
        max_rs = df_raw["risk_score"].max()
        df_raw["risk_score_pct"] = (
            df_raw["risk_score"] * 100 if max_rs <= 1.0 else df_raw["risk_score"]
        )
    else:
        df_raw["risk_score_pct"] = 0.0
if "risk_label" not in df_raw.columns:
    if "risk_tier" in df_raw.columns:
        df_raw["risk_label"] = df_raw["risk_tier"]
    else:
        df_raw["risk_label"] = df_raw["risk_score_pct"].apply(
            lambda s: "Critical" if s >= 75 else
                      "High"     if s >= 50 else
                      "Medium"   if s >= 25 else
                      "Low"      if s > 0  else "None"
        )
if "ship_date_dt" not in df_raw.columns:
    if "ship_date" in df_raw.columns:
        df_raw["ship_date_dt"] = pd.to_datetime(df_raw["ship_date"], errors="coerce")
    else:
        df_raw["ship_date_dt"] = pd.Timestamp.now()

# Check if we have PACE scored data in session
if st.session_state.get("upload_scored") is not None:
    pace_scored = st.session_state["upload_scored"]
    if "charge_type" in pace_scored.columns and "risk_score_pct" in pace_scored.columns:
        st.success(
            f"Showing PACE model predictions for {len(pace_scored):,} scored records. "
            "Upload new data on the Upload page to refresh.",
            icon="✅"
        )
        # Merge scored data into df_raw if dot_number available
        if "dot_number" in pace_scored.columns and "dot_number" in df_raw.columns:
            df_raw = df_raw.merge(
                pace_scored[["dot_number", "charge_type", "risk_score_pct", "risk_label"]],
                on="dot_number", how="left", suffixes=("", "_pace")
            )
            for col in ["charge_type", "risk_score_pct", "risk_label"]:
                if f"{col}_pace" in df_raw.columns:
                    df_raw[col] = df_raw[f"{col}_pace"].fillna(df_raw[col])
                    df_raw.drop(columns=[f"{col}_pace"], inplace=True)

df_all = df_raw.copy()


# ── Chart builders ────────────────────────────────────────────────

def _build_donut_fig(df_in: pd.DataFrame, total_acc: float,
                     height: int = 280) -> go.Figure:
    # Use PACE charge_type if available, fall back to accessorial_type
    """Handle build donut fig."""
    type_col = "charge_type" if "charge_type" in df_in.columns else "accessorial_type"
    type_data = (
        df_in.groupby(type_col)["accessorial_charge_usd"]
        .agg(total="sum", count="count")
        .reset_index()
        .sort_values("total", ascending=False)
    )
    colors = [CHARGE_COLORS.get(t, "#7C3AED")
              for t in type_data[type_col]]
    fig = go.Figure(go.Pie(
        labels=type_data[type_col],
        values=type_data["total"],
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate=(
            "<b>%{label}</b><br>Total: $%{value:,.0f}"
            "<br>Share: %{percent}<extra></extra>"
        ),
    ))
    fig.add_annotation(
        text=f"${total_acc:,.0f}", x=0.5, y=0.5,
        font_size=16, font_color="#E2E8F0", showarrow=False,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0), height=height,
        paper_bgcolor="#0f0a1e", showlegend=True,
        font=dict(color="#A78BFA"),
        legend=dict(
            orientation="v", x=1.0, y=0.5,
            bgcolor="rgba(15,10,30,0.7)",
            font=dict(color="#FFFFFF"),
        ),
    )
    return fig


def _build_risk_distribution_fig(df_in: pd.DataFrame,
                                  height: int = 260) -> go.Figure:
    """PACE risk score distribution by charge type."""
    if "risk_score_pct" not in df_in.columns:
        return go.Figure()
    type_col = "charge_type" if "charge_type" in df_in.columns else "accessorial_type"
    fig = go.Figure()
    for charge in CHARGE_TYPE_LABELS:
        subset = df_in[df_in[type_col] == charge]["risk_score_pct"]
        if len(subset) > 0:
            fig.add_trace(go.Box(
                y=subset,
                name=charge,
                marker_color=CHARGE_COLORS.get(charge, "#A78BFA"),
                boxmean=True,
            ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        xaxis=dict(color="#94A3B8",
                   gridcolor="rgba(150,50,200,0.15)"),
        yaxis=dict(title="Risk Score (%)", color="#94A3B8",
                   gridcolor="rgba(150,50,200,0.15)", range=[0, 100]),
        showlegend=False,
    )
    return fig


def _build_carrier_fig(df_in: pd.DataFrame,
                        height: int = 280) -> go.Figure:
    """Handle build carrier fig."""
    carrier_col = "carrier" if "carrier" in df_in.columns else "carrier_phy_state"
    if carrier_col not in df_in.columns:
        return go.Figure()
    carrier_data = (
        df_in.groupby(carrier_col)
        .agg(
            total=("accessorial_charge_usd", "sum"),
            count=("accessorial_charge_usd", "count"),
            avg_risk=("risk_score_pct", "mean") if "risk_score_pct" in df_in.columns
                     else ("accessorial_charge_usd", "count"),
        )
        .reset_index()
        .sort_values("total", ascending=True)
        .tail(15)
    )
    fig = go.Figure(go.Bar(
        x=carrier_data["total"],
        y=carrier_data[carrier_col],
        orientation="h",
        marker_color="#9333EA",
        text=carrier_data["total"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        textfont={"color": "#E2E8F0"},
    ))
    fig.update_layout(
        margin=dict(l=0, r=80, t=8, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        xaxis=dict(tickprefix="$",
                   gridcolor="rgba(150,50,200,0.15)", color="#94A3B8"),
        yaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8"),
    )
    return fig


def _build_trend_fig(df_in: pd.DataFrame, height: int = 260) -> go.Figure:
    """Handle build trend fig."""
    tmp = df_in.copy()
    tmp["week"] = tmp["ship_date_dt"].dt.to_period("W").dt.start_time
    weekly = (
        tmp.groupby("week")["accessorial_charge_usd"]
        .sum().reset_index()
        .rename(columns={"accessorial_charge_usd": "total"})
    )
    fig = go.Figure(go.Scatter(
        x=weekly["week"], y=weekly["total"],
        mode="lines+markers",
        line=dict(color=RISK_HIGH_FG, width=2),
        marker=dict(color=RISK_HIGH_FG, size=6),
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.08)",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        xaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8"),
        yaxis=dict(tickprefix="$",
                   gridcolor="rgba(150,50,200,0.15)", color="#94A3B8"),
    )
    return fig


def _build_risk_tier_fig(df_in: pd.DataFrame,
                          height: int = 260) -> go.Figure:
    """Bar chart of avg accessorial charge by PACE risk label."""
    if "risk_label" not in df_in.columns:
        return go.Figure()
    tier_data = (
        df_in.groupby("risk_label")
        .agg(avg_acc=("accessorial_charge_usd", "mean"),
             count=("accessorial_charge_usd", "count"))
        .reset_index()
    )
    order = ["None", "Low", "Medium", "High", "Critical"]
    tier_data["order"] = tier_data["risk_label"].map(
        {t: i for i, t in enumerate(order)}
    )
    tier_data = tier_data.sort_values("order")
    colors = [TIER_COLORS.get(t, "#94A3B8") for t in tier_data["risk_label"]]
    fig = go.Figure(go.Bar(
        x=tier_data["risk_label"],
        y=tier_data["avg_acc"],
        marker_color=colors,
        text=tier_data["avg_acc"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        textfont={"color": "#E2E8F0"},
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        xaxis=dict(color="#94A3B8",
                   gridcolor="rgba(150,50,200,0.15)"),
        yaxis=dict(tickprefix="$", color="#94A3B8",
                   gridcolor="rgba(150,50,200,0.15)"),
    )
    return fig


# ── Dialogs ───────────────────────────────────────────────────────

@st.dialog("Accessorial Costs by Charge Type", width="large")
def _popup_donut():
    """Handle popup donut."""
    df_acc = df_all[df_all["accessorial_charge_usd"] > 0].copy()
    total  = df_acc["accessorial_charge_usd"].sum()
    st.caption(f"{len(df_acc):,} shipments with accessorial charges")
    if df_acc.empty:
        st.info("No accessorial charges available.")
    else:
        st.plotly_chart(
            _build_donut_fig(df_acc, total, height=480),
            use_container_width=True,
        )


@st.dialog("Risk Score by Charge Type", width="large")
def _popup_risk_dist():
    """Handle popup risk dist."""
    df_acc = df_all[df_all["accessorial_charge_usd"] > 0].copy()
    st.caption("Distribution of PACE risk scores across charge types")
    if df_acc.empty:
        st.info("No data available.")
    else:
        st.plotly_chart(
            _build_risk_distribution_fig(df_acc, height=480),
            use_container_width=True,
        )


@st.dialog("Accessorial Costs by Carrier", width="large")
def _popup_carrier():
    """Handle popup carrier."""
    df_acc = df_all[df_all["accessorial_charge_usd"] > 0].copy()
    st.caption(f"{len(df_acc):,} shipments with accessorial charges")
    if df_acc.empty:
        st.info("No data available.")
    else:
        st.plotly_chart(
            _build_carrier_fig(df_acc, height=480),
            use_container_width=True,
        )


@st.dialog("Accessorial Cost Trend", width="large")
def _popup_trend():
    """Handle popup trend."""
    df_acc = df_all[df_all["accessorial_charge_usd"] > 0].copy()
    if df_acc.empty:
        st.info("No trend data available.")
    else:
        st.plotly_chart(
            _build_trend_fig(df_acc, height=480),
            use_container_width=True,
        )


@st.dialog("PACE Risk Score Detail", width="large")
def _show_risk_detail(row: dict):
    """Handle show risk detail."""
    score  = float(row.get("risk_score_pct", 0))
    label  = row.get("risk_label", "Unknown")
    charge = row.get("charge_type",
                     row.get("accessorial_type", "Unknown"))
    color  = TIER_COLORS.get(label, "#94A3B8")
    c_color = CHARGE_COLORS.get(charge, "#A78BFA")

    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(f"#### Shipment Risk Detail")
    with h2:
        st.markdown(
            f"<div style='text-align:right;padding-top:6px;'>"
            f"<span style='border:1px solid {color};color:{color};"
            f"border-radius:4px;padding:3px 10px;font-size:13px;"
            f"font-weight:700;'>{label}</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score", f"{score:.1f}%")
    c2.metric("Risk Label", label)
    c3.metric("Predicted Charge", charge)

    # Probabilities if available
    prob_cols = [c for c in row.keys() if c.startswith("prob_")]
    if prob_cols:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Charge Type Probabilities**")
        prob_data = {
            col.replace("prob_", "").replace("_", " ").title(): float(row[col])
            for col in prob_cols
        }
        prob_df = pd.DataFrame(
            prob_data.items(), columns=["Charge Type", "Probability"]
        ).sort_values("Probability", ascending=False)
        prob_fig = go.Figure(go.Bar(
            x=prob_df["Probability"] * 100,
            y=prob_df["Charge Type"],
            orientation="h",
            marker_color=[
                CHARGE_COLORS.get(ct, "#A78BFA")
                for ct in prob_df["Charge Type"]
            ],
            text=[f"{v*100:.1f}%" for v in prob_df["Probability"]],
            textposition="outside",
            textfont={"color": "#E2E8F0"},
        ))
        prob_fig.update_layout(
            margin=dict(l=0, r=60, t=8, b=0), height=200,
            plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
            font=dict(color="#A78BFA"),
            xaxis=dict(ticksuffix="%", range=[0, 110],
                       color="#94A3B8",
                       gridcolor="rgba(150,50,200,0.15)"),
            yaxis=dict(color="#94A3B8",
                       gridcolor="rgba(150,50,200,0.15)",
                       autorange="reversed"),
        )
        st.plotly_chart(prob_fig, use_container_width=True)

    # Raw row data
    with st.expander("Raw Record", expanded=False):
        st.json({k: str(v) for k, v in row.items()
                 if not k.startswith("_")})


# ── Filters ───────────────────────────────────────────────────────
type_col = "charge_type" if "charge_type" in df_all.columns else "accessorial_type"
all_types    = sorted(df_all[type_col].dropna().unique())
carrier_col  = "carrier" if "carrier" in df_all.columns else None
all_carriers = sorted(df_all[carrier_col].dropna().unique()) if carrier_col else []

min_date = df_all["ship_date_dt"].min()
max_date = df_all["ship_date_dt"].max()
if pd.isna(min_date): min_date = pd.Timestamp("2020-01-01")
if pd.isna(max_date): max_date = pd.Timestamp.now()

with st.expander("Filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_types = st.multiselect(
            "Charge Type", all_types, default=all_types
        )
    with f2:
        if all_carriers:
            sel_carriers = st.multiselect(
                "Carrier", all_carriers, default=all_carriers
            )
        else:
            sel_carriers = []
    with f3:
        date_range = st.date_input(
            "Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

# ── Apply filters ─────────────────────────────────────────────────
df = df_all.copy()
if sel_types:
    df = df[df[type_col].isin(sel_types)]
if sel_carriers and carrier_col:
    df = df[df[carrier_col].isin(sel_carriers)]
if len(date_range) == 2:
    df = df[
        (df["ship_date_dt"] >= pd.Timestamp(date_range[0])) &
        (df["ship_date_dt"] <= pd.Timestamp(date_range[1]))
    ]

df_with_acc = df[df["accessorial_charge_usd"] > 0].copy()

# ── Header ────────────────────────────────────────────────────────
st.markdown("## Accessorial Cost Tracker")
st.caption(
    "Track where unexpected charges come from, which carriers carry the "
    "most risk, and how PACE risk scores correlate with actual charges."
)

if not MODEL_READY:
    st.info(
        "PACE model not yet trained — showing historical charge data. "
        "Risk score columns will populate after training completes.",
        icon="ℹ️"
    )

st.divider()

# ── KPI row ───────────────────────────────────────────────────────
total_acc       = df_with_acc["accessorial_charge_usd"].sum()
shipments_w_acc = len(df_with_acc)
total_shipments = len(df)
acc_rate        = (shipments_w_acc / total_shipments * 100) if total_shipments else 0
avg_acc         = df_with_acc["accessorial_charge_usd"].mean() if shipments_w_acc else 0
avg_risk        = df["risk_score_pct"].mean() if "risk_score_pct" in df.columns else 0
pct_of_total    = (
    total_acc / df["total_cost_usd"].sum() * 100
    if "total_cost_usd" in df.columns and df["total_cost_usd"].sum() > 0
    else 0
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Accessorial", f"${total_acc:,.0f}")
with k2:
    st.metric("Affected Shipments", f"{shipments_w_acc:,} / {total_shipments:,}")
with k3:
    st.metric("Accessorial Rate", f"{acc_rate:.1f}%")
with k4:
    st.metric("Avg per Shipment", f"${avg_acc:,.2f}")
with k5:
    st.metric("Avg PACE Risk Score",
              f"{avg_risk:.1f}%" if MODEL_READY else "N/A")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Donut + Risk distribution ─────────────────────────────
col_l, col_r = st.columns(2, gap="medium")

with col_l:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Charges by Type")
            st.caption("Which PACE charge types are costing the most?")
        with btn:
            if st.button("⤢", key="exp_donut"):
                _popup_donut()
        if not df_with_acc.empty:
            st.plotly_chart(
                _build_donut_fig(df_with_acc, total_acc),
                use_container_width=True,
            )
            type_summary = (
                df_with_acc.groupby(type_col)["accessorial_charge_usd"]
                .agg(Total="sum", Count="count")
                .reset_index()
                .rename(columns={type_col: "Charge Type"})
                .sort_values("Total", ascending=False)
            )
            st.dataframe(
                type_summary,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Total": st.column_config.NumberColumn(format="$%.2f")
                },
            )
        else:
            st.info("No accessorial charges match current filters.")

with col_r:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Risk Score by Charge Type")
            st.caption("PACE risk score distribution per charge category")
        with btn:
            if st.button("⤢", key="exp_risk_dist"):
                _popup_risk_dist()
        if not df_with_acc.empty and MODEL_READY:
            st.plotly_chart(
                _build_risk_distribution_fig(df_with_acc),
                use_container_width=True,
            )
        elif not MODEL_READY:
            st.markdown(
                "<div style='text-align:center;padding:60px 20px;"
                "color:#9CA3AF;'>"
                "<div style='font-size:32px;'>📊</div>"
                "<div style='font-size:13px;margin-top:8px;'>"
                "Available after PACE model training</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No data for current filters.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Carrier bar + trend ────────────────────────────────────
col_a, col_b = st.columns(2, gap="medium")

with col_a:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Charges by Carrier")
            st.caption("Top 15 carriers by total accessorial spend")
        with btn:
            if st.button("⤢", key="exp_carrier"):
                _popup_carrier()
        if not df_with_acc.empty:
            st.plotly_chart(
                _build_carrier_fig(df_with_acc),
                use_container_width=True,
            )
        else:
            st.info("No data for current filters.")

with col_b:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Accessorial Cost Trend")
            st.caption("Weekly accessorial spend over time")
        with btn:
            if st.button("⤢", key="exp_trend"):
                _popup_trend()
        if not df_with_acc.empty:
            st.plotly_chart(
                _build_trend_fig(df_with_acc),
                use_container_width=True,
            )
        else:
            st.info("No trend data available.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: Avg charge by PACE risk tier ───────────────────────────
with st.container(border=True):
    st.markdown("#### Avg Accessorial Charge by PACE Risk Label")
    st.caption(
        "Higher PACE risk scores should correlate with higher actual charges. "
        "This validates model accuracy over time."
    )
    if MODEL_READY and "risk_label" in df.columns:
        tier_col, table_col = st.columns([3, 2], gap="large")
        with tier_col:
            st.plotly_chart(
                _build_risk_tier_fig(df_with_acc),
                use_container_width=True,
            )
        with table_col:
            tier_table = (
                df.groupby("risk_label")
                .agg(
                    Shipments   =("accessorial_charge_usd", "count"),
                    Avg_Acc     =("accessorial_charge_usd", "mean"),
                    Total_Acc   =("accessorial_charge_usd", "sum"),
                    Pct_Charged =("accessorial_charge_usd",
                                  lambda x: f"{(x > 0).mean()*100:.1f}%"),
                    Avg_Risk    =("risk_score_pct", "mean"),
                )
                .reset_index()
                .rename(columns={
                    "risk_label":  "Risk Label",
                    "Avg_Acc":     "Avg Charge ($)",
                    "Total_Acc":   "Total Charge ($)",
                    "Pct_Charged": "% Charged",
                    "Avg_Risk":    "Avg Risk %",
                })
            )
            order = ["None", "Low", "Medium", "High", "Critical"]
            tier_table["_ord"] = tier_table["Risk Label"].map(
                {t: i for i, t in enumerate(order)}
            )
            tier_table = tier_table.sort_values("_ord").drop(columns="_ord")
            st.dataframe(
                tier_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Avg Charge ($)":   st.column_config.NumberColumn(format="$%.2f"),
                    "Total Charge ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Avg Risk %":       st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
    else:
        st.info(
            "Risk tier analysis available after PACE model training. "
            "Upload scored data on the Upload page to populate this section.",
            icon="ℹ️"
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Scored records table ──────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### Shipment-Level PACE Predictions")
    st.caption(
        "Individual records with PACE risk scores and charge type predictions. "
        "Click a row to view full detail."
    )

    # Build display columns from whatever is available
    candidate_cols = [
        "unique_id", "shipment_id", "dot_number",
        "ship_date", "carrier", "carrier_phy_state",
        "risk_score_pct", "risk_label", "charge_type",
        "accessorial_charge_usd",
    ]
    display_cols = [c for c in candidate_cols if c in df.columns]

    if not display_cols:
        st.info("No displayable columns found in current dataset.")
    else:
        search = st.text_input(
            "Search by ID or carrier",
            placeholder="Type to filter...",
            label_visibility="collapsed",
        )
        display_df = df[display_cols].copy()

        if search.strip():
            str_cols = display_df.select_dtypes(include=["object"]).columns
            mask = pd.Series(False, index=display_df.index)
            for col in str_cols:
                mask |= (
                    display_df[col].astype(str)
                    .str.contains(search.strip(), case=False, na=False)
                )
            display_df = display_df[mask]

        col_config = {}
        if "risk_score_pct" in display_df.columns:
            col_config["risk_score_pct"] = st.column_config.ProgressColumn(
                "Risk Score",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            )

        event = st.dataframe(
            display_df.head(500),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=col_config,
            height=min(420, 40 + len(display_df.head(500)) * 35),
        )

        selected = (
            event.selection.get("rows", [])
            if hasattr(event, "selection") else []
        )
        if selected:
            row_data = df[display_cols].iloc[selected[0]].to_dict()
            _show_risk_detail(row_data)

        if len(display_df) > 500:
            st.caption(
                f"Showing first 500 of {len(display_df):,} records. "
                "Use filters above to narrow results."
            )
