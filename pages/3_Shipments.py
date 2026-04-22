import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.database import load_shipments_with_fallback
from utils.styling import (
    inject_css, sidebar_account,
    NAVY_500,
    TIER_BG_FG, CHARGE_COLORS,
    risk_score_to_label, search_dataframe,
)
from pipeline.config import CHARGE_TYPE_LABELS, is_pace_model_ready
from pages._pace_ui import get_pace_mode, show_mode_badge

st.set_page_config(
    page_title="PACE — Shipments",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

require_auth()
username = st.session_state.get("username", "User")
sidebar_account(username)
show_mode_badge()

# ── Model availability ────────────────────────────────────────────
MODEL_READY = is_pace_model_ready(mode=get_pace_mode())

# ── Load data ─────────────────────────────────────────────────────
df_all = load_shipments_with_fallback()

# ── Normalize schema ──────────────────────────────────────────────
# risk_score is already 0-100 after database.py normalization
if "risk_score_pct" not in df_all.columns:
    if "risk_score" in df_all.columns:
        df_all["risk_score_pct"] = df_all["risk_score"]  # already 0-100
    else:
        df_all["risk_score_pct"] = 0.0

if "risk_label" not in df_all.columns:
    if "risk_tier" in df_all.columns:
        df_all["risk_label"] = df_all["risk_tier"]
    else:
        df_all["risk_label"] = df_all["risk_score_pct"].apply(risk_score_to_label)

if "charge_type" not in df_all.columns:
    df_all["charge_type"] = df_all.get("accessorial_type", "Unknown")

# Merge in PACE scored data from session if available
if st.session_state.get("upload_scored") is not None:
    scored = st.session_state["upload_scored"]
    id_col = next(
        (c for c in ["unique_id", "dot_number", "shipment_id"]
         if c in scored.columns and c in df_all.columns),
        None
    )
    if id_col and "charge_type" in scored.columns:
        df_all = df_all.merge(
            scored[[id_col, "charge_type", "risk_score_pct", "risk_label"]],
            on=id_col, how="left", suffixes=("", "_pace")
        )
        for col in ["charge_type", "risk_score_pct", "risk_label"]:
            if f"{col}_pace" in df_all.columns:
                df_all[col] = df_all[f"{col}_pace"].fillna(df_all[col])
                df_all.drop(columns=[f"{col}_pace"], inplace=True)

# ── ID column detection ───────────────────────────────────────────
ID_COL = next(
    (c for c in ["shipment_id", "unique_id", "dot_number"]
     if c in df_all.columns),
    df_all.columns[0] if len(df_all.columns) > 0 else "id"
)

# ── Filters ───────────────────────────────────────────────────────
with st.expander("Filters", expanded=False):
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        carrier_col = "carrier" if "carrier" in df_all.columns else None
        if carrier_col:
            carriers = sorted(df_all[carrier_col].dropna().unique())
            sel_carriers = st.multiselect(
                "Carrier", carriers, default=carriers, key="ship_carriers"
            )
        else:
            sel_carriers = []

    with f2:
        facility_col = "facility" if "facility" in df_all.columns else None
        if facility_col:
            facilities = sorted(df_all[facility_col].dropna().unique())
            sel_facilities = st.multiselect(
                "Facility", facilities, default=facilities, key="ship_facilities"
            )
        else:
            sel_facilities = []

    with f3:
        all_labels = sorted(df_all["risk_label"].dropna().unique())
        sel_labels = st.multiselect(
            "Risk Label", all_labels, default=all_labels, key="ship_labels"
        )

    with f4:
        all_charges = sorted(df_all["charge_type"].dropna().unique())
        sel_charges = st.multiselect(
            "Charge Type", all_charges, default=all_charges, key="ship_charges"
        )

# ── Apply filters ─────────────────────────────────────────────────
df = df_all.copy()
if sel_carriers and carrier_col:
    df = df[df[carrier_col].isin(sel_carriers)]
if sel_facilities and facility_col:
    df = df[df[facility_col].isin(sel_facilities)]
if sel_labels:
    df = df[df["risk_label"].isin(sel_labels)]
if sel_charges:
    df = df[df["charge_type"].isin(sel_charges)]

# ── Session state ─────────────────────────────────────────────────
if "selected_shipment" not in st.session_state:
    st.session_state["selected_shipment"] = None


# ── Detail view ───────────────────────────────────────────────────
def render_detail(row: pd.Series):
    label   = str(row.get("risk_label", row.get("risk_tier", "Unknown")))
    score   = float(row.get("risk_score_pct", row.get("risk_score", 0) * 100))
    charge  = str(row.get("charge_type", row.get("accessorial_type", "Unknown")))
    bg, fg  = TIER_BG_FG.get(label, ("#1E293B", "#94A3B8"))
    c_color = CHARGE_COLORS.get(charge, "#A78BFA")
    ship_id = row.get(ID_COL, "Unknown")

    if st.button("← Back to Shipments"):
        st.session_state["selected_shipment"] = None
        st.rerun()

    st.markdown(
        f"<div style='font-size:12px;color:#94A3B8;margin-bottom:12px;'>"
        f"Shipments / <b style='color:#E2E8F0'>{ship_id}</b></div>",
        unsafe_allow_html=True,
    )

    # ── Header card ───────────────────────────────────────────────
    with st.container(border=True):
        h1, h2, h3 = st.columns([5, 1, 1])
        with h1:
            st.markdown(f"### {ship_id}")
        with h2:
            st.markdown(
                f"<div style='text-align:right;padding-top:6px;'>"
                f"<span style='background:{bg};color:{fg};padding:4px 12px;"
                f"border-radius:4px;font-size:12px;font-weight:700;'>"
                f"{label.upper()}</span></div>",
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                f"<div style='text-align:right;padding-top:6px;'>"
                f"<span style='border:1px solid {c_color};color:{c_color};"
                f"padding:4px 10px;border-radius:4px;font-size:11px;"
                f"font-weight:600;'>{charge}</span></div>",
                unsafe_allow_html=True,
            )

        # Key fields — show whatever is available
        field_map = {
            "carrier":           "Carrier",
            "facility":          "Facility",
            "ship_date":         "Ship Date",
            "dot_number":        "DOT Number",
            "carrier_phy_state": "State",
            "base_freight_usd":  "Base Freight",
            "accessorial_charge_usd": "Accessorial ($)",
        }
        available = {
            label: row[col]
            for col, label in field_map.items()
            if col in row.index and pd.notna(row.get(col))
        }
        if available:
            cols = st.columns(min(len(available), 5))
            for i, (lbl, val) in enumerate(list(available.items())[:5]):
                with cols[i]:
                    st.markdown(f"**{lbl}**")
                    if isinstance(val, float):
                        st.write(f"${val:,.2f}" if "($)" in lbl or "Freight" in lbl
                                 else f"{val:,.0f}")
                    else:
                        st.write(str(val)[:10] if "Date" in lbl else str(val))

    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 3], gap="medium")

    # ── Risk score panel ──────────────────────────────────────────
    with left_col:
        with st.container(border=True):
            st.markdown("#### PACE Risk Score")

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(score, 1),
                number={"suffix": "%", "font": {"size": 36, "color": fg}},
                title={"text": f"<b>{label}</b>",
                       "font": {"size": 14, "color": fg}},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickfont": {"color": "#94A3B8", "size": 10}},
                    "bar":  {"color": fg, "thickness": 0.25},
                    "bgcolor": "#0f0a1e",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  25], "color": "rgba(5,150,105,0.15)"},
                        {"range": [25, 50], "color": "rgba(251,146,60,0.10)"},
                        {"range": [50, 75], "color": "rgba(217,119,6,0.15)"},
                        {"range": [75,100], "color": "rgba(220,38,38,0.15)"},
                    ],
                },
            ))
            gauge_fig.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#0f0a1e",
                font={"color": "#A78BFA"},
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Predicted charge type
            st.markdown(
                f"<div style='background:rgba(0,0,0,0.3);border:1px solid "
                f"{c_color};border-radius:6px;padding:10px 14px;margin-top:8px;'>"
                f"<div style='color:#94A3B8;font-size:10px;font-weight:600;"
                f"letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;'>"
                f"Predicted Charge Type</div>"
                f"<div style='color:{c_color};font-size:16px;font-weight:700;'>"
                f"{charge}</div></div>",
                unsafe_allow_html=True,
            )

            # Charge probabilities if available
            prob_cols = [c for c in row.index if str(c).startswith("prob_")]
            if prob_cols:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='color:#94A3B8;font-size:11px;"
                    "margin:0 0 6px;'>Charge Probabilities</p>",
                    unsafe_allow_html=True,
                )
                for pc in sorted(prob_cols,
                                  key=lambda x: float(row[x]),
                                  reverse=True)[:4]:
                    pct = float(row[pc]) * 100
                    lbl = pc.replace("prob_", "").replace("_", " ").title()
                    bar_color = CHARGE_COLORS.get(lbl, "#A78BFA")
                    st.markdown(
                        f"<div style='display:flex;align-items:center;"
                        f"gap:8px;margin-bottom:4px;'>"
                        f"<span style='color:#94A3B8;font-size:11px;"
                        f"min-width:120px;'>{lbl}</span>"
                        f"<div style='flex:1;background:rgba(255,255,255,0.08);"
                        f"border-radius:3px;height:6px;'>"
                        f"<div style='width:{pct:.0f}%;background:{bar_color};"
                        f"height:6px;border-radius:3px;'></div></div>"
                        f"<span style='color:#E2E8F0;font-size:11px;"
                        f"min-width:36px;text-align:right;'>{pct:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Risk factors + recommendations ────────────────────────────
    with right_col:
        with st.container(border=True):
            st.markdown("#### Risk Factor Breakdown")

            # Use PACE violation data if available
            viol_factors = {}
            if "oos_total" in row.index:
                viol_factors["OOS Violations"]     = float(row.get("oos_total", 0))
            if "basic_viol" in row.index:
                viol_factors["Basic Violations"]   = float(row.get("basic_viol", 0))
            if "unsafe_viol" in row.index:
                viol_factors["Unsafe Violations"]  = float(row.get("unsafe_viol", 0))
            if "crash_count" in row.index:
                viol_factors["Crash History"]      = float(row.get("crash_count", 0)) * 5
            if "vh_maint_viol" in row.index:
                viol_factors["Vehicle Maint."]     = float(row.get("vh_maint_viol", 0))

            # Fall back to old schema factors
            if not viol_factors:
                w = float(row.get("weight_lbs", 0))
                m = float(row.get("miles", 0))
                viol_factors = {
                    "Carrier History":  0.30,
                    "Facility Profile": 0.25,
                    "Miles / Distance": min(m / 5000 * 0.20, 0.20),
                    "Shipment Weight":  min(w / 44000 * 0.15, 0.15),
                    "Base Freight":     0.10,
                }

            total_w = sum(viol_factors.values()) or 1
            for factor, weight in sorted(
                viol_factors.items(), key=lambda x: -x[1]
            )[:6]:
                pct = (weight / total_w) * 100
                fc1, fc2, fc3 = st.columns([3, 5, 1])
                with fc1:
                    st.markdown(
                        f"<span style='font-size:12px;color:#CBD5E1;'>"
                        f"{factor}</span>",
                        unsafe_allow_html=True,
                    )
                with fc2:
                    st.progress(pct / 100)
                with fc3:
                    st.markdown(
                        f"<span style='font-size:12px;font-weight:600;"
                        f"color:#E2E8F0;'>{pct:.0f}%</span>",
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Recommended Actions")

            oos    = float(row.get("oos_total", 0))
            viols  = float(row.get("basic_viol", 0)) + float(row.get("unsafe_viol", 0))
            acc_est = float(row.get("accessorial_charge_usd", 0))

            if label in ("Critical", "High"):
                actions = [
                    "Escalate to carrier compliance team before dispatch.",
                    f"Add accessorial buffer of ${acc_est * 0.8:,.0f}–"
                    f"${acc_est * 1.3:,.0f} to quote.",
                    "Request morning appointment window to reduce detention risk.",
                    "Verify dock availability and confirm appointment 24h in advance.",
                ]
                if oos > 5:
                    actions.append(
                        f"Carrier has {oos:.0f} OOS events — "
                        f"consider alternative carrier for this lane."
                    )
                if viols > 10:
                    actions.append(
                        f"High violation count ({viols:.0f}) — "
                        f"flag for safety review before booking."
                    )
            elif label == "Medium":
                actions = [
                    "Monitor appointment confirmation — delays increase detention risk.",
                    f"Consider ${acc_est * 0.5:,.0f}–${acc_est:,.0f} contingency buffer.",
                    "Verify carrier's recent performance on similar lanes.",
                ]
            else:
                actions = [
                    "Low risk profile — standard operating procedure applies.",
                    "No additional accessorial buffer needed for this load.",
                ]

            for i, action in enumerate(actions, 1):
                st.markdown(
                    f"<div style='display:flex;align-items:flex-start;"
                    f"gap:10px;margin:8px 0;'>"
                    f"<span style='background:{NAVY_500};color:white;"
                    f"border-radius:50%;width:22px;height:22px;min-width:22px;"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"font-size:12px;font-weight:700;'>{i}</span>"
                    f"<span style='font-size:13px;color:#E2E8F0;'>{action}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Similar shipments ─────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### Similar Shipments — Historical Comparison")
        carrier_val = row.get("carrier", row.get("carrier_phy_state", None))
        if carrier_val and carrier_col:
            similar = df_all[df_all[carrier_col] == carrier_val].head(10)
        else:
            similar = df_all.head(10)

        sim_cols = [c for c in [
            ID_COL, "ship_date", "facility", "carrier_phy_state",
            "risk_score_pct", "risk_label", "charge_type",
            "accessorial_charge_usd",
        ] if c in similar.columns]

        col_config = {}
        if "risk_score_pct" in sim_cols:
            col_config["risk_score_pct"] = st.column_config.ProgressColumn(
                "Risk Score", format="%.1f%%", min_value=0, max_value=100
            )
        if "accessorial_charge_usd" in sim_cols:
            col_config["accessorial_charge_usd"] = st.column_config.NumberColumn(
                "Actual Charge ($)", format="$%.2f"
            )

        st.dataframe(
            similar[sim_cols],
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
        )


# ── Main list view ────────────────────────────────────────────────
if st.session_state["selected_shipment"] is not None:
    sid   = st.session_state["selected_shipment"]
    match = df_all[df_all[ID_COL].astype(str) == str(sid)]
    if not match.empty:
        render_detail(match.iloc[0])
    else:
        st.warning("Shipment not found.")
        st.session_state["selected_shipment"] = None
        st.rerun()

else:
    st.markdown("## Shipments")
    st.caption(f"{len(df):,} shipments match current filters")

    if not MODEL_READY:
        st.info(
            "PACE model not yet trained — risk scores are placeholders. "
            "Scores will update automatically after training completes.",
            icon="ℹ️"
        )

    st.divider()

    search = st.text_input(
        "Search",
        placeholder="🔍 Search by shipment ID, carrier, or DOT number...",
        label_visibility="collapsed",
    )
    if search.strip():
        df = search_dataframe(df, search)

    # ── Build display columns ─────────────────────────────────────
    candidate_cols = [
        ID_COL, "ship_date", "carrier", "facility",
        "dot_number", "carrier_phy_state",
        "weight_lbs", "miles",
        "risk_score_pct", "risk_label", "charge_type",
        "accessorial_charge_usd", "base_freight_usd",
    ]
    display_cols = [c for c in candidate_cols if c in df.columns]

    col_config = {}
    if "risk_score_pct" in display_cols:
        col_config["risk_score_pct"] = st.column_config.ProgressColumn(
            "Risk Score", format="%.1f%%", min_value=0, max_value=100
        )
    if "base_freight_usd" in display_cols:
        col_config["base_freight_usd"] = st.column_config.NumberColumn(
            "Base Freight ($)", format="$%.2f"
        )
    if "accessorial_charge_usd" in display_cols:
        col_config["accessorial_charge_usd"] = st.column_config.NumberColumn(
            "Est. Accessorial ($)", format="$%.2f"
        )

    with st.container(border=True):
        event = st.dataframe(
            df[display_cols].head(500),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=col_config,
            height=500,
        )

    selected_rows = (
        event.selection.get("rows", [])
        if hasattr(event, "selection") else []
    )
    if selected_rows:
        row_idx = selected_rows[0]
        sid     = df[display_cols].head(500).iloc[row_idx][ID_COL]
        st.session_state["selected_shipment"] = sid
        st.rerun()

    st.caption("Click any row to view full shipment detail and PACE prediction.")

    if len(df) > 500:
        st.caption(
            f"Showing first 500 of {len(df):,} records. "
            "Use filters to narrow results."
        )
