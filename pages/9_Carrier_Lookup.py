# File: pages/9_Carrier_Lookup.py
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.styling import inject_css, sidebar_account, TIER_COLORS, CHARGE_COLORS
from pipeline.config import CHARGE_TYPE_LABELS, is_pace_model_ready

st.set_page_config(
    page_title="PACE — Carrier Lookup",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

require_auth()
username = st.session_state.get("username", "User")
sidebar_account(username)

# ── Constants ─────────────────────────────────────────────────────
MODEL_READY = is_pace_model_ready()

API_ENABLED = os.environ.get("PACE_ENV", "").lower() == "production"

FMCSA_VIOLATION_LABELS = {
    "basic_viol":        "Basic Violations",
    "unsafe_viol":       "Unsafe Driving",
    "fatigued_viol":     "HOS/Fatigued",
    "dr_fitness_viol":   "Driver Fitness",
    "subt_alcohol_viol": "Alcohol/Drug",
    "vh_maint_viol":     "Vehicle Maint.",
    "hm_viol":           "Hazmat",
}

# ── Header ────────────────────────────────────────────────────────
st.markdown("## Carrier DOT Lookup")
st.caption(
    "Enter a USDOT number to pull live FMCSA carrier data and run a "
    "PACE accessorial risk prediction."
)

if not MODEL_READY:
    st.info(
        "PACE model not yet trained — predictions will be available after "
        "training completes. FMCSA data lookup is still available.",
        icon="⏳"
    )

if not API_ENABLED:
    st.info(
        "Running in standard mode — predictions use Teradata historical data. "
        "Live FMCSA carrier enrichment activates when PACE_ENV=production is set.",
        icon="ℹ️"
    )

st.divider()

# ── Search bar ────────────────────────────────────────────────────
search_col, btn_col = st.columns([3, 1], gap="medium")

with search_col:
    dot_input = st.text_input(
        "USDOT Number",
        placeholder="e.g. 3483464",
        label_visibility="collapsed",
        help="Enter the USDOT number for any registered motor carrier",
    )

with btn_col:
    lookup_clicked = st.button(
        "Look Up Carrier →",
        type="primary",
        use_container_width=True,
        disabled=not dot_input.strip(),
    )

# ── Recent lookups ────────────────────────────────────────────────
if "dot_lookup_history" not in st.session_state:
    st.session_state["dot_lookup_history"] = []

if st.session_state["dot_lookup_history"]:
    st.markdown(
        "<p style='color:#64748B;font-size:12px;margin:4px 0;'>"
        "Recent lookups:</p>",
        unsafe_allow_html=True,
    )
    history_cols = st.columns(min(len(st.session_state["dot_lookup_history"]), 5))
    for i, prev_dot in enumerate(st.session_state["dot_lookup_history"][-5:]):
        with history_cols[i]:
            if st.button(str(prev_dot), key=f"hist_{prev_dot}",
                         use_container_width=True):
                dot_input = str(prev_dot)
                lookup_clicked = True

# ── Run lookup ────────────────────────────────────────────────────
if lookup_clicked and dot_input.strip():
    try:
        dot_number = int(dot_input.strip())
    except ValueError:
        st.error("Please enter a valid numeric DOT number.")
        st.stop()

    # Add to history
    history = st.session_state["dot_lookup_history"]
    if dot_number not in history:
        history.append(dot_number)
        st.session_state["dot_lookup_history"] = history[-10:]

    with st.spinner(f"Looking up DOT {dot_number}..."):
        result    = None
        fmcsa_raw = {}

        if MODEL_READY:
            try:
                from pipeline.inference import get_inference_engine
                engine = get_inference_engine()
                result = engine.predict_dot(
                    dot_number=dot_number,
                    origin_state=None,
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Also pull raw FMCSA data if API enabled
        if API_ENABLED:
            try:
                from pipeline.api_integration import get_enricher
                enricher  = get_enricher()
                fmcsa_raw = enricher.fmcsa.build_realtime_features(dot_number)
            except Exception:
                fmcsa_raw = {}

        st.session_state["dot_result"]    = result
        st.session_state["dot_fmcsa_raw"] = fmcsa_raw
        st.session_state["dot_number"]    = dot_number

# ── Display results ───────────────────────────────────────────────
if st.session_state.get("dot_result") or st.session_state.get("dot_fmcsa_raw"):
    result    = st.session_state.get("dot_result", {})
    fmcsa_raw = st.session_state.get("dot_fmcsa_raw", {})
    dot_number = st.session_state.get("dot_number", "")

    # Handle error results
    if result and "error" in result:
        st.warning(f"DOT {dot_number}: {result['error']}")
        result = {}

    # Merge data sources
    combined = {**fmcsa_raw, **(result or {})}

    score       = float(combined.get("risk_score", 0))
    label       = combined.get("risk_label", combined.get("risk_tier", "Unknown"))
    charge      = combined.get("charge_type", "Unknown")
    carrier_name = combined.get("carrier_name", f"DOT {dot_number}")
    data_source  = combined.get("data_source", "teradata_historical")
    color        = TIER_COLORS.get(label, "#94A3B8")
    c_color      = CHARGE_COLORS.get(charge, "#A78BFA")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Carrier header ────────────────────────────────────────────
    with st.container(border=True):
        h1, h2, h3 = st.columns([5, 1, 1])
        with h1:
            st.markdown(f"### {carrier_name}")
            st.markdown(
                f"<span style='color:#64748B;font-size:13px;'>"
                f"USDOT {dot_number}</span>",
                unsafe_allow_html=True,
            )
        with h2:
            src_color = "#34D399" if data_source == "live_fmcsa" else "#60A5FA"
            src_label = "Live FMCSA" if data_source == "live_fmcsa" else "Historical"
            st.markdown(
                f"<div style='text-align:right;padding-top:8px;'>"
                f"<span style='border:1px solid {src_color};color:{src_color};"
                f"border-radius:4px;padding:3px 8px;font-size:11px;'>"
                f"{src_label}</span></div>",
                unsafe_allow_html=True,
            )
        with h3:
            if label and label != "Unknown":
                st.markdown(
                    f"<div style='text-align:right;padding-top:8px;'>"
                    f"<span style='border:1px solid {color};color:{color};"
                    f"border-radius:4px;padding:3px 8px;font-size:11px;"
                    f"font-weight:700;'>{label}</span></div>",
                    unsafe_allow_html=True,
                )

        # Carrier profile fields
        profile_fields = {
            "Status":        combined.get("carrier_status_code", ""),
            "Operation":     combined.get("carrier_carrier_operation", ""),
            "Power Units":   combined.get("carrier_power_units", ""),
            "Total Drivers": combined.get("carrier_total_drivers", ""),
            "State":         combined.get("carrier_phy_state", ""),
            "Safety Rating": combined.get("carrier_safety_rating", ""),
            "HM Carrier":    combined.get("carrier_hm_ind", ""),
        }
        available = {k: v for k, v in profile_fields.items()
                     if v is not None
                     and str(v).strip() not in ("", "0", "UNKNOWN", "nan", "None", "NaN", "N/A")}
        if available:
            st.divider()
            pcols = st.columns(min(len(available), 7))
            for i, (lbl, val) in enumerate(available.items()):
                with pcols[i]:
                    st.markdown(f"**{lbl}**")
                    st.write(str(val))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two column layout: gauge + violations ─────────────────────
    left_col, right_col = st.columns([2, 3], gap="medium")

    with left_col:

        # Risk gauge
        if MODEL_READY and score > 0:
            with st.container(border=True):
                st.markdown("#### PACE Risk Score")
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(score, 1),
                    number={"suffix": "%",
                            "font": {"size": 38, "color": color}},
                    title={"text": f"<b>{label} Risk</b>",
                           "font": {"size": 14, "color": color}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickfont": {"color": "#94A3B8", "size": 10},
                        },
                        "bar":     {"color": color, "thickness": 0.26},
                        "bgcolor": "#0f0a1e",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  25],
                             "color": "rgba(5,150,105,0.15)"},
                            {"range": [25, 50],
                             "color": "rgba(251,146,60,0.10)"},
                            {"range": [50, 75],
                             "color": "rgba(217,119,6,0.15)"},
                            {"range": [75,100],
                             "color": "rgba(220,38,38,0.15)"},
                        ],
                    },
                ))
                gauge_fig.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=50, b=10),
                    paper_bgcolor="#0f0a1e",
                    font={"color": "#A78BFA"},
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

                # Predicted charge type
                st.markdown(
                    f"<div style='background:rgba(0,0,0,0.3);border:1px solid "
                    f"{c_color};border-radius:6px;padding:10px 14px;"
                    f"margin-top:4px;'>"
                    f"<div style='color:#94A3B8;font-size:10px;font-weight:600;"
                    f"letter-spacing:1px;text-transform:uppercase;"
                    f"margin-bottom:4px;'>Predicted Charge</div>"
                    f"<div style='color:{c_color};font-size:16px;"
                    f"font-weight:700;'>{charge}</div></div>",
                    unsafe_allow_html=True,
                )

        # SMS scores
        sms_fields = {
            "SMS Power Units": combined.get("sms_nbr_power_unit", 0),
            "SMS Drivers":     combined.get("sms_driver_total", 0),
            "Recent Mileage":  combined.get("sms_recent_mileage", 0),
        }
        sms_available = {k: v for k, v in sms_fields.items()
                         if v and float(v) > 0}
        if sms_available:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("#### SMS Profile")
                for lbl, val in sms_available.items():
                    st.metric(lbl, f"{int(float(val)):,}")

    with right_col:

        # Violation breakdown
        viol_data = {
            label: int(float(combined.get(col, 0)))
            for col, label in FMCSA_VIOLATION_LABELS.items()
        }
        has_viols = any(v > 0 for v in viol_data.values())

        with st.container(border=True):
            st.markdown("#### Violation Profile")
            oos_total   = int(float(combined.get("oos_total", 0)))
            driver_oos  = int(float(combined.get("driver_oos_total", 0)))
            vehicle_oos = int(float(combined.get("vehicle_oos_total", 0)))

            oos1, oos2, oos3 = st.columns(3)
            with oos1:
                st.metric("OOS Total", oos_total)
            with oos2:
                st.metric("Driver OOS", driver_oos)
            with oos3:
                st.metric("Vehicle OOS", vehicle_oos)

            st.divider()

            if has_viols:
                max_viol = max(viol_data.values()) or 1
                for viol_label, count in sorted(
                    viol_data.items(), key=lambda x: -x[1]
                ):
                    pct = count / max_viol
                    bar_color = (
                        "#F87171" if count > 5 else
                        "#FCD34D" if count > 0 else
                        "#374151"
                    )
                    v1, v2, v3 = st.columns([3, 5, 1])
                    with v1:
                        st.markdown(
                            f"<span style='font-size:12px;color:#CBD5E1;'>"
                            f"{viol_label}</span>",
                            unsafe_allow_html=True,
                        )
                    with v2:
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.08);"
                            f"border-radius:3px;height:8px;margin-top:8px;'>"
                            f"<div style='width:{pct*100:.0f}%;"
                            f"background:{bar_color};height:8px;"
                            f"border-radius:3px;'></div></div>",
                            unsafe_allow_html=True,
                        )
                    with v3:
                        st.markdown(
                            f"<span style='font-size:12px;font-weight:600;"
                            f"color:#E2E8F0;'>{count}</span>",
                            unsafe_allow_html=True,
                        )
            else:
                st.success("No violations on record.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Crash history
        with st.container(border=True):
            st.markdown("#### Crash History")
            crash_count  = int(float(combined.get("crash_count", 0)))
            crash_fatal  = int(float(combined.get("crash_fatalities_total", 0)))
            crash_injure = int(float(combined.get("crash_injuries_total", 0)))
            crash_tow    = int(float(combined.get("crash_towaway_total", 0)))
            crash_sev    = float(combined.get("crash_avg_severity", 0))

            cr1, cr2, cr3, cr4 = st.columns(4)
            with cr1:
                st.metric("Total Crashes", crash_count)
            with cr2:
                st.metric("Fatalities", crash_fatal)
            with cr3:
                st.metric("Injuries", crash_injure)
            with cr4:
                st.metric("Towaways", crash_tow)

            if crash_count == 0:
                st.success("No crashes on record.")
            elif crash_count > 5:
                st.warning(
                    f"High crash count ({crash_count}) — "
                    f"elevated risk for this carrier."
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charge type probabilities ─────────────────────────────────
    probs = combined.get("probabilities", {})
    if probs and MODEL_READY:
        with st.container(border=True):
            st.markdown("#### Charge Type Probabilities")
            st.caption(
                "PACE model confidence across all six accessorial charge types"
            )
            prob_items = sorted(
                probs.items(), key=lambda x: x[1], reverse=True
            )
            labels = [p[0] for p in prob_items]
            values = [round(p[1] * 100, 1) for p in prob_items]
            colors = [CHARGE_COLORS.get(l, "#A78BFA") for l in labels]

            prob_fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                textfont={"color": "#E2E8F0"},
            ))
            prob_fig.update_layout(
                margin=dict(l=0, r=60, t=8, b=0), height=240,
                plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
                font=dict(color="#A78BFA"),
                xaxis=dict(
                    ticksuffix="%", range=[0, 110],
                    color="#94A3B8",
                    gridcolor="rgba(150,50,200,0.15)",
                ),
                yaxis=dict(
                    color="#94A3B8",
                    gridcolor="rgba(150,50,200,0.15)",
                    autorange="reversed",
                ),
            )
            st.plotly_chart(prob_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommended actions ───────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### Recommended Actions")

        total_viols = sum(viol_data.values())
        actions = []

        if label in ("Critical", "High"):
            actions.append(
                "Flag carrier for compliance review before booking new loads."
            )
            if oos_total > 5:
                actions.append(
                    f"OOS count of {oos_total} is above threshold — "
                    f"consider alternative carrier."
                )
            if crash_count > 3:
                actions.append(
                    f"{crash_count} crashes on record — "
                    f"escalate to risk management team."
                )
            actions.append(
                "Add accessorial buffer of 20–30% to quotes involving this carrier."
            )
            actions.append(
                "Request morning appointment windows to reduce detention exposure."
            )
        elif label == "Medium":
            actions.append(
                "Monitor carrier performance on active loads."
            )
            actions.append(
                "Add a 10–15% accessorial buffer to quotes for this carrier."
            )
            if total_viols > 0:
                actions.append(
                    f"{total_viols} violations on record — "
                    f"verify compliance status before high-value loads."
                )
        else:
            actions.append(
                "Carrier profile is within acceptable risk parameters."
            )
            actions.append(
                "Standard operating procedure applies — no additional buffer needed."
            )

        if combined.get("carrier_hm_ind") == "Y":
            actions.append(
                "Hazmat carrier — ensure proper placarding and documentation "
                "on all HM shipments."
            )

        for i, action in enumerate(actions, 1):
            action_color = (
                "#F87171" if label in ("Critical", "High") else
                "#FCD34D" if label == "Medium" else
                "#34D399"
            )
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;"
                f"gap:10px;margin:8px 0;'>"
                f"<span style='background:rgba(147,51,234,0.3);"
                f"color:#A78BFA;border-radius:50%;width:22px;height:22px;"
                f"min-width:22px;display:flex;align-items:center;"
                f"justify-content:center;font-size:12px;"
                f"font-weight:700;'>{i}</span>"
                f"<span style='font-size:13px;color:#E2E8F0;'>{action}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Raw data expander ─────────────────────────────────────────
    with st.expander("🔧 Raw Data", expanded=False):
        st.json({k: str(v) for k, v in combined.items()
                 if not str(k).startswith("_")})

else:
    # ── Empty state ───────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;padding:100px 20px;color:#9CA3AF;'>"
        "<div style='font-size:56px;'>🔍</div>"
        "<div style='font-size:16px;font-weight:600;margin-top:16px;"
        "color:#E2E8F0;'>Search for a carrier</div>"
        "<div style='font-size:13px;margin-top:8px;max-width:400px;"
        "margin-left:auto;margin-right:auto;'>"
        "Enter a USDOT number above to pull live FMCSA carrier data "
        "and get a PACE accessorial risk prediction"
        "</div>"
        "<div style='margin-top:24px;font-size:12px;color:#475569;'>"
        "Try DOT numbers: 3483464 · 1234567 · 2891234"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )