import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.database import load_shipments_with_fallback
from utils.styling import (
    inject_css,
    sidebar_account,
    TIER_COLORS,
    CHARGE_COLORS,
)
from pipeline.config import is_pace_model_ready

st.set_page_config(
    page_title="PACE — Carrier Benchmarking",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

require_auth()
username = st.session_state.get("username", "User")
sidebar_account(username)

df_raw = load_shipments_with_fallback()
df_raw["ship_date_dt"] = pd.to_datetime(df_raw["ship_date"])
df_all = df_raw  # module-level alias used by dialogs

ALL_CARRIERS = sorted(df_all["carrier"].dropna().unique())

CARRIER_COLORS = [
    "#0F2B4A", "#2563A8", "#059669", "#D97706",
    "#DC2626", "#7C3AED", "#0891B2", "#BE185D",
]
color_map = {c: CARRIER_COLORS[i % len(CARRIER_COLORS)]
             for i, c in enumerate(ALL_CARRIERS)}

MODEL_READY = is_pace_model_ready()
API_ENABLED = os.environ.get("PACE_ENV", "").lower() == "production"
FMCSA_VIOLATION_LABELS = {
    "basic_viol": "Basic Violations",
    "unsafe_viol": "Unsafe Driving",
    "fatigued_viol": "HOS/Fatigued",
    "dr_fitness_viol": "Driver Fitness",
    "subt_alcohol_viol": "Alcohol/Drug",
    "vh_maint_viol": "Vehicle Maint.",
    "hm_viol": "Hazmat",
}



def _sort_buttons(chart_key: str):
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


# ── Carrier metrics builder ───────────────────────────────────────────────────
def _build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = dict(
        shipments        =("shipment_id",            "count"),
        avg_cost         =("total_cost_usd",          "mean"),
        total_spend      =("total_cost_usd",          "sum"),
        avg_cpm          =("cost_per_mile",           "mean"),
        avg_risk         =("risk_score",              "mean"),
        high_risk_count  =("risk_tier",
                           lambda x: (x == "High").sum()),
        total_accessorial=("accessorial_charge_usd",  "sum"),
        avg_accessorial  =("accessorial_charge_usd",  "mean"),
        avg_miles        =("miles",                   "mean"),
        avg_weight       =("weight_lbs",              "mean"),
    )
    # Include DOT number if available
    if "dot_number" in df.columns:
        agg_dict["dot_number"] = ("dot_number", "first")
    m = df.groupby("carrier").agg(**agg_dict).reset_index()
    m["high_risk_pct"]    = (m["high_risk_count"]   / m["shipments"] * 100).round(1)
    m["accessorial_rate"] = (m["total_accessorial"] / m["total_spend"] * 100).round(1)
    return m


# ── Chart-builder functions ───────────────────────────────────────────────────
def _build_cpm_fig(metrics: pd.DataFrame, height=280, sort_by="Value ↓") -> go.Figure:
    if sort_by == "Value ↑":
        sorted_m = metrics.sort_values("avg_cpm", ascending=True)
    elif sort_by == "Value ↓":
        sorted_m = metrics.sort_values("avg_cpm", ascending=False)
    else:
        sorted_m = metrics.sort_values("carrier")
    fig = go.Figure(go.Bar(
        x=sorted_m["carrier"],
        y=sorted_m["avg_cpm"],
        marker_color=[color_map.get(c, "#9333EA") for c in sorted_m["carrier"]],
        text=sorted_m["avg_cpm"].apply(lambda v: f"${v:.2f}"),
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=36, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        yaxis=dict(tickprefix="$", gridcolor="rgba(150,50,200,0.15)",
                   color="#94A3B8", linecolor="rgba(150,50,200,0.2)"),
        xaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
                   linecolor="rgba(150,50,200,0.2)"),
        showlegend=False,
    )
    return fig


def _build_high_risk_fig(metrics: pd.DataFrame, height=280, sort_by="Value ↓") -> go.Figure:
    if sort_by == "Value ↑":
        sorted_m2 = metrics.sort_values("high_risk_pct", ascending=True)
    elif sort_by == "Value ↓":
        sorted_m2 = metrics.sort_values("high_risk_pct", ascending=False)
    else:
        sorted_m2 = metrics.sort_values("carrier")
    bar_colors = ["#DC2626" if p > 40 else "#D97706" if p > 25 else "#059669"
                  for p in sorted_m2["high_risk_pct"]]
    fig = go.Figure(go.Bar(
        x=sorted_m2["carrier"],
        y=sorted_m2["high_risk_pct"],
        marker_color=bar_colors,
        text=sorted_m2["high_risk_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=36, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        yaxis=dict(ticksuffix="%", gridcolor="rgba(150,50,200,0.15)",
                   color="#94A3B8", linecolor="rgba(150,50,200,0.2)"),
        xaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
                   linecolor="rgba(150,50,200,0.2)"),
        showlegend=False,
    )
    return fig


def _build_acc_rate_fig(metrics: pd.DataFrame, height=260, sort_by="Value ↓") -> go.Figure:
    if sort_by == "Value ↑":
        sorted_m3 = metrics.sort_values("accessorial_rate", ascending=True)
    elif sort_by == "Value ↓":
        sorted_m3 = metrics.sort_values("accessorial_rate", ascending=False)
    else:
        sorted_m3 = metrics.sort_values("carrier")
    fig = go.Figure(go.Bar(
        x=sorted_m3["carrier"],
        y=sorted_m3["accessorial_rate"],
        marker_color=[color_map.get(c, "#9333EA") for c in sorted_m3["carrier"]],
        text=sorted_m3["accessorial_rate"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=36, b=0), height=height,
        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        yaxis=dict(ticksuffix="%", gridcolor="rgba(150,50,200,0.15)",
                   color="#94A3B8", linecolor="rgba(150,50,200,0.2)"),
        xaxis=dict(gridcolor="rgba(150,50,200,0.15)", color="#94A3B8",
                   linecolor="rgba(150,50,200,0.2)"),
        showlegend=False,
    )
    return fig


def _build_radar_fig(metrics: pd.DataFrame, height=260) -> go.Figure:
    def norm(series, invert=True):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        n = (series - mn) / (mx - mn)
        return 1 - n if invert else n

    radar_df = metrics.copy()
    radar_df["cost_score"]      = norm(radar_df["avg_cpm"],          invert=True)
    radar_df["risk_score_norm"] = norm(radar_df["avg_risk"],         invert=True)
    radar_df["acc_score"]       = norm(radar_df["accessorial_rate"], invert=True)
    radar_df["volume_score"]    = norm(radar_df["shipments"],        invert=False)

    categories = ["Cost Efficiency", "Low Risk", "Low Accessorial", "Volume"]
    radar_fig = go.Figure()
    for _, row in radar_df.iterrows():
        vals = [row["cost_score"], row["risk_score_norm"],
                row["acc_score"],  row["volume_score"]]
        vals += [vals[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["carrier"],
            line_color=color_map.get(row["carrier"], "#9333EA"),
            opacity=0.6,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=20, r=20, t=20, b=20),
        height=height,
        paper_bgcolor="#0f0a1e",
        font=dict(color="#A78BFA"),
        legend=dict(bgcolor="rgba(15,10,30,0.7)", font=dict(size=10, color="#FFFFFF")),
    )
    return radar_fig


# ── Helper: get active carriers for dialogs ───────────────────────────────────
def _active_carriers_df(base_df: pd.DataFrame) -> pd.DataFrame:
    active = st.session_state.get("active_carriers", ALL_CARRIERS)
    if not active:
        return base_df
    return base_df[base_df["carrier"].isin(active)]


def _first_non_empty(series: pd.Series, default=None):
    vals = series.dropna()
    if vals.empty:
        return default
    for v in vals:
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none"}:
            return v
    return default


def _render_single_carrier_detail(carrier_df: pd.DataFrame, carrier_name: str) -> None:
    """Render a lookup-style detail view for one selected carrier."""
    base = carrier_df.iloc[0].to_dict() if not carrier_df.empty else {}
    dot_number = _first_non_empty(carrier_df.get("dot_number", pd.Series(dtype=float)), None)
    if dot_number is not None:
        try:
            dot_number = int(float(dot_number))
        except Exception:
            dot_number = None

    result = {}
    fmcsa_raw = {}
    if MODEL_READY and dot_number is not None:
        try:
            from pipeline.inference import get_inference_engine

            engine = get_inference_engine()
            result = engine.predict_dot(dot_number=dot_number, origin_state=None) or {}
        except Exception:
            result = {}

    if API_ENABLED and dot_number is not None:
        try:
            from pipeline.api_integration import get_enricher

            enricher = get_enricher()
            fmcsa_raw = enricher.fmcsa.build_realtime_features(dot_number) or {}
        except Exception:
            fmcsa_raw = {}

    combined = {**base, **fmcsa_raw, **result}
    score = float(combined.get("risk_score", 0) or 0)
    label = combined.get("risk_label", combined.get("risk_tier", "Unknown"))
    charge = combined.get("charge_type", "Unknown")
    color = TIER_COLORS.get(label, "#94A3B8")
    c_color = CHARGE_COLORS.get(charge, "#A78BFA")

    with st.container(border=True):
        h1, h2, h3 = st.columns([5, 1, 1])
        with h1:
            st.markdown(f"### {carrier_name}")
            dot_txt = str(dot_number) if dot_number is not None else "N/A"
            st.markdown(
                f"<span style='color:#64748B;font-size:13px;'>USDOT {dot_txt}</span>",
                unsafe_allow_html=True,
            )
        with h2:
            src = combined.get("data_source", "historical")
            src_color = "#34D399" if src == "live_fmcsa" else "#60A5FA"
            src_label = "Live FMCSA" if src == "live_fmcsa" else "Historical"
            st.markdown(
                f"<div style='text-align:right;padding-top:8px;'>"
                f"<span style='border:1px solid {src_color};color:{src_color};"
                f"border-radius:4px;padding:3px 8px;font-size:11px;'>{src_label}</span></div>",
                unsafe_allow_html=True,
            )
        with h3:
            if label and label != "Unknown":
                st.markdown(
                    f"<div style='text-align:right;padding-top:8px;'>"
                    f"<span style='border:1px solid {color};color:{color};"
                    f"border-radius:4px;padding:3px 8px;font-size:11px;font-weight:700;'>{label}</span></div>",
                    unsafe_allow_html=True,
                )

        profile_fields = {
            "Status": combined.get("carrier_status_code", ""),
            "Operation": combined.get("carrier_carrier_operation", ""),
            "Power Units": combined.get("carrier_power_units", ""),
            "Total Drivers": combined.get("carrier_total_drivers", ""),
            "State": combined.get("carrier_phy_state", ""),
            "HM Carrier": combined.get("carrier_hm_ind", ""),
        }
        available = {
            k: v
            for k, v in profile_fields.items()
            if v is not None and str(v).strip() not in ("", "0", "UNKNOWN", "nan", "None", "NaN", "N/A")
        }
        if available:
            st.divider()
            pcols = st.columns(min(len(available), 6))
            for i, (lbl, val) in enumerate(available.items()):
                with pcols[i]:
                    st.markdown(f"**{lbl}**")
                    st.write(str(val))

    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 3], gap="medium")

    with left_col:
        with st.container(border=True):
            st.markdown("#### PACE Risk Score")
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(score, 1),
                    number={"suffix": "%", "font": {"size": 34, "color": color}},
                    title={"text": f"<b>{label} Risk</b>", "font": {"size": 14, "color": color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "#94A3B8", "size": 10}},
                        "bar": {"color": color, "thickness": 0.26},
                        "bgcolor": "#0f0a1e",
                        "borderwidth": 0,
                    },
                )
            )
            gauge_fig.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="#0f0a1e",
                font={"color": "#A78BFA"},
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown(
                f"<div style='background:rgba(0,0,0,0.3);border:1px solid {c_color};"
                f"border-radius:6px;padding:10px 14px;margin-top:4px;'>"
                f"<div style='color:#94A3B8;font-size:10px;font-weight:600;letter-spacing:1px;"
                f"text-transform:uppercase;margin-bottom:4px;'>Predicted Charge</div>"
                f"<div style='color:{c_color};font-size:16px;font-weight:700;'>{charge}</div></div>",
                unsafe_allow_html=True,
            )

    with right_col:
        viol_data = {label_: int(float(combined.get(col, 0) or 0)) for col, label_ in FMCSA_VIOLATION_LABELS.items()}
        with st.container(border=True):
            st.markdown("#### Violation Profile")
            o1, o2, o3 = st.columns(3)
            with o1:
                st.metric("OOS Total", int(float(combined.get("oos_total", 0) or 0)))
            with o2:
                st.metric("Driver OOS", int(float(combined.get("driver_oos_total", 0) or 0)))
            with o3:
                st.metric("Vehicle OOS", int(float(combined.get("vehicle_oos_total", 0) or 0)))
            st.divider()
            if any(v > 0 for v in viol_data.values()):
                for k, v in sorted(viol_data.items(), key=lambda x: -x[1]):
                    st.write(f"{k}: {v}")
            else:
                st.success("No violations on record.")

    probs = combined.get("probabilities", {})
    if probs:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### Charge Type Probabilities")
            prob_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            labels = [p[0] for p in prob_items]
            values = [round(p[1] * 100, 1) for p in prob_items]
            colors = [CHARGE_COLORS.get(l, "#A78BFA") for l in labels]
            prob_fig = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in values],
                    textposition="outside",
                )
            )
            prob_fig.update_layout(
                margin=dict(l=0, r=60, t=8, b=0),
                height=240,
                plot_bgcolor="#0f0a1e",
                paper_bgcolor="#0f0a1e",
                font=dict(color="#A78BFA"),
                xaxis=dict(ticksuffix="%", range=[0, 110], color="#94A3B8"),
                yaxis=dict(color="#94A3B8", autorange="reversed"),
            )
            st.plotly_chart(prob_fig, use_container_width=True)


# ── Expand dialogs (module-level) ─────────────────────────────────────────────
@st.dialog("Avg Cost per Mile", width="large")
def _popup_cpm():
    sort_by = _sort_buttons("cpm")
    m = _build_metrics(_active_carriers_df(df_all))
    st.caption(f"{len(_active_carriers_df(df_all)):,} shipments · all carriers")
    if m.empty:
        st.info("No data available.")
    else:
        st.plotly_chart(_build_cpm_fig(m, height=460, sort_by=sort_by), width="stretch")


@st.dialog("High Risk Shipment Rate", width="large")
def _popup_high_risk():
    sort_by = _sort_buttons("high_risk")
    m = _build_metrics(_active_carriers_df(df_all))
    st.caption(f"{len(_active_carriers_df(df_all)):,} shipments · all carriers")
    if m.empty:
        st.info("No data available.")
    else:
        st.plotly_chart(_build_high_risk_fig(m, height=460, sort_by=sort_by), width="stretch")


@st.dialog("Accessorial Cost Rate", width="large")
def _popup_acc_rate():
    sort_by = _sort_buttons("acc_rate")
    m = _build_metrics(_active_carriers_df(df_all))
    st.caption(f"{len(_active_carriers_df(df_all)):,} shipments · all carriers")
    if m.empty:
        st.info("No data available.")
    else:
        st.plotly_chart(_build_acc_rate_fig(m, height=460, sort_by=sort_by), width="stretch")


@st.dialog("Carrier Performance Radar", width="large")
def _popup_radar():
    m = _build_metrics(_active_carriers_df(df_all))
    st.caption(f"{len(_active_carriers_df(df_all)):,} shipments · all carriers")
    if m.empty:
        st.info("No data available.")
    else:
        st.plotly_chart(_build_radar_fig(m, height=500), width="stretch")


# ── Persist active carriers in session state ──────────────────────────────────
if "active_carriers" not in st.session_state:
    st.session_state["active_carriers"] = ALL_CARRIERS

# ── DOT number search ─────────────────────────────────────────────────────────
_has_dot = "dot_number" in df_raw.columns

dot_search_col, dot_btn_col, dot_clear_col = st.columns([3, 1, 1], gap="small")
with dot_search_col:
    dot_query = st.text_input(
        "Search by DOT Number",
        placeholder="e.g. 72011",
        label_visibility="collapsed",
        help="Enter a USDOT number to filter to the matching carrier",
        key="cc_dot_search",
    )
with dot_btn_col:
    dot_search_clicked = st.button(
        "Search DOT",
        type="primary",
        use_container_width=True,
        disabled=not dot_query.strip(),
    )
with dot_clear_col:
    if st.button("Clear", use_container_width=True, key="cc_dot_clear"):
        st.session_state["active_carriers"] = ALL_CARRIERS
        st.rerun()

if dot_search_clicked and dot_query.strip():
    try:
        dot_int = int(dot_query.strip())
    except ValueError:
        st.error("DOT number must be numeric.")
        st.stop()

    if _has_dot:
        dot_str = str(dot_int)
        matched = (
            df_raw[
                df_raw["dot_number"]
                .astype(str)
                .str.strip()
                .str.split(".")
                .str[0]  # handle "72011.0" float strings
                == dot_str
            ]["carrier"]
            .dropna()
            .unique()
            .tolist()
        )
        if matched:
            st.session_state["active_carriers"] = matched
            st.success(
                f"Showing carrier{'s' if len(matched) > 1 else ''}: "
                + ", ".join(matched)
            )
            st.rerun()
        else:
            st.warning(f"No carrier found for DOT {dot_int} in current dataset.")
    else:
        st.info(
            f"DOT lookup requires uploaded PACE data with a `dot_number` column.",
            icon="ℹ️",
        )

# ── Inline carrier selector ───────────────────────────────────────────────────
with st.expander("⚙️ Manage Carriers", expanded=False):
    f1, f2 = st.columns([5, 1])
    with f1:
        selected = st.multiselect(
            "Active Carriers",
            options=ALL_CARRIERS,
            default=[c for c in st.session_state["active_carriers"] if c in ALL_CARRIERS],
        )
        st.session_state["active_carriers"] = selected
    with f2:
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("All", width="stretch"):
                st.session_state["active_carriers"] = ALL_CARRIERS
                st.rerun()
        with col_b:
            if st.button("Clear", width="stretch"):
                st.session_state["active_carriers"] = []
                st.rerun()

# ── Guard: need at least 1 carrier ───────────────────────────────────────────
active_carriers = st.session_state["active_carriers"]
if not active_carriers:
    st.markdown("## Carrier Benchmarking")
    st.warning("No carriers selected. Use the filter above to add carriers to the comparison.")
    st.stop()

df = df_all[df_all["carrier"].isin(active_carriers)].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## Carrier Benchmarking")
st.caption(f"Comparing {len(active_carriers)} carrier{'s' if len(active_carriers) != 1 else ''} across cost, risk, and performance metrics.")
st.divider()

# ── Build carrier metrics table ───────────────────────────────────────────────
metrics = _build_metrics(df)

if len(active_carriers) == 1:
    st.info("Single-carrier mode: showing lookup-style detail view.", icon="🔍")
    _render_single_carrier_detail(df, active_carriers[0])
    st.stop()

# ── Summary metrics table ─────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### Carrier Summary")
    display = metrics.copy()
    for col in ["avg_cost", "avg_cpm", "avg_risk", "avg_accessorial", "total_spend"]:
        display[col] = display[col].round(2)
    display["avg_miles"]  = display["avg_miles"].round(0).astype(int)
    display["avg_weight"] = display["avg_weight"].round(0).astype(int)

    # Build display columns — include DOT if present
    _dot_cols = ["dot_number"] if "dot_number" in display.columns else []
    _base_cols = ["carrier"] + _dot_cols + [
        "shipments", "avg_cost", "avg_cpm", "avg_risk",
        "high_risk_pct", "avg_accessorial", "accessorial_rate", "total_spend",
    ]
    _rename = {
        "carrier":         "Carrier",
        "dot_number":      "DOT #",
        "shipments":       "Shipments",
        "avg_cost":        "Avg Total Cost",
        "avg_cpm":         "Avg $/Mile",
        "avg_risk":        "Avg Risk",
        "high_risk_pct":   "High Risk %",
        "avg_accessorial": "Avg Accessorial",
        "accessorial_rate":"Accessorial %",
        "total_spend":     "Total Spend",
    }
    _col_config = {
        "Avg Risk":       st.column_config.ProgressColumn(
            "Avg Risk", format="%.1f", min_value=0, max_value=100),
        "Avg Total Cost": st.column_config.NumberColumn(format="$%.2f"),
        "Avg $/Mile":     st.column_config.NumberColumn(format="$%.3f"),
        "Avg Accessorial":st.column_config.NumberColumn(format="$%.2f"),
        "Total Spend":    st.column_config.NumberColumn(format="$%.0f"),
    }
    if _dot_cols:
        _col_config["DOT #"] = st.column_config.NumberColumn(
            "DOT #", format="%d", help="USDOT number for this carrier")

    st.dataframe(
        display[_base_cols].rename(columns=_rename).sort_values("Avg $/Mile"),
        width="stretch",
        hide_index=True,
        column_config=_col_config,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Bar chart comparisons ─────────────────────────────────────────────────────
ch1, ch2 = st.columns(2, gap="medium")

with ch1:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Avg Cost per Mile")
        with btn:
            if st.button("⤢", key="exp_cpm", help="Expand chart"):
                _popup_cpm()
        st.plotly_chart(_build_cpm_fig(metrics), width="stretch")

with ch2:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### High Risk Shipment Rate")
        with btn:
            if st.button("⤢", key="exp_high_risk", help="Expand chart"):
                _popup_high_risk()
        st.plotly_chart(_build_high_risk_fig(metrics), width="stretch")

st.markdown("<br>", unsafe_allow_html=True)

ch3, ch4 = st.columns(2, gap="medium")

with ch3:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Accessorial Cost Rate")
            st.caption("Accessorial charges as % of total spend")
        with btn:
            if st.button("⤢", key="exp_acc_rate", help="Expand chart"):
                _popup_acc_rate()
        st.plotly_chart(_build_acc_rate_fig(metrics), width="stretch")

with ch4:
    with st.container(border=True):
        hdr, btn = st.columns([9, 1])
        with hdr:
            st.markdown("#### Carrier Performance Radar")
            st.caption("Normalized across cost, risk, and accessorial rate (lower = better)")
        with btn:
            if st.button("⤢", key="exp_radar", help="Expand chart"):
                _popup_radar()
        st.plotly_chart(_build_radar_fig(metrics), width="stretch")
