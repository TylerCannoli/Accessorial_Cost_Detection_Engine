import os
import sys
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.styling import inject_css, sidebar_account, inject_metric_wrap_css
from pages._pace_ui import show_mode_badge

from utils.database import load_shipments_with_fallback

from utils.legacy.cost_model import get_cost_model

st.set_page_config(
    page_title="PACE — Cost Estimate",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
require_auth()

username = st.session_state.get("username", "User")
sidebar_account(username)
show_mode_badge()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _safe_str(value, default=""):
    if value is None:
        return default
    try:
        s = str(value).strip()
        return s if s else default
    except Exception:
        return default


def _normalize_dot_series(series: pd.Series) -> pd.Series:
    """
    Keep only digit characters so DOT filters match cleanly even if source
    data has decimals, spaces, or mixed formatting.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.strip()
    )


def _ensure_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize schema so this page works with both DB data and mock fallback.
    The rest of the app commonly uses origin_city / destination_city / lane. 
    """
    df = df.copy()

    if "origin_city" not in df.columns:
        if "OriginRegion" in df.columns:
            df["origin_city"] = df["OriginRegion"]
        else:
            df["origin_city"] = "Unknown"

    if "destination_city" not in df.columns:
        if "DestRegion" in df.columns:
            df["destination_city"] = df["DestRegion"]
        else:
            df["destination_city"] = "Unknown"

    if "lane" not in df.columns:
        df["lane"] = (
            df["origin_city"].fillna("Unknown").astype(str).str.strip()
            + " → "
            + df["destination_city"].fillna("Unknown").astype(str).str.strip()
        )

    if "dot_number" not in df.columns:
        df["dot_number"] = ""

    df["dot_number"] = _normalize_dot_series(df["dot_number"])
    return df


def _prepare_base_model_df(shipments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build clean modeling dataset from raw shipments.
    """
    df = _ensure_filter_columns(shipments_df)

    required_cols = [
        "carrier",
        "facility",
        "weight_lbs",
        "miles",
        "total_cost_usd",
        "origin_city",
        "destination_city",
        "lane",
        "dot_number",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    model_df = df[required_cols].copy()
    model_df["carrier"] = model_df["carrier"].fillna("UNKNOWN").astype(str).str.strip()
    model_df["facility"] = model_df["facility"].fillna("UNKNOWN").astype(str).str.strip()
    model_df["origin_city"] = model_df["origin_city"].fillna("Unknown").astype(str).str.strip()
    model_df["destination_city"] = (
        model_df["destination_city"].fillna("Unknown").astype(str).str.strip()
    )
    model_df["lane"] = model_df["lane"].fillna("Unknown → Unknown").astype(str).str.strip()
    model_df["dot_number"] = _normalize_dot_series(model_df["dot_number"])

    model_df["weight_lbs"] = pd.to_numeric(model_df["weight_lbs"], errors="coerce").fillna(0.0)
    model_df["miles"] = pd.to_numeric(model_df["miles"], errors="coerce").fillna(0.0)
    model_df["total_cost_usd"] = pd.to_numeric(
        model_df["total_cost_usd"], errors="coerce"
    ).fillna(0.0)

    model_df = model_df[
        (model_df["carrier"] != "")
        & (model_df["facility"] != "")
        & (model_df["weight_lbs"] >= 0)
        & (model_df["miles"] >= 0)
        & np.isfinite(model_df["total_cost_usd"])
    ].copy()

    return model_df


def _compute_data_hash(df: pd.DataFrame) -> int:
    """
    Stable hash for Streamlit cache key.
    Includes filter columns so model retrains when training subset changes.
    """
    if df.empty:
        return 0

    cols = [
        "carrier",
        "facility",
        "weight_lbs",
        "miles",
        "total_cost_usd",
        "origin_city",
        "destination_city",
        "dot_number",
    ]
    base = df[cols].copy()

    for col in base.columns:
        base[col] = base[col].astype(str)

    digest = hashlib.md5(
        pd.util.hash_pandas_object(base, index=True).values.tobytes(),
        usedforsecurity=False,
    ).hexdigest()
    return int(digest[:15], 16)


def _get_tree_interval(model, X_pred: pd.DataFrame):
    """
    Compute mean + approximate 95% interval from individual RF trees.
    """
    try:
        pre = model.named_steps["pre"]
        rf = model.named_steps["rf"]
        X_t = pre.transform(X_pred)
        preds = np.array([tree.predict(X_t)[0] for tree in rf.estimators_], dtype=float)
        mean_pred = float(preds.mean())
        lower = float(np.percentile(preds, 2.5))
        upper = float(np.percentile(preds, 97.5))
        return mean_pred, lower, upper
    except Exception:
        pred = float(model.predict(X_pred)[0])
        return pred, pred, pred


def _get_feature_importance(model) -> pd.DataFrame:
    """
    Extract feature importances from the RF pipeline.
    """
    try:
        pre = model.named_steps["pre"]
        rf = model.named_steps["rf"]
        feature_names = pre.get_feature_names_out()
        importances = rf.feature_importances_

        fi = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        fi["feature"] = (
            fi["feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
        )
        return fi
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def _subset_with_min_rows(
    df: pd.DataFrame,
    filters: list[tuple[str, str]],
    min_rows: int,
) -> pd.DataFrame:
    subset = df.copy()
    for col, value in filters:
        if value and col in subset.columns:
            subset = subset[subset[col].astype(str) == str(value)]
    return subset if len(subset) >= min_rows else pd.DataFrame()


def _build_training_subset(
    df: pd.DataFrame,
    dot_number: str,
    origin_city: str,
    destination_city: str,
    min_rows_exact: int = 30,
    min_rows_relaxed: int = 20,
) -> tuple[pd.DataFrame, str]:
    """
    Progressive fallback:
    1) DOT + origin + destination
    2) origin + destination
    3) DOT only
    4) origin only
    5) destination only
    6) full dataset

    This keeps the model context-aware without failing on sparse lanes.
    """
    dot_number = _safe_str(dot_number)
    origin_city = _safe_str(origin_city)
    destination_city = _safe_str(destination_city)

    candidates = [
        (
            [("dot_number", dot_number), ("origin_city", origin_city), ("destination_city", destination_city)],
            min_rows_exact,
            "DOT + lane filtered training set",
        ),
        (
            [("origin_city", origin_city), ("destination_city", destination_city)],
            min_rows_exact,
            "Lane filtered training set",
        ),
        (
            [("dot_number", dot_number)],
            min_rows_relaxed,
            "DOT filtered training set",
        ),
        (
            [("origin_city", origin_city)],
            min_rows_relaxed,
            "Origin filtered training set",
        ),
        (
            [("destination_city", destination_city)],
            min_rows_relaxed,
            "Destination filtered training set",
        ),
    ]

    for filters, min_rows, label in candidates:
        active_filters = [(c, v) for c, v in filters if v]
        if not active_filters:
            continue
        subset = _subset_with_min_rows(df, active_filters, min_rows)
        if not subset.empty:
            return subset.copy(), label

    return df.copy(), "Full network training set (fallback)"


def _get_history_subset(
    df: pd.DataFrame,
    carrier: str,
    facility: str,
    dot_number: str,
    origin_city: str,
    destination_city: str,
) -> pd.DataFrame:
    """
    Use the currently active training subset, then find the most relevant
    historical comparison group with progressive fallback.
    """
    attempts = [
        (
            (df["carrier"] == carrier)
            & (df["facility"] == facility)
            & (df["origin_city"] == origin_city)
            & (df["destination_city"] == destination_city)
            & ((df["dot_number"] == dot_number) if dot_number else True)
        ),
        (
            (df["carrier"] == carrier)
            & (df["facility"] == facility)
            & (df["origin_city"] == origin_city)
            & (df["destination_city"] == destination_city)
        ),
        (
            (df["carrier"] == carrier)
            & (df["origin_city"] == origin_city)
            & (df["destination_city"] == destination_city)
        ),
        ((df["carrier"] == carrier) & (df["facility"] == facility)),
        (df["carrier"] == carrier),
        (df["facility"] == facility),
    ]

    for mask in attempts:
        subset = df[mask].copy()
        if len(subset) >= 5:
            return subset

    return df.copy()


def _fmt_filter_value(value: str) -> str:
    return value if value else "Any"


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
shipments_df = load_shipments_with_fallback()

# Detect whether live DB data or mock fallback is in use
from utils.database import get_connection_safe as _get_conn_safe
_db_live = _get_conn_safe() is not None

st.markdown("## Cost Estimator")
st.caption(
    "Estimate total shipment cost using the Random Forest cost model trained on "
    "historical shipment data. DOT, origin, and destination filters narrow the "
    "training subset to the most relevant historical shipments before fitting."
)

if _db_live:
    st.success(
        "Cost model is trained on **live shipment data** from the connected database.",
        icon="✅",
    )
else:
    st.warning(
        "Database unavailable — cost model is trained on **synthetic demo data**. "
        "Cost estimates are directionally correct but not calibrated to your actual "
        "shipment history. Connect Azure SQL and upload real shipments to calibrate.",
        icon="⚠️",
    )

st.divider()

# Ensure KPI-style metric labels/values wrap instead of truncating in narrow columns.
inject_metric_wrap_css()

if shipments_df is None or shipments_df.empty:
    st.error(
        "Cost model cannot run because shipment data is unavailable. "
        "Make sure the database is connected or fallback data is available."
    )
    st.stop()

base_model_df = _prepare_base_model_df(shipments_df)

if base_model_df.empty:
    st.error("No valid shipment rows are available to train the cost model.")
    st.stop()

carrier_options = sorted(base_model_df["carrier"].dropna().astype(str).unique().tolist())
facility_options = sorted(base_model_df["facility"].dropna().astype(str).unique().tolist())
origin_options = sorted(
    [v for v in base_model_df["origin_city"].dropna().astype(str).unique().tolist() if v]
)
destination_options = sorted(
    [v for v in base_model_df["destination_city"].dropna().astype(str).unique().tolist() if v]
)

# -------------------------------------------------------------------
# UI Layout
# -------------------------------------------------------------------
form_col, result_col = st.columns([2, 3], gap="large")

with form_col:
    with st.container(border=True):
        st.markdown("#### Shipment Inputs")

        default_carrier = 0 if carrier_options else None
        default_facility = 0 if facility_options else None

        carrier = st.selectbox(
            "Carrier",
            carrier_options,
            index=default_carrier,
            help="Carrier is a direct model input feature.",
        )

        facility = st.selectbox(
            "Facility",
            facility_options,
            index=default_facility,
            help="Facility is a direct model input feature.",
        )

        weight_lbs = st.number_input(
            "Weight (lbs)",
            min_value=0.0,
            max_value=200000.0,
            value=42000.0,
            step=500.0,
        )

        miles = st.number_input(
            "Miles",
            min_value=0.0,
            max_value=5000.0,
            value=850.0,
            step=25.0,
        )

        st.markdown("#### Refinement Filters")

        origin_city = st.selectbox(
            "Origin",
            options=[""] + origin_options,
            index=0,
            format_func=lambda x: "Any origin" if x == "" else x,
            help="Used to refine the historical training subset before training.",
        )

        destination_city = st.selectbox(
            "Destination",
            options=[""] + destination_options,
            index=0,
            format_func=lambda x: "Any destination" if x == "" else x,
            help="Used to refine the historical training subset before training.",
        )

        # Build filtered training set live from current refinement filters.
        filtered_training_df, training_scope = _build_training_subset(
            base_model_df,
            dot_number="",
            origin_city=origin_city,
            destination_city=destination_city,
        )

        train_model_df = filtered_training_df[
            ["carrier", "facility", "weight_lbs", "miles", "total_cost_usd", "origin_city", "destination_city", "lane", "dot_number"]
        ].copy()

        data_hash = _compute_data_hash(train_model_df)
        cost_model = get_cost_model(
            data_hash,
            train_model_df[["carrier", "facility", "weight_lbs", "miles", "total_cost_usd"]].copy(),
        )

        st.caption(
            f"Training scope: **{training_scope}** · "
            f"Rows used: **{len(train_model_df):,}**"
        )

        estimate_clicked = st.button(
            "Estimate Total Shipment Cost →",
            type="primary",
            use_container_width=True,
        )

        with st.expander("ℹ️ Model Info", expanded=False):
            st.markdown(
                f"""
**Model:** Random Forest Regressor
**Target:** `total_cost_usd`
**Direct input features:** `carrier`, `facility`, `weight_lbs`, `miles`
**Refinement filters:** `origin_city`, `destination_city`
**Training scope:** {training_scope}
**Training rows:** {len(train_model_df):,}
**Unique carriers:** {train_model_df['carrier'].nunique():,}
**Unique facilities:** {train_model_df['facility'].nunique():,}
**Origin selected:** {_fmt_filter_value(origin_city)}
**Destination selected:** {_fmt_filter_value(destination_city)}
"""
            )

# -------------------------------------------------------------------
# Run cost prediction
# -------------------------------------------------------------------
if estimate_clicked:
    clean_dot = ""

    X_pred = pd.DataFrame(
        [
            {
                "carrier": carrier,
                "facility": facility,
                "weight_lbs": float(weight_lbs),
                "miles": float(miles),
            }
        ]
    )

    try:
        pred_cost, ci_low, ci_high = _get_tree_interval(cost_model, X_pred)
        est_cpm = pred_cost / miles if miles > 0 else 0.0

        hist_subset = _get_history_subset(
            train_model_df,
            carrier=carrier,
            facility=facility,
            dot_number=clean_dot,
            origin_city=origin_city,
            destination_city=destination_city,
        )

        hist_avg_cost = (
            float(hist_subset["total_cost_usd"].mean()) if not hist_subset.empty else 0.0
        )
        hist_avg_cpm = (
            float(
                (
                    hist_subset["total_cost_usd"]
                    / hist_subset["miles"].replace(0, np.nan)
                ).mean()
            )
            if not hist_subset.empty
            else 0.0
        )
        fleet_avg_cpm = float(
            (
                base_model_df["total_cost_usd"]
                / base_model_df["miles"].replace(0, np.nan)
            ).mean()
        )

        st.session_state["ce_cost_result"] = {
            "carrier": carrier,
            "facility": facility,
            "weight_lbs": float(weight_lbs),
            "miles": float(miles),
            "dot_number": clean_dot,
            "origin_city": origin_city,
            "destination_city": destination_city,
            "pred_cost": pred_cost,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "est_cpm": est_cpm,
            "hist_avg_cost": hist_avg_cost,
            "hist_avg_cpm": hist_avg_cpm,
            "fleet_avg_cpm": fleet_avg_cpm,
            "history_rows": len(hist_subset),
            "history_df": hist_subset,
            "feature_importance": _get_feature_importance(cost_model),
            "training_scope": training_scope,
            "training_rows": len(train_model_df),
        }
    except Exception as e:
        st.error(f"Cost prediction failed: {e}")

# -------------------------------------------------------------------
# Results
# -------------------------------------------------------------------
with result_col:
    if "ce_cost_result" not in st.session_state:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center;padding:100px 20px;color:#9CA3AF;'>"
                "<div style='font-size:48px;'>💰</div>"
                "<div style='font-size:15px;font-weight:600;margin-top:16px;color:#E2E8F0;'>"
                "Enter shipment inputs to estimate cost</div>"
                "<div style='font-size:13px;margin-top:8px;'>"
                "This estimator uses a Random Forest cost model and refines the "
                "historical training set using DOT, origin, and destination filters."
                "</div></div>",
                unsafe_allow_html=True,
            )
    else:
        r = st.session_state["ce_cost_result"]

        pred_cost = _safe_float(r["pred_cost"])
        ci_low = _safe_float(r["ci_low"])
        ci_high = _safe_float(r["ci_high"])
        est_cpm = _safe_float(r["est_cpm"])
        hist_avg_cost = _safe_float(r["hist_avg_cost"])
        hist_avg_cpm = _safe_float(r["hist_avg_cpm"])
        fleet_avg_cpm = _safe_float(r["fleet_avg_cpm"])
        history_rows = int(r["history_rows"])
        history_df = r["history_df"]
        feature_importance = r["feature_importance"]
        training_scope = _safe_str(r.get("training_scope", "Training subset"))
        training_rows = int(r.get("training_rows", 0))

        with st.container(border=True):
            st.markdown("#### Estimated Total Shipment Cost")

            info1, info2 = st.columns(2)
            with info1:
                st.caption(f"Training scope: {training_scope}")
            with info2:
                st.caption(f"Filtered rows used to train model: {training_rows:,}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Total Cost", f"${pred_cost:,.2f}")
            m2.metric("Estimated Cost / Mile", f"${est_cpm:,.2f}")
            m3.metric("Historical Rows Used", f"{history_rows:,}")

            st.markdown("<br>", unsafe_allow_html=True)

            ci_fig = go.Figure(
                go.Indicator(
                    mode="number+delta",
                    value=pred_cost,
                    number={"prefix": "$", "valueformat": ",.2f"},
                    delta={
                        "reference": hist_avg_cost if hist_avg_cost > 0 else pred_cost,
                        "relative": False,
                        "valueformat": ",.2f",
                    },
                    title={"text": f"<b>95% Interval: ${ci_low:,.2f} — ${ci_high:,.2f}</b>"},
                )
            )
            ci_fig.update_layout(
                height=180,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="#0f0a1e",
                font={"color": "#E2E8F0"},
            )
            st.plotly_chart(ci_fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Active Refinement Filters")
            f1, f2 = st.columns(2)
            f1.metric("Origin", _fmt_filter_value(_safe_str(r.get("origin_city", ""))))
            f2.metric(
                "Destination",
                _fmt_filter_value(_safe_str(r.get("destination_city", ""))),
            )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Benchmark Comparison")
            b1, b2, b3 = st.columns(3)
            b1.metric("Historical Avg Cost", f"${hist_avg_cost:,.2f}")
            b2.metric("Historical Avg Cost / Mile", f"${hist_avg_cpm:,.2f}")
            b3.metric("Fleet Avg Cost / Mile", f"${fleet_avg_cpm:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Summary Analysis ──────────────────────────────────────
        with st.container(border=True):
            st.markdown("#### Summary Analysis")

            # Variance from historical average
            hist_delta = pred_cost - hist_avg_cost if hist_avg_cost > 0 else 0.0
            hist_delta_pct = (hist_delta / hist_avg_cost * 100) if hist_avg_cost > 0 else 0.0
            fleet_delta_pct = ((est_cpm - fleet_avg_cpm) / fleet_avg_cpm * 100) if fleet_avg_cpm > 0 else 0.0

            # Risk language
            if hist_delta_pct > 15:
                cost_sentiment = "significantly above"
                cost_color = "#F87171"
                rec = (
                    "Consider negotiating rate caps or booking with an alternative carrier. "
                    "Review accessorial history and request a morning appointment window to reduce detention risk."
                )
            elif hist_delta_pct > 5:
                cost_sentiment = "moderately above"
                cost_color = "#FCD34D"
                rec = (
                    "Add a 10–15% accessorial buffer to your quote. "
                    "Confirm appointment type and facility hours to minimize delays."
                )
            elif hist_delta_pct < -5:
                cost_sentiment = "below"
                cost_color = "#34D399"
                rec = (
                    "This estimate looks favorable. Verify lane availability and lock in the rate. "
                    "Standard operating procedure applies — no additional buffer needed."
                )
            else:
                cost_sentiment = "in line with"
                cost_color = "#34D399"
                rec = (
                    "This estimate is consistent with historical performance. "
                    "Proceed with standard quoting — no additional buffer needed."
                )

            lane_desc = ""
            if _safe_str(r.get("origin_city")) and _safe_str(r.get("destination_city")):
                lane_desc = f" on the **{r['origin_city']} → {r['destination_city']}** lane"
            elif _safe_str(r.get("origin_city")):
                lane_desc = f" from **{r['origin_city']}**"

            summary_html = (
                f"<div style='font-size:14px;line-height:1.7;color:#E2E8F0;'>"
                f"The model estimates a total shipment cost of "
                f"<span style='font-weight:700;color:#A78BFA;'>${pred_cost:,.2f}</span> "
                f"for <span style='font-weight:700;color:#E2E8F0;'>{_safe_str(r.get('carrier',''))}</span>"
                f"{lane_desc} "
                f"({_safe_float(r.get('weight_lbs',0)):,.0f} lbs, "
                f"{_safe_float(r.get('miles',0)):,.0f} mi). "
                f"This is <span style='font-weight:700;color:{cost_color};'>"
                f"{cost_sentiment} the historical average"
                f"</span> of ${hist_avg_cost:,.2f} "
                f"({'+' if hist_delta_pct >= 0 else ''}{hist_delta_pct:.1f}%), "
                f"with a 95% confidence interval of "
                f"<strong>${ci_low:,.2f} – ${ci_high:,.2f}</strong>. "
                f"Cost per mile is estimated at ${est_cpm:,.2f} "
                f"vs. a fleet average of ${fleet_avg_cpm:,.2f} "
                f"({'+' if fleet_delta_pct >= 0 else ''}{fleet_delta_pct:.1f}%). "
                f"Training was based on <strong>{training_rows:,} historical shipments</strong> "
                f"({training_scope.lower()})."
                f"</div>"
                f"<div style='margin-top:12px;padding:10px 14px;"
                f"background:rgba(147,51,234,0.12);border-left:3px solid #9333EA;"
                f"border-radius:4px;font-size:13px;color:#CBD5E1;'>"
                f"<strong>Recommendation:</strong> {rec}"
                f"</div>"
            )
            st.markdown(summary_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Historical Cost Distribution")
            hist_plot_df = history_df.copy()
            hist_plot_df = hist_plot_df[np.isfinite(hist_plot_df["total_cost_usd"])]

            if hist_plot_df.empty:
                st.info("No historical shipment distribution available for this selection.")
            else:
                hist_fig = go.Figure()
                hist_fig.add_histogram(
                    x=hist_plot_df["total_cost_usd"],
                    nbinsx=25,
                    name="Historical Shipments",
                    opacity=0.8,
                )
                hist_fig.add_vline(
                    x=pred_cost,
                    line_width=3,
                    annotation_text=f"Prediction: ${pred_cost:,.0f}",
                    annotation_position="top",
                )
                hist_fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="#0f0a1e",
                    plot_bgcolor="#0f0a1e",
                    font={"color": "#E2E8F0"},
                    xaxis_title="Total Cost (USD)",
                    yaxis_title="Count",
                )
                st.plotly_chart(hist_fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Feature Importance")
            if feature_importance.empty:
                st.info("Feature importance is not available.")
            else:
                top_fi = (
                    feature_importance.head(12)
                    .sort_values("importance", ascending=True)
                    .copy()
                )

                fi_fig = go.Figure(
                    go.Bar(
                        x=top_fi["importance"],
                        y=top_fi["feature"],
                        orientation="h",
                        text=[f"{v:.3f}" for v in top_fi["importance"]],
                        textposition="outside",
                    )
                )
                fi_fig.update_layout(
                    height=380,
                    margin=dict(l=10, r=70, t=10, b=10),
                    paper_bgcolor="#0f0a1e",
                    plot_bgcolor="#0f0a1e",
                    font={"color": "#E2E8F0"},
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                )
                st.plotly_chart(fi_fig, use_container_width=True)
