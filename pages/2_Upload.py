# File: pages/2_Upload.py
import os
import sys
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_TD_AVAILABLE = bool(os.getenv("TD_HOST", ""))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth
from utils.doc_parser import parse_uploaded_document
from utils.styling import inject_css, sidebar_account, TIER_COLORS
from pipeline.data_pipeline import get_data_pipeline
from pipeline.config import CHARGE_TYPE_LABELS, is_pace_model_ready
from utils.column_mapper import (
    get_column_mapper, find_unrecognized_columns,
    PACE_TARGET_COLS, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PACE — Upload",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Auth guard ────────────────────────────────────────────────────
require_auth()
username = st.session_state.get("username", "User")
sidebar_account(username)

# ── Model availability check ──────────────────────────────────────
MODEL_READY = is_pace_model_ready()

# ── Helper: tier color ────────────────────────────────────────────
def tier_badge(label: str) -> str:
    """Handle tier badge."""
    color = TIER_COLORS.get(label, "#94A3B8")
    return (
        f"<span style='background:rgba(0,0,0,0.3);border:1px solid {color};"
        f"color:{color};border-radius:4px;padding:2px 8px;"
        f"font-size:11px;font-weight:600;'>{label}</span>"
    )

# ── Schema badge ──────────────────────────────────────────────────
SCHEMA_LABELS = {
    "pace":           ("PACE Schema",     "#A78BFA"),
    "pace_aliased":   ("PACE (mapped)",   "#60A5FA"),
    "pace_converted": ("Legacy → PACE",   "#FCD34D"),
    "legacy":         ("Legacy Schema",   "#FB923C"),
    "unknown":        ("Unknown Schema",  "#F87171"),
}

# ── Page header ───────────────────────────────────────────────────
st.markdown("## Upload & Batch Score")
st.caption("Upload a CSV or Excel file to validate, clean, and score with the PACE model.")

if not MODEL_READY:
    st.info(
        "The PACE model is not yet trained. Uploads will be validated and "
        "processed — scoring will be available once training is complete.",
        icon="ℹ️"
    )

# ── File requirements expander ────────────────────────────────────
with st.expander("📋 Accepted Formats & Columns", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**PACE Schema (full)**
- `dot_number` — USDOT carrier number
- `oos_total`, `driver_oos_total`, `vehicle_oos_total`
- `basic_viol`, `unsafe_viol`, `vh_maint_viol`
- `crash_count`, `crash_avg_severity`
- All 152 FMCSA/EIA/FRED/weather columns

**Legacy Schema (auto-converted)**
- `shipment_id`, `ship_date`
- `carrier`, `facility`
- `weight_lbs`, `miles`
- `base_freight_usd`, `accessorial_charge_usd`
        """)
    with col_b:
        st.markdown("""
**Column aliases are auto-mapped**, for example:
- `dot` → `dot_number`
- `crashes` → `crash_count`
- `oos` → `oos_total`
- `carrier_state` → `carrier_phy_state`
- `diesel_price` → `eia_diesel_national`

**Accepted files:** `.csv`, `.xlsx`, `.xls`
**Max size:** 50 MB
        """)

st.divider()

# ── Upload zone ───────────────────────────────────────────────────
if _TD_AVAILABLE:
    up_col, sample_col, td_col = st.columns([3, 1, 1], gap="medium")
else:
    up_col, sample_col = st.columns([3, 1], gap="large")
    td_col = None

with up_col:
    uploaded_file = st.file_uploader(
        "Drag & drop your file here, or click to browse",
        type=["csv", "xlsx", "xls"],
        help="Accepted: CSV, Excel · Max 50 MB",
    )

with sample_col:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    use_sample = st.button("Use sample data", type="secondary", use_container_width=True)
    st.caption("Loads 50 rows of mock PACE data")

if td_col is not None:
    with td_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        load_teradata = st.button(
            "Load from Teradata",
            type="primary",
            use_container_width=True,
            help="Pull live Teradata records and score with PACE (no upload required)",
        )
        st.caption("Live inference on Teradata data")
else:
    load_teradata = False

# ── Load from Teradata ────────────────────────────────────────────
if load_teradata:
    from utils.database import load_teradata_for_inference
    with st.spinner("Connecting to Teradata and pulling records..."):
        td_df = load_teradata_for_inference(row_limit=5000)
    if td_df.empty:
        st.error(
            "Could not retrieve data from Teradata. "
            "Check that TD_HOST is configured and pace_training_v is accessible."
        )
    else:
        st.session_state["upload_raw_df"]  = td_df
        st.session_state["upload_result"]  = None
        st.session_state["upload_scored"]  = None
        st.session_state["upload_source"]  = "teradata"
        st.success(f"Loaded {len(td_df):,} records from Teradata.")
        st.rerun()

# ── Load sample ───────────────────────────────────────────────────
if use_sample:
    from utils.mock_data import generate_mock_shipments
    sample = generate_mock_shipments(50)
    sample = sample.drop(
        columns=["risk_score", "risk_tier", "accessorial_type"], errors="ignore"
    )
    st.session_state["upload_raw_df"]      = sample
    st.session_state["upload_result"]      = None
    st.session_state["upload_scored"]      = None
    st.session_state.pop("upload_source", None)
    st.rerun()

# ── Parse uploaded file ───────────────────────────────────────────
if uploaded_file is not None:
    try:
        raw_df = parse_uploaded_document(uploaded_file, uploaded_file.name)
        if raw_df.empty:
            st.error("Uploaded file contains no usable data.")
        else:
            st.session_state["upload_raw_df"] = raw_df
            st.session_state["upload_result"] = None
            st.session_state["upload_scored"] = None
    except Exception as e:
        st.error(f"Could not parse file: {e}")

# ── Main processing ───────────────────────────────────────────────
if st.session_state.get("upload_raw_df") is not None:
    raw_df   = st.session_state["upload_raw_df"]
    pipeline = get_data_pipeline()

    if st.session_state.get("upload_source") == "teradata":
        st.info(
            f"Showing **{len(raw_df):,} live Teradata records** pulled from `pace_training_v`. "
            "Click **Score with PACE →** to run inference.",
            icon="🗄️",
        )

    # ── AI Column Mapping ──────────────────────────────────────────────────────
    # Only shown when the static alias map still leaves unrecognized columns.
    _unrecognized = find_unrecognized_columns(raw_df)

    if _unrecognized:
        # Reset mapping state when the file changes
        if st.session_state.get("upload_unrecognized_cols") != _unrecognized:
            st.session_state["upload_unrecognized_cols"]     = _unrecognized
            st.session_state["upload_ai_suggestions"]        = {}
            st.session_state["upload_col_mapping_confirmed"] = {}

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🤖 AI Column Mapping", expanded=True):
            st.caption(
                f"**{len(_unrecognized)} column(s)** were not recognized by the static alias map. "
                "Use the AI mapper to get suggestions, then apply your confirmed mapping."
            )

            # ── Settings row ──────────────────────────────────────────────────
            set_c1, set_c2, set_c3 = st.columns([2, 2, 1])
            with set_c1:
                ai_method = st.selectbox(
                    "Mapping method",
                    options=["semantic", "ollama"],
                    index=0 if st.session_state.get("upload_ai_method", "semantic") == "semantic" else 1,
                    key="ai_method_select",
                    help="Semantic: offline cosine similarity (no install needed). "
                         "Ollama: local LLM — requires `ollama serve` and a pulled model.",
                )
                st.session_state["upload_ai_method"] = ai_method

            with set_c2:
                ollama_model = st.text_input(
                    "Ollama model",
                    value=st.session_state.get("upload_ollama_model", "llama3.2"),
                    disabled=(ai_method != "ollama"),
                    key="ollama_model_input",
                )
                st.session_state["upload_ollama_model"] = ollama_model

            with set_c3:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                run_ai = st.button("Run AI Mapper", type="secondary", use_container_width=True)

            # Ollama availability warning
            if ai_method == "ollama":
                _mapper_check = get_column_mapper()
                _ok, _msg = _mapper_check.check_ollama(ollama_model)
                if not _ok:
                    st.warning(_msg, icon="⚠️")

            # ── Run mapper ────────────────────────────────────────────────────
            if run_ai:
                _mapper_inst = get_column_mapper()
                with st.spinner(f"Mapping {len(_unrecognized)} column(s) via {ai_method}…"):
                    _suggestions = _mapper_inst.map_columns(
                        _unrecognized, method=ai_method, ollama_model=ollama_model
                    )
                st.session_state["upload_ai_suggestions"]        = _suggestions
                st.session_state["upload_col_mapping_confirmed"] = {}
                st.session_state["upload_result"]                = None
                st.rerun()

            _suggestions = st.session_state.get("upload_ai_suggestions", {})

            if not _suggestions:
                st.info(
                    f"Unrecognized columns: **{', '.join(_unrecognized)}**  \n"
                    "Click **Run AI Mapper** to get suggestions.",
                    icon="ℹ️",
                )
            else:
                # ── Confidence badge helper ───────────────────────────────────
                def _conf_badge(conf: float) -> str:
                    """Handle conf badge."""
                    if conf >= CONFIDENCE_HIGH:
                        return (
                            "<span style='color:#34D399;font-size:11px;font-weight:700;"
                            f"letter-spacing:.04em'>HIGH {conf:.0%}</span>"
                        )
                    elif conf >= CONFIDENCE_MEDIUM:
                        return (
                            "<span style='color:#FCD34D;font-size:11px;font-weight:700;"
                            f"letter-spacing:.04em'>MED {conf:.0%}</span>"
                        )
                    return (
                        "<span style='color:#F87171;font-size:11px;font-weight:700;"
                        f"letter-spacing:.04em'>LOW {conf:.0%}</span>"
                    )

                # ── Auto-accept high-confidence button ────────────────────────
                _high_cols = [
                    col for col, res in _suggestions.items()
                    if res["confidence"] >= CONFIDENCE_HIGH and res["pace_col"] is not None
                ]
                if _high_cols:
                    _auto_c, _ = st.columns([3, 5])
                    with _auto_c:
                        if st.button(
                            f"Auto-accept {len(_high_cols)} high-confidence mapping(s)",
                            type="secondary",
                        ):
                            _confirmed = dict(
                                st.session_state.get("upload_col_mapping_confirmed", {})
                            )
                            for _col in _high_cols:
                                _confirmed[_col] = _suggestions[_col]["pace_col"]
                            st.session_state["upload_col_mapping_confirmed"] = _confirmed
                            st.rerun()

                # ── Per-column dropdown table ─────────────────────────────────
                st.caption(
                    "Review suggestions below. Use dropdowns to override. "
                    "Select **(skip / ignore)** to leave a column unmapped."
                )
                _pace_options = ["(skip / ignore)"] + sorted(PACE_TARGET_COLS)
                _confirmed = dict(st.session_state.get("upload_col_mapping_confirmed", {}))

                _hdr_a, _hdr_b, _hdr_c = st.columns([2, 3, 1])
                with _hdr_a:
                    st.markdown("**Your column**")
                with _hdr_b:
                    st.markdown("**Map to PACE column**")
                with _hdr_c:
                    st.markdown("**Confidence**")

                for _col, _res in _suggestions.items():
                    _row_a, _row_b, _row_c = st.columns([2, 3, 1])

                    with _row_a:
                        st.markdown(f"`{_col}`")

                    with _row_b:
                        _default = _confirmed.get(_col) or _res["pace_col"] or "(skip / ignore)"
                        try:
                            _default_idx = _pace_options.index(_default)
                        except ValueError:
                            _default_idx = 0

                        _chosen = st.selectbox(
                            label="",
                            options=_pace_options,
                            index=_default_idx,
                            key=f"col_map_{_col}",
                            label_visibility="collapsed",
                        )
                        _confirmed[_col] = _chosen if _chosen != "(skip / ignore)" else None

                    with _row_c:
                        st.markdown(_conf_badge(_res["confidence"]), unsafe_allow_html=True)

                st.session_state["upload_col_mapping_confirmed"] = _confirmed

                # ── Apply Mapping button ──────────────────────────────────────
                st.divider()
                _apply_c, _ = st.columns([2, 5])
                with _apply_c:
                    _apply_clicked = st.button(
                        "Apply Mapping", type="primary", use_container_width=True
                    )

                if _apply_clicked:
                    _to_apply = {
                        old: new
                        for old, new in st.session_state["upload_col_mapping_confirmed"].items()
                        if new is not None
                    }
                    if _to_apply:
                        _renamed = st.session_state["upload_raw_df"].rename(columns=_to_apply)
                        st.session_state["upload_raw_df"]              = _renamed
                        st.session_state["upload_result"]              = None
                        st.session_state["upload_scored"]              = None
                        st.session_state["upload_ai_suggestions"]      = {}
                        st.session_state["upload_col_mapping_confirmed"] = {}
                        st.session_state["upload_unrecognized_cols"]   = []
                        st.success(
                            f"Applied {len(_to_apply)} column mapping(s). Re-running pipeline…"
                        )
                        st.rerun()
                    else:
                        st.info("No mappings selected — proceeding with original column names.")

    # ── Run pipeline if not already done ──────────────────────────────────────
    if st.session_state.get("upload_result") is None:
        with st.spinner("Analyzing file..."):
            result = pipeline.process_csv(raw_df)
            st.session_state["upload_result"] = result

    result       = st.session_state["upload_result"]
    errs         = result["errors"]
    warns        = result["warnings"]
    pass_count   = result["pass_count"]
    fail_count   = result["fail_count"]
    schema       = result["schema"]
    mapping      = result["mapping"]
    df_clean     = result["df_clean"]
    row_fail_mask = result["row_fail_mask"]

    schema_label, schema_color = SCHEMA_LABELS.get(schema, ("Unknown", "#F87171"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Validation summary ────────────────────────────────────────
    with st.container(border=True):
        hdr_left, hdr_right = st.columns([3, 1])
        with hdr_left:
            st.markdown("#### Validation Results")
        with hdr_right:
            st.markdown(
                f"<div style='text-align:right;padding-top:4px;'>"
                f"<span style='background:rgba(0,0,0,0.3);border:1px solid {schema_color};"
                f"color:{schema_color};border-radius:4px;padding:3px 10px;"
                f"font-size:12px;font-weight:600;'>{schema_label}</span></div>",
                unsafe_allow_html=True,
            )

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Rows", f"{len(raw_df):,}")
        with s2:
            st.metric("✅ Passed", f"{pass_count:,}")
        with s3:
            st.metric("❌ Failed", f"{fail_count:,}")
        with s4:
            st.metric("⚠️ Warnings", f"{len(warns)}")

        st.divider()

        if not errs and not warns:
            st.success("All rows passed validation — ready to score.")

        # Column mapping report
        if mapping:
            with st.expander(f"🔀 Column Aliases Applied ({len(mapping)})", expanded=False):
                alias_df = pd.DataFrame(
                    [(orig, canon) for orig, canon in mapping.items()],
                    columns=["Original Column", "Mapped To"]
                )
                st.dataframe(alias_df, hide_index=True, use_container_width=True)

        # Schema conversion note
        if schema == "pace_converted":
            st.info(
                "Legacy schema detected and auto-converted to PACE format. "
                "Some features were approximated from available columns. "
                "For best accuracy, use the full PACE schema with DOT numbers.",
                icon="🔄"
            )

        if errs:
            with st.expander(f"❌ Errors ({len(errs)})", expanded=True):
                for e in errs[:50]:
                    st.markdown(
                        f"<p style='margin:4px 0;font-size:13px;color:#F87171;'>• {e}</p>",
                        unsafe_allow_html=True,
                    )
                if len(errs) > 50:
                    st.caption(f"… and {len(errs) - 50} more errors.")

        if warns:
            with st.expander(f"⚠️ Warnings ({len(warns)})", expanded=False):
                for w in warns[:50]:
                    st.markdown(
                        f"<p style='margin:4px 0;font-size:13px;color:#FCD34D;'>• {w}</p>",
                        unsafe_allow_html=True,
                    )
                if len(warns) > 50:
                    st.caption(f"… and {len(warns) - 50} more warnings.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data preview + score button ───────────────────────────────
    with st.container(border=True):
        hdr_col, btn_col = st.columns([4, 1])

        with hdr_col:
            st.markdown(f"#### Data Preview — first {min(25, len(raw_df))} rows")

        with btn_col:
            score_clicked = st.button(
                "Score with PACE →",
                type="primary",
                disabled=bool(errs) or (pass_count == 0) or not MODEL_READY,
                use_container_width=True,
            )

        if errs:
            st.caption("⚠️ Resolve all errors before scoring.")
        elif not MODEL_READY:
            st.caption("⏳ Model training in progress — scoring available soon.")

        # Show scored or raw preview
        scored = st.session_state.get("upload_scored")
        preview = scored.head(25) if scored is not None else raw_df.head(25)

        col_config = {}
        if scored is not None and "risk_score_pct" in preview.columns:
            col_config["risk_score_pct"] = st.column_config.ProgressColumn(
                "Risk Score",
                format="%.0f%%",
                min_value=0,
                max_value=100,
            )

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
        )

        # ── Run scoring ───────────────────────────────────────────
        if score_clicked and not errs and MODEL_READY:
            try:
                from pipeline.inference import get_inference_engine
                engine = get_inference_engine()

                with st.spinner(f"Scoring {pass_count:,} rows with PACE model..."):
                    results_df = engine.predict_dataframe(df_clean)

                    # Combine with original identifiers
                    id_cols = [c for c in ["unique_id", "dot_number"]
                               if c in df_clean.columns]
                    if id_cols:
                        scored_df = pd.concat(
                            [df_clean[id_cols].reset_index(drop=True),
                             results_df.reset_index(drop=True)],
                            axis=1
                        )
                    else:
                        scored_df = results_df

                st.session_state["upload_scored"] = scored_df
                st.success(
                    f"✅ Scoring complete — {len(scored_df):,} rows scored."
                )
                st.rerun()

            except Exception as e:
                st.error(f"Scoring failed: {e}")

    # ── Results section (shown after scoring) ─────────────────────
    if st.session_state.get("upload_scored") is not None:
        scored = st.session_state["upload_scored"]
        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Score Summary")

            if "charge_type" in scored.columns and "risk_score_pct" in scored.columns:
                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    avg_risk = scored["risk_score_pct"].mean()
                    st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
                with m2:
                    high_risk = (scored["risk_score_pct"] >= 75).sum()
                    st.metric("Critical Risk", f"{high_risk:,}")
                with m3:
                    top_charge = scored["charge_type"].value_counts().index[0]
                    st.metric("Top Charge Type", top_charge)
                with m4:
                    no_charge = (scored["charge_type"] == "No Charge").sum()
                    st.metric("No Charge", f"{no_charge:,}")

                st.markdown("<br>", unsafe_allow_html=True)

                chart_col, dist_col = st.columns(2)

                with chart_col:
                    # Charge type distribution
                    ct_counts = scored["charge_type"].value_counts()
                    fig_ct = go.Figure(go.Bar(
                        x=ct_counts.index.tolist(),
                        y=ct_counts.values.tolist(),
                        marker_color="#9333EA",
                        text=ct_counts.values.tolist(),
                        textposition="outside",
                        textfont={"color": "#E2E8F0"},
                    ))
                    fig_ct.update_layout(
                        title="Charge Type Distribution",
                        margin=dict(l=0, r=0, t=40, b=0), height=280,
                        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
                        font=dict(color="#A78BFA"),
                        xaxis=dict(color="#94A3B8",
                                   gridcolor="rgba(150,50,200,0.15)"),
                        yaxis=dict(color="#94A3B8",
                                   gridcolor="rgba(150,50,200,0.15)"),
                    )
                    st.plotly_chart(fig_ct, use_container_width=True)

                with dist_col:
                    # Risk score histogram
                    fig_hist = go.Figure(go.Histogram(
                        x=scored["risk_score_pct"],
                        nbinsx=20,
                        marker_color="#6D28D9",
                        marker_line_color="#9333EA",
                        marker_line_width=1,
                    ))
                    fig_hist.add_vline(
                        x=avg_risk, line_dash="dash",
                        line_color="#F87171", line_width=2,
                        annotation_text=f"Avg: {avg_risk:.1f}%",
                        annotation_font_color="#F87171",
                    )
                    fig_hist.update_layout(
                        title="Risk Score Distribution",
                        margin=dict(l=0, r=0, t=40, b=0), height=280,
                        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
                        font=dict(color="#A78BFA"),
                        xaxis=dict(title="Risk Score (%)", color="#94A3B8",
                                   gridcolor="rgba(150,50,200,0.15)"),
                        yaxis=dict(title="Count", color="#94A3B8",
                                   gridcolor="rgba(150,50,200,0.15)"),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # ── Per-class model confidence ─────────────────────
                from pipeline.config import CHARGE_TYPE_LABELS as _CTL
                _prob_cols = {
                    label: f"prob_{label.lower().replace(' ', '_').replace('/', '_')}"
                    for label in _CTL
                }
                _avail = {l: c for l, c in _prob_cols.items() if c in scored.columns}
                if _avail:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### Per-Class Model Confidence")
                    st.caption(
                        "Average probability the model assigned to each charge type "
                        "across all scored rows. Bars close to uniform indicate uncertainty; "
                        "high peaks show strong model conviction."
                    )
                    _avg_probs = {l: float(scored[c].mean()) for l, c in _avail.items()}
                    _CLASS_COLORS = {
                        "No Charge":          "#34D399",
                        "Detention":          "#F59E0B",
                        "Safety Surcharge":   "#F87171",
                        "Compliance Fee":     "#A78BFA",
                        "Hazmat Fee":         "#60A5FA",
                        "High Risk / Multiple": "#EC4899",
                    }
                    _labels = list(_avg_probs.keys())
                    _values = [_avg_probs[l] for l in _labels]
                    _colors = [_CLASS_COLORS.get(l, "#9333EA") for l in _labels]
                    fig_conf = go.Figure(go.Bar(
                        x=_labels,
                        y=_values,
                        marker_color=_colors,
                        text=[f"{v:.1%}" for v in _values],
                        textposition="outside",
                        textfont={"color": "#E2E8F0", "size": 12},
                    ))
                    fig_conf.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0), height=260,
                        plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
                        font=dict(color="#A78BFA"),
                        xaxis=dict(color="#94A3B8",
                                   gridcolor="rgba(150,50,200,0.15)"),
                        yaxis=dict(title="Avg Probability", color="#94A3B8",
                                   gridcolor="rgba(150,50,200,0.15)",
                                   tickformat=".0%", range=[0, max(_values) * 1.25]),
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

            # ── Data lineage note ─────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "**How risk scores are computed:** Risk scores (0–100) and charge type "
                "predictions are produced by the PACE FT-Transformer, trained on FMCSA "
                "carrier inspection and safety violation patterns. Scores reflect the "
                "statistical likelihood of accessorial charges based on carrier safety "
                "history, fuel market conditions, weather, and economic signals — not "
                "from observed billing outcomes. Upload real accessorial invoices via "
                "**Admin → Train from CSV** to calibrate against actual charge history.",
                icon="ℹ️",
            )

            # ── Download results ──────────────────────────────────
            st.divider()
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                csv_buf = io.StringIO()
                scored.to_csv(csv_buf, index=False)
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_buf.getvalue(),
                    file_name="pace_scored_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with dl_col2:
                if st.button("🔄 Clear & Upload New File",
                             use_container_width=True):
                    for key in ["upload_raw_df", "upload_result",
                                "upload_scored"]:
                        st.session_state.pop(key, None)
                    st.rerun()
