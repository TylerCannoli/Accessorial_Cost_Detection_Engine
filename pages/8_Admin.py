# File: pages/8_Admin.py
import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_utils import require_auth, pace_role_is_admin          # ← your addition kept
from utils.database import (
    clear_db_cache,
    create_pace_user,
    delete_pace_user,
    get_carriers,
    get_connection_safe,
    get_db_config_status,
    get_pace_users,
    test_connection,
    update_carrier_dot_number,
)
from utils.styling import inject_css, sidebar_account, ACCENT_SOFT, TEXT_PRIMARY, TEXT_SECONDARY
import utils.model_config as mcfg
from utils.risk_model import retrain, incremental_update, rollback_to_version, list_saved_versions, get_risk_model
from utils.mock_data import generate_mock_shipments

st.set_page_config(
    page_title="PACE — Admin",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

require_auth()

username = st.session_state.get("username", "Admin")            # ← your default kept
if not pace_role_is_admin():                                     # ← your auth check kept
    sidebar_account(username)
    st.error("Access denied. Admins only.")
    st.page_link("pages/0_Home.py", label="Go to Home", icon="🏠")
    st.stop()

sidebar_account(username)

conn = get_connection_safe()
# ── DB status / retry ─────────────────────────────────────────────────────────
db_ok_cfg, db_cfg_msg = get_db_config_status()

top_a, top_b = st.columns([5, 1])
with top_a:
    st.markdown("## Admin Panel")
    st.caption(f"Logged in as **{username}** · admin")
with top_b:
    if st.button("Retry DB", use_container_width=True):
        clear_db_cache()
        st.rerun()

if conn is not None:
    st.session_state.pop("_azure_retries", None)   # reset retry counter on success
    st.success("Database connected.", icon="✅")
else:
    if not db_ok_cfg:
        st.warning(f"Database unavailable — configuration incomplete. {db_cfg_msg}", icon="⚠️")
    else:
        import re as _re, time as _time
        ok, msg = test_connection()
        if not ok:
            _code_match = _re.search(r"\((\d+),", msg)
            _code = _code_match.group(1) if _code_match else ""

            if _code == "40613":
                # Azure serverless is paused — auto-retry twice (25 s gap) before
                # giving up and asking the user to retry manually.
                _retries = st.session_state.get("_azure_retries", 0)
                if _retries < 2:
                    st.session_state["_azure_retries"] = _retries + 1
                    with st.spinner(
                        f"Azure SQL is waking up — waiting 25 s then retrying "
                        f"(attempt {_retries + 1}/2)…"
                    ):
                        _time.sleep(25)
                        clear_db_cache()
                    st.rerun()
                else:
                    st.session_state.pop("_azure_retries", None)
                    st.warning(
                        "Azure SQL is still unavailable after 2 automatic retries. "
                        "The serverless instance may need more time — click **Retry DB** "
                        "in 30 s, or resume the server in the Azure portal.",
                        icon="⚠️",
                    )
            elif "connection failed" in msg.lower() or "adaptive server" in msg.lower():
                _server = os.getenv("DB_SERVER", "")
                st.warning(
                    f"Cannot reach SQL server `{_server or 'DB_SERVER not set'}`. "
                    "Check firewall rules, server name, and that the server is running.",
                    icon="⚠️",
                )
                st.info(
                    "Verify `DB_SERVER`, `DB_DATABASE`, `DB_USERNAME`, and `DB_PASSWORD` "
                    "environment variables on the host.",
                    icon="💡",
                )
            else:
                _clean = msg[:180] + ("…" if len(msg) > 180 else "")
                st.warning(_clean, icon="⚠️")
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## User Management")


# ── User Management ───────────────────────────────────────────────────────────
col_form, col_users = st.columns([1, 2], gap="large")

with col_form:
    with st.container(border=True):
        st.markdown("#### Create User")
        with st.form("create_user_form"):
            new_username = st.text_input("Username").strip().lower()  # ← sanitized from teammate
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"])
            submitted = st.form_submit_button(
                "Create User",
                use_container_width=True,                         # ← fixed: width="stretch" is invalid
                type="primary",
            )
            if submitted:
                if not new_username or not new_password:
                    st.error("Username and password are required.")
                elif conn is None:
                    st.warning("No DB connection — cannot create user.")
                else:
                    ok, msg = create_pace_user(conn, new_username, new_password, new_role)
                    if ok:
                        st.success(msg)
                        clear_db_cache()                          # ← added from teammate
                        st.rerun()
                    else:
                        st.error(f"Failed: {msg}")

with col_users:
    with st.container(border=True):
        st.markdown("#### Current Users")
        if conn is None:
            st.warning("No database connection — showing fallback accounts only.")
            st.dataframe(
                [{"username": "admin", "role": "admin"}, {"username": "user", "role": "user"}],
                use_container_width=True,                         # ← fixed: width="stretch" is invalid
                hide_index=True,
            )
        else:
            users_df = get_pace_users(conn)
            if users_df.empty:
                st.info("No users found in PaceUsers table.")
            else:
                st.dataframe(users_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### Delete User")
        if conn is not None:
            users_df2 = get_pace_users(conn)
            deletable = []

            if not users_df2.empty and "username" in users_df2.columns:  # ← safer column check from teammate
                deletable = [
                    u for u in users_df2["username"].astype(str).tolist()
                    if u.lower() != username.lower()              # ← case-insensitive from teammate
                ]

            if deletable:
                del_user = st.selectbox("Select user to delete", deletable, key="del_user_sel")
                if st.button("Delete User", type="primary", use_container_width=True):
                    ok, msg = delete_pace_user(conn, del_user, current_username=username)  # ← safety param from teammate
                    if ok:
                        st.success(msg)
                        clear_db_cache()                          # ← added from teammate
                        st.rerun()
                    else:
                        st.error(f"Failed: {msg}")
            else:
                st.caption("No other users to delete.")
        else:
            st.caption("Database unavailable.")

st.divider()

# ── Model Management ───────────────────────────────────────────────────────────
st.markdown("## Model Management")
st.caption("Control how PACE learns from your data. All model activity stays within your deployment.")
st.markdown("<br>", unsafe_allow_html=True)

cfg = mcfg.load()

# ── Row 1: Status cards ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4, gap="medium")

mode_color = "#9333EA" if cfg["mode"] == "production" else "#64748B"
mode_label = "PRODUCTION" if cfg["mode"] == "production" else "DEMO"

with c1:
    with st.container(border=True):
        st.markdown(f"<p style='color:{mode_color};font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>{mode_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:22px;font-weight:700;margin:4px 0 0'>Model Mode</p>", unsafe_allow_html=True)
        st.caption("Demo uses mock data only")

with c2:
    with st.container(border=True):
        auc = cfg["metrics"].get("auc")
        auc_display = f"{auc:.2f}" if auc else "—"
        color = "#22C55E" if auc and auc >= 0.75 else "#F59E0B" if auc else "#64748B"
        st.markdown(f"<p style='color:{color};font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>AUC SCORE</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:28px;font-weight:700;margin:4px 0 0'>{auc_display}</p>", unsafe_allow_html=True)
        st.caption("Model accuracy (0–1)")

with c3:
    with st.container(border=True):
        n = cfg.get("records_trained_on", 0)
        st.markdown(f"<p style='color:{ACCENT_SOFT};font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>TRAINED ON</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:28px;font-weight:700;margin:4px 0 0'>{n:,}</p>", unsafe_allow_html=True)
        st.caption("Total shipment records")

with c4:
    with st.container(border=True):
        pending = cfg.get("pending_records", 0)
        p_color = "#22C55E" if pending >= cfg.get("auto_update_threshold", 100) else "#64748B"
        st.markdown(f"<p style='color:{p_color};font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>PENDING</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:28px;font-weight:700;margin:4px 0 0'>{pending:,}</p>", unsafe_allow_html=True)
        st.caption(f"New records since last update (threshold: {cfg.get('auto_update_threshold', 100)})")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Controls + Version History ─────────────────────────────────────────
col_controls, col_versions = st.columns([1, 1], gap="large")

with col_controls:
    with st.container(border=True):
        st.markdown("#### Model Settings")

        # Mode toggle
        new_mode = st.selectbox(
            "Model mode",
            ["demo", "production"],
            index=0 if cfg["mode"] == "demo" else 1,
            help="Demo uses mock data. Production learns from your real uploaded shipments.",
        )
        if new_mode != cfg["mode"]:
            mcfg.set_mode(new_mode)
            st.success(f"Switched to {new_mode} mode.")
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk tier thresholds
        st.markdown("**Risk Tier Thresholds**")
        st.caption("Adjust what score counts as High or Medium risk for your operation.")

        suggested = cfg.get("suggested_thresholds", {})
        sug_high  = suggested.get("high")
        sug_med   = suggested.get("medium")

        high_thresh = st.slider(
            "High risk cutoff", 0.50, 0.95,
            float(cfg["tier_thresholds"]["high"]), 0.01,
            help="Shipments above this score are flagged High Risk",
        )
        if sug_high:
            diff_h = round(high_thresh - sug_high, 2)
            arrow  = "↑ above" if diff_h > 0 else "↓ below" if diff_h < 0 else "matches"
            color  = "#22C55E" if abs(diff_h) <= 0.05 else "#F59E0B"
            st.markdown(
                f"<p style='font-size:11px;color:{color};margin:-8px 0 8px;'>"
                f"📊 Data suggests <strong>{sug_high}</strong> &nbsp;·&nbsp; "
                f"current is {abs(diff_h):.2f} {arrow} recommendation</p>",
                unsafe_allow_html=True,
            )

        med_thresh = st.slider(
            "Medium risk cutoff", 0.10, float(high_thresh) - 0.05,
            float(cfg["tier_thresholds"]["medium"]), 0.01,
            help="Shipments above this score are flagged Medium Risk",
        )
        if sug_med:
            diff_m = round(med_thresh - sug_med, 2)
            arrow  = "↑ above" if diff_m > 0 else "↓ below" if diff_m < 0 else "matches"
            color  = "#22C55E" if abs(diff_m) <= 0.05 else "#F59E0B"
            st.markdown(
                f"<p style='font-size:11px;color:{color};margin:-8px 0 8px;'>"
                f"📊 Data suggests <strong>{sug_med}</strong> &nbsp;·&nbsp; "
                f"current is {abs(diff_m):.2f} {arrow} recommendation</p>",
                unsafe_allow_html=True,
            )

        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button("Save Thresholds", type="primary", use_container_width=True):
                mcfg.set_thresholds(high_thresh, med_thresh)
                st.success("Thresholds saved.")
        with col_reset:
            if sug_high and sug_med:
                if st.button("Use Recommended", use_container_width=True):
                    mcfg.set_thresholds(sug_high, sug_med)
                    st.success(f"Set to recommended: High {sug_high} · Medium {sug_med}")
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Auto-update settings
        st.markdown("**Auto-Update Settings**")
        auto_enabled = st.toggle(
            "Auto-update model when new records arrive",
            value=cfg.get("auto_update_enabled", True),
        )
        auto_threshold = st.number_input(
            "Update every N new records",
            min_value=10, max_value=10000,
            value=int(cfg.get("auto_update_threshold", 100)),
            step=10,
            disabled=not auto_enabled,
        )
        if st.button("Save Auto-Update Settings"):
            mcfg.set_auto_update(auto_enabled, auto_threshold)
            st.success("Auto-update settings saved.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### Train from CSV")
        st.caption(
            "Upload any shipment CSV to train the model on real data. "
            "Column names are auto-normalized — `detention_fee`, `demurrage`, `surcharge`, etc. "
            "are all recognized as accessorial charges."
        )

        train_file = st.file_uploader(
            "Drop training CSV here",
            type=["csv", "xlsx", "xls"],
            key="admin_train_upload",
        )

        if train_file is not None:
            try:
                if train_file.name.endswith((".xlsx", ".xls")):
                    import pandas as pd
                    train_df = pd.read_excel(train_file)
                else:
                    import pandas as pd
                    train_df = pd.read_csv(train_file)

                from utils.doc_parser import ensure_expected_columns
                train_df = ensure_expected_columns(train_df)

                st.info(f"{len(train_df):,} rows loaded · columns: {', '.join(train_df.columns.tolist())}")

                t0, t1_btn = st.columns(2)
                with t0:
                    if st.button("Incremental Update from File", type="primary", use_container_width=True):
                        with st.spinner("Updating model..."):
                            try:
                                metrics = incremental_update(train_df)
                                st.success(f"Updated! AUC: {metrics['auc']} · F1: {metrics['f1']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Update failed: {e}")
                with t1_btn:
                    if st.button("Full Retrain from File", use_container_width=True):
                        with st.spinner("Retraining from scratch..."):
                            try:
                                metrics = retrain(train_df)
                                st.success(f"Retrained! AUC: {metrics['auc']} · F1: {metrics['f1']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Retrain failed: {e}")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### Manual Training (Mock Data)")
        st.caption("Train on generated mock data — useful for testing the pipeline without a real dataset.")

        t1, t2 = st.columns(2)
        with t1:
            if st.button("Incremental Update", type="primary", use_container_width=True):
                with st.spinner("Updating model..."):
                    try:
                        df = generate_mock_shipments(500)
                        metrics = incremental_update(df)
                        st.success(f"Updated! AUC: {metrics['auc']} · F1: {metrics['f1']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")

        with t2:
            if st.button("Full Retrain", use_container_width=True):
                with st.spinner("Retraining from scratch..."):
                    try:
                        df = generate_mock_shipments(1000)
                        metrics = retrain(df)
                        st.success(f"Retrained! AUC: {metrics['auc']} · F1: {metrics['f1']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retrain failed: {e}")

with col_versions:
    with st.container(border=True):
        st.markdown("#### Version History")
        st.caption("Last 3 model versions. Roll back if a new update hurts performance.")

        versions = list_saved_versions()
        if not versions:
            st.info("No saved versions yet. Run a training event to create one.")
        else:
            for v in versions:
                m = v.get("metrics", {})
                auc_v = m.get("auc")
                f1_v  = m.get("f1")
                acc_v = m.get("accuracy")
                update_type = m.get("update_type", "full retrain")
                is_current  = (v["version"] == cfg.get("version", 0))

                label = f"v{v['version']}{'  ← current' if is_current else ''}"
                with st.expander(label, expanded=is_current):
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("AUC",      f"{auc_v:.3f}" if auc_v else "—")
                    mc2.metric("F1",       f"{f1_v:.3f}"  if f1_v  else "—")
                    mc3.metric("Accuracy", f"{acc_v:.3f}" if acc_v else "—")
                    st.caption(f"Type: {update_type} · Records: {m.get('n_train', 0) + m.get('n_test', 0)}")

                    if not is_current:
                        if st.button(f"Roll back to v{v['version']}", key=f"rollback_{v['version']}"):
                            ok = rollback_to_version(v["version"])
                            if ok:
                                st.success(f"Rolled back to v{v['version']}")
                                st.rerun()
                            else:
                                st.error("Rollback failed — model file not found.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### Last Trained")
        last = cfg.get("last_trained")
        if last:
            st.markdown(f"<p style='font-size:20px;font-weight:600'>{last}</p>", unsafe_allow_html=True)
        else:
            st.info("Model has not been trained yet.")

        f1_val  = cfg["metrics"].get("f1")
        acc_val = cfg["metrics"].get("accuracy")
        if f1_val or acc_val:
            mc1, mc2 = st.columns(2)
            mc1.metric("F1 Score", f"{f1_val:.3f}"  if f1_val  else "—")
            mc2.metric("Accuracy", f"{acc_val:.3f}" if acc_val else "—")

st.divider()

# ── PACE FT-Transformer Model Metrics ─────────────────────────────────────────
st.markdown("## PACE FT-Transformer — Model Card")
st.caption(
    "Metrics from the last full training run of the FT-Transformer (pace_transformer.pt). "
    "These are separate from the lightweight risk model above."
)

import json as _json
_tm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "training_metrics.json")
if os.path.exists(_tm_path):
    with open(_tm_path) as _f:
        _tm = _json.load(_f)

    _test = _tm.get("test_metrics", {})
    _hp   = _tm.get("hyperparams",  {})
    _hist = _tm.get("history",      {})

    # ── Top metrics ────────────────────────────────────────────────────────────
    tm1, tm2, tm3, tm4 = st.columns(4)
    with tm1:
        with st.container(border=True):
            st.markdown("<p style='color:#A78BFA;font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>TEST MAE</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:28px;font-weight:700;margin:4px 0 0'>±{_test.get('test_mae', 0):.2f}</p>", unsafe_allow_html=True)
            st.caption("Risk score error (0–100 scale)")
    with tm2:
        with st.container(border=True):
            st.markdown("<p style='color:#A78BFA;font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>TEST RMSE</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:28px;font-weight:700;margin:4px 0 0'>{_test.get('test_rmse', 0):.2f}</p>", unsafe_allow_html=True)
            st.caption("Root mean squared error")
    with tm3:
        with st.container(border=True):
            _acc_pct = _test.get('test_acc', 0) * 100
            st.markdown("<p style='color:#A78BFA;font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>CHARGE TYPE ACC</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:28px;font-weight:700;margin:4px 0 0'>{_acc_pct:.1f}%</p>", unsafe_allow_html=True)
            st.caption("6-class charge classification")
    with tm4:
        with st.container(border=True):
            st.markdown("<p style='color:#A78BFA;font-size:11px;font-weight:700;letter-spacing:1px;margin:0'>TRAINED AT</p>", unsafe_allow_html=True)
            _trained_at = _tm.get("trained_at", "Unknown")
            st.markdown(f"<p style='font-size:14px;font-weight:600;margin:4px 0 0'>{_trained_at}</p>", unsafe_allow_html=True)
            st.caption(f"{_hp.get('epochs_run', '?')} epochs · {_hp.get('n_cat', '?')} cat + {_hp.get('n_cont', '?')} cont features")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Context note ────────────────────────────────────────────────────────────
    st.info(
        "**About these metrics:** The FT-Transformer was trained and evaluated on "
        "**CTGAN-generated synthetic data** derived from real FMCSA inspection records. "
        "High accuracy reflects performance within the synthetic distribution. "
        "Real-world validation against live accessorial billing data is the next milestone — "
        "upload production shipment data via the Upload page to begin building that ground truth.",
        icon="ℹ️",
    )

    # ── Architecture summary ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Architecture")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Layers",       _hp.get("n_layers",  "?"))
        a2.metric("Attn Heads",   _hp.get("n_heads",   "?"))
        a3.metric("Token Dim",    _hp.get("token_dim", "?"))
        a4.metric("Batch Size",   _hp.get("batch_size","?"))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charge type legend ──────────────────────────────────────────────────
        st.markdown("**Predicted Charge Types (6 classes)**")
        _charge_types = [
            ("No Charge",          "#34D399", "Carrier has no predicted accessorial"),
            ("Detention",          "#F59E0B", "Driver/truck held at facility beyond free time"),
            ("Safety Surcharge",   "#F87171", "Elevated OOS or violation history"),
            ("Compliance Fee",     "#A78BFA", "Regulatory or hours-of-service issues"),
            ("Hazmat Fee",         "#60A5FA", "Hazardous materials handling required"),
            ("High Risk / Multiple","#EC4899", "Multiple charge types likely; high overall risk"),
        ]
        _cols = st.columns(3)
        for _idx, (_label, _color, _desc) in enumerate(_charge_types):
            with _cols[_idx % 3]:
                st.markdown(
                    f"<div style='border:1px solid {_color};border-radius:6px;"
                    f"padding:8px 12px;margin-bottom:8px;'>"
                    f"<span style='color:{_color};font-weight:700;font-size:12px;'>{_label}</span>"
                    f"<div style='color:#94A3B8;font-size:11px;margin-top:4px;'>{_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Training curve ──────────────────────────────────────────────────────────
    _val_losses = _hist.get("val_loss", [])
    if _val_losses:
        import plotly.graph_objects as _go
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### Training Curve")
            _epochs = list(range(1, len(_val_losses) + 1))
            _train_losses = _hist.get("train_loss", [])
            _fig = _go.Figure()
            if _train_losses:
                _fig.add_scatter(x=_epochs, y=_train_losses, name="Train Loss",
                                 line=dict(color="#9333EA", width=2))
            _fig.add_scatter(x=_epochs, y=_val_losses, name="Val Loss",
                             line=dict(color="#60A5FA", width=2))
            _fig.update_layout(
                height=250, margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor="#0f0a1e", paper_bgcolor="#0f0a1e",
                font=dict(color="#A78BFA"),
                xaxis=dict(title="Epoch", color="#94A3B8",
                           gridcolor="rgba(150,50,200,0.15)"),
                yaxis=dict(title="Loss", color="#94A3B8",
                           gridcolor="rgba(150,50,200,0.15)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(_fig, use_container_width=True)
else:
    st.info("Training metrics not found. Run the FT-Transformer training pipeline to generate `models/training_metrics.json`.", icon="ℹ️")

st.divider()

# ── Carrier DOT Numbers ────────────────────────────────────────────────────────
st.markdown("## Carrier DOT Numbers")
st.caption(
    "Assign USDOT numbers to carriers stored in the database. "
    "These are used by the Carrier Benchmarking page for DOT number search and FMCSA lookups."
)

if conn is None:
    st.warning("Database unavailable — cannot manage carrier DOT numbers.", icon="⚠️")
else:
    carriers_df = get_carriers(conn)
    if carriers_df.empty:
        st.info("No carriers found in the Carriers table.", icon="ℹ️")
    else:
        # Normalize column names (case-insensitive)
        carriers_df.columns = [c.lower() for c in carriers_df.columns]

        dot_col, edit_col = st.columns([2, 1], gap="large")

        with dot_col:
            with st.container(border=True):
                st.markdown("#### Current Carrier DOT Numbers")
                display_cols = ["carrier_name", "dot_number"] if "dot_number" in carriers_df.columns else ["carrier_name"]
                rename_map = {"carrier_name": "Carrier", "dot_number": "DOT Number"}
                display_df = carriers_df[display_cols].rename(columns=rename_map).copy()
                if "DOT Number" in display_df.columns:
                    display_df["DOT Number"] = display_df["DOT Number"].fillna("—")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        with edit_col:
            with st.container(border=True):
                st.markdown("#### Update DOT Number")
                carrier_names = carriers_df["carrier_name"].dropna().tolist() if "carrier_name" in carriers_df.columns else []
                if carrier_names:
                    with st.form("update_dot_form"):
                        selected_carrier = st.selectbox("Carrier", carrier_names)
                        new_dot = st.text_input(
                            "USDOT Number",
                            placeholder="e.g. 72011",
                            help="Enter the official USDOT number for this carrier. Leave blank to clear.",
                        )
                        save_clicked = st.form_submit_button("Save", type="primary", use_container_width=True)
                        if save_clicked:
                            ok, msg = update_carrier_dot_number(conn, selected_carrier, new_dot)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(f"Failed: {msg}")
                else:
                    st.info("No carriers available.")