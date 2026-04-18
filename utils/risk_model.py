"""
utils/risk_model.py
PACE accessorial risk model — LightGBM binary classifier.

Target variable: had_accessorial (1 = shipment incurred any accessorial charge)
Output:          risk_score (0–1 probability), risk_tier (Low/Medium/High)

Feature set (all available without map data):
  Categorical : carrier, facility, appointment_type, origin_state, dest_state
  Numeric     : weight_lbs, miles, day_of_week, month, avg_dwell_hrs

Persistence:
  The trained model is saved to utils/pace_risk_model.joblib so it survives
  app restarts and can be retrained on demand from new data.

Adaptability:
  Call retrain(df) whenever new shipment data arrives — the model will
  update itself and save the new version to disk automatically.
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

_MODEL_DIR  = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_MODEL_DIR, "pace_risk_model.joblib")

# ── Feature definitions ───────────────────────────────────────────────────────
_CAT_COLS = ["carrier", "facility", "appointment_type", "origin_state", "dest_state"]
_NUM_COLS = ["weight_lbs", "miles", "day_of_week", "month", "avg_dwell_hrs"]
_ALL_COLS = _CAT_COLS + _NUM_COLS
_TARGET   = "had_accessorial"

# Columns that can be derived if missing (fallback logic)
_DERIVABLE = {"day_of_week", "month", "had_accessorial"}


# ── Tier thresholds ───────────────────────────────────────────────────────────
def score_to_tier(score: float) -> str:
    """Map a 0–1 risk probability to a Low/Medium/High tier label."""
    if score >= 0.67:
        return "High"
    if score >= 0.34:
        return "Medium"
    return "Low"


# ── Feature preparation ───────────────────────────────────────────────────────
def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and fill feature columns. Derives missing columns where possible.
    Returns a copy with all _ALL_COLS present.
    """
    df = df.copy()

    # Derive temporal features from ship_date if missing
    if "day_of_week" not in df.columns and "ship_date" in df.columns:
        df["day_of_week"] = pd.to_datetime(df["ship_date"], errors="coerce").dt.dayofweek
    if "month" not in df.columns and "ship_date" in df.columns:
        df["month"] = pd.to_datetime(df["ship_date"], errors="coerce").dt.month

    # Derive had_accessorial from accessorial_charge_usd if missing
    if "had_accessorial" not in df.columns and "accessorial_charge_usd" in df.columns:
        df["had_accessorial"] = (df["accessorial_charge_usd"] > 0).astype(int)

    # Fill optional columns with sensible defaults if absent
    if "appointment_type" not in df.columns:
        df["appointment_type"] = "Live"
    if "origin_state" not in df.columns:
        df["origin_state"] = "Unknown"
    if "dest_state" not in df.columns:
        df["dest_state"] = "Unknown"
    if "avg_dwell_hrs" not in df.columns:
        df["avg_dwell_hrs"] = 3.0   # fleet average fallback

    # Fill nulls
    for col in _CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    for col in _NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ── Model persistence ─────────────────────────────────────────────────────────
def _versioned_path(version: int) -> str:
    return os.path.join(_MODEL_DIR, f"pace_risk_model_v{version}.joblib")


def save_model(model, metrics: dict):
    from utils.model_config import load as cfg_load, record_training
    cfg = cfg_load()
    next_version = cfg.get("version", 0) + 1

    versioned = _versioned_path(next_version)
    payload   = {"model": model, "metrics": metrics, "version": next_version}
    joblib.dump(payload, versioned)
    joblib.dump(payload, _MODEL_PATH)

    n = metrics.get("n_train", 0) + metrics.get("n_test", 0)
    record_training(metrics, n)


def load_model_from_disk():
    """Returns (model, metrics) or (None, None) if no saved model exists."""
    if not os.path.exists(_MODEL_PATH):
        return None, None
    try:
        data = joblib.load(_MODEL_PATH)
        return data["model"], data["metrics"]
    except Exception:
        return None, None


def rollback_to_version(version: int) -> bool:
    """
    Restore a previous model version as the active model.
    Returns True on success.
    """
    path = _versioned_path(version)
    if not os.path.exists(path):
        return False
    try:
        data = joblib.load(path)
        joblib.dump(data, _MODEL_PATH)
        get_risk_model.clear()
        return True
    except Exception as e:
        import logging
        logging.warning("PACE: rollback_to_version(%s) failed: %s", version, e)
        return False


def list_saved_versions() -> list[dict]:
    """Return metadata for all saved versioned model files."""
    versions = []
    for fname in os.listdir(_MODEL_DIR):
        if fname.startswith("pace_risk_model_v") and fname.endswith(".joblib"):
            fpath = os.path.join(_MODEL_DIR, fname)
            try:
                data = joblib.load(fpath)
                versions.append({
                    "version":  data.get("version", "?"),
                    "metrics":  data.get("metrics", {}),
                    "file":     fname,
                })
            except Exception as e:
                import logging
                logging.warning("PACE: could not load model version file %s: %s", fname, e)
    return sorted(versions, key=lambda x: x["version"], reverse=True)


# ── Training ──────────────────────────────────────────────────────────────────
def _train_model(df: pd.DataFrame):
    """
    Internal: train LightGBM classifier on df, return (pipeline, metrics).
    Requires LightGBM and scikit-learn.
    """
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    df = _prepare_features(df)

    needed = _ALL_COLS + [_TARGET]
    df = df[[c for c in needed if c in df.columns]].dropna(subset=[_TARGET])

    if len(df) < 40:
        raise ValueError(
            f"Need at least 40 rows to train — only {len(df)} rows available."
        )

    X = df[_ALL_COLS]
    y = df[_TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ), _CAT_COLS),
        ("num", "passthrough", _NUM_COLS),
    ])

    model = Pipeline([
        ("pre", preprocessor),
        ("lgbm", LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            class_weight="balanced",   # handles imbalanced had_accessorial
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc":          round(float(roc_auc_score(y_test, y_prob)),  3),
        "f1":           round(float(f1_score(y_test, y_pred)),       3),
        "accuracy":     round(float(accuracy_score(y_test, y_pred)), 3),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "pos_rate":     round(float(y.mean()), 3),
        "features":     _ALL_COLS,
    }

    # ── Optimal threshold via Youden's J (maximises TPR - FPR on ROC curve) ──
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores   = tpr - fpr
    best_idx   = int(np.argmax(j_scores))
    opt_high   = round(float(thresholds[best_idx]), 2)

    # Medium cutoff = score at the 33rd percentile of predicted positives
    opt_medium = round(float(np.percentile(y_prob, 33)), 2)
    opt_medium = min(opt_medium, opt_high - 0.05)   # always stay below high

    metrics["suggested_thresholds"] = {
        "high":   opt_high,
        "medium": max(0.10, opt_medium),
    }

    return model, metrics


# ── Cached loader / trainer ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_risk_model(data_hash: int, _df: pd.DataFrame):
    """
    Returns (model, metrics).
    Load from disk if available, otherwise train fresh and save.
    data_hash is used as the Streamlit cache key.
    """
    model, metrics = load_model_from_disk()
    if model is not None:
        return model, metrics

    try:
        model, metrics = _train_model(_df)
        save_model(model, metrics)
        return model, metrics
    except Exception as e:
        return None, {"error": str(e)}


def retrain(df: pd.DataFrame) -> dict:
    """
    Full retrain from scratch on df. Saves versioned model to disk.
    Returns metrics dict.
    """
    model, metrics = _train_model(df)
    save_model(model, metrics)
    get_risk_model.clear()
    return metrics


def incremental_update(df_new: pd.DataFrame) -> dict:
    """
    Update the existing model with new records without retraining from scratch.
    Uses LightGBM's init_model to continue training from the current model.
    Falls back to full retrain if no saved model exists.
    Returns metrics dict.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from sklearn.model_selection import train_test_split

    existing_model, _ = load_model_from_disk()

    if existing_model is None:
        return retrain(df_new)

    df_new = _prepare_features(df_new)
    needed = _ALL_COLS + [_TARGET]
    df_new = df_new[[c for c in needed if c in df_new.columns]].dropna(subset=[_TARGET])

    if len(df_new) < 10:
        raise ValueError(f"Need at least 10 new rows to update — only {len(df_new)} provided.")

    X = df_new[_ALL_COLS]
    y = df_new[_TARGET].astype(int)

    lgbm_step = existing_model.named_steps["lgbm"]
    pre = existing_model.named_steps["pre"]
    X_transformed = pre.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.20, random_state=42
    )

    # Warm-start from the existing booster so prior knowledge is preserved.
    # Fewer trees here (100) because we're adding an incremental batch, not retraining.
    updated_lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    updated_lgbm.fit(
        X_train, y_train,
        init_model=lgbm_step.booster_,
    )

    from sklearn.pipeline import Pipeline
    updated_pipeline = Pipeline([
        ("pre",  pre),
        ("lgbm", updated_lgbm),
    ])

    y_pred = updated_pipeline.predict(X_test)
    y_prob = updated_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "auc":      round(float(roc_auc_score(y_test, y_prob)),  3),
        "f1":       round(float(f1_score(y_test, y_pred)),       3),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "n_train":  int(len(X_train)),
        "n_test":   int(len(X_test)),
        "update_type": "incremental",
    }

    save_model(updated_pipeline, metrics)
    get_risk_model.clear()
    return metrics


