"""
utils/model_config.py
Reads and writes PACE model metadata.

Stored in utils/model_config.json — one file per deployment (per company).
Never contains raw data, only model statistics and settings.
"""
import json
import os
from datetime import datetime

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")

_DEFAULTS = {
    "mode":                   "demo",       # "demo" | "production"
    "version":                0,
    "last_trained":           None,
    "records_trained_on":     0,
    "pending_records":        0,            # accumulated since last update
    "auto_update_threshold":  100,          # update model every N new records
    "auto_update_enabled":    True,
    "tier_thresholds": {
        "high":   0.67,
        "medium": 0.34,
    },
    "metrics": {
        "auc":      None,
        "f1":       None,
        "accuracy": None,
    },
    "version_history": [],                  # last 3 versions with metrics
}


def load() -> dict:
    """Load model config from disk, merging with defaults for any missing keys."""
    if not os.path.exists(_CONFIG_PATH):
        return dict(_DEFAULTS)
    try:
        with open(_CONFIG_PATH, "r") as f:
            data = json.load(f)
        # Merge with defaults so new keys always exist
        merged = dict(_DEFAULTS)
        merged.update(data)
        if "tier_thresholds" in data:
            merged["tier_thresholds"] = {**_DEFAULTS["tier_thresholds"], **data["tier_thresholds"]}
        if "metrics" in data:
            merged["metrics"] = {**_DEFAULTS["metrics"], **data["metrics"]}
        return merged
    except Exception as e:
        import logging
        logging.warning("PACE: model_config load failed (%s), using defaults: %s", _CONFIG_PATH, e)
        return dict(_DEFAULTS)


def save(config: dict):
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)


def record_training(metrics: dict, n_records: int):
    """Call after any training event to update config."""
    cfg = load()
    cfg["version"]            += 1
    cfg["last_trained"]        = datetime.now().strftime("%Y-%m-%d %H:%M")
    cfg["records_trained_on"]  = n_records
    cfg["pending_records"]     = 0
    cfg["metrics"]             = {
        "auc":      metrics.get("auc"),
        "f1":       metrics.get("f1"),
        "accuracy": metrics.get("accuracy"),
    }
    if "suggested_thresholds" in metrics:
        cfg["suggested_thresholds"] = metrics["suggested_thresholds"]

    # Keep last 3 versions in history
    history_entry = {
        "version":    cfg["version"],
        "date":       cfg["last_trained"],
        "n_records":  n_records,
        "metrics":    cfg["metrics"],
    }
    history = cfg.get("version_history", [])
    history.insert(0, history_entry)
    cfg["version_history"] = history[:3]

    save(cfg)
    return cfg


def add_pending_records(n: int) -> dict:
    """Call when new records are confirmed for model contribution."""
    cfg = load()
    cfg["pending_records"] = cfg.get("pending_records", 0) + n
    save(cfg)
    return cfg


def should_auto_update() -> bool:
    """Return True when pending_records has reached the configured threshold."""
    cfg = load()
    if not cfg.get("auto_update_enabled", True):
        return False
    threshold = cfg.get("auto_update_threshold", 100)
    pending   = cfg.get("pending_records", 0)
    return pending >= threshold


def set_mode(mode: str):
    """Switch between 'demo' and 'production'."""
    cfg = load()
    cfg["mode"] = mode
    save(cfg)


def set_thresholds(high: float, medium: float):
    cfg = load()
    cfg["tier_thresholds"] = {"high": high, "medium": medium}
    save(cfg)


def set_auto_update(enabled: bool, threshold: int):
    cfg = load()
    cfg["auto_update_enabled"]    = enabled
    cfg["auto_update_threshold"]  = threshold
    save(cfg)
