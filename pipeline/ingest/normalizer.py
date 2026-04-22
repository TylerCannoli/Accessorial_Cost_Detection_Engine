"""
pipeline/ingest/normalizer.py
==============================
DataNormalizer — fills in missing PACE features using training-data
statistics (median imputation) and static lookup tables for state-level
and time-series enrichments.

This runs AFTER CarrierSchema.normalize() and BEFORE inference.
Its job: ensure every required PACE column exists with a sensible value,
even when the carrier only provided DOT number + state + date.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

ENRICH_DIR = Path(__file__).parents[2] / "outputs" / "enrichment"
ARTIFACTS_DIR = Path(__file__).parents[2] / "models"


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_csv_lookup(filename: str, key_col: str) -> Dict[str, Dict[str, Any]]:
    """Load an enrichment CSV as a dict keyed by key_col."""
    path = ENRICH_DIR / filename
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    df[key_col] = df[key_col].astype(str).str.strip()
    return df.set_index(key_col).to_dict(orient="index")


class DataNormalizer:
    """
    Fills missing PACE features using:
      1. State-level enrichment lookups (parking, crash, ATRI, FAF)
      2. Carrier-level enrichment lookups (carrier history, SMS percentiles)
      3. Time-series lookups (freight rates by year+month)
      4. Column-level median imputation from training artifacts
      5. Zero/UNKNOWN fallbacks for anything still missing

    Usage:
        normalizer = DataNormalizer()
        df_ready   = normalizer.fill(df_normalized)
    """

    def __init__(self):
        self._state_parking   = _load_csv_lookup("state_parking_features.csv",    "state_abbr")
        self._state_crash     = _load_csv_lookup("state_fatal_crash_features.csv", "state_abbr")
        self._state_atri      = _load_csv_lookup("atri_detention_features.csv",    "state_abbr")
        self._state_faf       = _load_csv_lookup("faf_freight_features.csv",       "state_abbr")
        self._carrier_hist    = _load_csv_lookup("carrier_oos_history.csv",        "dot_number")
        self._sms_pct         = _load_csv_lookup("sms_percentile_features.csv",    "dot_number")
        self._atri_constants  = _load_json(ENRICH_DIR / "atri_national_constants.json")
        self._freight_rates   = self._load_freight_rates()
        self._col_medians: Dict[str, float] = {}
        self._load_medians()

    def _load_freight_rates(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Load freight_rate_features.csv keyed by (year, month) tuple."""
        path = ENRICH_DIR / "freight_rate_features.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        lookup = {}
        for _, row in df.iterrows():
            key = (int(row["year"]), int(row["month"]))
            lookup[key] = row.to_dict()
        return lookup

    def _load_medians(self):
        """Load per-column medians from training artifacts if available."""
        for candidate in ("artifacts_ltl.pkl", "artifacts_ftl.pkl", "artifacts.pkl"):
            artifacts_path = ARTIFACTS_DIR / candidate
            if artifacts_path.exists():
                break
        else:
            return
        try:
            import pickle
            with open(artifacts_path, "rb") as f:
                import __main__
                from pipeline.pace_transformer import CategoricalEncoder
                __main__.CategoricalEncoder = CategoricalEncoder
                artifacts = pickle.load(f)  # nosec B301
            scaler   = artifacts.get("scaler")
            cont_cols = artifacts.get("cont_cols", [])
            if scaler is not None and hasattr(scaler, "mean_"):
                for col, mean_val in zip(cont_cols, scaler.mean_):
                    self._col_medians[col] = float(mean_val)
        except Exception as e:
            import logging
            logging.warning("PACE: could not load training artifacts from %s: %s", artifacts_path, e)

    def fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill all missing PACE features into the DataFrame."""
        df = df.copy()
        df = self._fill_state_features(df)
        df = self._fill_carrier_features(df)
        df = self._fill_freight_rates(df)
        df = self._fill_atri_constants(df)
        df = self._fill_remaining(df)
        return df

    def _fill_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join state-level enrichments based on report_state."""
        state_col = "report_state"
        if state_col not in df.columns:
            return df

        lookups = [
            (self._state_parking, "state_parking_features"),
            (self._state_crash,   "state_fatal_crash_features"),
            (self._state_atri,    "atri_detention_features"),
            (self._state_faf,     "faf_freight_features"),
        ]

        for lookup, _ in lookups:
            if not lookup:
                continue
            # Only fill columns that aren't already present
            sample_key = next(iter(lookup))
            for col in lookup[sample_key].keys():
                if col not in df.columns:
                    df[col] = df[state_col].map(
                        lambda s, lk=lookup: lk.get(str(s), {}).get(col, np.nan)
                    )

        return df

    def _fill_carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join carrier-level enrichments based on dot_number."""
        dot_col = "dot_number"
        if dot_col not in df.columns:
            return df

        dot_str = df[dot_col].astype(str).str.strip()

        for lookup, cols_prefix in [
            (self._carrier_hist, "carrier_hist_"),
            (self._sms_pct,      "sms_"),
        ]:
            if not lookup:
                continue
            sample_key = next(iter(lookup))
            for col in lookup[sample_key].keys():
                if col not in df.columns:
                    df[col] = dot_str.map(
                        lambda d, lk=lookup: lk.get(d, {}).get(col, np.nan)
                    )

        return df

    def _fill_freight_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join freight rate features based on insp_year + insp_month."""
        if not self._freight_rates:
            return df
        if "insp_year" not in df.columns or "insp_month" not in df.columns:
            return df

        sample_key = next(iter(self._freight_rates))
        rate_cols = [c for c in self._freight_rates[sample_key].keys()
                     if c not in ("year", "month")]

        for col in rate_cols:
            if col not in df.columns:
                df[col] = df.apply(
                    lambda row: self._freight_rates.get(
                        (int(row["insp_year"]), int(row["insp_month"])), {}
                    ).get(col, np.nan),
                    axis=1,
                )

        return df

    def _fill_atri_constants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill national ATRI constants (scalar values, same for all rows)."""
        atri_map = {
            "atri_detention_freq_rate":   "detention_freq_rate",
            "atri_avg_detention_hours":   "avg_detention_hours",
            "atri_avg_detention_cost_usd":"avg_detention_cost_usd",
            "atri_cost_per_hour_usd":     "cost_per_hour",
            "atri_reefer_multiplier":     "reefer_multiplier",
        }
        for pace_col, json_key in atri_map.items():
            if pace_col not in df.columns and json_key in self._atri_constants:
                df[pace_col] = float(self._atri_constants[json_key])
        return df

    def _fill_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final pass: fill any remaining NaN continuous values.
        Uses training-data column means when available, otherwise 0.
        """
        from pipeline.config import CONTINUOUS_COLUMNS_V2
        for col in CONTINUOUS_COLUMNS_V2:
            if col not in df.columns:
                df[col] = self._col_medians.get(col, 0.0)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                    self._col_medians.get(col, 0.0)
                )
        return df

    def coverage_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return a dict showing fill coverage before and after normalization."""
        from pipeline.config import CONTINUOUS_COLUMNS_V2, CATEGORICAL_COLUMNS
        all_cols = list(CONTINUOUS_COLUMNS_V2) + list(CATEGORICAL_COLUMNS)
        before_present = sum(1 for c in all_cols if c in df.columns and df[c].notna().any())
        filled = self.fill(df)
        after_present  = sum(1 for c in all_cols if c in filled.columns and filled[c].notna().any())
        return {
            "total_expected": len(all_cols),
            "before_fill":    before_present,
            "after_fill":     after_present,
            "coverage_pct":   round(after_present / len(all_cols) * 100, 1),
            "still_missing":  [c for c in all_cols if c not in filled.columns or filled[c].isna().all()],
        }
