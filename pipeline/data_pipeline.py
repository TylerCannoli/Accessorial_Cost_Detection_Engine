"""
PACE Data Pipeline
pipeline/data_pipeline.py
 
Handles all data ingestion, validation, schema detection, cleaning,
and transformation for the three PACE input methods:
    1. CSV batch upload
    2. Manual shipment input
    3. DOT number lookup
 
Works on both the GPU cluster (aimlsrv) and Oracle Cloud (daxori).
"""
 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pipeline.config import (
    CONTINUOUS_COLUMNS,
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    DOT_COLUMN,
)
 
# ── Schema definitions ────────────────────────────────────────────
 
# Full PACE schema — 152 column FT-Transformer input
PACE_SCHEMA_COLS = set(CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + [ID_COLUMN, DOT_COLUMN])
 
# Legacy schema — old LightGBM app columns
LEGACY_SCHEMA_COLS = {
    "shipment_id", "ship_date", "carrier", "facility",
    "weight_lbs", "miles", "base_freight_usd", "accessorial_charge_usd",
    "appointment_type", "origin_state", "dest_state",
}
 
# Minimum columns required to attempt inference in PACE schema
PACE_REQUIRED_COLS = {
    "dot_number",
    "carrier_status_code", "carrier_carrier_operation",
    "carrier_power_units", "carrier_total_drivers",
    "oos_total", "driver_oos_total", "vehicle_oos_total",
    "basic_viol", "unsafe_viol", "vh_maint_viol",
    "crash_count", "crash_avg_severity",
}
 
# Minimum columns required to attempt inference in legacy schema
LEGACY_REQUIRED_COLS = {
    "carrier", "weight_lbs", "miles",
}
 
# ── Column alias map — auto-mapping similar names ─────────────────
# Maps common alternative column names → canonical PACE column names
COLUMN_ALIASES = {
    # DOT / ID
    "dot":                      "dot_number",
    "usdot":                    "dot_number",
    "usdot_number":             "dot_number",
    "dot_num":                  "dot_number",
    "carrier_dot":              "dot_number",
    "id":                       "unique_id",
    "unique_id":                "unique_id",
    "shipment_id":              "unique_id",
    "record_id":                "unique_id",
 
    # Carrier profile
    "carrier_id":               "dot_number",
    "carrier_op":               "carrier_carrier_operation",
    "operation":                "carrier_carrier_operation",
    "fleet_size":               "carrier_fleetsize",
    "power_units":              "carrier_power_units",
    "truck_units":              "carrier_truck_units",
    "total_drivers":            "carrier_total_drivers",
    "drivers":                  "carrier_total_drivers",
    "cdl_drivers":              "carrier_total_cdl",
    "mileage":                  "carrier_mcs150_mileage",
    "annual_mileage":           "carrier_mcs150_mileage",
    "crash_rate":               "carrier_recordable_crash_rate",
    "safety_rating":            "carrier_safety_rating",
    "hm_flag":                  "carrier_hm_ind",
    "hazmat_flag":              "carrier_hm_ind",
    "state":                    "carrier_phy_state",
    "carrier_state":            "carrier_phy_state",
    "origin_state":             "carrier_phy_state",
    "country":                  "carrier_phy_country",
 
    # SMS
    "sms_power_units":          "sms_nbr_power_unit",
    "sms_drivers":              "sms_driver_total",
    "sms_mileage":              "sms_recent_mileage",
 
    # Violations
    "oos":                      "oos_total",
    "out_of_service":           "oos_total",
    "driver_oos":               "driver_oos_total",
    "vehicle_oos":              "vehicle_oos_total",
    "hazmat_oos":               "hazmat_oos_total",
    "basic_violations":         "basic_viol",
    "unsafe_violations":        "unsafe_viol",
    "fatigue_violations":       "fatigued_viol",
    "driver_fitness":           "dr_fitness_viol",
    "alcohol_violations":       "subt_alcohol_viol",
    "vehicle_maintenance":      "vh_maint_viol",
    "hazmat_violations":        "hm_viol",
 
    # Crash
    "crashes":                  "crash_count",
    "num_crashes":              "crash_count",
    "fatalities":               "crash_fatalities_total",
    "injuries":                 "crash_injuries_total",
    "towaways":                 "crash_towaway_total",
    "severity":                 "crash_avg_severity",
    "hazmat_releases":          "crash_hazmat_releases",
 
    # Inspection
    "inspection_level":         "insp_level_id",
    "inspection_year":          "insp_year",
    "inspection_month":         "insp_month",
    "inspection_day":           "insp_day",
    "day_of_week":              "insp_dow",
 
    # Economic / fuel
    "diesel_price":             "eia_diesel_national",
    "diesel_national":          "eia_diesel_national",
    "crude_wti":                "eia_crude_wti_spot",
    "wti":                      "eia_crude_wti_spot",
    "freight_index":            "fred_TSIFRGHT",
 
    # Weather
    "high_temp":                "wx_avg_high_f",
    "low_temp":                 "wx_avg_low_f",
    "precipitation":            "wx_total_precip_in",
    "snowfall":                 "wx_total_snow_in",
    "wind_speed":               "wx_avg_wind_mph",
    "wind":                     "wx_avg_wind_mph",
 
}
 
# ── Boolean columns that need True/False → Y/N conversion ────────
BOOL_COLS = [
    "sms_hm_flag", "sms_pc_flag", "sms_private_only",
    "sms_authorized_for_hire", "sms_exempt_for_hire", "sms_private_property",
]
 
# ── Default fill values for missing columns ───────────────────────
NUMERIC_DEFAULTS = {col: 0.0 for col in CONTINUOUS_COLUMNS}
CATEGORICAL_DEFAULTS = {col: "UNKNOWN" for col in CATEGORICAL_COLUMNS}
FEATURE_DEFAULTS = {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS}
 
 
# ══════════════════════════════════════════════════════════════════
# Schema Detection
# ══════════════════════════════════════════════════════════════════
 
def detect_schema(df: pd.DataFrame) -> str:
    """
    Auto-detect whether a DataFrame uses the PACE schema or legacy schema.
 
    Returns:
        "pace"   — has PACE-specific columns (dot_number, oos_total, etc.)
        "legacy" — has old app columns (carrier, weight_lbs, miles, etc.)
        "unknown" — cannot determine
    """
    cols_lower = {c.lower().strip() for c in df.columns}
 
    pace_hits   = len(PACE_REQUIRED_COLS & cols_lower)
    legacy_hits = len(LEGACY_REQUIRED_COLS & cols_lower)
 
    if pace_hits >= 3:
        return "pace"
    if legacy_hits >= 2:
        return "legacy"
 
    # Check aliases
    aliased = {COLUMN_ALIASES.get(c, c) for c in cols_lower}
    pace_alias_hits = len(PACE_REQUIRED_COLS & aliased)
 
    if pace_alias_hits >= 3:
        return "pace_aliased"
 
    return "unknown"
 
 
# ══════════════════════════════════════════════════════════════════
# Column Mapping
# ══════════════════════════════════════════════════════════════════
 
def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize column names:
    - Strip whitespace
    - Lowercase
    - Apply alias map
 
    Returns (normalized_df, mapping_applied)
    """
    mapping = {}
    new_cols = {}
 
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_").replace("-", "_")
        canonical = COLUMN_ALIASES.get(clean, clean)
        if canonical != col:
            mapping[col] = canonical
        new_cols[col] = canonical
 
    df = df.rename(columns=new_cols)
    return df, mapping
 
 
def find_missing_required_cols(df: pd.DataFrame,
                                schema: str = "pace") -> List[str]:
    """Return list of required columns missing from df."""
    if schema == "legacy":
        required = LEGACY_REQUIRED_COLS
    else:
        required = PACE_REQUIRED_COLS
    return [c for c in required if c not in df.columns]
 
 
# ══════════════════════════════════════════════════════════════════
# Data Cleaning
# ══════════════════════════════════════════════════════════════════
 
def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean a DataFrame for PACE inference:
    - Strip string whitespace
    - Fix boolean columns
    - Coerce numeric columns
    - Fill missing values with defaults
    - Remove completely empty rows
 
    Returns (cleaned_df, list_of_warnings)
    """
    warnings = []
    df = df.copy()

    # Deduplicate column names (duplicate cols make df[col] return a DataFrame,
    # not a Series, which breaks .str access and causes AttributeError)
    if df.columns.duplicated().any():
        seen: dict = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
        warnings.append("Duplicate column names detected and renamed automatically.")

    # Strip string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "None": None, "": None})
 
    # Fix boolean columns
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].map({
                "TRUE": "Y", "FALSE": "N",
                "YES": "Y", "NO": "N",
                "1": "Y", "0": "N",
                "Y": "Y", "N": "N",
            }).fillna("N")
 
    # Coerce numeric columns
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns:
            original_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,\s%]", "", regex=True),
                errors="coerce"
            )
            new_nulls = df[col].isna().sum()
            if new_nulls > original_nulls:
                warnings.append(
                    f"Column '{col}': {new_nulls - original_nulls} non-numeric "
                    f"values coerced to 0"
                )
            df[col] = df[col].fillna(0.0)
 
    # Fill missing categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")
        else:
            df[col] = "UNKNOWN"
 
    # Fill missing continuous columns with 0
    for col in CONTINUOUS_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
 
    # Drop completely empty rows
    before = len(df)
    df = df.dropna(how="all")
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"Removed {dropped} completely empty rows")
 
    return df, warnings
 
 
# ══════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════
 
def validate_dataframe(df: pd.DataFrame,
                        schema: str = "pace") -> Tuple[List[str], List[str], pd.Series]:
    """
    Validate a DataFrame for PACE inference.
 
    Returns:
        errors        — list of blocking error messages
        warnings      — list of non-blocking warnings
        row_fail_mask — boolean Series (True = row has errors)
    """
    errors   = []
    warnings = []
 
    if df.empty:
        return ["File contains no data"], [], pd.Series(dtype=bool)
 
    row_fail_mask = pd.Series(False, index=df.index)
 
    if schema in ("pace", "pace_aliased"):
        # Check required PACE columns
        missing = find_missing_required_cols(df, schema="pace")
        if missing:
            errors.append(
                f"Missing required columns: {', '.join(sorted(missing))}. "
                f"These can be filled in manually or enriched via API."
            )
 
        # Validate dot_number
        if DOT_COLUMN in df.columns:
            bad_dot = ~pd.to_numeric(df[DOT_COLUMN], errors="coerce").notna()
            if bad_dot.any():
                row_fail_mask |= bad_dot
                errors.append(
                    f"{bad_dot.sum()} rows have invalid DOT numbers "
                    f"(must be numeric)"
                )
 
        # Validate violation counts are non-negative
        viol_cols = [
            "oos_total", "driver_oos_total", "vehicle_oos_total",
            "basic_viol", "unsafe_viol", "vh_maint_viol",
        ]
        for col in viol_cols:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                bad = numeric < 0
                if bad.any():
                    row_fail_mask |= bad
                    warnings.append(
                        f"Column '{col}': {bad.sum()} negative values "
                        f"will be set to 0"
                    )
 
        # Validate crash count
        if "crash_count" in df.columns:
            bad_crash = pd.to_numeric(df["crash_count"], errors="coerce") < 0
            if bad_crash.any():
                row_fail_mask |= bad_crash
                warnings.append(
                    f"crash_count: {bad_crash.sum()} negative values "
                    f"will be set to 0"
                )
 
    else:
        # Legacy schema validation
        missing = find_missing_required_cols(df, schema="legacy")
        if missing:
            errors.append(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )
 
        if "weight_lbs" in df.columns:
            weight = pd.to_numeric(df["weight_lbs"], errors="coerce")
            bad_weight = ~weight.between(0, 200_000)
            if bad_weight.any():
                row_fail_mask |= bad_weight
                errors.append(
                    f"weight_lbs: {bad_weight.sum()} values out of range (0–200,000)"
                )
 
        if "miles" in df.columns:
            miles = pd.to_numeric(df["miles"], errors="coerce")
            bad_miles = ~miles.between(0, 5_000)
            if bad_miles.any():
                row_fail_mask |= bad_miles
                errors.append(
                    f"miles: {bad_miles.sum()} values out of range (0–5,000)"
                )
 
    # General checks for all schemas
    if len(df) < 1:
        errors.append("File contains no rows after cleaning")
 
    if len(df) > 100_000:
        warnings.append(
            f"File contains {len(df):,} rows — large files may take "
            f"several minutes to score"
        )
 
    return errors, warnings, row_fail_mask
 
 
# ══════════════════════════════════════════════════════════════════
# Legacy Schema Conversion
# ══════════════════════════════════════════════════════════════════
 
def convert_legacy_to_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a legacy schema DataFrame to PACE schema.
 
    Maps old columns to their closest PACE equivalents using heuristics.
    Missing PACE columns are filled with defaults.
    """
    df = df.copy()
    pace_df = pd.DataFrame(index=df.index)
 
    # Direct mappings where we have reasonable proxies
    if "carrier" in df.columns:
        pace_df["carrier_status_code"]       = "A"
        pace_df["carrier_carrier_operation"] = "C"
        pace_df["carrier_safety_rating"]     = "UNKNOWN"
        pace_df["carrier_phy_state"]         = df.get("origin_state", "UNKNOWN")
        pace_df["carrier_phy_country"]       = "US"
        pace_df["carrier_hm_ind"]            = "N"
        pace_df["carrier_fleetsize"]         = "C"
 
    if "weight_lbs" in df.columns:
        weight = pd.to_numeric(df["weight_lbs"], errors="coerce").fillna(0)
        # Approximate truck/power units from weight
        pace_df["carrier_power_units"]    = np.ceil(weight / 44000).astype(int).clip(1, 100)
        pace_df["carrier_truck_units"]    = pace_df["carrier_power_units"]
        pace_df["carrier_total_drivers"]  = pace_df["carrier_power_units"]
        pace_df["carrier_total_cdl"]      = pace_df["carrier_power_units"]
 
    if "miles" in df.columns:
        miles = pd.to_numeric(df["miles"], errors="coerce").fillna(0)
        pace_df["carrier_mcs150_mileage"] = (miles * 52).astype(int)  # annualize
        pace_df["sms_recent_mileage"]     = pace_df["carrier_mcs150_mileage"]
 
    if "accessorial_charge_usd" in df.columns:
        acc = pd.to_numeric(
            df["accessorial_charge_usd"].astype(str)
            .str.replace(r"[$,\s]", "", regex=True),
            errors="coerce"
        ).fillna(0)
        # Infer OOS/violations from accessorial charge presence
        has_charge = (acc > 0).astype(int)
        pace_df["oos_total"]         = has_charge
        pace_df["basic_viol"]        = has_charge
        pace_df["vh_maint_viol"]     = 0
        pace_df["unsafe_viol"]       = 0
        pace_df["fatigued_viol"]     = 0
        pace_df["dr_fitness_viol"]   = 0
        pace_df["subt_alcohol_viol"] = 0
        pace_df["hm_viol"]           = 0
        pace_df["driver_oos_total"]  = 0
        pace_df["vehicle_oos_total"] = 0
        pace_df["hazmat_oos_total"]  = 0
    else:
        for col in ["oos_total", "driver_oos_total", "vehicle_oos_total",
                    "hazmat_oos_total", "basic_viol", "unsafe_viol",
                    "fatigued_viol", "dr_fitness_viol", "subt_alcohol_viol",
                    "vh_maint_viol", "hm_viol"]:
            pace_df[col] = 0
 
    # Crash defaults
    for col in ["crash_count", "crash_fatalities_total", "crash_injuries_total",
                "crash_towaway_total", "crash_hazmat_releases"]:
        pace_df[col] = 0
    pace_df["crash_avg_severity"] = 0.0
 
    # Date features
    if "ship_date" in df.columns:
        dates = pd.to_datetime(df["ship_date"], errors="coerce")
        pace_df["insp_year"]  = dates.dt.year.fillna(2024).astype(int)
        pace_df["insp_month"] = dates.dt.month.fillna(1).astype(int)
        pace_df["insp_day"]   = dates.dt.day.fillna(1).astype(int)
        pace_df["insp_dow"]   = dates.dt.dayofweek.fillna(0).astype(int)
    else:
        pace_df["insp_year"]  = 2024
        pace_df["insp_month"] = 1
        pace_df["insp_day"]   = 1
        pace_df["insp_dow"]   = 0
 
    pace_df["is_holiday"]      = 0
    pace_df["is_near_holiday"] = 0
 
    # Fill all remaining PACE columns with defaults
    for col in CONTINUOUS_COLUMNS:
        if col not in pace_df.columns:
            pace_df[col] = 0.0
 
    for col in CATEGORICAL_COLUMNS:
        if col not in pace_df.columns:
            pace_df[col] = "UNKNOWN"
 
    # Carry over ID columns
    if "shipment_id" in df.columns:
        pace_df["unique_id"] = df["shipment_id"]
    if "dot_number" in df.columns:
        pace_df["dot_number"] = df["dot_number"]
 
    return pace_df
 
 
# ══════════════════════════════════════════════════════════════════
# Main Entry Points
# ══════════════════════════════════════════════════════════════════
 
class PACEDataPipeline:
    """
    Main data pipeline class for PACE.
 
    Usage:
        pipeline = PACEDataPipeline()
 
        # CSV upload
        result = pipeline.process_csv(df)
 
        # Manual input
        result = pipeline.process_manual(user_inputs)
 
        # DOT lookup (returns feature dict directly)
        features = pipeline.process_dot(dot_number)
    """
 
    def process_csv(self, df: pd.DataFrame) -> Dict:
        """
        Full processing pipeline for CSV uploads.
 
        Returns dict with:
            schema      — detected schema type
            df_clean    — cleaned, model-ready DataFrame
            errors      — blocking errors
            warnings    — non-blocking warnings
            row_fail_mask — per-row error flags
            mapping     — column aliases that were applied
            ready       — bool, True if safe to run inference
        """
        # Step 1: Normalize column names and detect aliases
        df, mapping = normalize_column_names(df)
 
        # Step 2: Detect schema
        schema = detect_schema(df)
 
        # Step 3: Convert legacy → PACE if needed
        if schema == "legacy":
            df = convert_legacy_to_pace(df)
            schema = "pace_converted"
 
        # Step 4: Validate
        errors, warnings, row_fail_mask = validate_dataframe(df, schema)
 
        # Step 5: Clean
        df_clean, clean_warnings = clean_dataframe(df)
        warnings.extend(clean_warnings)
 
        return {
            "schema":        schema,
            "df_clean":      df_clean,
            "errors":        errors,
            "warnings":      warnings,
            "row_fail_mask": row_fail_mask,
            "mapping":       mapping,
            "ready":         len(errors) == 0,
            "row_count":     len(df_clean),
            "pass_count":    int((~row_fail_mask).sum()) if row_fail_mask is not None else len(df_clean),
            "fail_count":    int(row_fail_mask.sum()) if row_fail_mask is not None else 0,
        }
 
    def process_manual(self, user_inputs: Dict) -> Dict:
        """
        Process manually entered shipment/carrier data.
 
        Fills missing fields with defaults and validates.
        Returns cleaned feature dict ready for inference.
        """
        # Fill all missing fields with defaults
        features = dict(FEATURE_DEFAULTS)
        features.update({k: v for k, v in user_inputs.items() if v is not None})
 
        # Normalize aliases
        normalized = {}
        for k, v in features.items():
            canonical = COLUMN_ALIASES.get(k.lower().strip(), k)
            normalized[canonical] = v
 
        # Fix booleans
        for col in BOOL_COLS:
            if col in normalized:
                val = str(normalized[col]).strip().upper()
                normalized[col] = "Y" if val in ("TRUE", "YES", "1", "Y") else "N"
 
        # Coerce numerics
        for col in CONTINUOUS_COLUMNS:
            if col in normalized:
                try:
                    normalized[col] = float(
                        str(normalized[col]).replace("$", "").replace(",", "")
                    )
                except (ValueError, TypeError):
                    normalized[col] = 0.0
            else:
                normalized[col] = 0.0
 
        # Fill missing categoricals
        for col in CATEGORICAL_COLUMNS:
            if col not in normalized or normalized[col] is None:
                normalized[col] = "UNKNOWN"
 
        return normalized
 
    def process_dot(self, dot_number: int) -> Dict:
        """
        Process a DOT number lookup.
        Returns a minimal feature dict with DOT number set.
        On production, this gets enriched by api_integration.
        """
        features = dict(FEATURE_DEFAULTS)
        features["dot_number"] = int(dot_number)
        return features
 
    def get_column_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a column mapping report for display in the UI.
        Shows which columns were found, mapped, or are missing.
        """
        df_norm, mapping = normalize_column_names(df.copy())
        schema = detect_schema(df_norm)
 
        pace_found   = [c for c in PACE_SCHEMA_COLS if c in df_norm.columns]
        pace_missing = [c for c in PACE_REQUIRED_COLS if c not in df_norm.columns]
        aliased      = [(orig, canon) for orig, canon in mapping.items()]
 
        return {
            "schema":        schema,
            "total_cols":    len(df.columns),
            "pace_found":    pace_found,
            "pace_missing":  pace_missing,
            "aliased":       aliased,
            "coverage_pct":  round(len(pace_found) / len(PACE_SCHEMA_COLS) * 100, 1),
        }
 
 
# ── Singleton ─────────────────────────────────────────────────────
_pipeline: Optional[PACEDataPipeline] = None
 
def get_data_pipeline() -> PACEDataPipeline:
    """Return data pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PACEDataPipeline()
    return _pipeline
