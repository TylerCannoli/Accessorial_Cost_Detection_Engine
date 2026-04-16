"""
pipeline/ingest/carrier_schema.py
==================================
Carrier-agnostic schema normalization layer.

Each carrier submits data in their own format (different column names,
date formats, status codes, units). This module maps any carrier's raw
data to the PACE feature schema before inference.

USAGE
-----
    from pipeline.ingest.carrier_schema import CarrierSchema, normalize_carrier_data

    # Load a carrier's schema config and normalize their data
    schema = CarrierSchema.from_yaml("configs/carriers/acme_logistics.yaml")
    normalized_df = schema.normalize(raw_df)

    # Or use the convenience function (auto-detects schema by carrier_id)
    normalized_df = normalize_carrier_data(raw_df, carrier_id="acme_logistics")

CARRIER CONFIG FORMAT
---------------------
See configs/carriers/template.yaml for the full config spec.
At minimum, a carrier config needs:
    - carrier_id: unique string identifier
    - field_map: dict mapping their column names to PACE column names
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

# Default config directory — override with PACE_CARRIER_CONFIGS env var
CONFIG_DIR = Path(
    os.getenv("PACE_CARRIER_CONFIGS", Path(__file__).parents[2] / "configs" / "carriers")
)

# Categorical columns that should default to "UNKNOWN" when missing
_CAT_DEFAULTS = {
    "report_state":              "UNKNOWN",
    "county_code_state":         "UNKNOWN",
    "carrier_status_code":       "A",
    "carrier_carrier_operation": "C",
    "carrier_fleetsize":         "UNKNOWN",
    "carrier_hm_ind":            "N",
    "carrier_phy_state":         "UNKNOWN",
    "carrier_phy_country":       "US",
    "carrier_safety_rating":     "UNKNOWN",
    "carrier_crgo_genfreight":   "N",
    "carrier_crgo_household":    "N",
    "carrier_crgo_produce":      "N",
    "carrier_crgo_intermodal":   "N",
    "carrier_crgo_oilfield":     "N",
    "carrier_crgo_meat":         "N",
    "carrier_crgo_chem":         "N",
    "carrier_crgo_drybulk":      "N",
    "carrier_crgo_coldfood":     "N",
    "carrier_crgo_beverages":    "N",
    "carrier_crgo_construct":    "N",
    "sms_carrier_operation":     "UNKNOWN",
    "sms_hm_flag":               "N",
    "sms_pc_flag":               "N",
    "sms_phy_state":             "UNKNOWN",
    "sms_phy_country":           "US",
    "sms_private_only":          "N",
    "sms_authorized_for_hire":   "N",
    "sms_exempt_for_hire":       "N",
    "sms_private_property":      "N",
    "hazmat_placard_req":        "N",
    "unit_type_desc":            "UNKNOWN",
    "unit_make":                 "UNKNOWN",
    "unit_license_state":        "UNKNOWN",
}

# US state abbreviation normalization — handles full names, FIPS codes, mixed case
_STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "12": "FL", "13": "GA",
    "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO",
    "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ",
    "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC",
    "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT",
    "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
    "11": "DC", "72": "PR",
}

_STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC",
}


def _normalize_state(val: Any) -> str:
    """Normalize any state representation to 2-letter abbreviation."""
    if pd.isna(val):
        return "UNKNOWN"
    s = str(val).strip().upper()
    if len(s) == 2 and s.isalpha():
        return s
    if s.isdigit():
        return _STATE_FIPS.get(s.zfill(2), "UNKNOWN")
    return _STATE_NAMES.get(s, "UNKNOWN")


def _parse_date(val: Any, fmt: Optional[str] = None) -> Optional[datetime]:
    """Parse a date value into a datetime, trying common formats."""
    if pd.isna(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val)
    s = str(val).strip()
    formats_to_try = []
    if fmt:
        formats_to_try.append(fmt)
    formats_to_try += [
        "%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y",
        "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%y",
    ]
    for f in formats_to_try:
        try:
            return datetime.strptime(s, f)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(s, infer_datetime_format=True)
    except Exception:
        return None


class CarrierSchema:
    """
    Defines the mapping from a carrier's raw data format to the PACE feature schema.

    A schema config (YAML) specifies:
        carrier_id    — unique name for this carrier
        field_map     — dict of {their_column: pace_column}
        value_maps    — optional dict of {pace_column: {their_value: pace_value}}
        date_column   — which column holds the shipment/inspection date
        date_format   — strptime format string for their dates
        state_column  — which column holds the US state
        dot_column    — which column holds the DOT number (optional)
        defaults      — optional dict of {pace_column: default_value}
    """

    def __init__(self, config: Dict):
        self.carrier_id   = config.get("carrier_id", "unknown")
        self.field_map    = config.get("field_map", {})
        self.value_maps   = config.get("value_maps", {})
        self.date_column  = config.get("date_column")
        self.date_format  = config.get("date_format")
        self.state_column = config.get("state_column")
        self.dot_column   = config.get("dot_column", "dot_number")
        self.defaults     = config.get("defaults", {})

    @classmethod
    def from_yaml(cls, path: str) -> "CarrierSchema":
        """Load a CarrierSchema from a YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_carrier_id(cls, carrier_id: str) -> "CarrierSchema":
        """Load a CarrierSchema by carrier_id from the default config directory."""
        path = CONFIG_DIR / f"{carrier_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"No schema config found for carrier '{carrier_id}' at {path}. "
                f"Create {path} using configs/carriers/template.yaml as a guide."
            )
        return cls.from_yaml(str(path))

    @classmethod
    def passthrough(cls) -> "CarrierSchema":
        """
        Returns a no-op schema that assumes data is already in PACE format.
        Use this when data comes pre-normalized (e.g., from the FMCSA pipeline).
        """
        return cls({"carrier_id": "passthrough", "field_map": {}})

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a carrier's raw DataFrame into the PACE feature schema.

        Steps:
            1. Rename columns via field_map
            2. Apply value_maps (recode categorical values)
            3. Normalize state columns to 2-letter abbreviations
            4. Parse date column into insp_year, insp_month, insp_dow, insp_day
            5. Apply carrier-specific defaults
            6. Apply global categorical defaults for missing columns
        """
        df = df.copy()

        # Step 1 — rename columns
        df = df.rename(columns=self.field_map)

        # Step 2 — recode categorical values
        for pace_col, mapping in self.value_maps.items():
            if pace_col in df.columns:
                df[pace_col] = df[pace_col].astype(str).str.strip().map(
                    lambda v, m=mapping: m.get(v, m.get(v.upper(), v))
                )

        # Step 3 — normalize state columns
        state_cols = ["report_state", "carrier_phy_state", "sms_phy_state",
                      "unit_license_state"]
        if self.state_column and self.state_column in df.columns:
            if "report_state" not in df.columns:
                df["report_state"] = df[self.state_column]
        for col in state_cols:
            if col in df.columns:
                df[col] = df[col].apply(_normalize_state)

        # Step 4 — parse date into temporal features
        date_src = self.date_column or "insp_date"
        if date_src in df.columns:
            parsed = df[date_src].apply(lambda v: _parse_date(v, self.date_format))
            now = datetime.now()
            df["insp_year"]  = parsed.apply(lambda d: d.year  if d else now.year)
            df["insp_month"] = parsed.apply(lambda d: d.month if d else now.month)
            df["insp_dow"]   = parsed.apply(lambda d: d.weekday() if d else now.weekday())
            df["insp_day"]   = parsed.apply(lambda d: d.day    if d else now.day)
            df = df.drop(columns=[date_src], errors="ignore")
        elif "insp_year" not in df.columns:
            # No date at all — use current month as context
            now = datetime.now()
            df["insp_year"]  = now.year
            df["insp_month"] = now.month
            df["insp_dow"]   = now.weekday()
            df["insp_day"]   = now.day

        # Step 5 — carrier-specific defaults
        for col, val in self.defaults.items():
            if col not in df.columns:
                df[col] = val

        # Step 6 — global categorical defaults for any missing cat columns
        for col, default in _CAT_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        return df

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Check what PACE columns are present after normalization.
        Returns a report dict with 'present', 'missing', and 'coverage_pct'.
        """
        from pipeline.config import CONTINUOUS_COLUMNS_V2, CATEGORICAL_COLUMNS
        all_expected = set(CONTINUOUS_COLUMNS_V2) | set(CATEGORICAL_COLUMNS)
        normalized   = self.normalize(df)
        present  = [c for c in all_expected if c in normalized.columns]
        missing  = [c for c in all_expected if c not in normalized.columns]
        return {
            "carrier_id":    self.carrier_id,
            "rows":          len(df),
            "present":       sorted(present),
            "missing":       sorted(missing),
            "coverage_pct":  round(len(present) / len(all_expected) * 100, 1),
        }


def normalize_carrier_data(
    df: pd.DataFrame,
    carrier_id: Optional[str] = None,
    schema: Optional[CarrierSchema] = None,
) -> pd.DataFrame:
    """
    Convenience function — normalize a DataFrame using a carrier schema.

    Args:
        df:         Raw carrier DataFrame
        carrier_id: Carrier identifier (loads schema from configs/carriers/<id>.yaml)
        schema:     Pre-loaded CarrierSchema instance (alternative to carrier_id)

    Returns:
        Normalized DataFrame ready for PACEInference.predict()
    """
    if schema is None:
        if carrier_id:
            schema = CarrierSchema.from_carrier_id(carrier_id)
        else:
            schema = CarrierSchema.passthrough()
    return schema.normalize(df)
