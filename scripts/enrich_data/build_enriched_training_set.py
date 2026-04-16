"""
Master Enrichment Builder — build_enriched_training_set.py
===========================================================
Merges all enrichment feature files with ctgan_input.csv to produce
enriched_ctgan_input.csv — the Phase 2 training input.

USAGE
-----
    # Step 1: Run all enrichment scripts first
    python 01_carrier_oos_history.py
    python 02_sms_percentiles.py
    python 03_parking_scarcity.py
    python 04_fars_crash_rates.py
    python 05_atri_detention_lookup.py
    python 06_freight_rate_proxy.py
    python 08_faf_freight_flows.py

    # Step 2: Build the merged dataset
    python build_enriched_training_set.py

    # Optional: Only merge available enrichments (skip missing files)
    python build_enriched_training_set.py --permissive

OUTPUT
------
outputs/enriched_ctgan_input.csv   — drop-in replacement for ctgan_input.csv
outputs/enrichment_report.txt      — summary of what was added / coverage rates

NEW FEATURES ADDED (relative to original ctgan_input)
------------------------------------------------------
From carrier OOS history (01):
  carrier_hist_insp_count, carrier_hist_oos_rate, carrier_hist_driver_oos_rate,
  carrier_hist_vehicle_oos_rate, carrier_hist_hazmat_oos_rate,
  carrier_hist_avg_viol_per_insp, carrier_hist_driver_viol_rate,
  carrier_hist_vehicle_viol_rate, carrier_hist_hazmat_viol_rate,
  carrier_hist_post_acc_rate

From SMS percentiles (02):
  sms_insp_total, sms_driver_insp_total, sms_driver_oos_insp_total,
  sms_vehicle_insp_total, sms_vehicle_oos_insp_total,
  sms_unsafe_driv_measure, sms_unsafe_driv_alert, sms_hos_measure, sms_hos_alert,
  sms_driv_fit_measure, sms_driv_fit_alert, sms_contr_subst_measure, sms_contr_subst_alert,
  sms_veh_maint_measure, sms_veh_maint_alert, sms_alert_count, sms_max_measure

From parking scarcity (03):
  state_parking_spots_total, state_parking_spots_per_100mi,
  state_parking_scarcity_index

From FARS crash rates (04):
  state_fatal_crashes_2023, state_fatal_fatalities_2023,
  state_avg_fatals_per_crash, state_crashes_involving_trucks,
  state_fatal_crash_index

From ATRI detention (05):
  state_atri_detention_propensity, state_atri_severity_index

From freight rates (06):
  lmi_composite_index, lmi_transportation_capacity, lmi_transportation_utilization,
  lmi_transportation_prices, ftsi_index, ppi_trucking_mom_pct, ftsi_yoy_pct,
  freight_tightness_score

From FAF5 flows (08):
  faf_state_freight_intensity, faf_state_hazmat_share, faf_state_reefer_share_faf
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PACE_DATA = os.getenv(
    "PACE_DATA_DIR",
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
REPO_ROOT   = Path(__file__).parents[2]
ENRICH_DIR  = REPO_ROOT / "outputs" / "enrichment"
CTGAN_IN    = os.path.join(PACE_DATA, "ctgan_input", "ctgan_input.csv")
OUT_CSV     = REPO_ROOT / "outputs" / "enriched_ctgan_input.csv"
REPORT_FILE = REPO_ROOT / "outputs" / "enrichment_report.txt"


# ── Enrichment file registry ───────────────────────────────────────────
ENRICHMENTS = [
    {
        "name":      "Carrier OOS History",
        "file":      ENRICH_DIR / "carrier_oos_history.csv",
        "join_key":  "dot_number",
        "join_how":  "left",
        "required":  False,
        "fill_zero": [
            "carrier_hist_insp_count", "carrier_hist_oos_rate",
            "carrier_hist_driver_oos_rate", "carrier_hist_vehicle_oos_rate",
            "carrier_hist_hazmat_oos_rate", "carrier_hist_avg_viol_per_insp",
            "carrier_hist_driver_viol_rate", "carrier_hist_vehicle_viol_rate",
            "carrier_hist_hazmat_viol_rate", "carrier_hist_post_acc_rate",
        ],
    },
    {
        "name":      "SMS BASIC Percentiles",
        "file":      ENRICH_DIR / "sms_percentile_features.csv",
        "join_key":  "dot_number",
        "join_how":  "left",
        "required":  False,
        "fill_zero": [
            "sms_insp_total", "sms_driver_insp_total", "sms_driver_oos_insp_total",
            "sms_vehicle_insp_total", "sms_vehicle_oos_insp_total",
            "sms_unsafe_driv_measure", "sms_unsafe_driv_alert",
            "sms_hos_measure", "sms_hos_alert",
            "sms_driv_fit_measure", "sms_driv_fit_alert",
            "sms_contr_subst_measure", "sms_contr_subst_alert",
            "sms_veh_maint_measure", "sms_veh_maint_alert",
            "sms_alert_count", "sms_max_measure",
        ],
    },
    {
        "name":      "Parking Scarcity (state level)",
        "file":      ENRICH_DIR / "state_parking_features.csv",
        "join_key":  "report_state",          # join on report_state in base df
        "enrich_key":"state_abbr",            # column name in enrichment file
        "join_how":  "left",
        "required":  False,
        "fill_zero": [
            "state_parking_spots_total", "state_parking_spots_per_100mi",
            "state_parking_scarcity_index",
        ],
    },
    {
        "name":      "FARS Fatal Crash Rates",
        "file":      ENRICH_DIR / "state_fatal_crash_features.csv",
        "join_key":  "report_state",
        "enrich_key":"state_abbr",
        "join_how":  "left",
        "required":  False,
        "fill_zero": [
            "state_fatal_crashes_2023", "state_fatal_fatalities_2023",
            "state_avg_fatals_per_crash", "state_crashes_involving_trucks",
            "state_fatal_crash_index",
        ],
    },
    {
        "name":      "ATRI Detention Propensity",
        "file":      ENRICH_DIR / "atri_detention_features.csv",
        "join_key":  "report_state",
        "enrich_key":"state_abbr",
        "join_how":  "left",
        "required":  False,
        "fill_zero": [
            "state_atri_detention_propensity", "state_atri_severity_index",
        ],
    },
    {
        "name":      "Freight Rate Proxies (LMI/FTSI/PPI)",
        "file":      ENRICH_DIR / "freight_rate_features.csv",
        "join_key":  ["insp_year", "insp_month"],
        "enrich_key":["year", "month"],
        "join_how":  "left",
        "required":  False,
        "fill_zero": [
            "lmi_composite_index", "lmi_transportation_capacity",
            "lmi_transportation_utilization", "lmi_transportation_prices",
            "ftsi_index", "ppi_trucking_mom_pct", "ftsi_yoy_pct",
            "freight_tightness_score",
        ],
        "fill_forward": True,   # forward-fill monthly time series
    },
    {
        "name":      "FAF5 Freight Flows",
        "file":      ENRICH_DIR / "faf_freight_features.csv",
        "join_key":  "report_state",
        "enrich_key":"state_abbr",
        "join_how":  "left",
        "required":  False,
        "fill_zero": [
            "faf_state_freight_intensity", "faf_state_hazmat_share",
            "faf_state_reefer_share_faf",
        ],
    },
]


def apply_enrichment(base: pd.DataFrame, spec: dict, permissive: bool) -> pd.DataFrame:
    """Merge one enrichment file into base dataframe."""
    fpath = spec["file"]
    name  = spec["name"]

    if not os.path.exists(fpath):
        msg = f"  [{name}] MISSING: {fpath}"
        if spec.get("required") and not permissive:
            raise FileNotFoundError(msg + "\n  Run the corresponding enrichment script first.")
        print(msg + " — SKIPPING")
        return base

    enrich = pd.read_csv(fpath)
    print(f"  [{name}] Loaded {len(enrich):,} rows from {os.path.basename(fpath)}")

    # Determine join keys
    base_key   = spec["join_key"]
    enrich_key = spec.get("enrich_key", base_key)

    # Rename enrichment key column(s) to match base if they differ
    if isinstance(base_key, list):
        rename_map = {e: b for e, b in zip(enrich_key, base_key) if e != b}
    else:
        rename_map = {enrich_key: base_key} if enrich_key != base_key else {}
    if rename_map:
        enrich = enrich.rename(columns=rename_map)

    # Drop any columns that already exist in base (except join keys)
    join_keys = [base_key] if isinstance(base_key, str) else base_key
    overlap   = [c for c in enrich.columns if c in base.columns and c not in join_keys]
    if overlap:
        print(f"    Dropping {len(overlap)} overlapping columns: {overlap[:5]}{'...' if len(overlap)>5 else ''}")
        enrich = enrich.drop(columns=overlap)

    before_rows = len(base)
    base = base.merge(enrich, on=join_keys, how=spec["join_how"])
    after_rows  = len(base)
    if after_rows != before_rows:
        print(f"    WARNING: row count changed {before_rows:,} → {after_rows:,} after merge")

    # Fill missing values with zero for numeric enrichment cols
    for col in spec.get("fill_zero", []):
        if col in base.columns:
            base[col] = base[col].fillna(0)

    # Forward-fill time-series enrichments
    if spec.get("fill_forward"):
        base = base.sort_values(["insp_year","insp_month"])
        for col in spec.get("fill_zero", []):
            if col in base.columns:
                base[col] = base[col].ffill().bfill()

    # Coverage report
    sample_col = spec.get("fill_zero", [None])[0]
    if sample_col and sample_col in base.columns:
        cov = (base[sample_col] > 0).mean() * 100
        print(f"    Coverage ({sample_col}): {cov:.1f}% of rows have non-zero value")

    return base


def main(permissive: bool = False):
    (REPO_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    # ── Load base ──────────────────────────────────────────────────────
    print(f"Loading ctgan_input from {CTGAN_IN} ...")
    base = pd.read_csv(CTGAN_IN, low_memory=False)
    print(f"  Base shape: {base.shape}")

    # Normalise dot_number type for joining
    base["dot_number"] = base["dot_number"].astype(str).str.strip()

    # ── Apply each enrichment ──────────────────────────────────────────
    report_lines = [f"Enrichment Report\nBase: {base.shape[0]:,} rows × {base.shape[1]} cols\n"]
    cols_before  = set(base.columns)

    for spec in ENRICHMENTS:
        print(f"\n{'='*60}")
        print(f"Applying: {spec['name']}")
        try:
            base = apply_enrichment(base, spec, permissive)
            added = [c for c in base.columns if c not in cols_before]
            report_lines.append(f"{spec['name']}: +{len(added)} columns — {added}")
            cols_before = set(base.columns)
        except Exception as e:
            print(f"  ERROR: {e}")
            if not permissive:
                raise
            report_lines.append(f"{spec['name']}: FAILED — {e}")

    # ── ATRI national constants as scalar columns ──────────────────────
    const_path = ENRICH_DIR / "atri_national_constants.json"
    if const_path.exists():
        with open(const_path) as f:
            constants = json.load(f)
        for k, v in constants.items():
            if isinstance(v, (int, float)):
                base[k] = v
        print(f"\nAdded {len([k for k,v in constants.items() if isinstance(v,(int,float))])} ATRI scalar constants")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Final shape: {base.shape}")
    new_cols = [c for c in base.columns if c not in cols_before or True]
    print(f"Total columns: {len(base.columns)} (was {len(pd.read_csv(CTGAN_IN, nrows=0).columns)})")

    # Save
    print(f"\nSaving enriched dataset → {OUT_CSV}")
    base.to_csv(OUT_CSV, index=False)
    size_mb = OUT_CSV.stat().st_size / 1e6
    print(f"  Saved {size_mb:.0f} MB, {len(base):,} rows × {len(base.columns)} columns")

    # Write report
    report_lines.insert(0, f"Final: {base.shape[0]:,} rows × {base.shape[1]} columns\n")
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport → {REPORT_FILE}")

    print("\nNext step: upload enriched_ctgan_input.csv to Teradata and retrain.")
    print("Update pipeline/config.py with NEW_CONTINUOUS_COLUMNS defined in config_v2.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all enrichments into training set")
    parser.add_argument("--permissive", action="store_true",
                        help="Skip missing enrichment files instead of failing")
    args = parser.parse_args()
    main(permissive=args.permissive)
