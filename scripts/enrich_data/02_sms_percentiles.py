"""
Enrichment Script 02 — FMCSA SMS BASIC Percentile Scores
=========================================================
Source : SMS_AB_PassProperty_20260316.csv  (713 K carriers)
Output : outputs/enrichment/sms_percentile_features.csv

WHY THIS MATTERS
----------------
The SMS BASIC system is FMCSA's own carrier risk ranking.  Each carrier receives
a 0-100 percentile score in six safety categories (Unsafe Driving, HOS Compliance,
Driver Fitness, Controlled Substances, Vehicle Maintenance, Hazmat).  Carriers above
the alert threshold (AC = 'Y') are considered high-risk by FMCSA and are targeted
for intervention.

These percentile scores are:
  • Already normalised (0-100 scale)
  • More informative than raw violation counts (they weight by exposure/mileage)
  • What real shippers/brokers check before tendering a load
  • Available at booking time (not inspection outcome)

OUTPUT FEATURES (joined on dot_number)
---------------------------------------
sms_insp_total                 Total inspections on record in SMS
sms_driver_insp_total          Driver inspections
sms_driver_oos_insp_total      Driver OOS inspections
sms_vehicle_insp_total         Vehicle inspections
sms_vehicle_oos_insp_total     Vehicle OOS inspections
sms_unsafe_driv_measure        Unsafe Driving BASIC percentile (0-100)
sms_unsafe_driv_alert          Alert condition for Unsafe Driving (0/1)
sms_hos_measure                HOS Compliance BASIC percentile
sms_hos_alert                  Alert condition for HOS
sms_driv_fit_measure           Driver Fitness BASIC percentile
sms_driv_fit_alert             Alert condition
sms_contr_subst_measure        Controlled Substances BASIC percentile
sms_contr_subst_alert          Alert condition
sms_veh_maint_measure          Vehicle Maintenance BASIC percentile
sms_veh_maint_alert            Alert condition
sms_alert_count                Total number of alert conditions triggered (0-5)
sms_max_measure                Max BASIC percentile across all categories
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

PACE_DATA = os.getenv(
    "PACE_DATA_DIR",
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
SMS_FILE = os.path.join(PACE_DATA, "SMS_AB_PassProperty_20260316.csv")
OUT_DIR  = Path(__file__).parents[2] / "outputs" / "enrichment"
OUT_FILE = OUT_DIR / "sms_percentile_features.csv"

# Alert condition columns → binary
ALERT_COLS = {
    "UNSAFE_DRIV_AC": "sms_unsafe_driv_alert",
    "HOS_DRIV_AC":    "sms_hos_alert",
    "DRIV_FIT_AC":    "sms_driv_fit_alert",
    "CONTR_SUBST_AC": "sms_contr_subst_alert",
    "VEH_MAINT_AC":   "sms_veh_maint_alert",
}

MEASURE_MAP = {
    "UNSAFE_DRIV_MEASURE": "sms_unsafe_driv_measure",
    "HOS_DRIV_MEASURE":    "sms_hos_measure",
    "DRIV_FIT_MEASURE":    "sms_driv_fit_measure",
    "CONTR_SUBST_MEASURE": "sms_contr_subst_measure",
    "VEH_MAINT_MEASURE":   "sms_veh_maint_measure",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {SMS_FILE} ...")
    df = pd.read_csv(SMS_FILE, dtype={"DOT_NUMBER": str})
    print(f"  Loaded {len(df):,} carriers")

    # Rename to output schema
    out = pd.DataFrame()
    out["dot_number"] = df["DOT_NUMBER"].str.strip()

    # Inspection counts
    for src, dst in [
        ("INSP_TOTAL",             "sms_insp_total"),
        ("DRIVER_INSP_TOTAL",      "sms_driver_insp_total"),
        ("DRIVER_OOS_INSP_TOTAL",  "sms_driver_oos_insp_total"),
        ("VEHICLE_INSP_TOTAL",     "sms_vehicle_insp_total"),
        ("VEHICLE_OOS_INSP_TOTAL", "sms_vehicle_oos_insp_total"),
    ]:
        out[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0).astype(int)

    # Measure percentiles (0-100)
    for src, dst in MEASURE_MAP.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0).round(2)

    # Alert conditions → binary 0/1
    for src, dst in ALERT_COLS.items():
        out[dst] = (df[src].astype(str).str.strip().str.upper() == "Y").astype(int)

    # Derived: total alert count and max measure
    alert_cols_list = list(ALERT_COLS.values())
    measure_cols    = list(MEASURE_MAP.values())
    out["sms_alert_count"] = out[alert_cols_list].sum(axis=1).astype(int)
    out["sms_max_measure"] = out[measure_cols].max(axis=1).round(2)

    print(f"\nOutput shape: {out.shape}")
    print("\nAlert condition rates:")
    for c in alert_cols_list:
        print(f"  {c}: {out[c].mean()*100:.1f}% of carriers")
    print(f"\nCarriers with ANY alert: {(out['sms_alert_count'] > 0).mean()*100:.1f}%")
    print(f"Carriers with 2+ alerts:  {(out['sms_alert_count'] > 1).mean()*100:.1f}%")

    out.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
