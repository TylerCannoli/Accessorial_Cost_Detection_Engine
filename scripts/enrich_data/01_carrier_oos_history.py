"""
Enrichment Script 01 — Carrier Historical OOS Rates
=====================================================
Source : Vehicle_Inspection_File_20260312_NOT_SMS.csv  (8.2 M rows, 2023-2026)
Output : outputs/enrichment/carrier_oos_history.csv   (one row per DOT_NUMBER)

WHY THIS MATTERS
----------------
The current ctgan_input uses per-inspection violation counts (basic_viol, oos_total,
etc.) which DIRECTLY DETERMINE the accessorial_type label via the SQL CASE statement.
That is 100% target leakage.

This script replaces those per-inspection counts with CARRIER-LEVEL HISTORICAL RATES
computed from the last 3 years of real FMCSA inspections.  At booking time a shipper
already knows a carrier's track record — they would never know the specific outcome of
a future inspection.  Historical rates are the correct predictors.

OUTPUT FEATURES (joined to ctgan_input on dot_number)
------------------------------------------------------
carrier_hist_insp_count         Total inspections in window
carrier_hist_oos_rate           OOS events / total inspections
carrier_hist_driver_oos_rate    Driver-OOS events / total inspections
carrier_hist_vehicle_oos_rate   Vehicle-OOS events / total inspections
carrier_hist_hazmat_oos_rate    Hazmat-OOS events / total inspections
carrier_hist_avg_viol_per_insp  Average total violations per inspection
carrier_hist_driver_viol_rate   Driver violations / total inspections
carrier_hist_vehicle_viol_rate  Vehicle violations / total inspections
carrier_hist_hazmat_viol_rate   Hazmat violations / total inspections
carrier_hist_post_acc_rate      Post-accident inspections / total
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
PACE_DATA = os.getenv(
    "PACE_DATA_DIR",
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
INSP_FILE = os.path.join(PACE_DATA, "Vehicle_Inspection_File_20260312_NOT_SMS.csv")
OUT_DIR   = Path(__file__).parents[2] / "outputs" / "enrichment"
OUT_FILE  = OUT_DIR / "carrier_oos_history.csv"

# ── Columns we actually need ───────────────────────────────────────────
USE_COLS = [
    "DOT_NUMBER", "INSP_DATE",
    "OOS_TOTAL", "DRIVER_OOS_TOTAL", "VEHICLE_OOS_TOTAL", "HAZMAT_OOS_TOTAL",
    "VIOL_TOTAL", "DRIVER_VIOL_TOTAL", "VEHICLE_VIOL_TOTAL", "HAZMAT_VIOL_TOTAL",
    "POST_ACC_IND",
]

CHUNK_SIZE = 500_000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {INSP_FILE} in chunks of {CHUNK_SIZE:,} ...")

    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(
            INSP_FILE,
            usecols=USE_COLS,
            dtype={"DOT_NUMBER": str, "POST_ACC_IND": str},
            chunksize=CHUNK_SIZE,
            encoding="latin1",
        )
    ):
        # Normalise types
        chunk["DOT_NUMBER"] = chunk["DOT_NUMBER"].str.strip()
        for col in ["OOS_TOTAL","DRIVER_OOS_TOTAL","VEHICLE_OOS_TOTAL","HAZMAT_OOS_TOTAL",
                    "VIOL_TOTAL","DRIVER_VIOL_TOTAL","VEHICLE_VIOL_TOTAL","HAZMAT_VIOL_TOTAL"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
        chunk["is_oos"]          = (chunk["OOS_TOTAL"] > 0).astype(int)
        chunk["is_driver_oos"]   = (chunk["DRIVER_OOS_TOTAL"] > 0).astype(int)
        chunk["is_vehicle_oos"]  = (chunk["VEHICLE_OOS_TOTAL"] > 0).astype(int)
        chunk["is_hazmat_oos"]   = (chunk["HAZMAT_OOS_TOTAL"] > 0).astype(int)
        chunk["is_post_acc"]     = (chunk["POST_ACC_IND"].str.strip().str.upper() == "Y").astype(int)
        chunk["insp_count"]      = 1
        chunks.append(chunk)
        print(f"  Chunk {i+1} loaded ({len(chunk):,} rows)")

    print("Concatenating all chunks ...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total rows: {len(df):,}  |  Unique carriers: {df['DOT_NUMBER'].nunique():,}")

    print("Aggregating by carrier ...")
    agg = df.groupby("DOT_NUMBER").agg(
        carrier_hist_insp_count       = ("insp_count",       "sum"),
        _total_oos                    = ("is_oos",           "sum"),
        _total_driver_oos             = ("is_driver_oos",    "sum"),
        _total_vehicle_oos            = ("is_vehicle_oos",   "sum"),
        _total_hazmat_oos             = ("is_hazmat_oos",    "sum"),
        _total_viol                   = ("VIOL_TOTAL",       "sum"),
        _total_driver_viol            = ("DRIVER_VIOL_TOTAL","sum"),
        _total_vehicle_viol           = ("VEHICLE_VIOL_TOTAL","sum"),
        _total_hazmat_viol            = ("HAZMAT_VIOL_TOTAL","sum"),
        _total_post_acc               = ("is_post_acc",      "sum"),
    ).reset_index()

    n = agg["carrier_hist_insp_count"].clip(lower=1)   # avoid /0
    agg["carrier_hist_oos_rate"]          = agg["_total_oos"]         / n
    agg["carrier_hist_driver_oos_rate"]   = agg["_total_driver_oos"]  / n
    agg["carrier_hist_vehicle_oos_rate"]  = agg["_total_vehicle_oos"] / n
    agg["carrier_hist_hazmat_oos_rate"]   = agg["_total_hazmat_oos"]  / n
    agg["carrier_hist_avg_viol_per_insp"] = agg["_total_viol"]        / n
    agg["carrier_hist_driver_viol_rate"]  = agg["_total_driver_viol"] / n
    agg["carrier_hist_vehicle_viol_rate"] = agg["_total_vehicle_viol"]/ n
    agg["carrier_hist_hazmat_viol_rate"]  = agg["_total_hazmat_viol"] / n
    agg["carrier_hist_post_acc_rate"]     = agg["_total_post_acc"]    / n

    # Drop raw totals — only keep rates + count
    drop_cols = [c for c in agg.columns if c.startswith("_total")]
    agg = agg.drop(columns=drop_cols)

    # Round rates to 4 dp
    rate_cols = [c for c in agg.columns if "rate" in c or "avg_viol" in c]
    agg[rate_cols] = agg[rate_cols].round(4)

    print(f"\nOutput shape: {agg.shape}")
    print(agg.describe().to_string())
    agg.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
