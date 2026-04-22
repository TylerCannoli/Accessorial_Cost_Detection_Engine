"""Merge OSRM road distance/duration onto enriched LTL training data.

Reads:
  - od_zip_pairs_osrm.csv  (Shipper Zip, Consignee Zip, osrm_miles, osrm_minutes)
  - enriched_ltl_training.csv

Writes:
  - enriched_ltl_training.csv  (in place; adds osrm_miles, osrm_minutes, osrm_imputed)

Fallback: rows with NaN osrm_miles get haversine_miles * 1.175 (CONUS median detour factor
observed on this batch) and osrm_imputed = 1. Rows without haversine_miles keep NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
OSRM_CSV = OUT_DIR / "od_zip_pairs_osrm.csv"
TRAIN_CSV = OUT_DIR / "enriched_ltl_training.csv"

DETOUR_FACTOR = 1.175
AVG_SPEED_MPH = 45.0


def main() -> None:
    osrm = pd.read_csv(OSRM_CSV, dtype={"Shipper Zip": str, "Consignee Zip": str})
    osrm["Shipper Zip"] = osrm["Shipper Zip"].str.zfill(5)
    osrm["Consignee Zip"] = osrm["Consignee Zip"].str.zfill(5)

    resolved = osrm["osrm_miles"].notna().sum()
    total = len(osrm)
    print(f"OSRM pairs: {resolved}/{total} resolved ({resolved/total*100:.1f}%)")

    keep = ["Shipper Zip", "Consignee Zip", "osrm_miles", "osrm_minutes"]
    if "haversine_miles" in osrm.columns:
        keep.append("haversine_miles")
    osrm_keys = osrm[keep].drop_duplicates(subset=["Shipper Zip", "Consignee Zip"])

    train = pd.read_csv(TRAIN_CSV, dtype={"Shipper Zip": str, "Consignee Zip": str})
    train["Shipper Zip"] = train["Shipper Zip"].str.zfill(5)
    train["Consignee Zip"] = train["Consignee Zip"].str.zfill(5)

    for col in ("osrm_miles", "osrm_minutes", "osrm_imputed"):
        if col in train.columns:
            train = train.drop(columns=col)

    merged = train.merge(
        osrm_keys[["Shipper Zip", "Consignee Zip", "osrm_miles", "osrm_minutes"]],
        on=["Shipper Zip", "Consignee Zip"], how="left",
    )

    hit = merged["osrm_miles"].notna().sum()
    miss = merged["osrm_miles"].isna().sum()
    print(f"Training rows: {len(merged)}, OSRM hit: {hit} ({hit/len(merged)*100:.2f}%), miss: {miss}")

    merged["osrm_imputed"] = 0
    miss_mask = merged["osrm_miles"].isna()
    if "haversine_miles" in merged.columns:
        hav_fallback = miss_mask & merged["haversine_miles"].notna()
        merged.loc[hav_fallback, "osrm_miles"] = merged.loc[hav_fallback, "haversine_miles"] * DETOUR_FACTOR
        merged.loc[hav_fallback, "osrm_minutes"] = (
            merged.loc[hav_fallback, "osrm_miles"] / AVG_SPEED_MPH * 60.0
        )
        merged.loc[hav_fallback, "osrm_imputed"] = 1
        print(f"Imputed from haversine: {hav_fallback.sum()}")

    final_miss = merged["osrm_miles"].isna().sum()
    print(f"Final missing osrm_miles: {final_miss} ({final_miss/len(merged)*100:.2f}%)")

    merged.to_csv(TRAIN_CSV, index=False)
    print(f"Wrote {TRAIN_CSV}")

    real = merged[merged["osrm_imputed"] == 0]["osrm_miles"]
    if len(real):
        print(f"OSRM miles (real): min={real.min():.1f}, median={real.median():.1f}, "
              f"max={real.max():.1f}, mean={real.mean():.1f}")


if __name__ == "__main__":
    main()
