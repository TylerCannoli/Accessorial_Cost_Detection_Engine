"""
Enrichment Script 06 — Freight Rate & Market Condition Proxies
===============================================================
Sources (already in model):
  - FRED TSIFRGHT  (Freight Transportation Services Index)
  - FRED TRUCKD11  (Truck Tonnage Index)
  - FRED WPU057303 (PPI: Long-distance truckload freight)

Additional sources processed here:
  - market_data/lmi_timeseries.csv  (Logistics Managers Index — partial)
  - Freight Transportation Services Index.csv
  - Producer_Price_Index_by_Industry.csv

Note on DAT/Truckstop.com (commercial spot rates)
--------------------------------------------------
DAT One and Truckstop.com provide lane-level spot rate data ($/mile by
origin-destination pair) which is the gold standard for freight rate signals.
These are COMMERCIAL datasets requiring an API subscription.

As a free proxy, this script:
  1. Enriches the existing FRED freight indices with additional derived metrics
  2. Processes the partial LMI data that was successfully scraped
  3. Creates month-over-month rate change features (momentum signals)

When you obtain DAT or Truckstop API access, insert your key as:
    DAT_API_KEY = os.getenv("DAT_API_KEY", "")
and call: https://api.dat.com/freight-data/spot-rates

OUTPUT FEATURES (joined on insp_year + insp_month)
---------------------------------------------------
lmi_composite_index             Overall logistics conditions (50 = neutral)
lmi_transportation_capacity     Transport capacity sub-index
lmi_transportation_utilization  Utilization sub-index
lmi_transportation_prices       Pricing pressure index
ftsi_index                      Freight Transportation Services Index (BTS)
ppi_trucking_mom_pct            Month-over-month PPI trucking % change
ftsi_yoy_pct                    Year-over-year FTSI % change
freight_tightness_score         Composite: high = tight market (more detention)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

PACE_DATA = os.getenv(
    "PACE_DATA_DIR",
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
OUT_DIR  = Path(__file__).parents[2] / "outputs" / "enrichment"
OUT_FILE = OUT_DIR / "freight_rate_features.csv"


def load_lmi(pace_data: str) -> pd.DataFrame:
    path = os.path.join(pace_data, "market_data", "lmi_timeseries.csv")
    df   = pd.read_csv(path)
    df   = df[df["lmi_composite"].notna()].copy()
    df["year"]  = df["period"].str[:4].astype(int)
    df["month"] = df["period"].str[5:7].astype(int)
    return df[["year","month","lmi_composite",
               "transportation_capacity","transportation_utilization",
               "transportation_prices"]].rename(columns={
        "lmi_composite":            "lmi_composite_index",
        "transportation_capacity":   "lmi_transportation_capacity",
        "transportation_utilization":"lmi_transportation_utilization",
        "transportation_prices":     "lmi_transportation_prices",
    })


def load_ftsi(pace_data: str) -> pd.DataFrame:
    path = os.path.join(pace_data, "Freight Transportation Services Index.csv")
    df   = pd.read_csv(path)
    df["date"]  = pd.to_datetime(df["observation_date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df          = df.sort_values(["year","month"]).copy()
    df["ftsi_yoy_pct"] = df["TSIFRGHT"].pct_change(12).round(4)
    return df[["year","month","TSIFRGHT","ftsi_yoy_pct"]].rename(
        columns={"TSIFRGHT": "ftsi_index"}
    )


def load_ppi(pace_data: str) -> pd.DataFrame:
    path = os.path.join(pace_data, "Producer_Price_Index_by_Industry.csv")
    df   = pd.read_csv(path)
    df["date"]  = pd.to_datetime(df["observation_date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df          = df.sort_values(["year","month"]).copy()
    df["ppi_trucking_mom_pct"] = df["PCU484121484121"].pct_change(1).round(4)
    return df[["year","month","ppi_trucking_mom_pct"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lmi  = load_lmi(PACE_DATA)
    ftsi = load_ftsi(PACE_DATA)
    ppi  = load_ppi(PACE_DATA)

    print(f"LMI rows (non-null): {len(lmi)}")
    print(f"FTSI rows: {len(ftsi)}")
    print(f"PPI rows:  {len(ppi)}")

    # Merge all on year + month
    merged = ftsi.merge(ppi, on=["year","month"], how="left")
    merged = merged.merge(lmi, on=["year","month"], how="left")

    # Forward-fill LMI gaps (sparse data)
    lmi_fill_cols = ["lmi_composite_index","lmi_transportation_capacity",
                     "lmi_transportation_utilization","lmi_transportation_prices"]
    for c in lmi_fill_cols:
        if c in merged.columns:
            merged[c] = merged[c].ffill().bfill()
        else:
            merged[c] = 55.0  # neutral LMI value

    # Freight tightness composite score (0-100)
    # High utilization + high prices + low capacity = tight market = more detention
    util  = merged.get("lmi_transportation_utilization", pd.Series([55.0]*len(merged)))
    price = merged.get("lmi_transportation_prices",      pd.Series([55.0]*len(merged)))
    cap   = merged.get("lmi_transportation_capacity",    pd.Series([55.0]*len(merged)))
    # Capacity is inverted: low capacity = high tightness
    tightness = (util.fillna(55) + price.fillna(55) + (100 - cap.fillna(55))) / 3
    merged["freight_tightness_score"] = tightness.round(1)

    merged = merged.sort_values(["year","month"]).reset_index(drop=True)
    print(f"\nOutput shape: {merged.shape}")
    print(f"Year range: {merged['year'].min()} – {merged['year'].max()}")
    print(merged[["year","month","ftsi_index","freight_tightness_score"]].tail(6).to_string(index=False))

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")
    print("\n[DAT NOTE] For spot rate features, set DAT_API_KEY env var and run with --dat flag")


if __name__ == "__main__":
    main()
