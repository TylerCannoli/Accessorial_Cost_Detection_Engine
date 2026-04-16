"""
Enrichment Script 07 — Download FMCSA Historical Inspection Data (2016-2022)
=============================================================================
Source : FMCSA SAFER Data Downloads (publicly available, no authentication required)
Output : <PACE_DATA>/fmcsa_historical/inspection_YYYY.csv  (one file per year)
         <PACE_DATA>/fmcsa_historical/company_census_YYYY.csv

FMCSA DATA PORTAL
-----------------
FMCSA publishes Motor Carrier Safety data at:
  https://www.fmcsa.dot.gov/safety/data-and-statistics/downloads

The Motor Carrier Census and Inspection data are released as large ZIP files.
This script automates the download and extraction into a standard format
matching the ctgan_input column schema.

WHAT THIS ADDS
--------------
Currently the model only trains on synthetic data derived from 2024-2025
inspection records.  Adding 2016-2022 gives:
  - 8 additional years of carrier safety history
  - Pre/post COVID freight pattern variation
  - 2017 ELD mandate impact on violation patterns
  - 2022 high-inflation / driver shortage period
  - Economic cycle diversity the model cannot learn from 2-year window

HOW TO RUN
----------
    python 07_download_fmcsa_historical.py --years 2016 2017 2018 2019 2020 2021 2022
    python 07_download_fmcsa_historical.py --years 2016 2022  # just two years as a test

The script:
  1. Downloads ZIP files from FMCSA public storage (~200-600 MB per year)
  2. Extracts and normalises column names to match ctgan_input schema
  3. Saves annual CSVs ready to merge with existing training data

MANUAL DOWNLOAD (if automated download fails)
---------------------------------------------
Go to: https://www.fmcsa.dot.gov/safety/data-and-statistics/downloads
Download: "Motor Carrier Census Information File" and "Inspection File"
Place them in: <PACE_DATA>/fmcsa_historical/raw/YYYY/
Then re-run this script with --local-only flag.

NOTE: FMCSA may require accepting a data use agreement on the portal.
If automated requests are blocked, the --manual-only mode will guide you.
"""

import os
import sys
import argparse
import zipfile
import io
import re
from pathlib import Path

import requests
import pandas as pd
import numpy as np

PACE_DATA = os.getenv(
    "PACE_DATA_DIR",
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
OUT_BASE = Path(PACE_DATA) / "fmcsa_historical"

# ── FMCSA public data URLs ─────────────────────────────────────────────
# These are the standard FMCSA data release URLs.
# Format: replace YYYY with 4-digit year.
# If a URL fails, check https://www.fmcsa.dot.gov/safety/data-and-statistics/downloads
# for updated paths.
INSPECTION_URL_TMPL  = (
    "https://ai.fmcsa.dot.gov/AnalysisResearchAndData/SAFER/"
    "InspectionFile{YYYY}.zip"
)
COMPANY_URL_TMPL = (
    "https://ai.fmcsa.dot.gov/AnalysisResearchAndData/SAFER/"
    "CompanyCensus{YYYY}.zip"
)

# Column mapping: FMCSA raw name → ctgan_input name
INSP_COL_MAP = {
    "DOT_NUMBER":       "dot_number",
    "REPORT_STATE":     "report_state",
    "INSP_DATE":        "insp_date",
    "INSP_LEVEL_ID":    "insp_level_id",
    "COUNTY_CODE_STATE":"county_code_state",
    "TIME_WEIGHT":      "time_weight",
    "DRIVER_OOS_TOTAL": "driver_oos_total",
    "VEHICLE_OOS_TOTAL":"vehicle_oos_total",
    "TOTAL_HAZMAT_SENT":"total_hazmat_sent",
    "OOS_TOTAL":        "oos_total",
    "HAZMAT_OOS_TOTAL": "hazmat_oos_total",
    "HAZMAT_PLACARD_REQ":"hazmat_placard_req",
    "UNIT_TYPE_DESC":   "unit_type_desc",
    "UNIT_MAKE":        "unit_make",
    "UNIT_LICENSE_STATE":"unit_license_state",
}

COMPANY_COL_MAP = {
    "DOT_NUMBER":       "dot_number",
    "STATUS_CODE":      "carrier_status_code",
    "ADD_DATE":         "carrier_add_date",
    "CARRIER_OPERATION":"carrier_carrier_operation",
    "MCS150_MILEAGE":   "carrier_mcs150_mileage",
    "MCS150_MILEAGE_YEAR":"carrier_mcs150_mileage_year",
    "TRUCK_UNITS":      "carrier_truck_units",
    "POWER_UNITS":      "carrier_power_units",
    "FLEETSIZE":        "carrier_fleetsize",
    "RECORDABLE_CRASH_RATE":"carrier_recordable_crash_rate",
    "TOTAL_INTRASTATE_DRIVERS":"carrier_total_intrastate_drivers",
    "HM_IND":           "carrier_hm_ind",
    "INTERSTATE_BEYOND_100_MILES":"carrier_interstate_beyond_100_miles",
    "INTERSTATE_WITHIN_100_MILES":"carrier_interstate_within_100_miles",
    "INTRASTATE_BEYOND_100_MILES":"carrier_intrastate_beyond_100_miles",
    "INTRASTATE_WITHIN_100_MILES":"carrier_intrastate_within_100_miles",
    "TOTAL_CDL":        "carrier_total_cdl",
    "TOTAL_DRIVERS":    "carrier_total_drivers",
    "PHY_COUNTRY":      "carrier_phy_country",
    "PHY_STATE":        "carrier_phy_state",
    "SAFETY_RATING":    "carrier_safety_rating",
    "CRGO_GENFREIGHT":  "carrier_crgo_genfreight",
    "CRGO_HOUSEHOLD":   "carrier_crgo_household",
    "CRGO_PRODUCE":     "carrier_crgo_produce",
    "CRGO_INTERMODAL":  "carrier_crgo_intermodal",
    "CRGO_OILFIELD":    "carrier_crgo_oilfield",
    "CRGO_MEAT":        "carrier_crgo_meat",
    "CRGO_CHEM":        "carrier_crgo_chem",
    "CRGO_DRYBULK":     "carrier_crgo_drybulk",
    "CRGO_COLDFOOD":    "carrier_crgo_coldfood",
    "CRGO_BEVERAGES":   "carrier_crgo_beverages",
    "CRGO_CONSTRUCT":   "carrier_crgo_construct",
}


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download url → dest_path. Returns True on success."""
    print(f"  Downloading {desc or url} ...")
    try:
        r = requests.get(url, stream=True, timeout=120,
                         headers={"User-Agent": "PACE-Research/1.0"})
        r.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = dest_path.stat().st_size / 1e6
        print(f"    Saved {size_mb:.1f} MB → {dest_path}")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def extract_and_normalise_inspections(zip_path: Path, year: int) -> pd.DataFrame:
    """Extract inspection ZIP → normalised DataFrame."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".txt") or n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV/TXT found in {zip_path}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, sep="|", encoding="latin1", low_memory=False)

    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]
    keep = {k: v for k, v in INSP_COL_MAP.items() if k in df.columns}
    df   = df[list(keep.keys())].rename(columns=keep)

    # Parse date → year/month/day
    df["insp_date_raw"] = df["insp_date"].astype(str)
    # FMCSA format is typically YYYYMMDD or YYYY-MM-DD
    df["insp_date_parsed"] = pd.to_datetime(df["insp_date_raw"], errors="coerce", format="%Y%m%d")
    df.loc[df["insp_date_parsed"].isna(), "insp_date_parsed"] = pd.to_datetime(
        df.loc[df["insp_date_parsed"].isna(), "insp_date_raw"], errors="coerce"
    )
    df["insp_year"]  = df["insp_date_parsed"].dt.year.fillna(year).astype(int)
    df["insp_month"] = df["insp_date_parsed"].dt.month.fillna(1).astype(int)
    df["insp_dow"]   = df["insp_date_parsed"].dt.dayofweek.fillna(0).astype(int)
    df["insp_day"]   = df["insp_date_parsed"].dt.day.fillna(1).astype(int)
    df = df.drop(columns=["insp_date","insp_date_raw","insp_date_parsed"], errors="ignore")

    # Numeric coercion
    for col in ["driver_oos_total","vehicle_oos_total","oos_total","hazmat_oos_total",
                "total_hazmat_sent","insp_level_id","time_weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"    Extracted {len(df):,} inspections for {year}")
    return df


def process_year(year: int, local_only: bool = False):
    raw_dir = OUT_BASE / "raw" / str(year)
    raw_dir.mkdir(parents=True, exist_ok=True)

    insp_zip  = raw_dir / f"inspection_{year}.zip"
    out_csv   = OUT_BASE / f"inspection_{year}.csv"

    if out_csv.exists():
        print(f"  {year}: already processed at {out_csv}, skipping.")
        return

    # Download if not local
    if not insp_zip.exists():
        if local_only:
            print(f"  {year}: no local ZIP found and --local-only set.  Place ZIP at {insp_zip}")
            return
        url = INSPECTION_URL_TMPL.replace("{YYYY}", str(year))
        ok  = download_file(url, insp_zip, f"inspections {year}")
        if not ok:
            # Try alternate URL patterns
            alt_url = f"https://www.fmcsa.dot.gov/safety/data-statistics/InspectionFile{year}.zip"
            ok = download_file(alt_url, insp_zip, f"inspections {year} (alt)")
        if not ok:
            print(f"  {year}: Could not download.  Manual download instructions:")
            print(f"    1. Go to https://www.fmcsa.dot.gov/safety/data-and-statistics/downloads")
            print(f"    2. Download 'Motor Carrier Inspection File' for {year}")
            print(f"    3. Save to {insp_zip}")
            return

    # Extract and normalise
    try:
        df = extract_and_normalise_inspections(insp_zip, year)
        df.to_csv(out_csv, index=False)
        print(f"  {year}: Saved {len(df):,} rows → {out_csv}")
    except Exception as e:
        print(f"  {year}: Extraction failed — {e}")


def main():
    parser = argparse.ArgumentParser(description="Download FMCSA historical inspection data")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2016, 2023)),
                        help="Years to download (default: 2016-2022)")
    parser.add_argument("--local-only", action="store_true",
                        help="Only process already-downloaded files, no network requests")
    args = parser.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_BASE}")
    print(f"Years to process: {args.years}")
    print()

    for year in sorted(args.years):
        print(f"Processing {year} ...")
        process_year(year, local_only=args.local_only)

    # Summary
    csvs = list(OUT_BASE.glob("inspection_*.csv"))
    print(f"\nDone. {len(csvs)} year files available in {OUT_BASE}")
    total = sum(pd.read_csv(c, usecols=["insp_year"]).shape[0] for c in csvs)
    print(f"Total rows across all years: {total:,}")
    print()
    print("Next step: run  build_enriched_training_set.py  to merge with ctgan_input")


if __name__ == "__main__":
    main()
