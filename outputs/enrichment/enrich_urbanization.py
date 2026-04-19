"""ZIP-level urbanization / residential proxy features.

Uses pgeocode (offline) to map ZIPs to county FIPS, then joins to CBP
county-level establishment data from PACE DATA. Produces:
  - Warehouse / trucking establishment counts per ZIP's county
  - Commercial establishment density
  - Urban vs rural classification via thresholds on total establishments

Output enrichment file merged into enriched_ltl_training.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pgeocode

PACE_DATA = Path(r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA")
OUT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = OUT_DIR / "enriched_ltl_training.csv"
ZIP_URBAN_CSV = OUT_DIR / "zip_urbanization.csv"

# State abbreviation -> FIPS code (1-56).
STATE_ABBR_TO_FIPS = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10,
    "DC": 11, "FL": 12, "GA": 13, "HI": 15, "ID": 16, "IL": 17, "IN": 18,
    "IA": 19, "KS": 20, "KY": 21, "LA": 22, "ME": 23, "MD": 24, "MA": 25,
    "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31, "NV": 32,
    "NH": 33, "NJ": 34, "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39,
    "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47,
    "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54, "WI": 55,
    "WY": 56,
}


def zip_to_fips(state_code: str, county_code: float) -> int | None:
    fips = STATE_ABBR_TO_FIPS.get(state_code)
    if fips is None or pd.isna(county_code):
        return None
    return fips * 1000 + int(county_code)


def main() -> None:
    df = pd.read_csv(TRAIN_CSV, low_memory=False)

    all_zips = pd.unique(
        pd.concat([df["Shipper Zip"], df["Consignee Zip"]]).astype(str).str.zfill(5)
    )
    print(f"Unique zips: {len(all_zips)}")

    nomi = pgeocode.Nominatim("us")
    geo = nomi.query_postal_code(list(all_zips))
    geo["zip"] = geo["postal_code"].astype(str).str.zfill(5)
    geo["fips_full"] = geo.apply(
        lambda r: zip_to_fips(r["state_code"], r["county_code"]), axis=1
    )
    print(f"FIPS resolved: {geo['fips_full'].notna().sum()}/{len(geo)}")

    cbp = pd.read_csv(PACE_DATA / "facility_data" / "cbp_combined_facility_features.csv")
    print(f"CBP counties: {len(cbp)}")
    warehouse_cols = [
        "estab_484110", "estab_484121", "estab_484122", "estab_484220", "estab_484230",
        "estab_493110", "estab_493120", "estab_493130", "estab_493190",
    ]
    wh_cols_present = [c for c in warehouse_cols if c in cbp.columns]
    cbp["estab_trucking_warehouse"] = cbp[wh_cols_present].sum(axis=1)

    total_estab_cols = [c for c in cbp.columns if c.startswith("estab_")]
    cbp["estab_total"] = cbp[total_estab_cols].sum(axis=1)

    p90 = cbp["estab_total"].quantile(0.90)
    p25 = cbp["estab_total"].quantile(0.25)

    def urban_class(n: float) -> str:
        if pd.isna(n):
            return "unknown"
        if n >= p90:
            return "metro"
        if n >= p25:
            return "suburban"
        return "rural"

    cbp["urban_class"] = cbp["estab_total"].apply(urban_class)
    cbp_keep = cbp[["fips_full", "estab_total", "estab_trucking_warehouse",
                    "estab_warehousing_total", "reefer_share", "urban_class"]]

    zip_urban = (
        geo[["zip", "fips_full"]]
        .merge(cbp_keep, on="fips_full", how="left")
        .rename(columns={
            "estab_total": "county_estab_total",
            "estab_trucking_warehouse": "county_trucking_warehouse_estabs",
            "estab_warehousing_total": "county_warehousing_estabs",
            "reefer_share": "county_reefer_share",
            "urban_class": "zip_urban_class",
        })
    )
    zip_urban.to_csv(ZIP_URBAN_CSV, index=False)
    resolved = zip_urban["county_estab_total"].notna().sum()
    print(f"ZIPs with CBP match: {resolved}/{len(zip_urban)} ({resolved/len(zip_urban)*100:.1f}%)")
    print("Urban class distribution:")
    print(zip_urban["zip_urban_class"].value_counts())

    df["_shipper_zip_str"] = df["Shipper Zip"].astype(str).str.zfill(5)
    df["_consignee_zip_str"] = df["Consignee Zip"].astype(str).str.zfill(5)

    origin_cols = zip_urban.rename(columns={
        "zip": "_shipper_zip_str",
        "county_estab_total": "origin_county_estabs",
        "county_trucking_warehouse_estabs": "origin_trucking_warehouse_estabs",
        "county_warehousing_estabs": "origin_warehousing_estabs",
        "county_reefer_share": "origin_reefer_share",
        "zip_urban_class": "origin_urban_class",
    })[["_shipper_zip_str", "origin_county_estabs", "origin_trucking_warehouse_estabs",
        "origin_warehousing_estabs", "origin_reefer_share", "origin_urban_class"]]

    dest_cols = zip_urban.rename(columns={
        "zip": "_consignee_zip_str",
        "county_estab_total": "dest_county_estabs",
        "county_trucking_warehouse_estabs": "dest_trucking_warehouse_estabs",
        "county_warehousing_estabs": "dest_warehousing_estabs",
        "county_reefer_share": "dest_reefer_share",
        "zip_urban_class": "dest_urban_class",
    })[["_consignee_zip_str", "dest_county_estabs", "dest_trucking_warehouse_estabs",
        "dest_warehousing_estabs", "dest_reefer_share", "dest_urban_class"]]

    df = df.merge(origin_cols.drop_duplicates("_shipper_zip_str"), how="left", on="_shipper_zip_str")
    df = df.merge(dest_cols.drop_duplicates("_consignee_zip_str"), how="left", on="_consignee_zip_str")
    df = df.drop(columns=["_shipper_zip_str", "_consignee_zip_str"])

    df["likely_residential_dest"] = (df["dest_county_estabs"].fillna(0) < p25).astype(int)

    df.to_csv(TRAIN_CSV, index=False)
    print(f"\nFinal shape: {df.shape}")
    print(f"Origin urban class coverage: {df['origin_urban_class'].notna().mean()*100:.1f}%")
    print(f"Dest urban class coverage: {df['dest_urban_class'].notna().mean()*100:.1f}%")
    print(f"likely_residential_dest rate: {df['likely_residential_dest'].mean()*100:.1f}%")
    print(f"Wrote {TRAIN_CSV}")


if __name__ == "__main__":
    main()
