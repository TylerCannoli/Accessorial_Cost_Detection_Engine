"""ZIP-level commercial/industrial facility density via EPA FRS.

NATIONAL_SINGLE.CSV has ~4M regulated US facilities with POSTAL_CODE, SITE_TYPE_NAME,
and a NAICS code field. We aggregate counts to ZIP and merge onto shipper/consignee.

Feature intuition: EPA FRS is a reasonable proxy for "is this delivery address in a
commercial/industrial area?" — much more discriminative than CBP county totals.

Output adds to enriched_ltl_training.csv:
  - origin_frs_facilities, dest_frs_facilities        (all site types)
  - origin_frs_stationary, dest_frs_stationary        (stationary regulated sites)
  - frs_residential_dest                              (0 facilities in dest ZIP)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PACE_DATA = Path(r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA")
FRS_CSV = PACE_DATA / "EPA Facility Registry Service (FRS)" / "national_single" / "NATIONAL_SINGLE.CSV"

OUT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = OUT_DIR / "enriched_ltl_training.csv"
ZIP_FRS_CSV = OUT_DIR / "zip_frs_counts.csv"

CHUNK = 250_000
USE_COLS = ["POSTAL_CODE", "SITE_TYPE_NAME", "STATE_CODE"]


def build_zip_counts() -> pd.DataFrame:
    if ZIP_FRS_CSV.exists():
        print(f"Using cached {ZIP_FRS_CSV}")
        return pd.read_csv(ZIP_FRS_CSV, dtype={"zip": str})

    print(f"Streaming {FRS_CSV.name} in {CHUNK:,}-row chunks…")
    agg = {}
    rows_seen = 0
    for chunk in pd.read_csv(FRS_CSV, usecols=USE_COLS, dtype=str,
                             chunksize=CHUNK, low_memory=False):
        rows_seen += len(chunk)
        chunk = chunk.dropna(subset=["POSTAL_CODE"])
        chunk["zip"] = chunk["POSTAL_CODE"].str[:5].str.zfill(5)
        chunk = chunk[chunk["zip"].str.match(r"^\d{5}$", na=False)]
        chunk["is_stationary"] = (chunk["SITE_TYPE_NAME"] == "STATIONARY").astype(int)
        grp = chunk.groupby("zip").agg(
            facilities=("zip", "size"),
            stationary=("is_stationary", "sum"),
        )
        for z, row in grp.iterrows():
            cur = agg.get(z, (0, 0))
            agg[z] = (cur[0] + int(row["facilities"]), cur[1] + int(row["stationary"]))
        print(f"  scanned {rows_seen:,} rows, zips so far: {len(agg):,}")

    out = pd.DataFrame(
        [(z, f, s) for z, (f, s) in agg.items()],
        columns=["zip", "frs_facilities", "frs_stationary"],
    )
    out.to_csv(ZIP_FRS_CSV, index=False)
    print(f"Wrote {ZIP_FRS_CSV}: {len(out):,} zips")
    return out


def main() -> None:
    counts = build_zip_counts()
    counts["zip"] = counts["zip"].astype(str).str.zfill(5)

    df = pd.read_csv(TRAIN_CSV, low_memory=False)
    df["_shipper_zip_str"] = df["Shipper Zip"].astype(str).str.zfill(5)
    df["_consignee_zip_str"] = df["Consignee Zip"].astype(str).str.zfill(5)

    origin = counts.rename(columns={
        "zip": "_shipper_zip_str",
        "frs_facilities": "origin_frs_facilities",
        "frs_stationary": "origin_frs_stationary",
    })
    dest = counts.rename(columns={
        "zip": "_consignee_zip_str",
        "frs_facilities": "dest_frs_facilities",
        "frs_stationary": "dest_frs_stationary",
    })

    df = df.merge(origin, how="left", on="_shipper_zip_str")
    df = df.merge(dest, how="left", on="_consignee_zip_str")

    for col in ["origin_frs_facilities", "origin_frs_stationary",
                "dest_frs_facilities", "dest_frs_stationary"]:
        df[col] = df[col].fillna(0).astype(int)

    df["frs_residential_dest"] = (df["dest_frs_facilities"] == 0).astype(int)
    df = df.drop(columns=["_shipper_zip_str", "_consignee_zip_str"])

    df.to_csv(TRAIN_CSV, index=False)
    print(f"\nFinal shape: {df.shape}")
    print(f"Dest FRS coverage: {(df['dest_frs_facilities'] > 0).mean()*100:.1f}% have >=1 facility")
    print(f"frs_residential_dest rate: {df['frs_residential_dest'].mean()*100:.1f}%")
    print(f"Mean dest facilities: {df['dest_frs_facilities'].mean():.1f}, median: {df['dest_frs_facilities'].median():.0f}")
    print(f"Wrote {TRAIN_CSV}")


if __name__ == "__main__":
    main()
