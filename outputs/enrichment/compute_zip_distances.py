"""Compute O/D distance + lat/lon for all unique Shipper/Consignee ZIP pairs.

Baseline: haversine (great-circle). This runs offline in seconds via pgeocode.
OSRM road distance will layer on top later via separate script once server is up.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pgeocode

OUT_DIR = Path(__file__).resolve().parent
LTL_CSV = OUT_DIR / "enriched_ltl_training.csv"
PAIRS_CSV = OUT_DIR / "od_zip_pairs.csv"
ZIP_GEO_CSV = OUT_DIR / "zip_geocodes.csv"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.756  # earth radius in miles
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main() -> None:
    df = pd.read_csv(LTL_CSV, low_memory=False, usecols=["Shipper Zip", "Consignee Zip"])
    all_zips = pd.unique(pd.concat([df["Shipper Zip"], df["Consignee Zip"]]).astype(str).str.zfill(5))
    print(f"Unique zips: {len(all_zips)}")

    nomi = pgeocode.Nominatim("us")
    geo = nomi.query_postal_code(list(all_zips))
    geo = geo[["postal_code", "place_name", "state_code", "latitude", "longitude"]].rename(
        columns={"postal_code": "zip"}
    )
    geo["zip"] = geo["zip"].astype(str).str.zfill(5)
    geo.to_csv(ZIP_GEO_CSV, index=False)
    print(f"Zip geocodes written: {ZIP_GEO_CSV} (resolved {geo['latitude'].notna().sum()}/{len(geo)})")

    pairs = (
        df.assign(
            **{
                "Shipper Zip": df["Shipper Zip"].astype(str).str.zfill(5),
                "Consignee Zip": df["Consignee Zip"].astype(str).str.zfill(5),
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    g_lookup = geo.set_index("zip")[["latitude", "longitude"]]
    pairs = pairs.join(g_lookup, on="Shipper Zip").rename(
        columns={"latitude": "orig_lat", "longitude": "orig_lon"}
    )
    pairs = pairs.join(g_lookup, on="Consignee Zip").rename(
        columns={"latitude": "dest_lat", "longitude": "dest_lon"}
    )
    pairs["haversine_miles"] = haversine_miles(
        pairs["orig_lat"], pairs["orig_lon"], pairs["dest_lat"], pairs["dest_lon"]
    )
    pairs.to_csv(PAIRS_CSV, index=False)
    resolved = pairs["haversine_miles"].notna().sum()
    print(f"Pairs: {len(pairs)}, haversine resolved: {resolved} ({resolved/len(pairs)*100:.1f}%)")
    print(f"Distance stats (miles): min={pairs['haversine_miles'].min():.1f}, "
          f"median={pairs['haversine_miles'].median():.1f}, "
          f"max={pairs['haversine_miles'].max():.1f}")
    print(f"Wrote {PAIRS_CSV}")


if __name__ == "__main__":
    main()
