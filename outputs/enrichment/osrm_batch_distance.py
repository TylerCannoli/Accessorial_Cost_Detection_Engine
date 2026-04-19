"""Call a local OSRM server to compute road distance + duration for all O/D ZIP pairs.

Expects `od_zip_pairs.csv` (from compute_zip_distances.py) with orig_lat/orig_lon/dest_lat/dest_lon.
Writes `od_zip_pairs_osrm.csv` with added `osrm_miles` and `osrm_minutes` columns.

Usage:
  python osrm_batch_distance.py              # uses http://localhost:5000
  OSRM_URL=http://host:5000 python ...       # override server URL

Resumes automatically: if output exists, only missing rows are requested.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT_DIR = Path(__file__).resolve().parent
PAIRS_CSV = OUT_DIR / "od_zip_pairs.csv"
OSRM_OUT = OUT_DIR / "od_zip_pairs_osrm.csv"
OSRM_URL = os.environ.get("OSRM_URL", "http://localhost:5000")

METERS_PER_MILE = 1609.344


def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[502, 503, 504], allowed_methods=["GET"])
    s.mount("http://", HTTPAdapter(max_retries=retries, pool_maxsize=16))
    return s


def route_one(session: requests.Session, o_lat, o_lon, d_lat, d_lon):
    url = f"{OSRM_URL}/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}"
    try:
        r = session.get(url, params={"overview": "false", "alternatives": "false"}, timeout=15)
        if r.status_code != 200:
            return None, None
        j = r.json()
        if j.get("code") != "Ok" or not j.get("routes"):
            return None, None
        route = j["routes"][0]
        return route["distance"] / METERS_PER_MILE, route["duration"] / 60.0
    except (requests.RequestException, ValueError):
        return None, None


def main() -> None:
    pairs = pd.read_csv(PAIRS_CSV, dtype={"Shipper Zip": str, "Consignee Zip": str})
    pairs["Shipper Zip"] = pairs["Shipper Zip"].str.zfill(5)
    pairs["Consignee Zip"] = pairs["Consignee Zip"].str.zfill(5)

    if OSRM_OUT.exists():
        done = pd.read_csv(OSRM_OUT, dtype={"Shipper Zip": str, "Consignee Zip": str})
        done["Shipper Zip"] = done["Shipper Zip"].str.zfill(5)
        done["Consignee Zip"] = done["Consignee Zip"].str.zfill(5)
        merged = pairs.merge(
            done[["Shipper Zip", "Consignee Zip", "osrm_miles", "osrm_minutes"]],
            on=["Shipper Zip", "Consignee Zip"], how="left",
        )
    else:
        merged = pairs.copy()
        merged["osrm_miles"] = pd.NA
        merged["osrm_minutes"] = pd.NA

    needs = merged["osrm_miles"].isna() & merged[["orig_lat", "orig_lon", "dest_lat", "dest_lon"]].notna().all(axis=1)
    todo_idx = merged.index[needs].tolist()
    print(f"Total pairs: {len(merged)}, resolved: {(~needs).sum()}, to query: {len(todo_idx)}")

    # Health check.
    session = build_session()
    try:
        hc = session.get(f"{OSRM_URL}/route/v1/driving/-90.34,38.74;-80.82,35.29?overview=false", timeout=5)
        if hc.status_code != 200:
            raise SystemExit(f"OSRM health check failed: {hc.status_code} {hc.text[:200]}")
    except requests.RequestException as e:
        raise SystemExit(f"Cannot reach OSRM at {OSRM_URL}: {e}")
    print(f"OSRM reachable at {OSRM_URL}")

    checkpoint_every = 500
    t0 = time.time()
    for n, i in enumerate(todo_idx, 1):
        row = merged.loc[i]
        miles, minutes = route_one(session, row["orig_lat"], row["orig_lon"],
                                   row["dest_lat"], row["dest_lon"])
        merged.at[i, "osrm_miles"] = miles
        merged.at[i, "osrm_minutes"] = minutes
        if n % checkpoint_every == 0 or n == len(todo_idx):
            rate = n / max(time.time() - t0, 1e-6)
            eta = (len(todo_idx) - n) / max(rate, 1e-6)
            merged.to_csv(OSRM_OUT, index=False)
            print(f"  {n}/{len(todo_idx)} done ({rate:.1f}/s, ETA {eta/60:.1f} min)")

    merged.to_csv(OSRM_OUT, index=False)
    resolved = merged["osrm_miles"].notna().sum()
    print(f"\nResolved OSRM: {resolved}/{len(merged)} ({resolved/len(merged)*100:.1f}%)")
    if resolved:
        both = merged.dropna(subset=["osrm_miles", "haversine_miles"])
        ratio = (both["osrm_miles"] / both["haversine_miles"]).replace([float("inf")], pd.NA).dropna()
        print(f"OSRM miles stats: min={merged['osrm_miles'].min():.1f}, "
              f"median={merged['osrm_miles'].median():.1f}, max={merged['osrm_miles'].max():.1f}")
        print(f"Road/haversine ratio: median={ratio.median():.3f}, p95={ratio.quantile(0.95):.3f}")
    print(f"Wrote {OSRM_OUT}")


if __name__ == "__main__":
    main()
