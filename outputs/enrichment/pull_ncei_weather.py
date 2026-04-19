"""Pull 2025-2026 daily weather summaries from NOAA's CDO v2 API.

Replaces the stale ncei_daily_combined.csv (ends 2024-12-31) with fresh rows for
the set of NCEI cities used in build_enriched_ltl.py (STATE_TO_NCEI_CITY).

Register for a token: https://www.ncdc.noaa.gov/cdo-web/token
Then set it in env:
    export CDO_API_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Request budget: v2 allows 1,000 requests/day and 5 requests/second.
Strategy: one request per (station_id, 1-year window), datatypes=TMIN,TMAX,PRCP,SNOW,AWND.

Output columns: station, date, city, state, tmin_f, tmax_f, prcp_in, snow_in, wind_mph.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent
STATIONS_CSV = OUT_DIR / "ncei_stations.csv"    # optional hand-curated mapping
OUT_CSV = OUT_DIR / "ncei_daily_2025_2026.csv"

TOKEN = os.environ.get("CDO_API_TOKEN")
BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"

# Representative GHCND stations per city used in STATE_TO_NCEI_CITY.
# IDs verified against NCDC station search (GHCND:USW* are primary airport ASOS).
DEFAULT_STATIONS = [
    ("GHCND:USW00013874", "ATLANTA", "GA"),       # ATL
    ("GHCND:USW00014739", "BOSTON", "MA"),        # BOS
    ("GHCND:USW00014819", "CHICAGO", "IL"),       # ORD
    ("GHCND:USW00013881", "CHARLOTTE", "NC"),     # CLT
    ("GHCND:USW00013960", "DALLAS", "TX"),        # DAL
    ("GHCND:USW00023234", "SAN_FRANCISCO", "CA"), # SFO
    ("GHCND:USW00023174", "LOS_ANGELES", "CA"),   # LAX
    ("GHCND:USW00012960", "HOUSTON", "TX"),       # IAH
    ("GHCND:USW00013723", "GREENSBORO", "NC"),
    ("GHCND:USW00014922", "MINNEAPOLIS", "MN"),   # MSP
    ("GHCND:USW00014933", "DES_MOINES", "IA"),    # DSM
    ("GHCND:USW00094846", "CHICAGO_MDW", "IL"),
    ("GHCND:USW00093721", "BALTIMORE", "MD"),     # BWI
    ("GHCND:USW00013739", "PHILADELPHIA", "PA"),  # PHL
    ("GHCND:USW00094728", "NEW_YORK", "NY"),      # CPK
    ("GHCND:USW00003927", "DFW", "TX"),
    ("GHCND:USW00023183", "PHOENIX", "AZ"),       # PHX
    ("GHCND:USW00024233", "SEATTLE", "WA"),       # SEA
    ("GHCND:USW00024127", "SALT_LAKE_CITY", "UT"),
    ("GHCND:USW00003017", "DENVER", "CO"),        # DEN
]

DATATYPES = ["TMIN", "TMAX", "PRCP", "SNOW", "AWND"]


def load_stations() -> list[tuple[str, str, str]]:
    if STATIONS_CSV.exists():
        st = pd.read_csv(STATIONS_CSV)
        return list(zip(st["station_id"], st["city"], st["state"]))
    return DEFAULT_STATIONS


def request_year(session: requests.Session, station_id: str, year: int) -> pd.DataFrame:
    rows: list[dict] = []
    offset = 1
    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "datatypeid": DATATYPES,
            "units": "standard",
            "limit": 1000,
            "offset": offset,
        }
        r = session.get(f"{BASE}/data", params=params, timeout=60)
        if r.status_code == 429:
            print(f"    rate limited, sleeping 60s…")
            time.sleep(60)
            continue
        r.raise_for_status()
        j = r.json()
        results = j.get("results", [])
        rows.extend(results)
        total = j.get("metadata", {}).get("resultset", {}).get("count", 0)
        if offset + len(results) > total or not results:
            break
        offset += 1000
        time.sleep(0.25)  # stay under 5 req/s
    return pd.DataFrame(rows)


def pivot_long(long_df: pd.DataFrame, city: str, state: str) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.date
    wide = long_df.pivot_table(index=["station", "date"], columns="datatype",
                               values="value", aggfunc="first").reset_index()
    wide["city"] = city
    wide["state"] = state
    # Units (standard): PRCP/SNOW in inches, TMIN/TMAX in F, AWND in mph.
    wide = wide.rename(columns={
        "TMIN": "tmin_f", "TMAX": "tmax_f",
        "PRCP": "prcp_in", "SNOW": "snow_in", "AWND": "wind_mph",
    })
    return wide


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set CDO_API_TOKEN env var. Register at https://www.ncdc.noaa.gov/cdo-web/token")
    session = requests.Session()
    session.headers.update({"token": TOKEN})

    stations = load_stations()
    print(f"Stations: {len(stations)}")

    # Resume-aware: skip (station, year) already present in output.
    existing = pd.DataFrame()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV, parse_dates=["date"])
        print(f"Existing rows: {len(existing)}")

    frames: list[pd.DataFrame] = [existing] if not existing.empty else []
    for station_id, city, state in stations:
        for year in (2025, 2026):
            if not existing.empty:
                have = existing[(existing["station"] == station_id)
                                & (existing["date"].dt.year == year)]
                if len(have) > 300:  # treat as complete
                    continue
            print(f">> {station_id} {city},{state} {year}")
            try:
                long_df = request_year(session, station_id, year)
            except requests.HTTPError as e:
                print(f"   error: {e}; skipping")
                continue
            wide = pivot_long(long_df, city, state)
            if not wide.empty:
                frames.append(wide)
                out = pd.concat(frames, ignore_index=True)
                out.to_csv(OUT_CSV, index=False)
                print(f"   +{len(wide)} rows (total {len(out)})")

    if frames:
        final = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["station", "date"], keep="last")
        final.to_csv(OUT_CSV, index=False)
        print(f"\nFinal rows: {len(final)}")
        print(f"Date range: {final['date'].min()} → {final['date'].max()}")
        print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
