"""Merge NCEI 2025-2026 weather + carrier terminal features into enriched_ltl_training.csv.

Fills the 0% hit-rate weather columns (origin_wx_*, dest_wx_*) by assigning each
shipment ZIP to its nearest NCEI station (20 representative airport ASOS stations)
via haversine lookup, then joining on (station, pickup_date).

Adds 3 carrier terminal features:
  - carrier_has_terminal_in_origin_zip (0/1)
  - carrier_has_terminal_in_dest_zip   (0/1)
  - dist_to_nearest_carrier_terminal_mi (min over origin/dest ZIP -> carrier terminal ZIP)

Overwrites enriched_ltl_training.csv in place.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "enriched_ltl_training.csv"
NCEI = ROOT / "ncei_daily_2025_2026.csv"
TERMINALS = ROOT / "carrier_terminal_zips.csv"
ZIPS = ROOT / "zip_geocodes.csv"

# (station_id, city, state, lat, lon). From pull_ncei_weather.py DEFAULT_STATIONS.
NCEI_STATIONS = [
    ("GHCND:USW00013874", "ATLANTA",        "GA", 33.6367,  -84.4281),
    ("GHCND:USW00014739", "BOSTON",         "MA", 42.3656,  -71.0096),
    ("GHCND:USW00014819", "CHICAGO",        "IL", 41.9786,  -87.9047),
    ("GHCND:USW00013881", "CHARLOTTE",      "NC", 35.2144,  -80.9472),
    ("GHCND:USW00013960", "DALLAS",         "TX", 32.8471,  -96.8518),
    ("GHCND:USW00023234", "SAN_FRANCISCO",  "CA", 37.6213, -122.3790),
    ("GHCND:USW00023174", "LOS_ANGELES",    "CA", 33.9416, -118.4085),
    ("GHCND:USW00012960", "HOUSTON",        "TX", 29.9844,  -95.3414),
    ("GHCND:USW00013723", "GREENSBORO",     "NC", 36.0978,  -79.9373),
    ("GHCND:USW00014922", "MINNEAPOLIS",    "MN", 44.8848,  -93.2223),
    ("GHCND:USW00014933", "DES_MOINES",     "IA", 41.5341,  -93.6631),
    ("GHCND:USW00094846", "CHICAGO_MDW",    "IL", 41.7868,  -87.7522),
    ("GHCND:USW00093721", "BALTIMORE",      "MD", 39.1754,  -76.6683),
    ("GHCND:USW00013739", "PHILADELPHIA",   "PA", 39.8729,  -75.2437),
    ("GHCND:USW00094728", "NEW_YORK",       "NY", 40.7794,  -73.9692),
    ("GHCND:USW00003927", "DFW",            "TX", 32.8998,  -97.0403),
    ("GHCND:USW00023183", "PHOENIX",        "AZ", 33.4343, -112.0116),
    ("GHCND:USW00024233", "SEATTLE",        "WA", 47.4502, -122.3088),
    ("GHCND:USW00024127", "SALT_LAKE_CITY", "UT", 40.7884, -111.9778),
    ("GHCND:USW00003017", "DENVER",         "CO", 39.8561, -104.6737),
]

# LTL "Carrier Name" -> scraper carrier key (carrier_terminal_zips.csv).
CARRIER_NAME_TO_KEY = {
    "SAIA": "SAIA",
    "R&L Carriers": "RL",
    "Estes Express Lines": "ESTES",
    "XPO Logistics": "XPO",
    "FedEx Economy": "FEDEXFRT",
    "FedEx Priority": "FEDEXFRT",
    "Old Dominion Freight Line": "ODFL",
}


def haversine_mi(lat1, lon1, lat2, lon2):
    R = 3958.7613  # Earth radius, miles
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def assign_nearest_station(zip_lat: pd.Series, zip_lon: pd.Series) -> pd.Series:
    """Vectorized nearest-station lookup. Returns city name (NCEI's 'city' column)."""
    st_lat = np.array([s[3] for s in NCEI_STATIONS])
    st_lon = np.array([s[4] for s in NCEI_STATIONS])
    st_city = np.array([s[1] for s in NCEI_STATIONS])

    lat = zip_lat.to_numpy()[:, None]  # (N,1)
    lon = zip_lon.to_numpy()[:, None]
    d = haversine_mi(lat, lon, st_lat[None, :], st_lon[None, :])  # (N, 20)
    # NaN-safe argmin: rows with NaN lat/lon -> all NaN distances -> emit None.
    valid = ~np.isnan(d).all(axis=1)
    d_safe = np.where(np.isnan(d), np.inf, d)
    idx = np.argmin(d_safe, axis=1)
    out = st_city[idx]
    out = np.where(valid, out, None)
    return pd.Series(out, index=zip_lat.index)


def main():
    print(f"Loading {CSV.name}...")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"  shape: {df.shape}")

    # --- Drop old 100%-null weather columns ---
    old_wx = [c for c in df.columns if c.startswith(("origin_wx_", "dest_wx_"))]
    if old_wx:
        df = df.drop(columns=old_wx)
        print(f"  dropped {len(old_wx)} stale wx cols")

    df["pickup_dt"] = pd.to_datetime(df["pickup_dt"], errors="coerce")

    # --- ZIP geocode lookup ---
    print("Loading ZIP geocodes...")
    zg = pd.read_csv(ZIPS, dtype={"zip": str})[["zip", "latitude", "longitude"]]
    zg["zip"] = zg["zip"].str.zfill(5)
    zg = zg.drop_duplicates(subset="zip", keep="first")
    zg_map = zg.set_index("zip")

    df["_ship_zip5"] = df["Shipper Zip"].astype(str).str.zfill(5)
    df["_cons_zip5"] = df["Consignee Zip"].astype(str).str.zfill(5)

    df["_ship_lat"] = df["_ship_zip5"].map(zg_map["latitude"])
    df["_ship_lon"] = df["_ship_zip5"].map(zg_map["longitude"])
    df["_cons_lat"] = df["_cons_zip5"].map(zg_map["latitude"])
    df["_cons_lon"] = df["_cons_zip5"].map(zg_map["longitude"])
    print(f"  ship ZIP geocode hit: {df['_ship_lat'].notna().mean()*100:.1f}%")
    print(f"  cons ZIP geocode hit: {df['_cons_lat'].notna().mean()*100:.1f}%")

    # --- NCEI station assignment ---
    print("Assigning nearest NCEI station per ZIP...")
    df["origin_ncei_city"] = assign_nearest_station(df["_ship_lat"], df["_ship_lon"])
    df["dest_ncei_city"]   = assign_nearest_station(df["_cons_lat"], df["_cons_lon"])

    # --- NCEI weather join ---
    print(f"Loading {NCEI.name}...")
    w = pd.read_csv(NCEI, parse_dates=["date"])
    print(f"  weather rows: {len(w)}, stations: {w['station'].nunique()}, "
          f"date range: {w['date'].min().date()} .. {w['date'].max().date()}")

    # aggregate to (city, date) — 1 station per city here but this is safe
    w_agg = w.groupby(["city", "date"], as_index=False).agg(
        wind_mph=("wind_mph", "max"),
        prcp_in=("prcp_in", "max"),
        snow_in=("snow_in", "max"),
        tmax_f=("tmax_f", "max"),
        tmin_f=("tmin_f", "min"),
    )

    origin_wx = w_agg.add_prefix("origin_wx_").rename(
        columns={"origin_wx_city": "origin_ncei_city", "origin_wx_date": "pickup_dt"}
    )
    dest_wx = w_agg.add_prefix("dest_wx_").rename(
        columns={"dest_wx_city": "dest_ncei_city", "dest_wx_date": "pickup_dt"}
    )

    df = df.merge(origin_wx, how="left", on=["origin_ncei_city", "pickup_dt"])
    df = df.merge(dest_wx,   how="left", on=["dest_ncei_city",   "pickup_dt"])

    orig_hit = df["origin_wx_prcp_in"].notna().mean() * 100
    dest_hit = df["dest_wx_prcp_in"].notna().mean() * 100
    print(f"  origin weather hit: {orig_hit:.1f}%")
    print(f"  dest weather hit:   {dest_hit:.1f}%")

    # severe-weather flag
    df["severe_weather_flag"] = (
        (df["origin_wx_prcp_in"].fillna(0) > 1.0)
        | (df["origin_wx_snow_in"].fillna(0) > 2.0)
        | (df["origin_wx_wind_mph"].fillna(0) > 25)
        | (df["dest_wx_prcp_in"].fillna(0) > 1.0)
        | (df["dest_wx_snow_in"].fillna(0) > 2.0)
        | (df["dest_wx_wind_mph"].fillna(0) > 25)
    ).astype(int)
    print(f"  severe_weather_flag fires on {df['severe_weather_flag'].mean()*100:.2f}% of rows")

    # --- Carrier terminal features ---
    print(f"Loading {TERMINALS.name}...")
    term = pd.read_csv(TERMINALS, dtype={"zip": str})
    term["zip"] = term["zip"].str.zfill(5)
    # carrier -> set of ZIPs
    carrier_zips = term.groupby("carrier")["zip"].apply(set).to_dict()
    # carrier -> (N,2) lat/lon of terminal ZIPs (via zip_geocodes)
    carrier_coords: dict[str, np.ndarray] = {}
    for ck, zs in carrier_zips.items():
        coords = zg[zg["zip"].isin(zs)][["latitude", "longitude"]].to_numpy()
        carrier_coords[ck] = coords
        print(f"  {ck:<10s} terminals={len(zs):>4d}  geocoded={len(coords):>4d}")

    df["_carrier_key"] = df["Carrier Name"].map(CARRIER_NAME_TO_KEY)
    # default all features to NaN for carriers not in the 6-LTL set
    df["carrier_has_terminal_in_origin_zip"] = np.nan
    df["carrier_has_terminal_in_dest_zip"] = np.nan
    df["dist_to_nearest_carrier_terminal_mi"] = np.nan

    for ck, zips in carrier_zips.items():
        mask = df["_carrier_key"].eq(ck).to_numpy()
        if not mask.any():
            continue
        sub = df.loc[mask]
        df.loc[mask, "carrier_has_terminal_in_origin_zip"] = (
            sub["_ship_zip5"].isin(zips).astype(int).to_numpy()
        )
        df.loc[mask, "carrier_has_terminal_in_dest_zip"] = (
            sub["_cons_zip5"].isin(zips).astype(int).to_numpy()
        )
        # distance: min over {origin, dest} to any carrier terminal
        coords = carrier_coords[ck]
        if len(coords) == 0:
            continue
        s_lat = sub["_ship_lat"].to_numpy()[:, None]
        s_lon = sub["_ship_lon"].to_numpy()[:, None]
        c_lat = sub["_cons_lat"].to_numpy()[:, None]
        c_lon = sub["_cons_lon"].to_numpy()[:, None]
        t_lat = coords[:, 0][None, :]
        t_lon = coords[:, 1][None, :]
        d_origin = haversine_mi(s_lat, s_lon, t_lat, t_lon)  # (n, T)
        d_dest   = haversine_mi(c_lat, c_lon, t_lat, t_lon)
        dmin = np.minimum(np.nanmin(d_origin, axis=1), np.nanmin(d_dest, axis=1))
        df.loc[mask, "dist_to_nearest_carrier_terminal_mi"] = dmin

    covered = df["carrier_has_terminal_in_origin_zip"].notna().mean() * 100
    exact_origin = df["carrier_has_terminal_in_origin_zip"].fillna(0).sum()
    exact_dest = df["carrier_has_terminal_in_dest_zip"].fillna(0).sum()
    print(f"  carrier coverage:  {covered:.1f}% of rows have one of the 6 LTL carriers")
    print(f"  terminal-in-origin-ZIP exact matches: {int(exact_origin):,}")
    print(f"  terminal-in-dest-ZIP exact matches:   {int(exact_dest):,}")
    dist = df["dist_to_nearest_carrier_terminal_mi"].dropna()
    if len(dist):
        print(f"  dist_to_nearest (miles): median={dist.median():.1f}, p90={dist.quantile(0.9):.1f}, max={dist.max():.1f}")

    # --- Cleanup + write ---
    drop_tmp = ["_ship_zip5", "_cons_zip5", "_ship_lat", "_ship_lon",
                "_cons_lat", "_cons_lon", "origin_ncei_city", "dest_ncei_city",
                "_carrier_key"]
    df = df.drop(columns=[c for c in drop_tmp if c in df.columns])

    print(f"Writing {CSV.name} (shape {df.shape})...")
    df.to_csv(CSV, index=False)
    print("done.")


if __name__ == "__main__":
    main()
