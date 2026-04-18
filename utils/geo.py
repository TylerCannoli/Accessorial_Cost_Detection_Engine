"""
utils/geo.py
Free geospatial utilities for PACE.

- Geocoding:  geopy + Nominatim (OpenStreetMap) — no API key required
- Routing:    OpenRouteService (ORS) free tier — 2,000 req/day, no credit card
              Falls back to Haversine straight-line if ORS key is not configured.

Setup (one-time):
  1. pip install geopy openrouteservice
  2. Get a free ORS key at https://openrouteservice.org/dev/#/signup
  3. Add to .env:  ORS_API_KEY=your_key_here

Without an ORS key the module still works — it returns Haversine distance
multiplied by 1.25 (industry-standard road-factor for US lanes).
"""
import os
import math
import time
import functools

ORS_API_KEY = os.getenv("ORS_API_KEY", "")

# ORS rate limit: 40 req/min on free tier — we stay well under
_ORS_DELAY = 1.6   # seconds between calls


# ── Haversine fallback ─────────────────────────────────────────────────────────
def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in miles between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Geocoding ─────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=512)
def geocode(location: str) -> tuple[float, float] | None:
    """
    Return (lat, lon) for a location string using Nominatim (OpenStreetMap).
    Results are cached in-process so repeated lookups are free.
    Returns None if geocoding fails.
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
    except ImportError:
        raise ImportError("geopy is required. Run: pip install geopy")

    geolocator = Nominatim(user_agent="pace_accessorial_engine")
    try:
        # Be polite to the free Nominatim service
        time.sleep(1.1)
        loc = geolocator.geocode(location, timeout=10)
        if loc:
            return (loc.latitude, loc.longitude)
        return None
    except GeocoderTimedOut:
        return None
    except Exception:
        return None


# ── Driving distance via OpenRouteService ─────────────────────────────────────
def _driving_miles_ors(
    origin_lat: float, origin_lon: float,
    dest_lat: float,   dest_lon: float,
) -> float | None:
    """
    Call ORS directions API for real driving distance in miles.
    Returns None on any failure.
    """
    if not ORS_API_KEY:
        return None
    try:
        import openrouteservice
        from openrouteservice.exceptions import ApiError
    except ImportError:
        return None

    client = openrouteservice.Client(key=ORS_API_KEY)
    try:
        time.sleep(_ORS_DELAY)
        route = client.directions(
            coordinates=[[origin_lon, origin_lat], [dest_lon, dest_lat]],
            profile="driving-hgv",   # heavy goods vehicle — more accurate for trucks
            format="geojson",
        )
        meters = route["features"][0]["properties"]["summary"]["distance"]
        return round(meters * 0.000621371, 1)  # meters → miles
    except ApiError:
        return None
    except Exception:
        return None


# ── Main entry point ──────────────────────────────────────────────────────────
def driving_miles(origin: str, destination: str) -> dict:
    """
    Compute driving distance between two location strings.

    Returns a dict:
        miles        — float, driving distance in miles
        method       — "ors_driving" | "haversine_estimated"
        origin_coords   — (lat, lon) or None
        dest_coords     — (lat, lon) or None
        error        — str or None
    """
    result = {
        "miles":          None,
        "method":         None,
        "origin_coords":  None,
        "dest_coords":    None,
        "error":          None,
    }

    orig_coords = geocode(origin)
    dest_coords = geocode(destination)

    if orig_coords is None:
        result["error"] = f"Could not geocode origin: '{origin}'"
        return result
    if dest_coords is None:
        result["error"] = f"Could not geocode destination: '{destination}'"
        return result

    result["origin_coords"] = orig_coords
    result["dest_coords"]   = dest_coords

    # Try ORS first
    ors = _driving_miles_ors(*orig_coords, *dest_coords)
    if ors is not None:
        result["miles"]  = ors
        result["method"] = "ors_driving"
        return result

    # Fall back to Haversine × 1.25 road factor
    straight = _haversine_miles(*orig_coords, *dest_coords)
    result["miles"]  = round(straight * 1.25, 1)
    result["method"] = "haversine_estimated"
    return result


def enrich_dataframe_miles(df, origin_col: str = "origin_city",
                            dest_col: str = "destination_city") -> object:
    """
    Add a `driving_miles` column to a DataFrame by geocoding origin/dest pairs.
    Skips rows where geocoding fails (keeps existing `miles` value).

    Only processes unique lane pairs to minimise API calls.
    Returns the enriched DataFrame.
    """
    import pandas as pd

    df = df.copy()
    if origin_col not in df.columns or dest_col not in df.columns:
        return df

    # Build unique lane → miles mapping
    lane_cache: dict[tuple, float] = {}
    unique_lanes = df[[origin_col, dest_col]].drop_duplicates().values

    for orig, dest in unique_lanes:
        key = (str(orig), str(dest))
        if key in lane_cache:
            continue
        res = driving_miles(str(orig), str(dest))
        if res["miles"] is not None:
            lane_cache[key] = res["miles"]

    def _lookup(row):
        key = (str(row[origin_col]), str(row[dest_col]))
        return lane_cache.get(key, row.get("miles", None))

    df["miles"] = df.apply(_lookup, axis=1)
    return df
