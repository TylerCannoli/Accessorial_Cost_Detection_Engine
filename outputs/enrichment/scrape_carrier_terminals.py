"""Scrape LTL carrier terminal locations into carrier_terminal_zips.csv.

Each carrier is handled by a dedicated function because their locator sites
use different tech stacks (Contentful CMS, custom XHR, JSON endpoints, etc).
All functions return list[Terminal]; `main` concatenates and writes CSV.

Currently implemented:
  - SAIA: paginate the Contentful Delivery API (token lifted from the
    locator page via headless Playwright).
  - ESTES: iterate the public sitemap /api/vtl/sitemap/sitemap.xml, fetch
    each /terminals/{slug} page, regex-extract "Terminal:" + "Address:" blocks.
  - ODFL: ODFL's own site hides detail behind reCAPTCHA, so we fetch the
    sitemap to enumerate (state, code) per terminal, then cross-reference
    with OpenStreetMap Overpass API (operator~"Old Dominion") for addresses.
    Any terminals Overpass misses fall through to the Google Places API
    if GOOGLE_PLACES_API_KEY is set in the environment.

Known blockers (documented in EXTERNAL_BLOCKERS.md):
  - R&L, XPO, FedEx Freight: no public directory / sitemap for terminals.
    Fall back to FMCSA carrier census (HQ-biased) if needed.

To add a carrier: write `fetch_<key>(...)` returning list[Terminal], add to
PARSERS. If the site is a plain HTTP/JSON call use `requests`; if it needs
to fire client-side code to expose an API token or XHR payload, use the
Playwright helper `_capture_from_page` as SAIA does.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from playwright.sync_api import sync_playwright

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

OUT_DIR = Path(__file__).resolve().parent
RAW_DIR = OUT_DIR / "terminals_raw"
RAW_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "carrier_terminal_zips.csv"


@dataclass
class Terminal:
    carrier: str
    code: str
    name: str
    street: str
    city: str
    state: str
    zip: str


# ---------------------------------------------------------------------------
# SAIA — Contentful-backed site
# ---------------------------------------------------------------------------

def fetch_saia() -> list[Terminal]:
    """Lift the Contentful Authorization token from the locator page then
    paginate the Delivery API for content_type=terminal."""
    locator_url = "https://www.saia.com/tools/terminal-locator"
    token: str | None = None
    space_url: str | None = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        def on_request(req):
            nonlocal token, space_url
            if "cdn.contentful.com" in req.url and "/entries" in req.url:
                auth = req.headers.get("authorization") or req.headers.get("Authorization")
                if auth and not token:
                    token = auth
                    space_url = req.url.split("/entries")[0] + "/entries"

        page.on("request", on_request)
        try:
            page.goto(locator_url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000)
        finally:
            browser.close()

    if not token or not space_url:
        print("  SAIA: failed to capture Contentful auth token")
        return []

    items: list[dict] = []
    skip = 0
    while True:
        r = requests.get(space_url, headers={"Authorization": token},
                         params={"content_type": "terminal", "limit": 1000, "skip": skip},
                         timeout=30)
        r.raise_for_status()
        j = r.json()
        batch = j.get("items", [])
        items.extend(batch)
        total = j.get("total", 0)
        skip += len(batch)
        if skip >= total or not batch:
            break

    (RAW_DIR / "saia_terminals_all.json").write_text(json.dumps({"items": items}, indent=2))

    out: list[Terminal] = []
    for it in items:
        f = it.get("fields", {})
        zip5 = str(f.get("addressPostalCode", ""))[:5].zfill(5)
        if not zip5.isdigit():
            continue
        out.append(Terminal(
            carrier="SAIA",
            code=str(f.get("code", "")).upper(),
            name=f.get("name", ""),
            street=f.get("streetAddress", ""),
            city=f.get("addressLocation", ""),
            state=f.get("addressRegion", ""),
            zip=zip5,
        ))
    return out


# ---------------------------------------------------------------------------
# Estes — public sitemap + simple HTML regex
# ---------------------------------------------------------------------------

ESTES_SITEMAP = "https://www.estes-express.com/api/vtl/sitemap/sitemap.xml"
ESTES_UA = {"User-Agent": "Mozilla/5.0 (research) contact claytonrjosef@gmail.com"}
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")
# "Terminal:" block: name ("Hagerstown") - code ("HAG") (numeric id)
ESTES_TERMINAL_RE = re.compile(
    r"Terminal:\s*</strong>[^<]*?([A-Z][A-Za-z0-9\s\-\.\&',]+?)\s*-\s*([A-Z0-9]{2,5})\s*\((\d+)\)"
)
# "Address:" block: street, city, ST ZIP
ESTES_ADDRESS_RE = re.compile(
    r"Address:\s*</strong>[^<]*?([^,<]+),\s*([A-Za-z\.\s]+),\s*([A-Z]{2})\s+(\d{5})"
)


def fetch_estes() -> list[Terminal]:
    sm = requests.get(ESTES_SITEMAP, headers=ESTES_UA, timeout=30)
    sm.raise_for_status()
    all_urls = LOC_RE.findall(sm.text)
    term_urls = [u for u in all_urls
                 if "/terminals/" in u and not u.rstrip("/").endswith("terminal-detail")]
    print(f"  Estes sitemap: {len(term_urls)} terminal URLs")

    out: list[Terminal] = []
    session = requests.Session()
    session.headers.update(ESTES_UA)
    for i, url in enumerate(term_urls, 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                continue
            html = r.text
            t_match = ESTES_TERMINAL_RE.search(html)
            a_match = ESTES_ADDRESS_RE.search(html)
            if not a_match:
                continue
            street, city, state, zip5 = (x.strip() for x in a_match.groups())
            name = t_match.group(1).strip() if t_match else city
            code = t_match.group(2).strip() if t_match else ""
            out.append(Terminal(
                carrier="ESTES", code=code, name=name,
                street=street, city=city, state=state, zip=zip5,
            ))
        except requests.RequestException:
            continue
        if i % 25 == 0:
            print(f"    {i}/{len(term_urls)} pages scanned, {len(out)} terminals")
        time.sleep(0.1)  # be polite
    return out


# ---------------------------------------------------------------------------
# ODFL — sitemap + OSM Overpass + (optional) Google Places fallback
# ---------------------------------------------------------------------------

ODFL_SITEMAP = "https://www.odfl.com/sitemap.xml"
ODFL_URL_RE = re.compile(r"/service-center-locator/([A-Z]{2})/([A-Z0-9]{2,5})\.html")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_UA = {"User-Agent": "PACE-research (contact claytonrjosef@gmail.com)"}


def _odfl_sitemap_codes() -> list[tuple[str, str, str]]:
    """Return list of (state, code, url) from ODFL's public sitemap."""
    r = requests.get(ODFL_SITEMAP, headers=OVERPASS_UA, timeout=30)
    r.raise_for_status()
    out = []
    for u in LOC_RE.findall(r.text):
        m = ODFL_URL_RE.search(u)
        if m:
            out.append((m.group(1), m.group(2), u))
    # Dedupe by (state, code)
    seen = set()
    uniq = []
    for s, c, u in out:
        if (s, c) in seen:
            continue
        seen.add((s, c))
        uniq.append((s, c, u))
    return uniq


def _odfl_fetch_h1_city(url: str, session: requests.Session) -> str:
    """Fetch a sitemap page and pull the marketing city from the <h1>."""
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return ""
        m = re.search(r"<h1[^>]*>([^<]+)</h1>", r.text)
        if not m:
            return ""
        # Typical format: "North Houston, Texas" or "Atlanta, Georgia"
        h1 = m.group(1).strip()
        city = h1.split(",")[0].strip()
        return city
    except requests.RequestException:
        return ""


STATE_NAME_TO_ABBR = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA",
    "colorado":"CO","connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA",
    "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS",
    "kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA",
    "michigan":"MI","minnesota":"MN","mississippi":"MS","missouri":"MO","montana":"MT",
    "nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ","new mexico":"NM",
    "new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC",
    "south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT",
    "virginia":"VA","washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY",
    "district of columbia":"DC",
}


def _nearest_zip(lat: float, lon: float, zip_df: pd.DataFrame) -> tuple[str, str, str]:
    """Return (zip, city, state) of closest ZIP centroid in zip_geocodes.csv."""
    # Quick bounding prefilter for speed (~0.5 deg ≈ 35 mi).
    near = zip_df[(zip_df["latitude"].between(lat - 0.5, lat + 0.5))
                  & (zip_df["longitude"].between(lon - 0.5, lon + 0.5))]
    if near.empty:
        near = zip_df
    dlat = near["latitude"] - lat
    dlon = near["longitude"] - lon
    d2 = dlat * dlat + dlon * dlon
    i = d2.idxmin()
    row = near.loc[i]
    return str(row["zip"]).zfill(5), row["place_name"], row["state_code"]


def _query_overpass(zip_df: pd.DataFrame) -> list[dict]:
    """Query OSM for features operated/named as Old Dominion Freight Line in US.
    Returns one record per feature; fills missing ZIP/city/state via nearest-ZIP."""
    q = """
    [out:json][timeout:180];
    area["ISO3166-1"="US"][admin_level=2]->.us;
    (
      nwr["operator"~"Old Dominion Freight",i](area.us);
      nwr["name"~"Old Dominion Freight",i](area.us);
    );
    out center tags;
    """
    r = requests.post(OVERPASS_URL, data={"data": q},
                      headers=OVERPASS_UA, timeout=200)
    r.raise_for_status()
    elems = r.json().get("elements", [])
    results = []
    for e in elems:
        tags = e.get("tags", {}) or {}
        name = (tags.get("name") or "").lower()
        op = (tags.get("operator") or "").lower()
        # Require "Old Dominion Freight" (not just "Old Dominion" — excludes
        # historic markers, street names, etc.).
        if "old dominion freight" not in name and "old dominion freight" not in op:
            continue
        # Exclude non-terminal features.
        if tags.get("historic") or tags.get("highway") or tags.get("place"):
            continue
        lat = e.get("lat") or (e.get("center") or {}).get("lat")
        lon = e.get("lon") or (e.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue

        zip5 = str(tags.get("addr:postcode", ""))[:5]
        city = tags.get("addr:city", "")
        state = tags.get("addr:state", "")

        # Normalize state to 2-letter abbr.
        if state:
            s = state.strip()
            if len(s) > 2:
                state = STATE_NAME_TO_ABBR.get(s.lower(), s[:2].upper())
            else:
                state = s.upper()

        # Fill any missing address via nearest ZIP centroid.
        if not zip5.isdigit() or not state or not city:
            nz, ncity, nstate = _nearest_zip(lat, lon, zip_df)
            if not zip5.isdigit():
                zip5 = nz
            if not city:
                city = ncity
            if not state:
                state = nstate

        results.append({
            "name": tags.get("name", ""),
            "state": state, "city": city,
            "street": f"{tags.get('addr:housenumber','')} {tags.get('addr:street','')}".strip(),
            "zip": zip5, "lat": lat, "lon": lon,
        })
    return results


def _google_places_lookup(code: str, city: str, state: str, api_key: str) -> dict | None:
    """Query Google Places Text Search for a specific ODFL terminal."""
    url = "https://places.googleapis.com/v1/places:searchText"
    query = f"Old Dominion Freight Line {city} {state}" if city else \
            f"Old Dominion Freight Line service center {state}"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,"
                            "places.addressComponents,places.location",
    }
    r = requests.post(url, json={"textQuery": query}, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    places = r.json().get("places", [])
    if not places:
        return None
    p = places[0]
    comps = {c["types"][0]: c.get("shortText") or c.get("longText")
             for c in p.get("addressComponents", []) if c.get("types")}
    zip5 = str(comps.get("postal_code", ""))[:5]
    if not zip5.isdigit():
        return None
    return {
        "name": p.get("displayName", {}).get("text", ""),
        "street": f"{comps.get('street_number','')} {comps.get('route','')}".strip(),
        "city": comps.get("locality", city),
        "state": comps.get("administrative_area_level_1", state),
        "zip": zip5,
    }


def fetch_odfl() -> list[Terminal]:
    print("  fetching ODFL sitemap...")
    sitemap = _odfl_sitemap_codes()
    print(f"  sitemap has {len(sitemap)} unique (state, code) entries")

    zip_df = pd.read_csv(OUT_DIR / "zip_geocodes.csv",
                         dtype={"zip": str})
    zip_df["zip"] = zip_df["zip"].str.zfill(5)

    print("  querying OSM Overpass (operator='Old Dominion')...")
    osm = _query_overpass(zip_df)
    print(f"  Overpass returned {len(osm)} ODFL-tagged features (post-geocode)")

    session = requests.Session()
    session.headers.update(OVERPASS_UA)

    # Bucket OSM features by 2-letter state (now normalized).
    by_state: dict[str, list[dict]] = {}
    for o in osm:
        by_state.setdefault((o.get("state") or "")[:2].upper(), []).append(o)

    out: list[Terminal] = []
    unmatched: list[tuple[str, str, str, str]] = []  # (state, code, url, h1_city)
    used_osm_ids: set[int] = set()

    for state, code, url in sitemap:
        h1_city = _odfl_fetch_h1_city(url, session).lower()
        time.sleep(0.05)
        pool = [o for o in by_state.get(state, []) if id(o) not in used_osm_ids]

        hit = None
        if h1_city and pool:
            tokens = [t for t in h1_city.replace("-", " ").split() if len(t) >= 4]
            for tok in tokens:
                for o in pool:
                    blob = f"{o.get('city','')} {o.get('name','')}".lower()
                    if tok in blob:
                        hit = o
                        break
                if hit:
                    break

        if hit:
            used_osm_ids.add(id(hit))
            out.append(Terminal(
                carrier="ODFL", code=code,
                name=hit.get("name") or f"ODFL - {h1_city.title()}, {state}",
                street=hit.get("street", ""),
                city=hit.get("city") or h1_city.title(),
                state=state, zip=hit["zip"],
            ))
        else:
            unmatched.append((state, code, url, h1_city))

    print(f"  OSM matched: {len(out)} / {len(sitemap)}")
    print(f"  unmatched: {len(unmatched)}")

    # Google Places fallback if a key is configured
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY")
    if unmatched and api_key:
        print(f"  using Google Places for {len(unmatched)} unmatched...")
        hits = 0
        for i, (state, code, url, h1_city) in enumerate(unmatched, 1):
            res = _google_places_lookup(code, h1_city, state, api_key)
            if res:
                out.append(Terminal(
                    carrier="ODFL", code=code,
                    name=res["name"] or f"ODFL - {h1_city.title()}, {state}",
                    street=res["street"], city=res["city"],
                    state=state, zip=res["zip"],
                ))
                hits += 1
            if i % 20 == 0:
                print(f"    {i}/{len(unmatched)} queried, {hits} resolved")
            time.sleep(0.05)
        print(f"  Google Places resolved: {hits} / {len(unmatched)}")
    elif unmatched:
        print(f"  set GOOGLE_PLACES_API_KEY to resolve the remaining {len(unmatched)}")

    return out


# ---------------------------------------------------------------------------
# FMCSA census — HQ-level fallback for carriers with no public directory
# ---------------------------------------------------------------------------

PACE_DATA = Path(r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA")
FMCSA_CENSUS = PACE_DATA / "FMCSA_Company_Census.csv"
FMCSA_USECOLS = ["DOT_NUMBER", "LEGAL_NAME", "DBA_NAME",
                 "PHY_STREET", "PHY_CITY", "PHY_STATE", "PHY_ZIP"]


def _fetch_fmcsa_hqs(crosswalk_carrier: str, output_key: str) -> list[Terminal]:
    """Look up each DOT number associated with `crosswalk_carrier` in the LTL
    crosswalk and pull its FMCSA census physical address. Returns one record
    per DOT (HQ / regional office)."""
    xw = pd.read_csv(OUT_DIR / "carrier_dot_crosswalk.csv")
    dots = xw[xw["ltl_carrier_name"].str.contains(
        crosswalk_carrier, case=False, regex=False, na=False
    )]["dot_number"].astype(str).tolist()
    if not dots:
        print(f"  no DOT matches for {crosswalk_carrier!r}")
        return []

    census = pd.read_csv(FMCSA_CENSUS, usecols=FMCSA_USECOLS, dtype=str,
                         low_memory=False)
    census["DOT_NUMBER"] = census["DOT_NUMBER"].astype(str)
    hits = census[census["DOT_NUMBER"].isin(dots)]

    def _s(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    out: list[Terminal] = []
    for _, row in hits.iterrows():
        zip5 = _s(row.get("PHY_ZIP"))[:5].zfill(5)
        if not zip5.isdigit():
            continue
        name = _s(row.get("DBA_NAME")) or _s(row.get("LEGAL_NAME"))
        state = _s(row.get("PHY_STATE")).upper()[:2]
        out.append(Terminal(
            carrier=output_key,
            code=str(row["DOT_NUMBER"]),
            name=name,
            street=_s(row.get("PHY_STREET")),
            city=_s(row.get("PHY_CITY")),
            state=state,
            zip=zip5,
        ))
    print(f"  {output_key}: {len(out)} HQ/regional ZIPs from {len(dots)} DOT #s")
    return out


def fetch_rl() -> list[Terminal]:
    return _fetch_fmcsa_hqs("R&L Carriers", "RL")


def fetch_xpo() -> list[Terminal]:
    return _fetch_fmcsa_hqs("XPO Logistics", "XPO")


def fetch_fedex_freight() -> list[Terminal]:
    # FedEx Economy / Priority are the two freight subdivisions.
    a = _fetch_fmcsa_hqs("FedEx Economy", "FEDEXFRT")
    b = _fetch_fmcsa_hqs("FedEx Priority", "FEDEXFRT")
    return a + b


# ---------------------------------------------------------------------------
# Registry & main
# ---------------------------------------------------------------------------

PARSERS: dict[str, Callable[[], list[Terminal]]] = {
    "SAIA": fetch_saia,
    "ESTES": fetch_estes,
    "ODFL": fetch_odfl,
    "RL": fetch_rl,
    "XPO": fetch_xpo,
    "FEDEXFRT": fetch_fedex_freight,
}


def main() -> None:
    all_terms: list[Terminal] = []
    for key, fn in PARSERS.items():
        print(f">> {key}")
        try:
            terms = fn()
        except Exception as e:
            print(f"   failed: {type(e).__name__}: {e}")
            continue
        print(f"   got {len(terms)} terminals")
        all_terms.extend(terms)

    if not all_terms:
        print("No terminals fetched.")
        return

    df = pd.DataFrame([asdict(t) for t in all_terms])
    df = df[df["zip"].str.match(r"^\d{5}$", na=False)].drop_duplicates(
        subset=["carrier", "zip", "code"]
    )
    df.to_csv(OUT_CSV, index=False)
    print(f"\nTotal terminals: {len(df)}")
    print(df.groupby("carrier").size())
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
