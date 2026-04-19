"""Fetch carrier terminal addresses and convert to ZIP-level coverage maps.

Scope: major LTL carriers that appear in the LTL training data. For each carrier,
hit a publicly-accessible terminal locator / service center API and record every
terminal ZIP. Output `carrier_terminal_zips.csv` drives downstream features:

  - carrier_has_terminal_in_origin_zip      (bool per row)
  - carrier_has_terminal_in_dest_zip
  - dist_to_nearest_carrier_terminal        (miles, via terminal lat/lon vs ZIP lat/lon)

Endpoints change often — each parser is in its own function so failures are isolated.
Robots.txt: all listed locators are public web pages or JSON endpoints that serve
customer-facing content. Rate-limited to 1 req/sec per host.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "carrier_terminal_zips.csv"

UA = "Mozilla/5.0 (research / academic; contact claytonrjosef@gmail.com)"
HEADERS = {"User-Agent": UA, "Accept": "application/json, text/html;q=0.9"}


@dataclass
class Terminal:
    carrier: str
    code: str
    name: str
    city: str
    state: str
    zip: str


def fetch_saia(session: requests.Session) -> list[Terminal]:
    """SAIA publishes terminal list via their locations API."""
    url = "https://www.saia.com/api/services/terminals"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []
    out: list[Terminal] = []
    for t in data if isinstance(data, list) else data.get("terminals", []):
        out.append(Terminal("SAIA", str(t.get("code", "")), t.get("name", ""),
                            t.get("city", ""), t.get("state", ""),
                            str(t.get("zip", ""))[:5].zfill(5)))
    return out


def fetch_estes(session: requests.Session) -> list[Terminal]:
    """Estes terminal directory (HTML page with embedded JSON)."""
    url = "https://www.estes-express.com/resources/terminal-finder"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        m = re.search(r'window\.__TERMINALS__\s*=\s*(\[.*?\]);', r.text, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(1))
    except (requests.RequestException, json.JSONDecodeError):
        return []
    return [Terminal("ESTES", str(t.get("terminalCode", "")), t.get("name", ""),
                     t.get("city", ""), t.get("state", ""),
                     str(t.get("postalCode", ""))[:5].zfill(5)) for t in data]


def fetch_rl_carriers(session: requests.Session) -> list[Terminal]:
    """R&L Carriers service center list."""
    url = "https://www.rlcarriers.com/api/ServiceCenters"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []
    return [Terminal("R&L", str(t.get("centerCode", "")), t.get("name", ""),
                     t.get("city", ""), t.get("state", ""),
                     str(t.get("zip", ""))[:5].zfill(5)) for t in data]


def fetch_xpo(session: requests.Session) -> list[Terminal]:
    """XPO/RXO terminal map data."""
    url = "https://xpo.com/api/terminal-locations"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []
    return [Terminal("XPO", str(t.get("id", "")), t.get("name", ""),
                     t.get("city", ""), t.get("state", ""),
                     str(t.get("zipCode", ""))[:5].zfill(5)) for t in data]


PARSERS = {
    "SAIA": fetch_saia,
    "ESTES": fetch_estes,
    "R&L": fetch_rl_carriers,
    "XPO": fetch_xpo,
}


def main() -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    all_terms: list[Terminal] = []
    for carrier, fn in PARSERS.items():
        print(f">> {carrier}")
        try:
            terms = fn(session)
        except Exception as e:  # parsers are best-effort
            print(f"   failed: {e}")
            terms = []
        print(f"   got {len(terms)} terminals")
        all_terms.extend(terms)
        time.sleep(1)

    if not all_terms:
        print("\nNo terminals fetched. Endpoints may have moved — extend PARSERS.")
        return

    df = pd.DataFrame([t.__dict__ for t in all_terms])
    df = df[df["zip"].str.match(r"^\d{5}$", na=False)].drop_duplicates()
    df.to_csv(OUT_CSV, index=False)
    print(f"\nTotal terminals: {len(df)}")
    print(df.groupby("carrier").size())
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
