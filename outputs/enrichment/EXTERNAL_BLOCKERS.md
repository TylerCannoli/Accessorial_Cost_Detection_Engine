# External / Manual-Pull Blockers

Features that depend on data outside PACE DATA or the LTL CSV. Status as of 2026-04-19.

## 1. Carrier terminal service ZIPs — BLOCKED

**Goal:** Per-carrier set of ZIPs where they have a service center, for features
`carrier_has_terminal_in_origin_zip`, `carrier_has_terminal_in_dest_zip`,
`dist_to_nearest_carrier_terminal`.

**Why blocked:** All major LTL carrier locators (SAIA, Estes, R&L, XPO, ODFL,
FedEx Freight, TForce) are Next.js / React SPAs that render terminal lists from
XHR calls authenticated with anti-bot tokens. HTTP `GET` on their public URLs
returns an empty HTML shell. `scrape_carrier_terminals.py` is scaffolded but the
parsers need one of these follow-ups:

- **Option A (fast):** Manually open each locator in a browser, copy the XHR JSON
  response into `terminals_raw/<carrier>.json`, then post-process.
- **Option B (robust):** Add Playwright and do `page.goto()` → wait for network idle
  → read window object / intercept fetch. Cost: headless Chromium in the pipeline.
- **Option C (pragmatic):** Use the FMCSA carrier census "terminals" field for
  registered authority terminals — HQ-biased but covers most carriers.

## 2. Appointment-required flag — EXTERNAL TMS

**Goal:** Binary flag `appointment_required` per shipment (drives C/Appointment Notif.,
C/Appointment Delivery which fire on 1.6% / 0.3% of rows).

**Why blocked:** This field exists upstream in the TMS order entry (BOL stop
instructions, shipper notes), not in the LTL CSV export. No public data source.

**Fix path:** Re-pull from TMS with `Stop Instructions` / `Special Instructions`
columns included, then regex-extract `appt|appointment|call before|schedule delivery`.
Until available, the model will infer from destination features (residential proxy,
time-of-day, commodity class).

## 3. NCEI weather 2025-2026 — NEEDS TOKEN

**Goal:** Fresh daily TMIN/TMAX/PRCP/SNOW/AWND for shipment dates (current
`ncei_daily_combined.csv` ends 2024-12-31, LTL starts 2025-01-02, hit rate = 0%).

**Why blocked:** NOAA CDO v2 API requires a free token.

**Fix path:**
1. Register at https://www.ncdc.noaa.gov/cdo-web/token (instant issuance).
2. `export CDO_API_TOKEN=<token>` (or set in .env).
3. `python pull_ncei_weather.py` — script is written, resume-aware, respects 5 req/s
   rate limit. Pulls 20 representative stations × 2 years ≈ 40 API calls.

## 4. Consignee facility type — PARTIALLY RESOLVED

**Goal:** Distinguish commercial vs residential delivery addresses.

**Status:** `enrich_epa_frs.py` completed via EPA FRS NATIONAL_SINGLE.CSV (offline,
4M facilities). Added `dest_frs_facilities`, `dest_frs_stationary`, `frs_residential_dest`
(0.9% rate — much more precise than CBP's 3.6%).

**Still external:** USPS DPV (Delivery Point Validation) would disambiguate rural-
route-vs-commercial more precisely but requires a USPS API Certified Mailer account.

## 5. OSRM road distance — IN PROGRESS

**Status:** WSL Ubuntu 24.04 + Docker confirmed. `us-latest.osm.pbf` (11.85 GB)
downloading from Geofabrik (~25% at 2026-04-19 12:14 CT). Once complete:
1. `bash setup_osrm.sh prep` — extract/partition/customize (~30-60 min, 8-12 GB RAM).
2. `bash setup_osrm.sh serve` — start server on port 5000.
3. `python osrm_batch_distance.py` — 6,462 O/D pairs at ~50 req/s ≈ 2 min.

Outputs `od_zip_pairs_osrm.csv` with `osrm_miles` and `osrm_minutes`. Then merge
into training data alongside `haversine_miles`.
