# External / Manual-Pull Blockers

Features that depend on data outside PACE DATA or the LTL CSV. Status as of 2026-04-19.

## 1. Carrier terminal service ZIPs — RESOLVED (all 6 covered)

**Goal:** Per-carrier set of ZIPs where they have a service center, for features
`carrier_has_terminal_in_origin_zip`, `carrier_has_terminal_in_dest_zip`,
`dist_to_nearest_carrier_terminal`.

**Status:** `scrape_carrier_terminals.py` is production, 719 terminals total in
`carrier_terminal_zips.csv`:

| Carrier      | Count | Source |
|--------------|-------|--------|
| SAIA         | 215   | Playwright lifts Contentful auth token, paginates Delivery API (`content_type=terminal`). Precise street+ZIP. |
| ESTES        | 224   | Public sitemap `/api/vtl/sitemap/sitemap.xml` → 226 terminal pages, regex-extract `<strong>Terminal:</strong>` + `<strong>Address:</strong>` blocks. |
| ODFL         | 272   | Sitemap gives 285 (state, code). OSM Overpass (`operator='Old Dominion'`) matches 22 via nearest-ZIP on lat/lon. Remaining 263 resolved by Google Places API (New) text-search: 250 / 263 precise ZIPs. |
| XPO          | 5     | FMCSA Company Census lookup by DOT number (via `carrier_dot_crosswalk.csv`) — HQ / regional office addresses. |
| FedEx Freight| 2     | FMCSA census (FedEx Economy + FedEx Priority DOTs). |
| R&L          | 1     | FMCSA census (HQ only). |

Google Places key lives in `.env` as `GOOGLE_PLACES_API_KEY` and is autoloaded
by `python-dotenv`. Restricted to Places API (New) with a billing cap; 263
lookups cost ~$1.30 against the $200/mo free credit.

Limitation: XPO / FedEx / R&L coverage is HQ-biased (1-5 ZIPs per carrier
rather than a full network). Features using these will be weaker for
those 3 carriers than for SAIA / Estes / ODFL. No public terminal directory
exists for them — this is the ceiling without pentesting their locator auth.

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

## 5. OSRM road distance — RESOLVED

**Status (2026-04-22):** Pivoted from local OSRM build to public demo server
`router.project-osrm.org` after full-US PBF extract OOM'd on 32 GB Windows box
(WSL2 capped at 24 GB) and GPU cluster had no Docker/Apptainer/sudo.

`osrm_batch_distance.py` ran 6,357 pairs at ~1.1 s/req (conservative rate limit),
total wall time ~2h. Result: **6,328 / 6,462 pairs resolved (97.9%)**.
`merge_osrm.py` joined onto training: **24,210 / 24,510 rows (98.78%)** have real
OSRM miles/minutes. The 300 misses are Canadian postal codes and other invalid
ZIPs that also fail haversine. Road/haversine ratio median = 1.175, p95 = 1.298
(textbook CONUS detour factor).

Artifacts:
- `od_zip_pairs_osrm.csv` — raw OSRM output (6,462 pairs).
- `enriched_ltl_training.csv` — joined columns `osrm_miles`, `osrm_minutes`,
  `osrm_imputed` (0 for real, 1 for haversine-fallback; currently all 0 since
  training file lacks haversine column — tree models handle NaN natively).
