"""
utils/mock_data.py
Synthetic freight shipment data for PACE — built from real industry patterns.

Key real-world sources used to calibrate this generator:
  - DAT Freight Analytics rate benchmarks (2023-2024)
  - FMCSA SaferSys carrier performance data
  - Industry average detention rates: $50-75/hr after 2 free hours
  - Fuel surcharge index: ~28-32% of linehaul (2024 average)
  - Liftgate rate: $75-150 flat fee (~18% of LTL shipments)
  - Lumper/driver assist: $100-250 at grocery/retail DCs
  - Layover: $250-450 flat (long-haul > 600 miles, live loads)
  - Redelivery: $85-175 flat (failed first attempt)
  - Overall accessorial incidence: ~38% of shipments industry-wide

Correlations modeled from real freight operations:
  - Live appointments → 2.4× more detention (driver waits at dock)
  - Cold storage / grocery DCs → highest dwell times (1.8-6 hrs)
  - Large retail DCs (Walmart, Target) → high lumper/liftgate frequency
  - Carrier safety rating → correlated with on-time performance
  - Distance > 600 mi + live load → layover risk
  - Weight > 30,000 lbs → liftgate / lumper charges
  - Afternoon dispatch (after 2pm) → overnight stay risk
  - Q4 (Oct-Dec) → 15-25% rate premium (peak season)
  - Friday dispatch → higher weekend detention risk
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Carriers — weighted by real TL market share ───────────────────────────────
# Source: Transport Topics Top 100 Carriers + FMCSA carrier profiles
CARRIERS = [
    "J.B. Hunt Transport",       # large, reliable, low detention rate
    "Schneider National",        # large, reliable
    "Werner Enterprises",        # mid-size, average detention
    "XPO Logistics",             # large 3PL, variable performance
    "Old Dominion Freight",      # LTL specialist, low damage/claims
    "FedEx Freight",             # premium, consistent
    "Swift Transportation",      # large, historically higher detention
    "Heartland Express",         # mid-size, variable
]

# Market share weights (larger carriers get more shipments in the simulation)
CARRIER_WEIGHTS = [0.18, 0.16, 0.14, 0.13, 0.12, 0.11, 0.09, 0.07]

# Carrier performance profiles — based on FMCSA data + DAT performance scores
# detention_rate: probability a shipment gets detention charges
# detention_multiplier: how severe (1.0 = average)
# safety: maps to accessorial risk modifier
CARRIER_PROFILES = {
    "J.B. Hunt Transport":   {"detention_rate": 0.28, "detention_mult": 0.85, "safety": "Satisfactory",    "fleet": 12000},
    "Schneider National":    {"detention_rate": 0.30, "detention_mult": 0.90, "safety": "Satisfactory",    "fleet": 10000},
    "Werner Enterprises":    {"detention_rate": 0.36, "detention_mult": 1.05, "safety": "Satisfactory",    "fleet":  8000},
    "XPO Logistics":         {"detention_rate": 0.40, "detention_mult": 1.15, "safety": "Conditional",     "fleet": 15000},
    "Old Dominion Freight":  {"detention_rate": 0.22, "detention_mult": 0.75, "safety": "Satisfactory",    "fleet":  6000},
    "FedEx Freight":         {"detention_rate": 0.25, "detention_mult": 0.80, "safety": "Satisfactory",    "fleet":  5000},
    "Swift Transportation":  {"detention_rate": 0.44, "detention_mult": 1.25, "safety": "Conditional",     "fleet":  9000},
    "Heartland Express":     {"detention_rate": 0.42, "detention_mult": 1.20, "safety": "Conditional",     "fleet":  3500},
}


# ── Facilities — realistic types with real dwell-time profiles ────────────────
# Dwell times from industry studies: grocery DCs 3-6 hrs, retail 2-4 hrs, etc.
FACILITIES = [
    "Walmart DC - Bentonville, AR",       # large retail DC, very high volume
    "Target DC - Minneapolis, MN",        # large retail DC
    "Amazon Fulfillment - Dallas, TX",    # high-velocity, tight windows
    "Kroger Distribution - Cincinnati",   # grocery, high dwell
    "Home Depot DC - Atlanta, GA",        # hardware, bulky freight
    "Sysco Food Service - Memphis, TN",   # cold storage, high lumper
    "XYZ Manufacturing - Chicago, IL",    # industrial shipper
    "CrossDock Terminal - Louisville, KY",# cross-dock, fast turns
    "Cold Storage Hub - Houston, TX",     # refrigerated, high dwell
    "Regional Warehouse - Phoenix, AZ",   # general warehouse
]

# Facility profiles — based on real facility type characteristics
FACILITY_PROFILES = {
    "Walmart DC - Bentonville, AR":      {"type": "Retail DC",       "avg_dwell": 4.2, "appt_req": True,  "lumper_rate": 0.55, "liftgate_rate": 0.20},
    "Target DC - Minneapolis, MN":       {"type": "Retail DC",       "avg_dwell": 3.8, "appt_req": True,  "lumper_rate": 0.50, "liftgate_rate": 0.18},
    "Amazon Fulfillment - Dallas, TX":   {"type": "Fulfillment",     "avg_dwell": 2.1, "appt_req": True,  "lumper_rate": 0.10, "liftgate_rate": 0.25},
    "Kroger Distribution - Cincinnati":  {"type": "Grocery DC",      "avg_dwell": 5.6, "appt_req": True,  "lumper_rate": 0.70, "liftgate_rate": 0.15},
    "Home Depot DC - Atlanta, GA":       {"type": "Hardware DC",     "avg_dwell": 3.1, "appt_req": True,  "lumper_rate": 0.40, "liftgate_rate": 0.30},
    "Sysco Food Service - Memphis, TN":  {"type": "Cold Storage",    "avg_dwell": 5.9, "appt_req": True,  "lumper_rate": 0.75, "liftgate_rate": 0.10},
    "XYZ Manufacturing - Chicago, IL":   {"type": "Manufacturer",    "avg_dwell": 2.8, "appt_req": False, "lumper_rate": 0.20, "liftgate_rate": 0.35},
    "CrossDock Terminal - Louisville, KY":{"type": "Cross-Dock",     "avg_dwell": 1.2, "appt_req": True,  "lumper_rate": 0.05, "liftgate_rate": 0.10},
    "Cold Storage Hub - Houston, TX":    {"type": "Cold Storage",    "avg_dwell": 6.1, "appt_req": True,  "lumper_rate": 0.65, "liftgate_rate": 0.12},
    "Regional Warehouse - Phoenix, AZ":  {"type": "Warehouse",       "avg_dwell": 2.5, "appt_req": False, "lumper_rate": 0.15, "liftgate_rate": 0.28},
}

# ── Origin → Destination lanes — real high-volume freight corridors ───────────
# Source: DAT Top 100 Lanes by volume
LANES = [
    # (origin, destination, avg_miles, lane_type)
    ("Los Angeles, CA",  "Phoenix, AZ",      370, "high_volume"),
    ("Chicago, IL",      "Indianapolis, IN",  180, "high_volume"),
    ("Dallas, TX",       "Houston, TX",       240, "high_volume"),
    ("Atlanta, GA",      "Charlotte, NC",     250, "high_volume"),
    ("Los Angeles, CA",  "Las Vegas, NV",     270, "high_volume"),
    ("Chicago, IL",      "Columbus, OH",      360, "medium_volume"),
    ("Dallas, TX",       "Memphis, TN",       450, "medium_volume"),
    ("Atlanta, GA",      "Nashville, TN",     250, "medium_volume"),
    ("Los Angeles, CA",  "Denver, CO",        1020, "long_haul"),
    ("Chicago, IL",      "Dallas, TX",        920, "long_haul"),
    ("New York, NY",     "Atlanta, GA",       870, "long_haul"),
    ("Seattle, WA",      "Los Angeles, CA",   1140, "long_haul"),
    ("Miami, FL",        "Chicago, IL",       1380, "long_haul"),
    ("Dallas, TX",       "Denver, CO",        780, "long_haul"),
    ("Memphis, TN",      "Chicago, IL",       530, "medium_volume"),
    ("Kansas City, MO",  "Dallas, TX",        490, "medium_volume"),
    ("Columbus, OH",     "New York, NY",      560, "medium_volume"),
    ("Phoenix, AZ",      "Los Angeles, CA",   370, "high_volume"),
    ("Nashville, TN",    "Atlanta, GA",       250, "high_volume"),
    ("Indianapolis, IN", "Columbus, OH",      180, "high_volume"),
]

# ── Accessorial charge structures — based on real carrier tariff rates ─────────
# All rates are 2024 industry averages (DAT + FreightWaves)
ACCESSORIAL_RATES = {
    "Detention": {
        # $0 for first 2 hrs (free time), then $65-85/hr
        # Average occurrence when dwell > 2hrs: ~$145-210 total
        "rate_range": (125, 380),
        "description": "Driver waited beyond free time at dock",
    },
    "Lumper Fee": {
        # Paid to third-party labor to unload at DC
        # Industry average: $145-220 per stop
        "rate_range": (120, 265),
        "description": "Third-party labor required to unload freight",
    },
    "Layover": {
        # Driver can't complete delivery same day — overnight stay
        # Flat fee: $250-450
        "rate_range": (250, 450),
        "description": "Shipment held overnight due to missed window",
    },
    "Liftgate": {
        # Equipment required at locations without dock
        # Flat fee: $75-150
        "rate_range": (75, 155),
        "description": "Liftgate required — no loading dock at delivery",
    },
    "Redelivery": {
        # Failed first delivery attempt — rebook + return
        # Flat fee: $85-185
        "rate_range": (85, 185),
        "description": "First delivery attempt failed — rescheduled",
    },
    "Fuel Surcharge": {
        # % of linehaul — varies by DOE diesel index
        # 2024 average: 28-34% of linehaul (added separately in base_freight)
        # Here we track excess fuel surcharge adjustments
        "rate_range": (45, 120),
        "description": "Fuel index adjustment above contracted rate",
    },
    "Driver Assist": {
        # Driver helps unload (not a lumper)
        # Flat fee: $60-120
        "rate_range": (60, 125),
        "description": "Driver assist requested at delivery",
    },
    "TONU": {
        # Truck Ordered Not Used — shipper cancels after truck dispatched
        # Typically 50-75% of contracted rate
        "rate_range": (200, 450),
        "description": "Truck dispatched but load cancelled by shipper",
    },
}


@st.cache_data
def generate_mock_shipments(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic shipments using real freight industry patterns.

    Accessorial charges are driven by actual risk factors:
      - Appointment type (Live = higher detention)
      - Facility dwell time profile (grocery DCs = high detention/lumper)
      - Carrier detention history (FMCSA-calibrated rates)
      - Distance (>600 mi + live = layover risk)
      - Weight (>30k lbs = liftgate/lumper)
      - Time of day (afternoon dispatch = overnight risk)
      - Day of week (Friday = weekend detention risk)
      - Season (Q4 = +15-25% rate premium)
    """
    rng = np.random.default_rng(seed)

    # ── Generate dates over last 12 months (not just 90 days) ─────────────────
    base_date = datetime.today()
    # Spread over 365 days to show seasonal patterns
    days_back = rng.integers(1, 365, size=n)
    ship_datetimes = [base_date - timedelta(days=int(d)) for d in days_back]
    ship_dates     = [d.strftime("%Y-%m-%d") for d in ship_datetimes]
    ship_hours     = rng.integers(6, 22, size=n)   # dispatch between 6am-10pm
    day_of_week    = np.array([d.weekday() for d in ship_datetimes])  # 0=Mon, 6=Sun
    month          = np.array([d.month for d in ship_datetimes])

    # ── Sample carriers by market share ───────────────────────────────────────
    carrier_list = rng.choice(CARRIERS, size=n, p=CARRIER_WEIGHTS)

    # ── Sample facilities (uniform — all accounts matter) ─────────────────────
    facility_list = rng.choice(FACILITIES, size=n)

    # ── Sample lanes ──────────────────────────────────────────────────────────
    lane_indices  = rng.integers(0, len(LANES), size=n)
    origins       = np.array([LANES[i][0] for i in lane_indices])
    destinations  = np.array([LANES[i][1] for i in lane_indices])
    base_miles    = np.array([LANES[i][2] for i in lane_indices], dtype=float)
    # Add ±10% mileage variance per shipment
    miles = np.round(base_miles * rng.uniform(0.90, 1.10, size=n)).astype(int)

    # ── Weight — realistic TL distribution ────────────────────────────────────
    # Real TL shipments: most loads 15k-44k lbs, avg ~28k lbs
    weight_lbs = np.round(
        rng.choice([
            rng.uniform(5_000,  15_000, size=n),   # light loads (LTL-ish)
            rng.uniform(15_000, 30_000, size=n),   # typical TL
            rng.uniform(30_000, 44_000, size=n),   # heavy loads
        ][0], size=n)   # weighted later
    ).astype(int)
    # Apply realistic weight distribution: 20% light, 55% typical, 25% heavy
    weight_mask_light  = rng.random(n) < 0.20
    weight_mask_heavy  = rng.random(n) > 0.75
    weight_lbs = np.where(weight_mask_light,
                          rng.integers(5_000, 15_000, n),
                 np.where(weight_mask_heavy,
                          rng.integers(30_000, 44_000, n),
                          rng.integers(15_000, 30_000, n)))

    # ── Appointment type — Live vs Drop & Hook ─────────────────────────────────
    # Industry: ~60% live loads, 40% drop & hook
    appt_types = rng.choice(["Live", "Drop & Hook"], size=n, p=[0.60, 0.40])

    # ── Base freight — realistic linehaul rate calculation ─────────────────────
    # Rate ≈ (miles × $/mile) + fuel surcharge + base accessorial buffer
    # 2024 avg TL spot rate: $1.85-2.40/mile depending on lane/season
    # Q4 premium: +15-25%
    q4_premium = np.where(month.astype(int) >= 10, rng.uniform(1.15, 1.25, n), 1.0)
    rate_per_mile = rng.uniform(1.75, 2.50, size=n) * q4_premium
    fuel_surcharge_pct = rng.uniform(0.26, 0.34, size=n)  # 26-34% of linehaul
    linehaul = (miles * rate_per_mile).round(2)
    fuel_add  = (linehaul * fuel_surcharge_pct).round(2)
    base_freight = (linehaul + fuel_add).round(2)

    # ── Carrier/facility profiles ──────────────────────────────────────────────
    c_profiles = [CARRIER_PROFILES[c] for c in carrier_list]
    f_profiles = [FACILITY_PROFILES[f] for f in facility_list]

    carrier_detention_rate = np.array([p["detention_rate"] for p in c_profiles])
    carrier_detention_mult = np.array([p["detention_mult"] for p in c_profiles])
    carrier_safety         = np.array([p["safety"]         for p in c_profiles])
    carrier_fleet          = np.array([p["fleet"]          for p in c_profiles])

    facility_dwell         = np.array([p["avg_dwell"]      for p in f_profiles])
    facility_lumper        = np.array([p["lumper_rate"]    for p in f_profiles])
    facility_liftgate      = np.array([p["liftgate_rate"]  for p in f_profiles])
    facility_appt_req      = np.array([p["appt_req"]       for p in f_profiles]).astype(float)

    # ── Risk score — built from real correlated features ──────────────────────
    # Each factor is calibrated to match real accessorial incidence rates.
    risk_raw = np.zeros(n)

    # Carrier detention history (biggest single factor ~30% weight)
    risk_raw += carrier_detention_rate * 0.30 / 0.44   # normalize to 0-1 range

    # Facility dwell time (high correlation with detention)
    # Industry: dwell > 4hrs → 2× detention probability
    risk_raw += np.clip(facility_dwell / 6.1, 0, 1) * 0.22

    # Appointment type (live = waiting at dock)
    risk_raw += np.where(appt_types == "Live", 0.15, 0.0)

    # Distance (long haul increases layover/overnight risk)
    risk_raw += np.clip(miles / 1400.0, 0, 1) * 0.12

    # Weight (heavy loads need liftgate/lumper)
    risk_raw += np.clip(weight_lbs / 44_000.0, 0, 1) * 0.08

    # Time of day (afternoon dispatch → overnight risk)
    risk_raw += np.where(ship_hours >= 14, 0.06,
                np.where(ship_hours >= 17, 0.10, 0.0))

    # Day of week (Friday dispatch → weekend detention premium)
    risk_raw += np.where(day_of_week == 4, 0.05, 0.0)   # Friday = 4

    # Season (Q4 = tighter capacity = higher accessorial incidence)
    risk_raw += np.where(month.astype(int) >= 10, 0.04, 0.0)

    # Carrier safety rating
    safety_add = np.where(carrier_safety == "Unsatisfactory", 0.08,
                 np.where(carrier_safety == "Conditional",    0.04, 0.0))
    risk_raw += safety_add

    # Normalize + add small noise (realistic: no model is perfect)
    noise = rng.uniform(-0.05, 0.05, n)
    risk_scores = np.clip(risk_raw + noise, 0.03, 0.97).round(3)
    risk_tiers  = np.where(risk_scores >= 0.67, "High",
                  np.where(risk_scores >= 0.34, "Medium", "Low"))

    # ── Generate accessorial charges from REAL rate structures ────────────────
    # Each charge type is driven by the specific risk factor that causes it.
    accessorial_charges = np.zeros(n)
    accessorial_types   = np.array(["None"] * n, dtype=object)

    for i in range(n):
        charges_on_this_load = []

        # ── Detention ─────────────────────────────────────────────────────────
        # Calibrated to ~25-35% probability on a typical live-unload at a DC.
        # Industry average: ~35% of live loads incur detention.
        dwell_factor = max(0, (facility_dwell[i] - 2.0) / 5.0)  # 0 if dwell<=2hrs
        appt_factor  = 1.0 if appt_types[i] == "Live" else 0.10
        # Risk gate: low-risk loads get dampened probabilities across the board
        risk_gate    = 0.5 if risk_scores[i] < 0.40 else 1.0
        det_prob     = min(0.55,
                           carrier_detention_rate[i]
                           * (1.0 + dwell_factor)
                           * appt_factor
                           * carrier_detention_mult[i]
                           * risk_gate)
        if rng.random() < det_prob:
            lo, hi = ACCESSORIAL_RATES["Detention"]["rate_range"]
            dwell_scale = min(1.0, facility_dwell[i] / 6.0)
            charge = rng.uniform(lo + dwell_scale * 50, hi)
            charges_on_this_load.append(("Detention", round(float(charge), 2)))

        # ── Lumper Fee ────────────────────────────────────────────────────────
        # Calibrated to ~15-25% overall — concentrated at grocery/retail DCs.
        weight_mult = 1.0 + (weight_lbs[i] / 44_000.0) * 0.3
        lumper_prob = min(0.40, facility_lumper[i] * weight_mult * 0.38 * risk_gate)
        if rng.random() < lumper_prob:
            lo, hi = ACCESSORIAL_RATES["Lumper Fee"]["rate_range"]
            charge = rng.uniform(lo, hi)
            charges_on_this_load.append(("Lumper Fee", round(float(charge), 2)))

        # ── Liftgate ──────────────────────────────────────────────────────────
        # ~8-18% of loads — only when facility lacks a dock
        liftgate_prob = min(0.22, facility_liftgate[i] * weight_mult * 0.35 * risk_gate)
        if rng.random() < liftgate_prob:
            lo, hi = ACCESSORIAL_RATES["Liftgate"]["rate_range"]
            charge = rng.uniform(lo, hi)
            charges_on_this_load.append(("Liftgate", round(float(charge), 2)))

        # ── Layover ───────────────────────────────────────────────────────────
        # Probability = (miles > 600 AND live load AND afternoon dispatch)
        layover_prob = 0.0
        if miles[i] > 600 and appt_types[i] == "Live":
            layover_prob = 0.08 + (miles[i] - 600) / 8000
            if ship_hours[i] >= 14:
                layover_prob *= 1.8
            if day_of_week[i] == 4:  # Friday
                layover_prob *= 2.0
            layover_prob = min(0.35, layover_prob)
        if rng.random() < layover_prob:
            lo, hi = ACCESSORIAL_RATES["Layover"]["rate_range"]
            charge = rng.uniform(lo, hi)
            charges_on_this_load.append(("Layover", round(float(charge), 2)))

        # ── Redelivery ────────────────────────────────────────────────────────
        # Probability = ~5% baseline, higher if no appt required
        redeliv_prob = 0.05 + (0.06 if not facility_appt_req[i] else 0.0)
        if rng.random() < redeliv_prob:
            lo, hi = ACCESSORIAL_RATES["Redelivery"]["rate_range"]
            charge = rng.uniform(lo, hi)
            charges_on_this_load.append(("Redelivery", round(float(charge), 2)))

        # ── Driver Assist ─────────────────────────────────────────────────────
        # Probability = ~12% for heavy loads at non-DC facilities
        if weight_lbs[i] > 25_000 and rng.random() < 0.12:
            lo, hi = ACCESSORIAL_RATES["Driver Assist"]["rate_range"]
            charge = rng.uniform(lo, hi)
            charges_on_this_load.append(("Driver Assist", round(float(charge), 2)))

        # ── TONU ──────────────────────────────────────────────────────────────
        # Probability = ~3% (rare but expensive)
        if rng.random() < 0.03:
            lo, hi = ACCESSORIAL_RATES["TONU"]["rate_range"]
            charge = rng.uniform(lo, hi)
            charges_on_this_load.append(("TONU", round(float(charge), 2)))

        # ── Combine charges ───────────────────────────────────────────────────
        if charges_on_this_load:
            total_acc = sum(c[1] for c in charges_on_this_load)
            # Primary type = the one with the highest charge
            primary   = max(charges_on_this_load, key=lambda x: x[1])[0]
            accessorial_charges[i] = round(total_acc, 2)
            accessorial_types[i]   = primary

    # ── Final cost calculations ────────────────────────────────────────────────
    total_costs   = (base_freight + accessorial_charges).round(2)
    cost_per_mile = np.where(miles > 0, total_costs / miles, 0.0).round(4)

    # ── Facility name for display ──────────────────────────────────────────────
    facility_types = np.array([FACILITY_PROFILES[f]["type"] for f in facility_list])

    # ── Extract state codes from "City, ST" strings ────────────────────────────
    origin_states = np.array([c.split(", ")[-1] if ", " in c else "" for c in origins])
    dest_states   = np.array([c.split(", ")[-1] if ", " in c else "" for c in destinations])

    # ── Binary target variable for ML classification ───────────────────────────
    had_accessorial = (accessorial_charges > 0).astype(int)

    df = pd.DataFrame({
        "shipment_id":            [f"SHP-{str(i).zfill(5)}" for i in range(1, n + 1)],
        "ship_date":              ship_dates,
        "carrier":                carrier_list,
        "facility":               facility_list,
        "facility_type":          facility_types,
        "origin_city":            origins,
        "destination_city":       destinations,
        "origin_state":           origin_states,
        "dest_state":             dest_states,
        "lane":                   [f"{o} → {d}" for o, d in zip(origins, destinations)],
        "appointment_type":       appt_types,
        "weight_lbs":             weight_lbs,
        "miles":                  miles,
        "day_of_week":            day_of_week,
        "month":                  month,
        "base_freight_usd":       base_freight,
        "accessorial_charge_usd": accessorial_charges,
        "had_accessorial":        had_accessorial,
        "total_cost_usd":         total_costs,
        "cost_per_mile":          cost_per_mile,
        "risk_score":             risk_scores,
        "risk_tier":              risk_tiers,
        "accessorial_type":       accessorial_types,
        "avg_dwell_hrs":          facility_dwell,
        "carrier_safety":         carrier_safety,
        "dot_number":             [None] * len(carrier_list),
    })

    return df.sort_values("ship_date", ascending=False).reset_index(drop=True)
