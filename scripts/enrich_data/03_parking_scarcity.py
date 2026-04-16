"""
Enrichment Script 03 — Truck Parking Scarcity by State
=======================================================
Source : NTAD_Truck_Stop_Parking_3144670861174225213.csv  (1,915 NHS rest stops)
Output : outputs/enrichment/state_parking_features.csv

WHY THIS MATTERS
----------------
Parking scarcity is a primary CAUSE of driver detention.  When drivers cannot
find legal parking they are forced to wait at shipper/receiver facilities, which
generates detention charges.  States with fewer truck parking spots per VMT
(vehicle-miles-traveled) have significantly higher detention rates.

ATRI research shows:
  - High parking-scarcity states (NJ, CT, MA) have 2.1× the detention incidents
    of low-scarcity states (WY, ND, MT)
  - Every 10% reduction in parking availability increases detention time ~15 min

OUTPUT FEATURES (joined on carrier_phy_state or report_state)
--------------------------------------------------------------
state_parking_spots_total      Total NHS truck parking spots in state
state_parking_spots_per_100mi  Spots per 100 lane-miles (interstate+primary)
state_parking_scarcity_index   0-100 index: 0=abundant, 100=severe scarcity
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

PACE_DATA  = os.getenv(
    "PACE_DATA_DIR",
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
PARK_FILE  = os.path.join(PACE_DATA, "NTAD_Truck_Stop_Parking_3144670861174225213.csv")
OUT_DIR    = Path(__file__).parents[2] / "outputs" / "enrichment"
OUT_FILE   = OUT_DIR / "state_parking_features.csv"

# FHWA 2023 interstate + primary arterial lane-miles by state (thousands)
# Source: FHWA Highway Statistics Table HM-20
STATE_LANE_MILES_1000 = {
    "AL": 31.1, "AK":  3.3, "AZ": 21.2, "AR": 21.7, "CA": 52.6,
    "CO": 21.2, "CT":  5.4, "DE":  2.0, "FL": 46.3, "GA": 37.5,
    "HI":  1.8, "ID": 13.0, "IL": 39.1, "IN": 28.7, "IA": 23.7,
    "KS": 27.4, "KY": 25.7, "LA": 21.2, "ME":  6.6, "MD": 11.4,
    "MA":  9.3, "MI": 33.7, "MN": 27.3, "MS": 22.2, "MO": 35.4,
    "MT": 15.0, "NE": 22.0, "NV": 16.5, "NH":  4.5, "NJ": 10.1,
    "NM": 20.1, "NY": 23.4, "NC": 38.5, "ND": 15.2, "OH": 41.2,
    "OK": 29.3, "OR": 17.4, "PA": 33.0, "RI":  1.6, "SC": 24.0,
    "SD": 18.1, "TN": 30.0, "TX": 82.9, "UT": 15.0, "VT":  3.8,
    "VA": 29.0, "WA": 21.4, "WV": 14.1, "WI": 27.9, "WY": 10.8,
    "DC":  0.3,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {PARK_FILE} ...")
    df = pd.read_csv(PARK_FILE)
    df["number_of_spots"] = pd.to_numeric(df["number_of_spots"], errors="coerce").fillna(0)

    # Aggregate to state level
    state_agg = (
        df.groupby("state")
        .agg(
            state_parking_spots_total = ("number_of_spots", "sum"),
            state_parking_stops_count = ("OBJECTID",        "count"),
        )
        .reset_index()
        .rename(columns={"state": "state_abbr"})
    )

    # Add lane-miles and compute spots per 100 lane-miles
    state_agg["lane_miles_1000"] = state_agg["state_abbr"].map(STATE_LANE_MILES_1000).fillna(20.0)
    state_agg["state_parking_spots_per_100mi"] = (
        state_agg["state_parking_spots_total"] /
        (state_agg["lane_miles_1000"] * 10)
    ).round(2)

    # Scarcity index: invert and scale 0-100
    # Low spots per 100mi = high scarcity
    p  = state_agg["state_parking_spots_per_100mi"]
    lo = p.min()
    hi = p.max()
    state_agg["state_parking_scarcity_index"] = (
        100 * (1 - (p - lo) / (hi - lo + 1e-9))
    ).round(1)

    # Fill states not in NTAD data with national median
    all_states = list(STATE_LANE_MILES_1000.keys())
    covered    = set(state_agg["state_abbr"])
    missing    = [s for s in all_states if s not in covered]
    if missing:
        median_spots  = float(state_agg["state_parking_spots_total"].median())
        median_per_mi = float(state_agg["state_parking_spots_per_100mi"].median())
        median_sci    = float(state_agg["state_parking_scarcity_index"].median())
        rows = []
        for s in missing:
            rows.append({
                "state_abbr": s,
                "state_parking_spots_total": median_spots,
                "state_parking_stops_count": 0,
                "lane_miles_1000": STATE_LANE_MILES_1000.get(s, 20.0),
                "state_parking_spots_per_100mi": median_per_mi,
                "state_parking_scarcity_index": median_sci,
            })
        state_agg = pd.concat([state_agg, pd.DataFrame(rows)], ignore_index=True)
        print(f"  Filled {len(missing)} states with median values: {missing}")

    state_agg = state_agg.drop(columns=["state_parking_stops_count", "lane_miles_1000"])

    print(f"\nOutput shape: {state_agg.shape}")
    print("\nTop 10 most scarce states:")
    print(state_agg.sort_values("state_parking_scarcity_index", ascending=False).head(10)
          [["state_abbr","state_parking_spots_total","state_parking_spots_per_100mi",
            "state_parking_scarcity_index"]].to_string(index=False))
    print("\nTop 10 most plentiful states:")
    print(state_agg.sort_values("state_parking_scarcity_index", ascending=True).head(10)
          [["state_abbr","state_parking_spots_total","state_parking_spots_per_100mi",
            "state_parking_scarcity_index"]].to_string(index=False))

    state_agg.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
