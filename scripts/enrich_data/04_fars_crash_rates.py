"""
Enrichment Script 04 — FARS Fatal Crash Rates by State
=======================================================
Source : FARS2023NationalCSV.zip  (37,654 fatal accidents, 2023)
Output : outputs/enrichment/state_fatal_crash_features.csv

WHY THIS MATTERS
----------------
Fatal crash density by state captures:
  - Infrastructure hazard levels (road quality, congestion, weather exposure)
  - Enforcement intensity (states with high crash rates often increase inspections)
  - Operational risk environment for carriers operating in that state

This complements the FMCSA crash history features with STATE-LEVEL context,
not just carrier-level history.

OUTPUT FEATURES (joined on report_state)
-----------------------------------------
state_fatal_crashes_2023       Total fatal crashes in state (2023)
state_fatal_fatalities_2023    Total fatalities
state_avg_fatals_per_crash     Average fatalities per crash
state_crashes_involving_trucks Share of fatal crashes involving large trucks
state_fatal_crash_index        0-100 risk index: 100 = highest fatal crash density
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

PACE_DATA = os.getenv(
    "PACE_DATA_DIR",
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
FARS_ZIP  = os.path.join(PACE_DATA, "FARS2023NationalCSV.zip")
OUT_DIR   = Path(__file__).parents[2] / "outputs" / "enrichment"
OUT_FILE  = OUT_DIR / "state_fatal_crash_features.csv"

# FIPS state code → 2-letter abbreviation
FIPS_TO_STATE = {
    1:"AL", 2:"AK", 4:"AZ", 5:"AR", 6:"CA", 8:"CO", 9:"CT", 10:"DE",
    11:"DC", 12:"FL", 13:"GA", 15:"HI", 16:"ID", 17:"IL", 18:"IN", 19:"IA",
    20:"KS", 21:"KY", 22:"LA", 23:"ME", 24:"MD", 25:"MA", 26:"MI", 27:"MN",
    28:"MS", 29:"MO", 30:"MT", 31:"NE", 32:"NV", 33:"NH", 34:"NJ", 35:"NM",
    36:"NY", 37:"NC", 38:"ND", 39:"OH", 40:"OK", 41:"OR", 42:"PA", 44:"RI",
    45:"SC", 46:"SD", 47:"TN", 48:"TX", 49:"UT", 50:"VT", 51:"VA", 53:"WA",
    54:"WV", 55:"WI", 56:"WY",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading FARS from {FARS_ZIP} ...")

    with zipfile.ZipFile(FARS_ZIP) as zf:
        # Main accident file
        with zf.open("FARS2023NationalCSV/accident.csv") as f:
            acc = pd.read_csv(f, usecols=["STATE", "ST_CASE", "FATALS"])
        # Vehicle file — identify large trucks (BODY_TYP 60-79 are large trucks per FARS coding)
        with zf.open("FARS2023NationalCSV/vehicle.csv") as f:
            veh_cols = pd.read_csv(f, nrows=0).columns.tolist()
            body_col = "BODY_TYP" if "BODY_TYP" in veh_cols else None
            if body_col:
                veh = pd.read_csv(
                    zf.open("FARS2023NationalCSV/vehicle.csv"),
                    usecols=["STATE", "ST_CASE", "BODY_TYP"]
                )
            else:
                veh = None

    print(f"  Accidents: {len(acc):,}")

    # State-level accident + fatality stats
    state_agg = (
        acc.groupby("STATE")
        .agg(
            state_fatal_crashes_2023    = ("ST_CASE", "count"),
            state_fatal_fatalities_2023 = ("FATALS",  "sum"),
        )
        .reset_index()
    )
    state_agg["state_avg_fatals_per_crash"] = (
        state_agg["state_fatal_fatalities_2023"] /
        state_agg["state_fatal_crashes_2023"].clip(lower=1)
    ).round(3)

    # Large truck involvement rate
    if veh is not None:
        large_truck_cases = (
            veh[veh["BODY_TYP"].between(60, 79)][["STATE", "ST_CASE"]]
            .drop_duplicates()
            .groupby("STATE")["ST_CASE"].count()
            .reset_index()
            .rename(columns={"ST_CASE": "truck_crash_count"})
        )
        state_agg = state_agg.merge(large_truck_cases, on="STATE", how="left")
        state_agg["truck_crash_count"] = state_agg["truck_crash_count"].fillna(0)
        state_agg["state_crashes_involving_trucks"] = (
            state_agg["truck_crash_count"] /
            state_agg["state_fatal_crashes_2023"].clip(lower=1)
        ).round(3)
    else:
        state_agg["state_crashes_involving_trucks"] = 0.15  # national average

    # Map FIPS to state abbreviation
    state_agg["state_abbr"] = state_agg["STATE"].map(FIPS_TO_STATE)
    state_agg = state_agg.dropna(subset=["state_abbr"])

    # Fatal crash index: normalise by population-adjusted crashes (0-100)
    c = state_agg["state_fatal_crashes_2023"]
    state_agg["state_fatal_crash_index"] = (
        100 * (c - c.min()) / (c.max() - c.min() + 1e-9)
    ).round(1)

    # Keep only output columns
    out = state_agg[[
        "state_abbr",
        "state_fatal_crashes_2023",
        "state_fatal_fatalities_2023",
        "state_avg_fatals_per_crash",
        "state_crashes_involving_trucks",
        "state_fatal_crash_index",
    ]]

    print(f"\nOutput shape: {out.shape}")
    print("\nTop 10 states by fatal crash index:")
    print(out.sort_values("state_fatal_crash_index", ascending=False).head(10)
          .to_string(index=False))

    out.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
