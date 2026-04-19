"""Enrichment pipeline for LTL PACE model training data.

Executes tractable items from the LTL data gathering TODO list:
  1  Drop Unnamed: 24, filter zero-cost rows
  2  Cap Total Pallets outlier
  3  Temporal features from Pickup Date
  4  Impute Top NMFC nulls from commodity class mode
  5  Carrier Name -> DOT crosswalk (fuzzy match vs FMCSA_SMS_Motor_Carrier_Census)
  6  FMCSA BASIC scores join via DOT
  7  EIA regional diesel price join (State -> PADD -> weekly)
  8  NCEI daily weather join (nearest-city by state) on Pickup Date
  9  LMI + FRED monthly market demand features
 10  Lane-level historical accessorial rates (empirical Bayes shrinkage)
 11  Pool low-frequency accessorials into any_low_freq_accessorial flag

Skipped (require external resources): OSRM distances, TMS appointment flag,
USPS/EPA consignee type classification, carrier terminal zip APIs,
itemized invoices, dimensions, customer IDs, CTGAN augmentation.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LTL_PATH = Path(
    r"C:\Users\clayt\Downloads\CJ Data File for LTL (COST.. Accessorials) "
    r"1_1_2025-Present-Report_Builder_1776457275142.csv"
)
PACE_DATA = Path(
    r"C:\Users\clayt\OneDrive - University of Arkansas\Senior Year\PACE DATA"
)
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "enriched_ltl_training.csv"
OUT_REPORT = OUT_DIR / "enriched_ltl_report.txt"

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


# PADD mapping (EIA Petroleum Administration for Defense Districts).
STATE_TO_PADD = {
    # PADD 1 - East Coast
    "CT": 1, "DC": 1, "DE": 1, "FL": 1, "GA": 1, "MA": 1, "MD": 1, "ME": 1,
    "NC": 1, "NH": 1, "NJ": 1, "NY": 1, "PA": 1, "RI": 1, "SC": 1, "VA": 1,
    "VT": 1, "WV": 1,
    # PADD 2 - Midwest
    "IA": 2, "IL": 2, "IN": 2, "KS": 2, "KY": 2, "MI": 2, "MN": 2, "MO": 2,
    "ND": 2, "NE": 2, "OH": 2, "OK": 2, "SD": 2, "TN": 2, "WI": 2,
    # PADD 3 - Gulf Coast
    "AL": 3, "AR": 3, "LA": 3, "MS": 3, "NM": 3, "TX": 3,
    # PADD 4 - Rocky Mountain
    "CO": 4, "ID": 4, "MT": 4, "UT": 4, "WY": 4,
    # PADD 5 - West Coast
    "AK": 5, "AZ": 5, "CA": 5, "HI": 5, "NV": 5, "OR": 5, "WA": 5,
}

PADD_DIESEL_FILES = {
    1: "eia_A_diesel_padd1_east_coast.csv",
    2: "eia_A_diesel_padd2_midwest.csv",
    3: "eia_A_diesel_padd3_gulf_coast.csv",
    4: "eia_A_diesel_padd4_rocky_mountain.csv",
    5: "eia_A_diesel_padd5_west_coast.csv",
}

# Map state -> nearest NCEI city in PACE DATA weather_data (20+ cities).
STATE_TO_NCEI_CITY = {
    "GA": "atlanta", "NC": "charlotte", "IL": "chicago", "OH": "columbus",
    "TX": "houston", "CO": "denver", "MI": "detroit", "AR": "fayetteville",
    "FL": "jacksonville", "MO": "kansas_city", "CA": "los_angeles",
    "TN": "memphis", "MN": "minneapolis", "NE": "omaha", "AZ": "phoenix",
    "OR": "portland", "WA": "seattle",
    # approximate neighbors
    "SC": "charlotte", "VA": "charlotte", "IN": "columbus", "KY": "nashville",
    "WV": "columbus", "PA": "columbus", "WI": "chicago", "IA": "omaha",
    "KS": "kansas_city", "OK": "dallas", "NM": "phoenix", "UT": "denver",
    "WY": "denver", "ID": "portland", "MT": "denver", "ND": "minneapolis",
    "SD": "omaha", "NV": "los_angeles", "AL": "atlanta", "MS": "memphis",
    "LA": "houston", "MD": "charlotte", "NJ": "charlotte", "NY": "charlotte",
    "CT": "charlotte", "MA": "charlotte", "ME": "charlotte", "NH": "charlotte",
    "RI": "charlotte", "VT": "charlotte", "DE": "charlotte", "DC": "charlotte",
}

# 2025 US federal holidays (supplemented since us_holidays_2026.csv only has 2026).
HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-10-13",  # Columbus Day
    "2025-11-11",  # Veterans Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
])


def step_header(title: str) -> None:
    log("")
    log("=" * 72)
    log(title)
    log("=" * 72)


def main() -> None:
    t0 = time.time()
    step_header("STEP 1: LOAD LTL FILE")
    df = pd.read_csv(LTL_PATH, low_memory=False)
    log(f"Loaded shape: {df.shape}")
    original_rows = len(df)

    # --- Drop Unnamed: 24, filter zero-cost rows, cap pallet outlier ---
    step_header("STEP 2: CLEANUP (drop null col, filter $0 rows, cap pallets)")
    if "Unnamed: 24" in df.columns:
        df = df.drop(columns=["Unnamed: 24"])
        log("Dropped Unnamed: 24 (100% null)")
    zero_cost_before = (df["Cost"] == 0).sum()
    df = df[df["Cost"] > 0].reset_index(drop=True)
    log(f"Filtered {zero_cost_before} zero-cost rows (data errors)")

    pallet_cap = 36
    outliers = (df["Total Pallets"] > pallet_cap).sum()
    df["Total Pallets"] = df["Total Pallets"].clip(upper=pallet_cap)
    log(f"Capped Total Pallets at {pallet_cap} ({outliers} rows winsorized)")

    # --- Temporal features ---
    step_header("STEP 3: TEMPORAL FEATURES")
    df["pickup_dt"] = pd.to_datetime(df["Pickup Date"], errors="coerce")
    df["pickup_month"] = df["pickup_dt"].dt.month
    df["pickup_dow"] = df["pickup_dt"].dt.dayofweek
    df["pickup_week"] = df["pickup_dt"].dt.isocalendar().week.astype(int)
    df["pickup_quarter"] = df["pickup_dt"].dt.quarter
    df["is_month_end"] = df["pickup_dt"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["pickup_dt"].dt.is_month_start.astype(int)

    holidays_2026 = pd.read_csv(PACE_DATA / "us_holidays_2026.csv")
    hols_2026 = pd.to_datetime(holidays_2026["date"])
    all_holidays = pd.concat([pd.Series(HOLIDAYS_2025), pd.Series(hols_2026)]).sort_values().reset_index(drop=True)

    def days_to_nearest_holiday(d: pd.Timestamp) -> int:
        if pd.isna(d):
            return -1
        diffs = (all_holidays - d).dt.days
        return int(diffs.abs().min())

    df["days_to_nearest_holiday"] = df["pickup_dt"].apply(days_to_nearest_holiday)
    df["is_holiday_week"] = (df["days_to_nearest_holiday"] <= 3).astype(int)
    log(f"Added 8 temporal features: month, dow, week, quarter, is_month_end/start, "
        f"days_to_nearest_holiday, is_holiday_week")

    # --- NMFC imputation ---
    step_header("STEP 4: NMFC IMPUTATION FROM COMMODITY CLASS MODE")
    nmfc_null_before = df["Top NMFC"].isna().sum()
    nmfc_by_class = (
        df.dropna(subset=["Top NMFC", "Top Commodity Class"])
          .groupby("Top Commodity Class")["Top NMFC"]
          .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    )
    df["nmfc_imputed"] = df["Top NMFC"].isna().astype(int)
    df["Top NMFC"] = df.apply(
        lambda r: r["Top NMFC"] if pd.notna(r["Top NMFC"])
        else nmfc_by_class.get(r["Top Commodity Class"], "UNKNOWN"),
        axis=1,
    )
    nmfc_null_after = df["Top NMFC"].isna().sum()
    log(f"NMFC nulls: {nmfc_null_before} -> {nmfc_null_after} "
        f"(imputed {nmfc_null_before - nmfc_null_after} from commodity class mode)")

    # --- Carrier Name -> DOT crosswalk ---
    step_header("STEP 5: CARRIER NAME -> DOT CROSSWALK")
    carriers_in_data = df["Carrier Name"].value_counts().to_dict()
    log(f"Distinct carriers: {len(carriers_in_data)}")

    carrier_keywords = {
        "SAIA": ["SAIA", "SAIA MOTOR"],
        "R&L Carriers": ["R AND L CARRIERS", "R+L CARRIERS", "R & L CARRIERS", "R L CARRIERS", "RANDL CARRIERS"],
        "Estes Express Lines": ["ESTES EXPRESS", "ESTES TRUCKING"],
        "XPO Logistics": ["XPO LOGISTICS", "XPO LTL"],
        "Dayton Freight Lines": ["DAYTON FREIGHT"],
        "Dohrn Transfer": ["DOHRN TRANSFER"],
        "Sutton Transport": ["SUTTON TRANSPORT"],
        "UPS TForce Freight": ["TFORCE FREIGHT", "UPS FREIGHT", "TFI INTERNATIONAL", "TFORCE"],
        "Old Dominion Freight Line": ["OLD DOMINION FREIGHT"],
        "ABF Freight": ["ABF FREIGHT", "ARKBEST", "ARCBEST"],
        "I 44 Express": ["I 44 EXPRESS", "I-44 EXPRESS", "I44 EXPRESS"],
        "Central Transport": ["CENTRAL TRANSPORT"],
        "AAA Cooper Transportation": ["AAA COOPER"],
        "FedEx Economy": ["FEDEX FREIGHT"],
        "FedEx Priority": ["FEDEX FREIGHT"],
        "Oak Harbor Freight Lines": ["OAK HARBOR FREIGHT"],
        "J.B. Hunt Transport Inc": ["J B HUNT", "JB HUNT", "J.B. HUNT"],
    }

    def name_matches(census_name: str, keywords: list[str]) -> bool:
        if not isinstance(census_name, str):
            return False
        u = census_name.upper()
        # Word-boundary match to avoid false positives like "SRL CARRIERS" matching "RL CARRIERS".
        return any(re.search(rf"\b{re.escape(kw)}\b", u) for kw in keywords)

    census_path = PACE_DATA / "FMCSA_SMS_Motor_Carrier_Census.csv"
    crosswalk_rows = []
    if census_path.exists():
        log(f"Streaming {census_path.name} for name matches...")
        cols = ["dot_number", "legal_name", "dba_name", "phy_state",
                "nbr_power_unit", "driver_total", "carrier_operation", "hm_flag"]
        try:
            chunks = pd.read_csv(census_path, usecols=cols, chunksize=200_000,
                                 low_memory=False, on_bad_lines="skip")
            for chunk_i, chunk in enumerate(chunks):
                for carrier_name, keywords in carrier_keywords.items():
                    mask = (
                        chunk["legal_name"].apply(lambda x: name_matches(x, keywords)) |
                        chunk["dba_name"].apply(lambda x: name_matches(x, keywords))
                    )
                    hits = chunk[mask]
                    if len(hits) > 0:
                        for _, row in hits.iterrows():
                            crosswalk_rows.append({
                                "ltl_carrier_name": carrier_name,
                                "dot_number": row["dot_number"],
                                "legal_name": row["legal_name"],
                                "dba_name": row["dba_name"],
                                "phy_state": row["phy_state"],
                                "nbr_power_unit": row["nbr_power_unit"],
                                "driver_total": row["driver_total"],
                                "carrier_operation": row["carrier_operation"],
                                "hm_flag": row["hm_flag"],
                            })
                if chunk_i % 5 == 0:
                    log(f"  processed chunk {chunk_i}, crosswalk rows so far: {len(crosswalk_rows)}")
        except Exception as exc:
            log(f"  FMCSA census stream failed: {exc}")

        crosswalk_df = pd.DataFrame(crosswalk_rows)
        if len(crosswalk_df) > 0:
            # For each LTL carrier, pick the DOT with the most power units
            # (largest fleet = most likely the national carrier).
            crosswalk_df["nbr_power_unit"] = pd.to_numeric(crosswalk_df["nbr_power_unit"], errors="coerce").fillna(0)
            best = (
                crosswalk_df.sort_values("nbr_power_unit", ascending=False)
                            .drop_duplicates("ltl_carrier_name")
                            .set_index("ltl_carrier_name")
            )
            (OUT_DIR / "carrier_dot_crosswalk.csv").write_text(
                crosswalk_df.to_csv(index=False)
            )
            log(f"Crosswalk resolved {len(best)}/{len(carrier_keywords)} target carriers")
            log(f"Top matches by fleet size:")
            for name in list(carrier_keywords.keys())[:10]:
                if name in best.index:
                    row = best.loc[name]
                    log(f"  {name:30s} -> DOT {row['dot_number']} "
                        f"({row['legal_name']}, {row['nbr_power_unit']:.0f} PU)")
                else:
                    log(f"  {name:30s} -> NOT FOUND")
            df["dot_number"] = df["Carrier Name"].map(best["dot_number"]).astype("Int64")
            df["carrier_fleet_size"] = df["Carrier Name"].map(best["nbr_power_unit"])
            df["carrier_driver_count"] = df["Carrier Name"].map(
                pd.to_numeric(best["driver_total"], errors="coerce")
            )
            df["carrier_hm_authorized"] = df["Carrier Name"].map(best["hm_flag"])
        else:
            log("No carrier matches found in FMCSA census.")
            df["dot_number"] = pd.NA
    else:
        log(f"FMCSA census file not found at {census_path}; skipping crosswalk")
        df["dot_number"] = pd.NA

    # --- FMCSA BASIC scores join via DOT ---
    step_header("STEP 6: FMCSA BASIC SCORES JOIN")
    basic_path = PACE_DATA / "SMS_AB_PassProperty_20260316.csv"
    if basic_path.exists() and df["dot_number"].notna().any():
        basic = pd.read_csv(basic_path, low_memory=False)
        basic.columns = [c.lower() for c in basic.columns]
        basic_subset = basic[[
            "dot_number", "insp_total", "driver_oos_inspt_total" if "driver_oos_inspt_total" in basic.columns else "driver_oos_insp_total",
            "vehicle_oos_insp_total", "unsafe_driv_measure", "hos_driv_measure",
            "driv_fit_measure", "contr_subst_measure", "veh_maint_measure",
        ]].rename(columns={
            "unsafe_driv_measure": "basic_unsafe_driving",
            "hos_driv_measure": "basic_hos",
            "driv_fit_measure": "basic_driver_fitness",
            "contr_subst_measure": "basic_contr_subst",
            "veh_maint_measure": "basic_vehicle_maint",
        })
        before = len(df)
        df = df.merge(basic_subset, how="left", left_on="dot_number", right_on="dot_number")
        log(f"Joined FMCSA BASIC scores. Rows unchanged: {before} -> {len(df)}")
        log(f"BASIC score null rates:")
        for col in ["basic_unsafe_driving", "basic_hos", "basic_driver_fitness",
                    "basic_contr_subst", "basic_vehicle_maint"]:
            null_pct = df[col].isna().mean() * 100
            log(f"  {col}: {null_pct:.1f}% null")
    else:
        log("BASIC scores file missing or no DOT matches; skipping join")

    # --- EIA regional diesel price ---
    step_header("STEP 7: EIA REGIONAL DIESEL PRICE JOIN")
    eia_frames = []
    for padd, fname in PADD_DIESEL_FILES.items():
        path = PACE_DATA / "eia_data" / fname
        if path.exists():
            e = pd.read_csv(path)
            e = e[["period", "value"]].rename(columns={"period": "week", "value": "diesel_price"})
            e["week"] = pd.to_datetime(e["week"])
            e["padd"] = padd
            eia_frames.append(e)
    eia = pd.concat(eia_frames, ignore_index=True) if eia_frames else pd.DataFrame()
    if not eia.empty:
        df["origin_padd"] = df["Shipper State"].map(STATE_TO_PADD).fillna(-1).astype(int)
        eia["padd"] = eia["padd"].astype(int)
        # merge_asof with `by` requires both sides globally sorted by the `on` key.
        left_sorted = df.sort_values("pickup_dt").reset_index().rename(columns={"index": "orig_idx"})
        eia_sorted = eia.sort_values("week").reset_index(drop=True)
        merged = pd.merge_asof(
            left_sorted,
            eia_sorted,
            left_on="pickup_dt",
            right_on="week",
            left_by="origin_padd",
            right_by="padd",
            direction="backward",
        ).set_index("orig_idx").sort_index()
        df["diesel_price_padd"] = merged["diesel_price"]
        log(f"Joined EIA diesel. Non-null rate: {df['diesel_price_padd'].notna().mean()*100:.1f}%")
        log(f"Diesel price range: ${df['diesel_price_padd'].min():.2f} - ${df['diesel_price_padd'].max():.2f}")
    else:
        log("EIA diesel files missing; skipping")

    # --- NCEI daily weather join ---
    step_header("STEP 8: NCEI DAILY WEATHER JOIN")
    ncei_path = PACE_DATA / "weather_data" / "ncei_daily_combined.csv"
    if ncei_path.exists():
        w = pd.read_csv(ncei_path, low_memory=False)
        w["DATE"] = pd.to_datetime(w["DATE"], errors="coerce")
        w["city_lc"] = w["city"].str.lower().str.replace(" ", "_")
        # Aggregate to one row per city+date (NCEI can have multiple stations per city).
        w_agg = w.groupby(["city_lc", "DATE"], as_index=False).agg(
            prcp_in=("PRCP", "max"),
            snow_in=("SNOW", "max"),
            tmax_f=("TMAX", "max"),
            tmin_f=("TMIN", "min"),
            wind_mph=("AWND", "max"),
            gust_mph=("WSF5", "max"),
        )
        df["origin_ncei_city"] = df["Shipper State"].map(STATE_TO_NCEI_CITY)
        df["dest_ncei_city"] = df["Consignee State"].map(STATE_TO_NCEI_CITY)
        df = df.merge(
            w_agg.add_prefix("origin_wx_").rename(columns={
                "origin_wx_city_lc": "origin_ncei_city",
                "origin_wx_DATE": "pickup_dt",
            }),
            how="left", on=["origin_ncei_city", "pickup_dt"],
        )
        df = df.merge(
            w_agg.add_prefix("dest_wx_").rename(columns={
                "dest_wx_city_lc": "dest_ncei_city",
                "dest_wx_DATE": "pickup_dt",
            }),
            how="left", on=["dest_ncei_city", "pickup_dt"],
        )
        hit = df["origin_wx_prcp_in"].notna().mean() * 100
        log(f"Joined NCEI weather. Origin hit rate: {hit:.1f}%")
        df["severe_weather_flag"] = (
            (df["origin_wx_prcp_in"].fillna(0) > 1.0) |
            (df["origin_wx_snow_in"].fillna(0) > 2.0) |
            (df["origin_wx_gust_mph"].fillna(0) > 35) |
            (df["dest_wx_prcp_in"].fillna(0) > 1.0) |
            (df["dest_wx_snow_in"].fillna(0) > 2.0) |
            (df["dest_wx_gust_mph"].fillna(0) > 35)
        ).astype(int)
        log(f"Severe weather flag fires on {df['severe_weather_flag'].mean()*100:.1f}% of rows")
    else:
        log("NCEI weather file missing; skipping")

    # --- LMI + FRED monthly market demand ---
    step_header("STEP 9: LMI + FRED MONTHLY MARKET DEMAND")
    df["pickup_month_key"] = df["pickup_dt"].dt.to_period("M").dt.to_timestamp()

    lmi_path = PACE_DATA / "market_data" / "lmi_timeseries.csv"
    if lmi_path.exists():
        lmi = pd.read_csv(lmi_path)
        lmi["period"] = pd.to_datetime(lmi["period"], errors="coerce")
        keep = ["period", "lmi_composite", "transportation_capacity",
                "transportation_utilization", "transportation_prices"]
        lmi = lmi[[c for c in keep if c in lmi.columns]]
        df = df.merge(
            lmi.rename(columns={"period": "pickup_month_key"}),
            how="left", on="pickup_month_key",
        )
        log(f"Joined LMI. Composite null rate: {df['lmi_composite'].isna().mean()*100:.1f}%")

    for fred_file, out_col in [
        ("TRUCKD11.csv", "fred_truck_tonnage"),
        ("TSIFRGHT.csv", "fred_freight_tsi"),
        ("PCU484121484121.csv", "fred_ltl_ppi"),
    ]:
        p = PACE_DATA / "fred_data" / fred_file
        if p.exists():
            f = pd.read_csv(p)
            date_col = "date" if "date" in f.columns else "observation_date"
            val_col = [c for c in f.columns if c != date_col][0]
            f[date_col] = pd.to_datetime(f[date_col], errors="coerce")
            f = f.rename(columns={date_col: "pickup_month_key", val_col: out_col})
            df = df.merge(f[["pickup_month_key", out_col]], how="left", on="pickup_month_key")
            log(f"Joined FRED {fred_file}: {out_col}, null rate: {df[out_col].isna().mean()*100:.1f}%")

    # --- Lane-level historical accessorial rates (empirical Bayes) ---
    step_header("STEP 10: LANE-LEVEL HISTORICAL ACCESSORIAL RATES (EB shrinkage)")
    accessorial_cols = [c for c in df.columns if c.startswith("C/")]
    trainable_labels = {
        "C/Lift Gate Delivery": "hist_lift_gate_rate",
        "C/Delivery Appointment": "hist_delivery_appt_rate",
        "C/Limited Access Delivery": "hist_limited_access_rate",
        "C/Excessive Length Charge": "hist_excessive_length_rate",
        "C/HAZARDOUS MATERIALS": "hist_hazmat_rate",
        "C/Inside Delivery": "hist_inside_delivery_rate",
        "C/Residential Delivery": "hist_residential_rate",
    }

    k = 20  # shrinkage pseudo-count
    for acc_col, out_col in trainable_labels.items():
        if acc_col not in df.columns:
            continue
        df[f"_{acc_col}_fired"] = (df[acc_col].fillna(0) != 0).astype(int)
        global_rate = df[f"_{acc_col}_fired"].mean()
        lane = df.groupby(["Shipper Zip", "Consignee Zip"])[f"_{acc_col}_fired"].agg(["sum", "count"])
        lane["rate_shrunk"] = (lane["sum"] + k * global_rate) / (lane["count"] + k)
        lane = lane.reset_index()[["Shipper Zip", "Consignee Zip", "rate_shrunk"]].rename(
            columns={"rate_shrunk": out_col}
        )
        df = df.merge(lane, how="left", on=["Shipper Zip", "Consignee Zip"])
        df = df.drop(columns=[f"_{acc_col}_fired"])
    log(f"Added {len(trainable_labels)} lane-level historical rate features (EB k={k})")

    # --- Pool low-frequency accessorials ---
    step_header("STEP 11: POOL LOW-FREQUENCY ACCESSORIALS")
    low_freq_with_signal = [
        "C/Detention", "C/Inspection", "C/Single Shipment",
        "C/Re Weigh Fee", "C/Lumper Fee", "C/SORT/SEGREGATING FEE",
    ]
    drop_near_zero = [
        "C/Reconsignment", "C/Redelivery Fee", "C/Notify Before Delivery",
        "C/High Cost Delivery", "C/TONU (Truck Ordered Not Used)",
        "C/CALIFORNIA COMPLIANCE SURCHARGE 671", "C/Guaranteed Services",
        "C/Guaranteed Charges",
    ]
    pool = np.zeros(len(df), dtype=int)
    for c in low_freq_with_signal:
        if c in df.columns:
            pool |= (df[c].fillna(0) != 0).astype(int).values
    df["any_low_freq_accessorial"] = pool
    log(f"any_low_freq_accessorial fires on {df['any_low_freq_accessorial'].mean()*100:.2f}% of rows")

    df = df.drop(columns=[c for c in drop_near_zero if c in df.columns])
    log(f"Dropped {len(drop_near_zero)} near-zero accessorial columns")

    # --- Write output ---
    step_header("STEP 12: WRITE OUTPUT")
    # Clean up helper columns we don't need in final output
    drop_final = ["pickup_month_key", "origin_ncei_city", "dest_ncei_city"]
    df = df.drop(columns=[c for c in drop_final if c in df.columns])

    df.to_csv(OUT_CSV, index=False)
    log(f"Wrote {OUT_CSV}")
    log(f"Final shape: {df.shape}")
    log(f"Rows retained: {len(df)} / {original_rows} ({len(df)/original_rows*100:.1f}%)")
    log(f"Total columns: {len(df.columns)}")
    log(f"Elapsed: {time.time() - t0:.1f}s")

    OUT_REPORT.write_text("\n".join(LOG_LINES), encoding="utf-8")
    log(f"Wrote report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
