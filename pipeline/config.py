# ── PACE Pipeline Configuration ───────────────────────────────────

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Teradata connection
TD_HOST     = os.getenv("TD_HOST", "")
TD_USERNAME = os.getenv("TD_USERNAME", "")
TD_PASSWORD = os.getenv("TD_PASSWORD", "")
TD_DATABASE = os.getenv("TD_DATABASE", "CTGAN")
TD_SOURCE_TABLE = "ctgan_input"
TD_SYNTHETIC_TABLE = "ctgan_synthetic"
TD_VIEW = "pace_synthetic_v"

# CTGAN settings
CTGAN_TRAIN_ROWS = 500000
CTGAN_SYNTHETIC_ROWS = 3000000
CTGAN_EPOCHS = 300
CTGAN_BATCH_SIZE = 10000
CTGAN_MODEL_PATH = "models/ctgan_model.pkl"
SYNTHETIC_CSV_PATH = "outputs/synthetic_data.csv"

# Column definitions
ID_COLUMN = "unique_id"
DOT_COLUMN = "dot_number"
DATE_COLUMN = "insp_year"
REGRESSION_TARGET = "accessorial_risk_score"
MULTICLASS_TARGET = "accessorial_type"
N_CLASSES = 6

CONTINUOUS_COLUMNS = [
    # Carrier size
    "carrier_truck_units", "carrier_power_units", "carrier_total_drivers",
    "carrier_total_cdl", "carrier_mcs150_mileage", "carrier_mcs150_mileage_year",
    "carrier_recordable_crash_rate", "carrier_total_intrastate_drivers",
    "carrier_interstate_beyond_100_miles", "carrier_interstate_within_100_miles",
    "carrier_intrastate_beyond_100_miles", "carrier_intrastate_within_100_miles",
    # SMS
    "sms_nbr_power_unit", "sms_driver_total", "sms_recent_mileage",
    "sms_recent_mileage_year",
    # Inspection/violation counts
    "insp_level_id", "time_weight", "oos_total", "driver_oos_total",
    "vehicle_oos_total", "hazmat_oos_total", "total_hazmat_sent",
    "basic_viol", "unsafe_viol", "fatigued_viol", "dr_fitness_viol",
    "subt_alcohol_viol", "vh_maint_viol", "hm_viol",
    # Crash history
    "crash_count", "crash_fatalities_total", "crash_injuries_total",
    "crash_towaway_total", "crash_avg_severity", "crash_hazmat_releases",
    # EIA fuel/energy
    "eia_diesel_national", "eia_diesel_padd1_east_coast", "eia_diesel_padd2_midwest",
    "eia_diesel_padd3_gulf_coast", "eia_diesel_padd4_rocky_mountain",
    "eia_diesel_padd5_west_coast", "eia_diesel_california", "eia_gasoline_national",
    "eia_crude_wti_spot", "eia_crude_brent_spot", "eia_diesel_no2_spot_ny",
    "eia_jet_fuel_spot_ny", "eia_heating_oil_spot_ny", "eia_distillate_stocks_us",
    "eia_distillate_supplied_us", "eia_distillate_stocks_padd1",
    "eia_distillate_stocks_padd2", "eia_distillate_stocks_padd3",
    "eia_refinery_utilization_us", "eia_crude_inputs_to_refineries",
    "eia_steo_diesel_price_forecast", "eia_steo_wti_price_forecast",
    "eia_steo_gasoline_retail_forecast", "eia_steo_liquid_fuels_consumption",
    "eia_natgas_henry_hub_spot", "eia_natgas_industrial_price",
    "eia_natgas_commercial_price",
    # FRED economic
    "fred_TSIFRGHT", "fred_PCU484121484121", "fred_PCU4841248412",
    "fred_AMTMVS", "fred_INDPRO", "fred_GASDESW",
    "fred_FRGEXPUSM649NCIS", "fred_FRGSHPUSM649NCIS",
    "fred_PCU4841224841221", "fred_TRUCKD11", "fred_CES4348400001",
    "fred_CES4349300001", "fred_RAILFRTINTERMODAL", "fred_WPU057303",
    "fred_WPU3012", "fred_CES4300000001", "fred_CES4348100001",
    # Weather
    "wx_avg_high_f", "wx_avg_low_f", "wx_total_precip_in",
    "wx_total_snow_in", "wx_avg_wind_mph",
    # Facility counts
    "fac_estab_311612", "fac_estab_311991", "fac_estab_311999",
    "fac_estab_312111", "fac_estab_312120", "fac_estab_424410",
    "fac_estab_424420", "fac_estab_424430", "fac_estab_424450",
    "fac_estab_424460", "fac_estab_424480", "fac_estab_424490",
    "fac_estab_484110", "fac_estab_484121", "fac_estab_484122",
    "fac_estab_484220", "fac_estab_484230", "fac_estab_493110",
    "fac_estab_493120", "fac_estab_493130", "fac_estab_493190",
    "fac_estab_warehousing_total", "fac_reefer_share",
    # Reefer/rail
    "usda_reefer_availability", "stb_avg_dwell_hours",
    # Temporal
    "insp_month", "insp_dow", "insp_day", "is_holiday", "is_near_holiday",
    "carrier_add_date",
]

CATEGORICAL_COLUMNS = [
    "report_state", "county_code_state", "carrier_status_code",
    "carrier_carrier_operation", "carrier_fleetsize", "carrier_hm_ind",
    "carrier_phy_state", "carrier_phy_country", "carrier_safety_rating",
    "carrier_crgo_genfreight", "carrier_crgo_household", "carrier_crgo_produce",
    "carrier_crgo_intermodal", "carrier_crgo_oilfield", "carrier_crgo_meat",
    "carrier_crgo_chem", "carrier_crgo_drybulk", "carrier_crgo_coldfood",
    "carrier_crgo_beverages", "carrier_crgo_construct",
    "sms_carrier_operation", "sms_hm_flag", "sms_pc_flag",
    "sms_phy_state", "sms_phy_country", "sms_private_only",
    "sms_authorized_for_hire", "sms_exempt_for_hire", "sms_private_property",
    "hazmat_placard_req", "unit_type_desc", "unit_make", "unit_license_state",
]

CTGAN_DISCRETE_COLUMNS = CATEGORICAL_COLUMNS + [
    "is_holiday", "is_near_holiday",
]

# ── Phase 2 enrichment columns ─────────────────────────────────────────
# Non-leaking predictors: all available at carrier booking time,
# none derived from inspection outcomes.

# Carrier-level historical OOS rates (from Vehicle_Inspection_NOT_SMS)
CARRIER_HISTORY_COLUMNS = [
    "carrier_hist_insp_count",
    "carrier_hist_oos_rate",
    "carrier_hist_driver_oos_rate",
    "carrier_hist_vehicle_oos_rate",
    "carrier_hist_hazmat_oos_rate",
    "carrier_hist_avg_viol_per_insp",
    "carrier_hist_driver_viol_rate",
    "carrier_hist_vehicle_viol_rate",
    "carrier_hist_hazmat_viol_rate",
    "carrier_hist_post_acc_rate",
]

# FMCSA SMS BASIC percentile measures (from SMS_AB_PassProperty)
SMS_PERCENTILE_COLUMNS = [
    "sms_insp_total",
    "sms_driver_insp_total",
    "sms_driver_oos_insp_total",
    "sms_vehicle_insp_total",
    "sms_vehicle_oos_insp_total",
    "sms_unsafe_driv_measure",
    "sms_unsafe_driv_alert",
    "sms_hos_measure",
    "sms_hos_alert",
    "sms_driv_fit_measure",
    "sms_driv_fit_alert",
    "sms_contr_subst_measure",
    "sms_contr_subst_alert",
    "sms_veh_maint_measure",
    "sms_veh_maint_alert",
    "sms_alert_count",
    "sms_max_measure",
]

# State-level contextual features (parking, crash rates, detention propensity, freight)
STATE_CONTEXT_COLUMNS = [
    "state_parking_spots_total",
    "state_parking_spots_per_100mi",
    "state_parking_scarcity_index",
    "state_fatal_crashes_2023",
    "state_fatal_fatalities_2023",
    "state_avg_fatals_per_crash",
    "state_crashes_involving_trucks",
    "state_fatal_crash_index",
    "state_atri_detention_propensity",
    "state_atri_severity_index",
    "faf_state_freight_intensity",
    "faf_state_hazmat_share",
    "faf_state_reefer_share_faf",
]

# Market/freight condition signals (time-series, joined on insp_year + insp_month)
MARKET_SIGNAL_COLUMNS = [
    "lmi_composite_index",
    "lmi_transportation_capacity",
    "lmi_transportation_utilization",
    "lmi_transportation_prices",
    "ftsi_index",
    "ppi_trucking_mom_pct",
    "ftsi_yoy_pct",
    "freight_tightness_score",
    "atri_detention_freq_rate",
    "atri_avg_detention_hours",
    "atri_avg_detention_cost_usd",
    "atri_cost_per_hour_usd",
    "atri_reefer_multiplier",
]

PHASE2_NEW_COLUMNS = (
    CARRIER_HISTORY_COLUMNS +
    SMS_PERCENTILE_COLUMNS +
    STATE_CONTEXT_COLUMNS +
    MARKET_SIGNAL_COLUMNS
)

# Columns to REMOVE from model inputs in Phase 2 (direct target leakers).
# These are per-inspection violation outcomes — not available at booking time.
# They directly determine accessorial_type via the SQL CASE statement, causing
# 100% classification accuracy (target leakage).
LEAKING_COLUMNS = [
    "oos_total",
    "driver_oos_total",
    "vehicle_oos_total",
    "hazmat_oos_total",
    "basic_viol",
    "unsafe_viol",
    "fatigued_viol",
    "dr_fitness_viol",
    "subt_alcohol_viol",
    "vh_maint_viol",
    "hm_viol",
]

# CONTINUOUS_COLUMNS without target-leaking violation counts, plus Phase 2 enrichment.
CONTINUOUS_COLUMNS_V2 = (
    [c for c in CONTINUOUS_COLUMNS if c not in LEAKING_COLUMNS] +
    PHASE2_NEW_COLUMNS
)

# ── LTL shipment feature columns (enriched_ltl_training.csv) ──────────────────
# Used when pace_transformer.py is invoked with --csv on the LTL enriched file.
# Column names match outputs/enrichment/build_enriched_ltl.py output exactly.

LTL_DATE_COLUMN = "pickup_year"  # derived from pickup_dt at load time
LTL_REGRESSION_TARGET = "Cost"  # actual shipment dollar cost; better regression signal than derived score

LTL_CONTINUOUS_COLUMNS = [
    # Shipment basics
    "Total Weight", "Total Pallets", "osrm_miles", "osrm_minutes",
    # Temporal
    "pickup_month", "pickup_dow", "pickup_week", "pickup_quarter",
    "days_to_nearest_holiday",
    # Carrier safety profile (from FMCSA SMS crosswalk)
    "carrier_fleet_size", "carrier_driver_count", "carrier_hm_authorized",
    "insp_total", "driver_oos_insp_total", "vehicle_oos_insp_total",
    "basic_unsafe_driving", "basic_hos", "basic_driver_fitness",
    # basic_contr_subst excluded: 100% zero (FMCSA measure not populated for these carriers)
    "basic_vehicle_maint",
    # Market / macro signals (17-23% null; median-imputed in pipeline, not zero-filled)
    "diesel_price_padd",
    "lmi_composite", "transportation_capacity", "transportation_utilization",
    "transportation_prices", "fred_truck_tonnage", "fred_freight_tsi", "fred_ltl_ppi",
    # Lane-level historical accessorial rates
    "hist_lift_gate_rate", "hist_delivery_appt_rate", "hist_limited_access_rate",
    "hist_excessive_length_rate", "hist_hazmat_rate", "hist_inside_delivery_rate",
    "hist_residential_rate", "hist_detention_rate",
    # Origin geography
    "origin_county_estabs", "origin_trucking_warehouse_estabs",
    "origin_warehousing_estabs", "origin_reefer_share",
    # Destination geography
    "dest_county_estabs", "dest_trucking_warehouse_estabs",
    "dest_warehousing_estabs", "dest_reefer_share",
    # Facility / residential flags (numeric)
    "origin_frs_facilities", "origin_frs_stationary",
    "dest_frs_facilities", "dest_frs_stationary",
    # Weather at origin and destination
    "origin_wx_wind_mph", "origin_wx_prcp_in", "origin_wx_snow_in",
    "origin_wx_tmax_f", "origin_wx_tmin_f",
    "dest_wx_wind_mph", "dest_wx_prcp_in", "dest_wx_snow_in",
    "dest_wx_tmax_f", "dest_wx_tmin_f",
    # Terminal proximity (49% null for carriers without full directory; median-imputed)
    "dist_to_nearest_carrier_terminal_mi",
    "carrier_has_terminal_in_origin_zip", "carrier_has_terminal_in_dest_zip",
    # Binary flags (treated as numeric)
    "severe_weather_flag", "is_month_end", "is_month_start", "is_holiday_week",
    "nmfc_imputed", "any_low_freq_accessorial", "likely_residential_dest",
    "frs_residential_dest",
    # osrm_imputed excluded: always 0 (all routes resolved by OSRM, no haversine fallback)
]

LTL_CATEGORICAL_COLUMNS = [
    "Shipper State", "Consignee State", "Carrier Name",
    "Top Commodity Class",
    # Shipment Mode excluded: 100% LTL, zero variance
    "origin_padd",
    "origin_urban_class", "dest_urban_class",
]

# FT-Transformer settings
MODEL_WEIGHTS_PATH_LTL   = "models/pace_transformer_ltl.pt"
MODEL_ARTIFACTS_PATH_LTL = "models/artifacts_ltl.pkl"
MODEL_WEIGHTS_PATH_FTL   = "models/pace_transformer_ftl.pt"
MODEL_ARTIFACTS_PATH_FTL = "models/artifacts_ftl.pkl"

# Backward-compat aliases (point to LTL paths)
MODEL_WEIGHTS_PATH   = MODEL_WEIGHTS_PATH_LTL
MODEL_ARTIFACTS_PATH = MODEL_ARTIFACTS_PATH_LTL

RESULTS_DIR = "outputs"


def model_paths(mode: str = "ltl") -> tuple:
    """Return (weights_path, artifacts_path) for the given mode."""
    if mode == "ftl":
        return MODEL_WEIGHTS_PATH_FTL, MODEL_ARTIFACTS_PATH_FTL
    return MODEL_WEIGHTS_PATH_LTL, MODEL_ARTIFACTS_PATH_LTL


def is_pace_model_ready(mode: str = "ltl") -> bool:
    """Return True only when both PACE model artifact files exist on disk."""
    weights, artifacts = model_paths(mode)
    return os.path.exists(weights) and os.path.exists(artifacts)
CHUNK_SIZE = 100000
NUM_THREADS = 4

# Charge type labels
CHARGE_TYPE_LABELS = [
    "No Charge",
    "Detention",
    "Safety Surcharge",
    "Compliance Fee",
    "Hazmat Fee",
    "High Risk / Multiple",
]

# Risk score weights (violation counts, not binary)
RISK_WEIGHTS = {
    "oos_total": 15.0,
    "driver_oos_total": 10.0,
    "vehicle_oos_total": 10.0,
    "basic_viol": 8.0,
    "unsafe_viol": 8.0,
    "vh_maint_viol": 7.0,
    "fatigued_viol": 7.0,
    "dr_fitness_viol": 6.0,
    "subt_alcohol_viol": 6.0,
    "hm_viol": 5.0,
    "crash_count": 8.0,
    "crash_avg_severity": 6.0,
    "crash_fatalities_total": 4.0,
    "crash_injuries_total": 3.0,
    "crash_towaway_total": 2.0,
}
