-- =============================================================
-- PACE Phase 2: Updated pace_synthetic_v with enrichment joins
-- Run this AFTER uploading all enrichment CSVs to Teradata:
--
--   1. Upload outputs/enrichment/carrier_oos_history.csv
--      → CTGAN.pace_carrier_history
--
--   2. Upload outputs/enrichment/sms_percentile_features.csv
--      → CTGAN.pace_sms_percentiles
--
--   3. Upload outputs/enrichment/state_parking_features.csv
--      → CTGAN.pace_parking_features
--
--   4. Upload outputs/enrichment/state_fatal_crash_features.csv
--      → CTGAN.pace_fatal_crash_features
--
--   5. Upload outputs/enrichment/atri_detention_features.csv
--      → CTGAN.pace_atri_features
--
--   6. Upload outputs/enrichment/freight_rate_features.csv
--      → CTGAN.pace_freight_rates
--
--   7. Upload outputs/enrichment/faf_freight_features.csv
--      → CTGAN.pace_faf_features
--
-- Usage:
--   bteq < scripts/create_teradata_view_v2.sql
-- =============================================================

-- Drop existing views
DROP VIEW CTGAN.pace_synthetic_v;
DROP VIEW CTGAN.pace_training_v;

-- =============================================================
-- PHASE 2 TRAINING VIEW
-- Key changes vs v1:
--   • Joins 7 enrichment tables (carrier history, SMS, parking,
--     crash rates, ATRI detention, freight rates, FAF5 freight)
--   • accessorial_type CASE is restructured to fix the
--     Detention class imbalance (was 0.03%, now ~5%)
--   • Leaking violation columns KEPT in data but model will be
--     configured via LEAKING_COLUMNS in config.py to skip them
-- =============================================================
CREATE VIEW CTGAN.pace_synthetic_v AS
SELECT
    -- ── Identity ──────────────────────────────────────────────────────
    s.unique_id,
    s.dot_number,
    s.insp_year,

    -- ── Original feature columns (pass-through) ───────────────────────
    s.report_state,
    s.county_code_state,
    s.insp_level_id,
    s.time_weight,
    s.oos_total,
    s.driver_oos_total,
    s.vehicle_oos_total,
    s.hazmat_oos_total,
    s.total_hazmat_sent,
    s.basic_viol,
    s.unsafe_viol,
    s.fatigued_viol,
    s.dr_fitness_viol,
    s.subt_alcohol_viol,
    s.vh_maint_viol,
    s.hm_viol,
    s.hazmat_placard_req,
    s.unit_type_desc,
    s.unit_make,
    s.unit_license_state,
    s.insp_month,
    s.insp_dow,
    s.insp_day,

    -- Carrier profile
    s.carrier_carrier_operation,
    s.carrier_total_drivers,
    s.carrier_power_units,
    s.carrier_truck_units,
    s.carrier_fleetsize,
    s.carrier_mcs150_mileage,
    s.carrier_mcs150_mileage_year,
    s.carrier_hm_ind,
    s.carrier_phy_state,
    s.carrier_phy_country,
    s.carrier_interstate_beyond_100_miles,
    s.carrier_interstate_within_100_miles,
    s.carrier_intrastate_beyond_100_miles,
    s.carrier_intrastate_within_100_miles,
    s.carrier_total_cdl,
    s.carrier_total_intrastate_drivers,
    s.carrier_crgo_genfreight,
    s.carrier_crgo_household,
    s.carrier_crgo_produce,
    s.carrier_crgo_coldfood,
    s.carrier_crgo_beverages,
    s.carrier_crgo_meat,
    s.carrier_crgo_chem,
    s.carrier_crgo_drybulk,
    s.carrier_crgo_construct,
    s.carrier_crgo_intermodal,
    s.carrier_crgo_oilfield,
    s.carrier_recordable_crash_rate,
    s.carrier_safety_rating,
    s.carrier_status_code,
    s.carrier_add_date,

    -- SMS
    s.sms_carrier_operation,
    s.sms_hm_flag,
    s.sms_pc_flag,
    s.sms_phy_state,
    s.sms_phy_country,
    s.sms_nbr_power_unit,
    s.sms_driver_total,
    s.sms_recent_mileage,
    s.sms_recent_mileage_year,
    s.sms_private_only,
    s.sms_authorized_for_hire,
    s.sms_exempt_for_hire,
    s.sms_private_property,

    -- Crash history
    s.crash_count,
    s.crash_fatalities_total,
    s.crash_injuries_total,
    s.crash_towaway_total,
    s.crash_avg_severity,
    s.crash_hazmat_releases,

    -- EIA energy (unchanged)
    s.eia_diesel_national,
    s.eia_diesel_padd1_east_coast,
    s.eia_diesel_padd2_midwest,
    s.eia_diesel_padd3_gulf_coast,
    s.eia_diesel_padd4_rocky_mountain,
    s.eia_diesel_padd5_west_coast,
    s.eia_diesel_california,
    s.eia_gasoline_national,
    s.eia_crude_wti_spot,
    s.eia_crude_brent_spot,
    s.eia_diesel_no2_spot_ny,
    s.eia_jet_fuel_spot_ny,
    s.eia_heating_oil_spot_ny,
    s.eia_distillate_stocks_us,
    s.eia_distillate_supplied_us,
    s.eia_distillate_stocks_padd1,
    s.eia_distillate_stocks_padd2,
    s.eia_distillate_stocks_padd3,
    s.eia_refinery_utilization_us,
    s.eia_crude_inputs_to_refineries,
    s.eia_steo_diesel_price_forecast,
    s.eia_steo_wti_price_forecast,
    s.eia_steo_gasoline_retail_forecast,
    s.eia_steo_liquid_fuels_consumption,
    s.eia_natgas_henry_hub_spot,
    s.eia_natgas_industrial_price,
    s.eia_natgas_commercial_price,

    -- FRED economic (unchanged)
    s.fred_TSIFRGHT,
    s.fred_PCU484121484121,
    s.fred_PCU4841248412,
    s.fred_AMTMVS,
    s.fred_INDPRO,
    s.fred_GASDESW,
    s.fred_FRGEXPUSM649NCIS,
    s.fred_FRGSHPUSM649NCIS,
    s.fred_PCU4841224841221,
    s.fred_TRUCKD11,
    s.fred_CES4348400001,
    s.fred_CES4349300001,
    s.fred_RAILFRTINTERMODAL,
    s.fred_WPU057303,
    s.fred_WPU3012,
    s.fred_CES4300000001,
    s.fred_CES4348100001,

    -- Weather (unchanged)
    s.wx_avg_high_f,
    s.wx_avg_low_f,
    s.wx_total_precip_in,
    s.wx_total_snow_in,
    s.wx_avg_wind_mph,

    -- Facility density (unchanged)
    s.fac_estab_311612,
    s.fac_estab_311991,
    s.fac_estab_311999,
    s.fac_estab_312111,
    s.fac_estab_312120,
    s.fac_estab_424410,
    s.fac_estab_424420,
    s.fac_estab_424430,
    s.fac_estab_424450,
    s.fac_estab_424460,
    s.fac_estab_424480,
    s.fac_estab_424490,
    s.fac_estab_484110,
    s.fac_estab_484121,
    s.fac_estab_484122,
    s.fac_estab_484220,
    s.fac_estab_484230,
    s.fac_estab_493110,
    s.fac_estab_493120,
    s.fac_estab_493130,
    s.fac_estab_493190,
    s.fac_estab_warehousing_total,
    s.fac_reefer_share,
    s.usda_reefer_availability,
    s.stb_avg_dwell_hours,
    s.is_holiday,
    s.is_near_holiday,

    -- ── NEW: Carrier historical OOS rates (Script 01) ─────────────────
    COALESCE(ch.carrier_hist_insp_count,        0)    AS carrier_hist_insp_count,
    COALESCE(ch.carrier_hist_oos_rate,          0.0)  AS carrier_hist_oos_rate,
    COALESCE(ch.carrier_hist_driver_oos_rate,   0.0)  AS carrier_hist_driver_oos_rate,
    COALESCE(ch.carrier_hist_vehicle_oos_rate,  0.0)  AS carrier_hist_vehicle_oos_rate,
    COALESCE(ch.carrier_hist_hazmat_oos_rate,   0.0)  AS carrier_hist_hazmat_oos_rate,
    COALESCE(ch.carrier_hist_avg_viol_per_insp, 0.0)  AS carrier_hist_avg_viol_per_insp,
    COALESCE(ch.carrier_hist_driver_viol_rate,  0.0)  AS carrier_hist_driver_viol_rate,
    COALESCE(ch.carrier_hist_vehicle_viol_rate, 0.0)  AS carrier_hist_vehicle_viol_rate,
    COALESCE(ch.carrier_hist_hazmat_viol_rate,  0.0)  AS carrier_hist_hazmat_viol_rate,
    COALESCE(ch.carrier_hist_post_acc_rate,     0.0)  AS carrier_hist_post_acc_rate,

    -- ── NEW: SMS BASIC percentile measures (Script 02) ────────────────
    COALESCE(sm.sms_insp_total,               0)    AS sms_insp_total,
    COALESCE(sm.sms_driver_insp_total,        0)    AS sms_driver_insp_total,
    COALESCE(sm.sms_driver_oos_insp_total,    0)    AS sms_driver_oos_insp_total,
    COALESCE(sm.sms_vehicle_insp_total,       0)    AS sms_vehicle_insp_total,
    COALESCE(sm.sms_vehicle_oos_insp_total,   0)    AS sms_vehicle_oos_insp_total,
    COALESCE(sm.sms_unsafe_driv_measure,      0.0)  AS sms_unsafe_driv_measure,
    COALESCE(sm.sms_unsafe_driv_alert,        0)    AS sms_unsafe_driv_alert,
    COALESCE(sm.sms_hos_measure,              0.0)  AS sms_hos_measure,
    COALESCE(sm.sms_hos_alert,                0)    AS sms_hos_alert,
    COALESCE(sm.sms_driv_fit_measure,         0.0)  AS sms_driv_fit_measure,
    COALESCE(sm.sms_driv_fit_alert,           0)    AS sms_driv_fit_alert,
    COALESCE(sm.sms_contr_subst_measure,      0.0)  AS sms_contr_subst_measure,
    COALESCE(sm.sms_contr_subst_alert,        0)    AS sms_contr_subst_alert,
    COALESCE(sm.sms_veh_maint_measure,        0.0)  AS sms_veh_maint_measure,
    COALESCE(sm.sms_veh_maint_alert,          0)    AS sms_veh_maint_alert,
    COALESCE(sm.sms_alert_count,              0)    AS sms_alert_count,
    COALESCE(sm.sms_max_measure,              0.0)  AS sms_max_measure,

    -- ── NEW: State-level parking scarcity (Script 03) ─────────────────
    COALESCE(pk.state_parking_spots_total,      0.0) AS state_parking_spots_total,
    COALESCE(pk.state_parking_spots_per_100mi,  0.0) AS state_parking_spots_per_100mi,
    COALESCE(pk.state_parking_scarcity_index,  50.0) AS state_parking_scarcity_index,

    -- ── NEW: State-level fatal crash rates (Script 04) ────────────────
    COALESCE(fc.state_fatal_crashes_2023,         0)    AS state_fatal_crashes_2023,
    COALESCE(fc.state_fatal_fatalities_2023,      0)    AS state_fatal_fatalities_2023,
    COALESCE(fc.state_avg_fatals_per_crash,       1.0)  AS state_avg_fatals_per_crash,
    COALESCE(fc.state_crashes_involving_trucks,   0.15) AS state_crashes_involving_trucks,
    COALESCE(fc.state_fatal_crash_index,          50.0) AS state_fatal_crash_index,

    -- ── NEW: ATRI detention propensity (Script 05) ────────────────────
    COALESCE(at.state_atri_detention_propensity, 45.0) AS state_atri_detention_propensity,
    COALESCE(at.state_atri_severity_index,        1.0) AS state_atri_severity_index,

    -- ── NEW: Freight market signals (Script 06) ───────────────────────
    COALESCE(fr.lmi_composite_index,            55.0) AS lmi_composite_index,
    COALESCE(fr.lmi_transportation_capacity,    55.0) AS lmi_transportation_capacity,
    COALESCE(fr.lmi_transportation_utilization, 55.0) AS lmi_transportation_utilization,
    COALESCE(fr.lmi_transportation_prices,      55.0) AS lmi_transportation_prices,
    COALESCE(fr.ftsi_index,                    130.0) AS ftsi_index,
    COALESCE(fr.ppi_trucking_mom_pct,            0.0) AS ppi_trucking_mom_pct,
    COALESCE(fr.ftsi_yoy_pct,                    0.0) AS ftsi_yoy_pct,
    COALESCE(fr.freight_tightness_score,        55.0) AS freight_tightness_score,

    -- ATRI national constants (scalar — same for all rows)
    0.634  AS atri_detention_freq_rate,
    6.3    AS atri_avg_detention_hours,
    1281.0 AS atri_avg_detention_cost_usd,
    203.0  AS atri_cost_per_hour_usd,
    1.40   AS atri_reefer_multiplier,

    -- ── NEW: FAF5 freight flow intensity (Script 08) ──────────────────
    COALESCE(faf.faf_state_freight_intensity, 1.0)  AS faf_state_freight_intensity,
    COALESCE(faf.faf_state_hazmat_share,      0.10) AS faf_state_hazmat_share,
    COALESCE(faf.faf_state_reefer_share_faf,  0.08) AS faf_state_reefer_share_faf,

    -- ── Computed target: regression (unchanged) ───────────────────────
    CAST(
        LEAST(100.0,
            (s.oos_total            * 15.0) +
            (s.driver_oos_total     * 10.0) +
            (s.vehicle_oos_total    * 10.0) +
            (s.basic_viol           *  8.0) +
            (s.unsafe_viol          *  8.0) +
            (s.vh_maint_viol        *  7.0) +
            (s.fatigued_viol        *  7.0) +
            (s.dr_fitness_viol      *  6.0) +
            (s.subt_alcohol_viol    *  6.0) +
            (s.hm_viol              *  5.0) +
            (s.crash_count          *  8.0) +
            (s.crash_avg_severity   *  6.0) +
            (s.crash_fatalities_total * 4.0) +
            (s.crash_injuries_total *  3.0) +
            (s.crash_towaway_total  *  2.0)
        ) AS FLOAT
    ) AS accessorial_risk_score,

    -- ── Computed target: multiclass (FIXED — balanced Detention class) ─
    -- Key change: Detention fires on DRIVER_OOS > 0 regardless of other
    -- violations, instead of only when it's the exclusive violation.
    -- This fixes the 0.03% class imbalance for Detention.
    -- Priority order reflects real-world charge precedence.
    CASE
        WHEN (s.oos_total * 15.0 + s.driver_oos_total * 10.0 +
              s.vehicle_oos_total * 10.0 + s.basic_viol * 8.0 +
              s.unsafe_viol * 8.0 + s.vh_maint_viol * 7.0 +
              s.fatigued_viol * 7.0 + s.dr_fitness_viol * 6.0 +
              s.subt_alcohol_viol * 6.0 + s.hm_viol * 5.0 +
              s.crash_count * 8.0) >= 80
            THEN 5  -- High Risk / Multiple
        WHEN s.hm_viol > 0 OR CAST(s.hazmat_placard_req AS INTEGER) > 0
            THEN 4  -- Hazmat Fee
        WHEN s.driver_oos_total > 0
            THEN 1  -- Detention  ← moved BEFORE safety surcharge to fix imbalance
        WHEN s.dr_fitness_viol > 0 OR s.fatigued_viol > 0
            THEN 3  -- Compliance Fee
        WHEN s.unsafe_viol > 0 OR s.basic_viol > 0 OR s.vh_maint_viol > 0
            THEN 2  -- Safety Surcharge
        ELSE 0      -- No Charge
    END AS accessorial_type

FROM CTGAN.ctgan_synthetic s

-- Carrier historical OOS rates (left join — not all carriers will match)
LEFT JOIN CTGAN.pace_carrier_history ch
    ON CAST(s.dot_number AS VARCHAR(20)) = CAST(ch.dot_number AS VARCHAR(20))

-- SMS BASIC percentiles
LEFT JOIN CTGAN.pace_sms_percentiles sm
    ON CAST(s.dot_number AS VARCHAR(20)) = CAST(sm.dot_number AS VARCHAR(20))

-- State-level parking (join on inspection report state)
LEFT JOIN CTGAN.pace_parking_features pk
    ON s.report_state = pk.state_abbr

-- State-level fatal crash rates
LEFT JOIN CTGAN.pace_fatal_crash_features fc
    ON s.report_state = fc.state_abbr

-- ATRI detention propensity (carrier home state)
LEFT JOIN CTGAN.pace_atri_features at
    ON s.carrier_phy_state = at.state_abbr

-- Freight market signals (monthly, join on year+month)
LEFT JOIN CTGAN.pace_freight_rates fr
    ON s.insp_year = fr.year AND s.insp_month = fr.month

-- FAF5 freight flows (carrier home state)
LEFT JOIN CTGAN.pace_faf_features faf
    ON s.carrier_phy_state = faf.state_abbr;
