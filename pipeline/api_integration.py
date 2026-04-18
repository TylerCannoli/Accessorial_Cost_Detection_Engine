"""
PACE Real-Time Data Integration Module
pipeline/api_integration.py

Fetches live data from all relevant APIs to enrich inference requests
with real-time signals before feeding into the FT-Transformer.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# ── API Keys ──────────────────────────────────────────────────────
FRED_API_KEY = "a9894999d90e9f5d08ebd26fe633927f"
EIA_API_KEY  = "m0L8lnt78ncuKxNy9XaLTU0RqyMeMhchq9wDyLb2"
OWM_API_KEY  = "d4426e1b18fe884f2fc25089952d8d3c"

# ── Base URLs ─────────────────────────────────────────────────────
FRED_BASE_URL  = "https://api.stlouisfed.org/fred/series/observations"
FMCSA_BASE_URL = "https://data.transportation.gov/resource"
NWS_BASE_URL   = "https://api.weather.gov"
BTS_BASE_URL   = "https://data.bts.gov/resource"
CENSUS_BASE_URL = "https://api.census.gov/data/2023/cbp"
EIA_BASE_URL   = "https://api.eia.gov/v2"
OWM_BASE_URL   = "https://api.openweathermap.org/data/2.5"

# ── FMCSA Endpoint Map ────────────────────────────────────────────
FMCSA_ENDPOINTS = {
    "company_census":     f"{FMCSA_BASE_URL}/az4n-8mr2.json",
    "crash_census":       f"{FMCSA_BASE_URL}/4wxs-vbns.json",
    "vehicle_inspection": f"{FMCSA_BASE_URL}/fx4q-ay7w.json",
    "carrier_census":     f"{FMCSA_BASE_URL}/kjg3-diqy.json",
    "inspection_sms":     f"{FMCSA_BASE_URL}/rbkj-cgst.json",
    "violation":          f"{FMCSA_BASE_URL}/8mt8-2mdr.json",
    "sms_ab":             f"{FMCSA_BASE_URL}/4y6x-dmck.json",
}

# ── FRED Series ───────────────────────────────────────────────────
FRED_SERIES = [
    "TSIFRGHT",           # Freight Transportation Services Index
    "FRGSHPUSM649NCIS",   # Cass Freight Index: Shipments
    "FRGEXPUSM649NCIS",   # Cass Freight Index: Expenditures
    "PCU484121484121",    # PPI: Long-Distance Truckload
    "PCU4841224841221",   # PPI: Long-Distance LTL
    "PCU4841248412",      # PPI: General Freight Trucking
    "TRUCKD11",           # ATA Truck Tonnage Index
    "CES4348400001",      # Trucking Employment
    "CES4349300001",      # Warehousing Employment
    "RAILFRTINTERMODAL",  # Rail Intermodal Traffic
    "WPU3012",            # Diesel fuel PPI
    "WPU057303",          # Petroleum products PPI
    "GASDESW",            # US Diesel retail price
    "CES4300000001",      # Transportation & warehousing employment
    "CES4348100001",      # Truck transportation employment
    "AMTMVS",             # Value of manufacturers' shipments
    "INDPRO",             # Industrial Production Index
]

# ── EIA Diesel Products by PADD Region ───────────────────────────
EIA_DIESEL_PRODUCTS = {
    "eia_diesel_national":             "EMD_EPD2D_PTE_NUS_DPG",
    "eia_diesel_padd1_east_coast":     "EMD_EPD2D_PTE_R10_DPG",
    "eia_diesel_padd2_midwest":        "EMD_EPD2D_PTE_R20_DPG",
    "eia_diesel_padd3_gulf_coast":     "EMD_EPD2D_PTE_R30_DPG",
    "eia_diesel_padd4_rocky_mountain": "EMD_EPD2D_PTE_R40_DPG",
    "eia_diesel_padd5_west_coast":     "EMD_EPD2D_PTE_R50_DPG",
    "eia_diesel_california":           "EMD_EPD2D_PTE_SCA_DPG",
}


# ══════════════════════════════════════════════════════════════════
# FMCSA Client
# ══════════════════════════════════════════════════════════════════

class FMCSAClient:
    """Fetches real-time FMCSA carrier safety data by DOT number."""

    def get_carrier_profile(self, dot_number: int) -> Dict[str, Any]:
        """Return carrier profile."""
        try:
            r = requests.get(
                FMCSA_ENDPOINTS["company_census"],
                params={"dot_number": dot_number, "$limit": 1},
                timeout=10,
            )
            data = r.json()
            return data[0] if data else {}
        except Exception:
            return {}

    def get_carrier_name(self, dot_number: int) -> Optional[str]:
        """Return carrier name."""
        try:
            r = requests.get(
                FMCSA_ENDPOINTS["carrier_census"],
                params={"dot_number": dot_number, "$limit": 1},
                timeout=10,
            )
            data = r.json()
            if data:
                return data[0].get("legal_name") or data[0].get("dba_name")
            return None
        except Exception:
            return None

    def get_sms_scores(self, dot_number: int) -> Dict[str, Any]:
        """Return sms scores."""
        try:
            r = requests.get(
                FMCSA_ENDPOINTS["sms_ab"],
                params={"dot_number": dot_number, "$limit": 1},
                timeout=10,
            )
            data = r.json()
            return data[0] if data else {}
        except Exception:
            return {}

    def get_recent_inspections(self, dot_number: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent inspections."""
        try:
            r = requests.get(
                FMCSA_ENDPOINTS["inspection_sms"],
                params={
                    "dot_number": dot_number,
                    "$limit": limit,
                    "$order": "inspection_date DESC",
                },
                timeout=10,
            )
            return r.json()
        except Exception:
            return []

    def get_recent_violations(self, dot_number: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent violations."""
        try:
            r = requests.get(
                FMCSA_ENDPOINTS["violation"],
                params={
                    "dot_number": dot_number,
                    "$limit": limit,
                    "$order": "inspection_date DESC",
                },
                timeout=10,
            )
            return r.json()
        except Exception:
            return []

    def get_crash_history(self, dot_number: int) -> List[Dict[str, Any]]:
        """Return crash history."""
        try:
            r = requests.get(
                FMCSA_ENDPOINTS["crash_census"],
                params={"dot_number": dot_number, "$limit": 50},
                timeout=10,
            )
            return r.json()
        except Exception:
            return []

    def build_realtime_features(self, dot_number: int) -> Dict[str, Any]:
        """Build complete PACE-compatible feature dict from FMCSA data."""
        profile     = self.get_carrier_profile(dot_number)
        sms         = self.get_sms_scores(dot_number)
        inspections = self.get_recent_inspections(dot_number)
        crashes     = self.get_crash_history(dot_number)

        viol_counts = {
            "basic_viol": 0, "unsafe_viol": 0, "fatigued_viol": 0,
            "dr_fitness_viol": 0, "subt_alcohol_viol": 0,
            "vh_maint_viol": 0, "hm_viol": 0,
        }
        oos_total = driver_oos = vehicle_oos = 0

        for insp in inspections:
            for k in viol_counts:
                viol_counts[k] += int(insp.get(k, 0) or 0)
            oos_total   += int(insp.get("oos_total", 0) or 0)
            driver_oos  += int(insp.get("driver_oos_total", 0) or 0)
            vehicle_oos += int(insp.get("vehicle_oos_total", 0) or 0)

        crash_count      = len(crashes)
        crash_fatalities = sum(int(c.get("fatalities", 0) or 0) for c in crashes)
        crash_injuries   = sum(int(c.get("injuries", 0) or 0) for c in crashes)
        crash_towaways   = sum(int(c.get("towaways", 0) or 0) for c in crashes)

        return {
            "dot_number":                dot_number,
            "carrier_name":              self.get_carrier_name(dot_number),
            "carrier_status_code":       profile.get("carrier_status_code", "A"),
            "carrier_carrier_operation": profile.get("carrier_operation", "C"),
            "carrier_power_units":       int(profile.get("power_units", 0) or 0),
            "carrier_total_drivers":     int(profile.get("total_drivers", 0) or 0),
            "carrier_total_cdl":         int(profile.get("total_cdl_drivers", 0) or 0),
            "carrier_phy_state":         profile.get("phy_state", ""),
            "carrier_phy_country":       profile.get("phy_country", "US"),
            "carrier_safety_rating":     profile.get("safety_rating", "UNKNOWN"),
            "carrier_hm_ind":            profile.get("hm_flag", "N"),
            "sms_nbr_power_unit":        int(sms.get("power_units", 0) or 0),
            "sms_driver_total":          int(sms.get("driver_total", 0) or 0),
            **viol_counts,
            "oos_total":                 oos_total,
            "driver_oos_total":          driver_oos,
            "vehicle_oos_total":         vehicle_oos,
            "crash_count":               crash_count,
            "crash_fatalities_total":    crash_fatalities,
            "crash_injuries_total":      crash_injuries,
            "crash_towaway_total":       crash_towaways,
            "crash_avg_severity": (
                (crash_fatalities * 3 + crash_injuries * 2 + crash_towaways)
                / max(crash_count, 1)
            ),
        }


# ══════════════════════════════════════════════════════════════════
# FRED Client
# ══════════════════════════════════════════════════════════════════

class FREDClient:
    """Fetches latest FRED economic series."""

    def get_latest(self, series_id: str) -> Optional[float]:
        """Return latest."""
        try:
            r = requests.get(
                FRED_BASE_URL,
                params={
                    "series_id":  series_id,
                    "api_key":    FRED_API_KEY,
                    "file_type":  "json",
                    "sort_order": "desc",
                    "limit":      1,
                },
                timeout=10,
            )
            obs = r.json().get("observations", [])
            if obs and obs[0]["value"] != ".":
                return float(obs[0]["value"])
            return None
        except Exception:
            return None

    def get_all_latest(self) -> Dict[str, float]:
        """Return all latest."""
        results = {}
        for series_id in FRED_SERIES:
            val = self.get_latest(series_id)
            results[f"fred_{series_id}"] = val if val is not None else 0.0
        return results


# ══════════════════════════════════════════════════════════════════
# EIA Client
# ══════════════════════════════════════════════════════════════════

class EIAClient:
    """
    EIA live diesel prices by PADD region + WTI crude spot.
    API Key: m0L8lnt78ncuKxNy9XaLTU0RqyMeMhchq9wDyLb2
    """

    def get_diesel_price(self, product_code: str) -> Optional[float]:
        """Return diesel price."""
        try:
            r = requests.get(
                f"{EIA_BASE_URL}/petroleum/pri/gnd/data/",
                params={
                    "api_key":             EIA_API_KEY,
                    "frequency":           "weekly",
                    "data[0]":             "value",
                    "facets[product][]":   product_code,
                    "length":              1,
                    "sort[0][column]":     "period",
                    "sort[0][direction]":  "desc",
                },
                timeout=10,
            )
            data = r.json().get("response", {}).get("data", [])
            return float(data[0]["value"]) if data else None
        except Exception:
            return None

    def get_crude_wti(self) -> Optional[float]:
        """Return crude wti."""
        try:
            r = requests.get(
                f"{EIA_BASE_URL}/petroleum/pri/spt/data/",
                params={
                    "api_key":             EIA_API_KEY,
                    "frequency":           "weekly",
                    "data[0]":             "value",
                    "facets[product][]":   "EPCWTI",
                    "length":              1,
                    "sort[0][column]":     "period",
                    "sort[0][direction]":  "desc",
                },
                timeout=10,
            )
            data = r.json().get("response", {}).get("data", [])
            return float(data[0]["value"]) if data else None
        except Exception:
            return None

    def get_all_latest(self) -> Dict[str, float]:
        """Return all latest."""
        results = {}
        for col_name, product_code in EIA_DIESEL_PRODUCTS.items():
            val = self.get_diesel_price(product_code)
            results[col_name] = val if val is not None else 0.0
        results["eia_crude_wti_spot"] = self.get_crude_wti() or 0.0
        return results


# ══════════════════════════════════════════════════════════════════
# NWS Client (no key required)
# ══════════════════════════════════════════════════════════════════

class NWSClient:
    """National Weather Service — free, no key required."""

    def get_forecast_by_coords(self, lat: float, lon: float) -> Dict[str, Any]:
        """Return forecast by coords."""
        try:
            point_r = requests.get(
                f"{NWS_BASE_URL}/points/{lat},{lon}",
                headers={"User-Agent": "PACE-App/1.0"},
                timeout=10,
            )
            forecast_url = point_r.json()["properties"]["forecast"]
            forecast_r = requests.get(
                forecast_url,
                headers={"User-Agent": "PACE-App/1.0"},
                timeout=10,
            )
            current = forecast_r.json()["properties"]["periods"][0]
            return {
                "wx_temp_f":             current.get("temperature", 0),
                "wx_wind_mph":           self._parse_wind(current.get("windSpeed", "0 mph")),
                "wx_short_forecast":     current.get("shortForecast", ""),
                "wx_precip_probability": current.get("probabilityOfPrecipitation", {}).get("value", 0) or 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_alerts(self, state: str) -> List[Dict[str, Any]]:
        """Return alerts."""
        try:
            r = requests.get(
                f"{NWS_BASE_URL}/alerts/active",
                params={"area": state},
                headers={"User-Agent": "PACE-App/1.0"},
                timeout=10,
            )
            return [
                {
                    "event":    f["properties"].get("event"),
                    "severity": f["properties"].get("severity"),
                    "headline": f["properties"].get("headline"),
                }
                for f in r.json().get("features", [])[:5]
            ]
        except Exception:
            return []

    def _parse_wind(self, wind_str: str) -> float:
        try:
            return float(wind_str.split()[0])
        except Exception:
            return 0.0

    def build_weather_features(self, lat: float, lon: float) -> Dict[str, Any]:
        forecast = self.get_forecast_by_coords(lat, lon)
        return {
            "wx_avg_high_f":      float(forecast.get("wx_temp_f") or 0),
            "wx_avg_low_f":       float(forecast.get("wx_temp_f") or 0),
            "wx_avg_wind_mph":    float(forecast.get("wx_wind_mph") or 0),
            "wx_total_precip_in": float(forecast.get("wx_precip_probability") or 0) / 100,
            "wx_total_snow_in":   0.0,
            "wx_forecast":        forecast.get("wx_short_forecast", ""),
        }


# ══════════════════════════════════════════════════════════════════
# OpenWeatherMap Client
# ══════════════════════════════════════════════════════════════════

class OWMClient:
    """
    OpenWeatherMap — better route-level weather resolution than NWS.
    API Key: d4426e1b18fe884f2fc25089952d8d3c
    Free tier: 1,000 calls/day
    """

    def get_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Return current weather."""
        try:
            r = requests.get(
                f"{OWM_BASE_URL}/weather",
                params={"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "imperial"},
                timeout=10,
            )
            data    = r.json()
            main    = data.get("main", {})
            wind    = data.get("wind", {})
            rain    = data.get("rain", {})
            snow    = data.get("snow", {})
            return {
                "wx_avg_high_f":      float(main.get("temp_max", 0)),
                "wx_avg_low_f":       float(main.get("temp_min", 0)),
                "wx_avg_wind_mph":    float(wind.get("speed", 0)),
                "wx_total_precip_in": float(rain.get("1h", 0)) / 25.4,
                "wx_total_snow_in":   float(snow.get("1h", 0)) / 25.4,
                "wx_conditions":      data.get("weather", [{}])[0].get("description", ""),
                "wx_humidity":        float(main.get("humidity", 0)),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_weather_by_city(self, city: str, state: str = "US") -> Dict[str, Any]:
        """Return weather by city."""
        try:
            r = requests.get(
                f"{OWM_BASE_URL}/weather",
                params={"q": f"{city},{state}", "appid": OWM_API_KEY, "units": "imperial"},
                timeout=10,
            )
            data  = r.json()
            main  = data.get("main", {})
            wind  = data.get("wind", {})
            rain  = data.get("rain", {})
            snow  = data.get("snow", {})
            return {
                "wx_avg_high_f":      float(main.get("temp_max", 0)),
                "wx_avg_low_f":       float(main.get("temp_min", 0)),
                "wx_avg_wind_mph":    float(wind.get("speed", 0)),
                "wx_total_precip_in": float(rain.get("1h", 0)) / 25.4,
                "wx_total_snow_in":   float(snow.get("1h", 0)) / 25.4,
                "wx_conditions":      data.get("weather", [{}])[0].get("description", ""),
                "wx_humidity":        float(main.get("humidity", 0)),
            }
        except Exception as e:
            return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════
# BTS Client
# ══════════════════════════════════════════════════════════════════

class BTSClient:
    """Bureau of Transportation Statistics freight indicators."""

    def get_freight_indicators(self) -> Dict[str, Any]:
        """Return freight indicators."""
        try:
            r = requests.get(
                f"{BTS_BASE_URL}/y5ut-ibwt.json",
                params={"$limit": 1, "$order": "date DESC"},
                timeout=10,
            )
            data = r.json()
            return data[0] if data else {}
        except Exception:
            return {}


# ══════════════════════════════════════════════════════════════════
# Census CBP Client
# ══════════════════════════════════════════════════════════════════

class CensusClient:
    """Census County Business Patterns — facility/warehouse density."""

    NAICS_MAP = {
        "484110": "fac_estab_484110",
        "484121": "fac_estab_484121",
        "484122": "fac_estab_484122",
        "493110": "fac_estab_493110",
        "493120": "fac_estab_493120",
        "493130": "fac_estab_493130",
        "493190": "fac_estab_493190",
    }

    def get_establishments_by_state(self, state_fips: str) -> Dict[str, int]:
        """Return establishments by state."""
        results = {}
        for naics, col_name in self.NAICS_MAP.items():
            try:
                r = requests.get(
                    CENSUS_BASE_URL,
                    params={
                        "get":       "ESTAB,NAICS2017_LABEL",
                        "for":       f"state:{state_fips}",
                        "NAICS2017": naics,
                    },
                    timeout=10,
                )
                data = r.json()
                results[col_name] = int(data[1][0]) if len(data) > 1 else 0
            except Exception:
                results[col_name] = 0
        return results


# ══════════════════════════════════════════════════════════════════
# Real-Time Enrichment Pipeline
# ══════════════════════════════════════════════════════════════════

class RealTimeEnrichment:
    """
    Main enrichment class — combines all data sources into a single
    feature dict ready for PACE model inference.

    Usage:
        enricher = RealTimeEnrichment()

        # DOT number lookup
        features = enricher.enrich_dot(dot_number=1234567)

        # Manual shipment input
        features = enricher.enrich_manual(user_inputs, origin_city="Chicago", origin_state="IL")

        # CSV batch upload
        df = enricher.enrich_dataframe(df)
    """

    def __init__(self):
        self.fmcsa  = FMCSAClient()
        self.fred   = FREDClient()
        self.eia    = EIAClient()
        self.nws    = NWSClient()
        self.owm    = OWMClient()
        self.bts    = BTSClient()
        self.census = CensusClient()

        self._fred_cache      = None
        self._fred_cache_time = None
        self._eia_cache       = None
        self._eia_cache_time  = None

    def _get_fred_features(self) -> Dict[str, float]:
        """Return fred features."""
        now = datetime.now()
        if self._fred_cache is None or (now - self._fred_cache_time).seconds > 3600:
            print("  Refreshing FRED indicators...")
            self._fred_cache = self.fred.get_all_latest()
            self._fred_cache_time = now
        return self._fred_cache

    def _get_eia_features(self) -> Dict[str, float]:
        """Return eia features."""
        now = datetime.now()
        if self._eia_cache is None or (now - self._eia_cache_time).seconds > 3600:
            print("  Refreshing EIA diesel prices...")
            self._eia_cache = self.eia.get_all_latest()
            self._eia_cache_time = now
        return self._eia_cache

    def enrich_dot(self, dot_number: int,
                   origin_lat: float = None,
                   origin_lon: float = None,
                   origin_state: str = None) -> Dict[str, Any]:
        """Full enrichment for a DOT number lookup."""
        print(f"Enriching DOT {dot_number}...")
        features = self.fmcsa.build_realtime_features(dot_number)
        features.update(self._get_fred_features())
        features.update(self._get_eia_features())

        if origin_lat and origin_lon:
            weather = self.owm.get_current_weather(origin_lat, origin_lon)
            if "error" not in weather:
                features.update(weather)
            else:
                features.update(self.nws.build_weather_features(origin_lat, origin_lon))

        if origin_state:
            alerts = self.nws.get_alerts(origin_state)
            features["weather_alerts"]    = alerts
            features["has_weather_alert"] = len(alerts) > 0

        return features

    def enrich_manual(self, user_inputs: Dict[str, Any],
                      origin_lat: float = None,
                      origin_lon: float = None,
                      origin_city: str = None,
                      origin_state: str = None) -> Dict[str, Any]:
        """Enrich manually entered shipment data with live signals."""
        features = dict(user_inputs)
        features.update(self._get_fred_features())
        features.update(self._get_eia_features())

        if origin_lat and origin_lon:
            weather = self.owm.get_current_weather(origin_lat, origin_lon)
            if "error" not in weather:
                features.update(weather)
        elif origin_city:
            weather = self.owm.get_weather_by_city(origin_city, origin_state or "US")
            if "error" not in weather:
                features.update(weather)

        if "dot_number" in user_inputs:
            fmcsa_features = self.fmcsa.build_realtime_features(int(user_inputs["dot_number"]))
            for k, v in fmcsa_features.items():
                if k not in features or features[k] is None:
                    features[k] = v

        return features

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch enrich a CSV upload with current economic + price signals."""
        combined = {**self._get_fred_features(), **self._get_eia_features()}
        for col, val in combined.items():
            if col not in df.columns:
                df[col] = val
        return df


# ══════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════

_enricher: Optional[RealTimeEnrichment] = None

def get_enricher() -> RealTimeEnrichment:
    """Return enricher."""
    global _enricher
    if _enricher is None:
        _enricher = RealTimeEnrichment()
    return _enricher
