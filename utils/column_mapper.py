"""
utils/column_mapper.py

AI-powered column header mapper for the PACE upload pipeline.

Provides two mapping strategies:
  - 'semantic': sentence-transformers cosine similarity (offline, always available)
  - 'ollama':   local LLM via Ollama HTTP API (optional, degrades to semantic)

Usage:
    mapper = get_column_mapper()
    results = mapper.map_columns(unrecognized_cols, method="semantic")
    # {"orig_col": {"pace_col": "dot_number", "confidence": 0.82, "method": "semantic"}}
"""

import os
import sys
import json
import re
import requests
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, ID_COLUMN, DOT_COLUMN

# ── PACE target column set (ordered, deduplicated) ───────────────────────────
PACE_TARGET_COLS: List[str] = list(dict.fromkeys(
    list(CONTINUOUS_COLUMNS) + list(CATEGORICAL_COLUMNS) + [ID_COLUMN, DOT_COLUMN]
))

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONFIDENCE_HIGH   = 0.75   # green badge — auto-accept candidate
CONFIDENCE_MEDIUM = 0.50   # yellow badge — show for review
# below 0.50 → red badge, defaults to "(skip / ignore)"

# ── Ollama settings ───────────────────────────────────────────────────────────
OLLAMA_BASE    = "http://localhost:11434"
OLLAMA_TIMEOUT = 45

# ── Sentence-transformer model cache ─────────────────────────────────────────
_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_ST_CACHE: Dict[str, Dict[str, Any]] = {}

# ── Rich descriptions for commonly user-provided columns ─────────────────────
# Expand abbreviations and add synonyms so cosine similarity is more accurate.
# All other PACE columns get auto-generated descriptions from their name.
_RICH_DESCRIPTIONS: Dict[str, str] = {
    "dot_number":               "DOT number USDOT carrier identifier federal motor carrier",
    "unique_id":                "unique id shipment id record identifier",
    "carrier_power_units":      "power units number of trucks tractors fleet size vehicles",
    "carrier_truck_units":      "truck units semi trucks tractor count fleet vehicles",
    "carrier_total_drivers":    "total drivers number of drivers CDL operators",
    "carrier_total_cdl":        "CDL drivers commercial driver license count",
    "carrier_mcs150_mileage":   "annual mileage MCS-150 miles driven yearly distance",
    "carrier_mcs150_mileage_year": "mileage year MCS-150 report year",
    "carrier_recordable_crash_rate": "crash rate recordable crashes per million miles",
    "carrier_total_intrastate_drivers": "intrastate drivers within state drivers",
    "carrier_interstate_beyond_100_miles": "interstate beyond 100 miles long-haul drivers",
    "carrier_interstate_within_100_miles": "interstate within 100 miles short-haul drivers",
    "carrier_intrastate_beyond_100_miles": "intrastate beyond 100 miles",
    "carrier_intrastate_within_100_miles": "intrastate within 100 miles local drivers",
    "carrier_status_code":      "carrier status active inactive authority operating status",
    "carrier_carrier_operation":"carrier operation type for-hire private exempt",
    "carrier_safety_rating":    "safety rating satisfactory conditional unsatisfactory",
    "carrier_phy_state":        "physical state carrier state location origin state",
    "carrier_phy_country":      "physical country carrier country",
    "carrier_hm_ind":           "hazmat hazardous material flag indicator",
    "carrier_fleetsize":        "fleet size fleet category small medium large",
    "carrier_add_date":         "carrier add date registration date",
    "sms_nbr_power_unit":       "SMS number of power units",
    "sms_driver_total":         "SMS total drivers",
    "sms_recent_mileage":       "SMS recent mileage annual miles",
    "sms_recent_mileage_year":  "SMS mileage year",
    "sms_carrier_operation":    "SMS carrier operation type",
    "sms_hm_flag":              "SMS hazmat flag hazardous material",
    "sms_pc_flag":              "SMS passenger carrier flag",
    "sms_phy_state":            "SMS physical state",
    "sms_phy_country":          "SMS physical country",
    "sms_private_only":         "SMS private carrier only",
    "sms_authorized_for_hire":  "SMS authorized for hire",
    "sms_exempt_for_hire":      "SMS exempt for hire",
    "sms_private_property":     "SMS private property",
    "insp_level_id":            "inspection level type roadside check",
    "time_weight":              "time weight inspection recency weight",
    "oos_total":                "out of service total OOS inspections violations",
    "driver_oos_total":         "driver out of service OOS driver violations",
    "vehicle_oos_total":        "vehicle out of service OOS vehicle violations",
    "hazmat_oos_total":         "hazmat out of service OOS hazardous material",
    "total_hazmat_sent":        "total hazmat shipments hazardous material sent",
    "basic_viol":               "BASIC violations basic safety measurement total violations",
    "unsafe_viol":              "unsafe driving violations speeding reckless",
    "fatigued_viol":            "fatigued driving hours of service HOS violations",
    "vh_maint_viol":            "vehicle maintenance violations brake light tire defects",
    "dr_fitness_viol":          "driver fitness violations CDL medical license",
    "subt_alcohol_viol":        "substance alcohol drug violations DUI",
    "hm_viol":                  "hazmat violations hazardous material",
    "crash_count":              "crash count number of crashes accidents",
    "crash_fatalities_total":   "crash fatalities deaths fatal accidents",
    "crash_injuries_total":     "crash injuries injured persons accidents",
    "crash_towaway_total":      "crash towaways towed vehicles accidents",
    "crash_avg_severity":       "crash severity average severity score",
    "crash_hazmat_releases":    "hazmat releases hazardous material spills crashes",
    "eia_diesel_national":      "diesel price national EIA fuel cost per gallon",
    "eia_diesel_california":    "california diesel price EIA fuel",
    "eia_gasoline_national":    "gasoline price national EIA fuel",
    "eia_crude_wti_spot":       "WTI crude oil price spot price per barrel",
    "eia_crude_brent_spot":     "Brent crude oil price spot price per barrel",
    "eia_natgas_henry_hub_spot":"natural gas Henry Hub spot price",
    "fred_TSIFRGHT":            "freight transportation services index TSI FRED",
    "fred_TRUCKD11":            "truck tonnage index FRED",
    "wx_avg_high_f":            "high temperature weather Fahrenheit degrees",
    "wx_avg_low_f":             "low temperature weather Fahrenheit degrees",
    "wx_total_precip_in":       "precipitation rain inches weather",
    "wx_total_snow_in":         "snowfall snow inches winter weather",
    "wx_avg_wind_mph":          "wind speed mph weather",
    "insp_month":               "inspection month date month",
    "insp_day":                 "inspection day date day",
    "insp_dow":                 "day of week weekday inspection",
    "is_holiday":               "holiday federal holiday flag",
    "is_near_holiday":          "near holiday flag near federal holiday",
    "report_state":             "report state inspection state",
    "county_code_state":        "county state county code",
    "hazmat_placard_req":       "hazmat placard required hazardous material placards",
    "unit_type_desc":           "unit type description vehicle type",
    "unit_make":                "unit make vehicle manufacturer brand",
    "unit_license_state":       "unit license state vehicle registration state",
    "usda_reefer_availability": "USDA reefer refrigerated trailer availability",
    "stb_avg_dwell_hours":      "STB average dwell hours rail intermodal",
    "carrier_crgo_genfreight":  "general freight cargo carrier commodity",
    "carrier_crgo_household":   "household goods cargo carrier",
    "carrier_crgo_produce":     "produce fresh food cargo carrier",
    "carrier_crgo_intermodal":  "intermodal cargo carrier",
    "carrier_crgo_oilfield":    "oilfield equipment cargo carrier",
    "carrier_crgo_meat":        "meat cargo carrier refrigerated",
    "carrier_crgo_chem":        "chemicals cargo carrier hazmat",
    "carrier_crgo_drybulk":     "dry bulk cargo carrier",
    "carrier_crgo_coldfood":    "cold food refrigerated cargo carrier",
    "carrier_crgo_beverages":   "beverages cargo carrier",
    "carrier_crgo_construct":   "construction cargo carrier",
}


def _make_description(col_name: str) -> str:
    """Return a human-readable description for embedding."""
    if col_name in _RICH_DESCRIPTIONS:
        return _RICH_DESCRIPTIONS[col_name]
    readable = col_name.replace("_", " ").replace("-", " ")
    return f"{col_name} {readable}"


def find_unrecognized_columns(df) -> List[str]:
    """
    Return column names that survive the static alias map but still don't
    match any PACE schema column. These are candidates for AI mapping.
    """
    from pipeline.data_pipeline import normalize_column_names, PACE_SCHEMA_COLS

    df_norm, _ = normalize_column_names(df.copy())
    return [c for c in df_norm.columns if c not in PACE_SCHEMA_COLS]


class ColumnMapper:
    """
    Maps arbitrary user column names to PACE schema column names.

    Two methods:
      - 'semantic': offline sentence-transformers cosine similarity
      - 'ollama':   Ollama local LLM, falls back to semantic if unavailable

    All public methods return per-column dicts:
        {"pace_col": str | None, "confidence": float, "method": str}
    """

    # ── Sentence-transformers ─────────────────────────────────────────────────

    def _load_st_model(self) -> Tuple[object, List[str], object]:
        """Load/cache SentenceTransformer model and PACE column embeddings."""
        if _ST_MODEL_NAME in _ST_CACHE:
            c = _ST_CACHE[_ST_MODEL_NAME]
            return c["model"], c["keys"], c["embeddings"]

        from sentence_transformers import SentenceTransformer, util as st_util  # noqa: F401

        model = SentenceTransformer(_ST_MODEL_NAME)
        keys  = list(PACE_TARGET_COLS)
        texts = [_make_description(k) for k in keys]
        embs  = model.encode(texts, convert_to_tensor=True)

        _ST_CACHE[_ST_MODEL_NAME] = {"model": model, "keys": keys, "embeddings": embs}
        return model, keys, embs

    def semantic(self, user_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Map user_columns to PACE columns via greedy cosine-similarity assignment.
        Each PACE column can only be assigned to one user column.
        """
        if not user_columns:
            return {}

        try:
            from sentence_transformers import util as st_util
            model, pace_keys, pace_emb = self._load_st_model()
        except ImportError:
            return {col: {"pace_col": None, "confidence": 0.0, "method": "none"}
                    for col in user_columns}

        col_texts = [c.lower().replace("_", " ").replace("-", " ") for c in user_columns]
        col_emb   = model.encode(col_texts, convert_to_tensor=True)
        similarity = st_util.cos_sim(col_emb, pace_emb)

        # Build all (score, user_idx, pace_idx) and sort descending
        candidates: List[Tuple[float, int, int]] = [
            (float(similarity[i][j]), i, j)
            for i in range(len(user_columns))
            for j in range(len(pace_keys))
        ]
        candidates.sort(reverse=True)

        used_pace: set  = set()
        used_input: set = set()
        results: Dict[str, Dict[str, Any]] = {}

        for score, i, j in candidates:
            col      = user_columns[i]
            pace_col = pace_keys[j]
            if col in used_input or pace_col in used_pace:
                continue
            used_input.add(col)
            used_pace.add(pace_col)
            results[col] = {
                "pace_col":   pace_col if score >= CONFIDENCE_MEDIUM else None,
                "confidence": round(score, 4),
                "method":     "semantic",
            }

        for col in user_columns:
            if col not in results:
                results[col] = {"pace_col": None, "confidence": 0.0, "method": "semantic"}

        return results

    # ── Ollama ────────────────────────────────────────────────────────────────

    def check_ollama(self, model_name: str = "llama3.2") -> Tuple[bool, str]:
        """Probe Ollama and verify the requested model is pulled."""
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
            if r.status_code != 200:
                return False, "Ollama returned an unexpected response."
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(model_name in m for m in models):
                return False, (
                    f"Ollama is running but '{model_name}' is not pulled. "
                    f"Run: ollama pull {model_name}"
                )
            return True, f"Ollama ready ({model_name})"
        except requests.exceptions.ConnectionError:
            return False, "Ollama is not running. Start with: ollama serve"
        except Exception as e:
            return False, f"Cannot reach Ollama: {e}"

    def ollama(self, user_columns: List[str], model_name: str = "llama3.2") -> Dict[str, Dict[str, Any]]:
        """
        Map user_columns using a local Ollama LLM.
        Falls back to semantic() if Ollama is unavailable or returns bad JSON.
        """
        if not user_columns:
            return {}

        available, _ = self.check_ollama(model_name)
        if not available:
            fallback = self.semantic(user_columns)
            for col in fallback:
                fallback[col]["method"] = "semantic_fallback"
            return fallback

        pace_col_list = "\n".join(f"  - {c}" for c in PACE_TARGET_COLS)
        user_col_list = "\n".join(f"  - {c}" for c in user_columns)

        prompt = f"""You are a freight data schema expert for the PACE (Predictive Accessorial Cost Engine) system.

Your task: map each user-provided column name to the most appropriate PACE schema column name.

PACE schema columns (map ONLY to one of these, or null):
{pace_col_list}

User columns to map:
{user_col_list}

Rules:
- Return ONLY a valid JSON object. No explanation, no markdown fences, no commentary.
- Each key is a user column name (exactly as provided).
- Each value is either a PACE column name from the list above, or null if no reasonable match.
- Do not invent column names. Only use names from the PACE schema list.
- null means the column has no PACE equivalent (e.g. an internal tracking field).

Example output format:
{{
  "DOT": "dot_number",
  "num_trucks": "carrier_power_units",
  "internal_ref": null
}}

JSON object:"""

        payload = {"model": model_name, "prompt": prompt, "stream": False}
        try:
            r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return self.semantic(user_columns)

        raw = r.json().get("response", "").strip()
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
        raw = re.sub(r"```$",           "", raw, flags=re.MULTILINE).strip()

        try:
            mapping: dict = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    mapping = json.loads(match.group())
                except json.JSONDecodeError:
                    return self.semantic(user_columns)
            else:
                return self.semantic(user_columns)

        pace_col_set = set(PACE_TARGET_COLS)
        results: Dict[str, Dict[str, Any]] = {}
        for col in user_columns:
            suggested = mapping.get(col)
            if suggested and suggested in pace_col_set:
                results[col] = {"pace_col": suggested, "confidence": 0.80, "method": "ollama"}
            else:
                results[col] = {"pace_col": None,      "confidence": 0.0,  "method": "ollama"}

        return results

    # ── Unified entry point ───────────────────────────────────────────────────

    def map_columns(
        self,
        user_columns: List[str],
        method:       str = "semantic",
        ollama_model: str = "llama3.2",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map unrecognized column names to PACE schema columns.

        Args:
            user_columns:  Column names not recognized by the static alias map.
            method:        "semantic" (default) or "ollama".
            ollama_model:  Ollama model tag, only used when method="ollama".

        Returns:
            Dict[str, Dict[str, Any]] where each value has keys:
                pace_col   (str | None)
                confidence (float 0.0–1.0)
                method     (str)
        """
        if method == "ollama":
            return self.ollama(user_columns, model_name=ollama_model)
        return self.semantic(user_columns)


# ── Module-level singleton ────────────────────────────────────────────────────
_mapper: Optional[ColumnMapper] = None


def get_column_mapper() -> ColumnMapper:
    """Return the module-level ColumnMapper singleton."""
    global _mapper
    if _mapper is None:
        _mapper = ColumnMapper()
    return _mapper
