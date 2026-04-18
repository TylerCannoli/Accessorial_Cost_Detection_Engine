"""
utils/doc_parser.py
Parses shipping documents (PDF, Excel, image) into a PACE-compatible DataFrame.
Uses Ollama (local LLM) for unstructured text extraction.
Excel files with recognizable columns are mapped directly without AI.
"""
import io
import json
import re

import pandas as pd
import requests

OLLAMA_BASE   = "http://localhost:11434"
OLLAMA_MODEL  = "llama3.2"
OLLAMA_TIMEOUT = 120  # seconds

# Fields Ollama must extract
PACE_FIELDS = {
    "shipment_id":            "unique shipment identifier (BOL number, PRO number, or any reference ID)",
    "ship_date":              "date of shipment — normalize to YYYY-MM-DD",
    "carrier":                "carrier or trucking company name",
    "facility":               "destination facility, warehouse, or DC name",
    "weight_lbs":             "shipment weight in pounds — numeric only, no units",
    "miles":                  "distance in miles — numeric only",
    "base_freight_usd":       "base freight or linehaul cost in USD — numeric only, no $ sign",
    "accessorial_charge_usd": "accessorial charges in USD — numeric only, 0 if not present",
}

# Excel column synonyms → PACE column names
_EXCEL_ALIASES = {
    "shipment_id":            ["shipment_id", "shipment id", "bol", "bol number", "pro", "pro number",
                               "reference", "ref #", "shipment #", "order id", "order number"],
    "ship_date":              ["ship_date", "ship date", "shipdate", "date", "pickup date",
                               "dispatch date", "shipped date"],
    "carrier":                ["carrier", "carrier name", "trucking company", "scac", "vendor"],
    "facility":               ["facility", "destination", "dest", "consignee", "delivery location",
                               "ship to", "warehouse"],
    "weight_lbs":             ["weight_lbs", "weight", "weight (lbs)", "lbs", "gross weight",
                               "total weight"],
    "miles":                  ["miles", "distance", "distance (mi)", "mileage", "transit miles"],
    "base_freight_usd":       ["base_freight_usd", "base freight", "linehaul", "linehaul cost",
                               "freight charge", "freight cost", "base cost", "rate"],
    "accessorial_charge_usd": ["accessorial_charge_usd", "accessorial", "accessorials",
                               "accessorial charges", "additional charges", "surcharge", "extras"],
}


# ── Ollama health check ────────────────────────────────────────────────────────
def check_ollama() -> tuple[bool, str]:
    """Returns (is_running, status_message)."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            has_model = any(OLLAMA_MODEL in m for m in models)
            if not has_model:
                return False, (
                    f"Ollama is running but model '{OLLAMA_MODEL}' is not pulled. "
                    f"Run: `ollama pull {OLLAMA_MODEL}`"
                )
            return True, f"Ollama ready ({OLLAMA_MODEL})"
        return False, "Ollama returned an unexpected response."
    except requests.exceptions.ConnectionError:
        return False, (
            "Ollama is not running. Start it with: `ollama serve`  "
            "then pull a model: `ollama pull llama3.2`"
        )
    except Exception as e:
        return False, f"Could not reach Ollama: {e}"


# ── Text extraction ────────────────────────────────────────────────────────────
def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF parsing. Run: pip install pdfplumber")

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No text could be extracted from the PDF. It may be a scanned image — try uploading as an image file instead.")
    return text


def _extract_image_text(file_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ImportError(
            "Pillow and pytesseract are required for image parsing. "
            "Run: pip install Pillow pytesseract  "
            "and install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    img = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(img).strip()
    if not text:
        raise ValueError("No text could be extracted from the image.")
    return text


# ── Excel direct mapping ───────────────────────────────────────────────────────
def _map_excel_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Try to map Excel columns to PACE schema using known aliases.
    Returns a remapped DataFrame if enough columns matched, else None.
    """
    col_lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    for pace_col, aliases in _EXCEL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in col_lower:
                mapping[col_lower[alias.lower()]] = pace_col
                break

    # Require at least shipment_id + 3 others to consider it a match
    mapped_pace_cols = set(mapping.values())
    must_have = {"shipment_id", "carrier", "ship_date"}
    if not must_have.issubset(mapped_pace_cols):
        return None

    remapped = df.rename(columns=mapping)
    # Keep only PACE columns that were found; fill missing ones with None
    for col in PACE_FIELDS:
        if col not in remapped.columns:
            remapped[col] = None
    return remapped[list(PACE_FIELDS.keys())]


# ── Semantic column mapping via sentence-transformers ─────────────────────────
# Rich keyword descriptions per PACE field — more synonyms = better matching
_PACE_MATCH_DESCRIPTIONS = {
    "shipment_id":            "shipment id BOL PRO order number reference load id tracking number bill of lading",
    "ship_date":              "ship date pickup date dispatch date shipment date order date",
    "carrier":                "carrier trucking company provider scac vendor transport name",
    "facility":               "facility destination warehouse distribution center DC consignee ship to",
    "weight_lbs":             "weight pounds lbs gross weight shipment weight total weight",
    "miles":                  "miles distance mileage route distance transit miles",
    "base_freight_usd":       "base freight linehaul cost rate charge dollar amount freight cost base rate",
    "accessorial_charge_usd": "accessorial detention surcharge penalty demurrage extra charges fees additional cost layover",
}

# Module-level cache so PACE embeddings are only computed once per process
_pace_embeddings_cache: dict = {}


def _st_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use sentence-transformers cosine similarity to map CSV/Excel column names
    to PACE field names.  Works offline, no API key, runs on Streamlit Cloud.
    Falls back to ensure_expected_columns() if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return ensure_expected_columns(df)


    model_name = "all-MiniLM-L6-v2"
    if model_name not in _pace_embeddings_cache:
        _model = SentenceTransformer(model_name)
        pace_keys  = list(_PACE_MATCH_DESCRIPTIONS.keys())
        pace_texts = list(_PACE_MATCH_DESCRIPTIONS.values())
        _pace_embeddings_cache[model_name] = {
            "model":       _model,
            "keys":        pace_keys,
            "embeddings":  _model.encode(pace_texts, convert_to_tensor=True),
        }

    cached      = _pace_embeddings_cache[model_name]
    st_model    = cached["model"]
    pace_keys   = cached["keys"]
    pace_emb    = cached["embeddings"]

    # Normalize uploaded column names: lowercase + strip punctuation for matching
    col_names   = df.columns.tolist()
    col_texts   = [c.lower().replace("_", " ").replace("-", " ") for c in col_names]
    col_emb     = st_model.encode(col_texts, convert_to_tensor=True)

    similarity  = util.cos_sim(col_emb, pace_emb)   # shape: (n_cols, n_pace_fields)

    # Greedy assignment: take highest-confidence pairs first, each field mapped once
    THRESHOLD = 0.35
    used_pace  = set()
    candidates = []
    for i, col in enumerate(col_names):
        for j, pace in enumerate(pace_keys):
            score = float(similarity[i][j])
            if score >= THRESHOLD:
                candidates.append((score, i, j))

    candidates.sort(reverse=True)
    rename = {}
    for score, i, j in candidates:
        pace = pace_keys[j]
        col  = col_names[i]
        if pace not in used_pace and col not in rename:
            if col != pace:           # skip if already the right name
                rename[col] = pace
            used_pace.add(pace)

    remapped = df.rename(columns=rename)
    for col in PACE_FIELDS:
        if col not in remapped.columns:
            remapped[col] = None

    remapped.attrs["column_mapping"] = rename
    return remapped[[c for c in PACE_FIELDS if c in remapped.columns] +
                    [c for c in remapped.columns if c not in PACE_FIELDS]]


# ── Ollama extraction ─────────────────────────────────────────────────────────
def _call_ollama(text: str) -> list[dict]:
    fields_desc = "\n".join(f'  "{k}": {v}' for k, v in PACE_FIELDS.items())
    prompt = (
        "You are a logistics data extraction assistant. "
        "Extract all shipment records from the document text below and return them as a JSON array.\n\n"
        "Each object in the array must have exactly these fields:\n"
        f"{fields_desc}\n\n"
        "Rules:\n"
        "- Return ONLY a valid JSON array, no explanation, no markdown fences\n"
        "- If a field cannot be found, use null\n"
        "- Extract ALL shipments found — there may be more than one\n"
        "- Normalize all dates to YYYY-MM-DD\n"
        "- Strip currency symbols from numeric fields\n"
        "- accessorial_charge_usd should be 0 if no accessorial charges are mentioned\n\n"
        f"Document:\n{text[:6000]}\n\n"
        "JSON array:"
    )

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out. The model may still be loading — try again in a moment.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Lost connection to Ollama during parsing.")

    raw = r.json().get("response", "").strip()

    # Strip markdown fences if model included them anyway
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first [...] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise RuntimeError(
                "Ollama returned a response that could not be parsed as JSON. "
                "Try again — the model may need a moment to warm up."
            )

    if isinstance(data, dict):
        # Unwrap if model returned {"shipments": [...]}
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]

    if isinstance(data, list):
        return data

    raise RuntimeError("Unexpected response structure from Ollama.")


# ── Main entry point ──────────────────────────────────────────────────────────
def parse_document(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse a shipping document into a PACE-compatible DataFrame.

    - Excel files: direct column mapping (no AI needed if columns are recognizable)
    - PDF / image files: text extraction → Ollama → structured JSON → DataFrame

    Raises ValueError / RuntimeError with user-friendly messages on failure.
    """
    ext = filename.lower().rsplit(".", 1)[-1]

    # ── CSV ────────────────────────────────────────────────────────────────────
    if ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Could not read CSV file: {e}")
        return _st_map_columns(df)

    # ── Excel ──────────────────────────────────────────────────────────────────
    if ext in ("xlsx", "xls"):
        try:
            raw = pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")

        mapped = _map_excel_columns(raw)
        if mapped is not None:
            return mapped

        # Columns didn't match — use semantic column mapping
        return _st_map_columns(raw)

    # ── PDF ────────────────────────────────────────────────────────────────────
    if ext == "pdf":
        text = _extract_pdf_text(file_bytes)
        records = _call_ollama(text)
        return _records_to_df(records)

    # ── Image ──────────────────────────────────────────────────────────────────
    if ext in ("png", "jpg", "jpeg"):
        text = _extract_image_text(file_bytes)
        records = _call_ollama(text)
        return _records_to_df(records)

    raise ValueError(f"Unsupported file type: .{ext}")


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert Ollama's list of dicts into a clean DataFrame with PACE columns."""
    if not records:
        raise ValueError("No shipment records were found in the document.")

    df = pd.DataFrame(records)

    # Ensure all PACE columns exist
    for col in PACE_FIELDS:
        if col not in df.columns:
            df[col] = None

    return df[list(PACE_FIELDS.keys())]


# ── Column normalization utilities (compatible with Bui's flexible mapping) ────
_RENAME_MAP = {
    # shipment id
    "load_id": "shipment_id", "order_id": "shipment_id",
    "id": "shipment_id", "shipment_number": "shipment_id",
    # date
    "date": "ship_date", "pickup_date": "ship_date",
    "shipment_date": "ship_date", "ship_dt": "ship_date",
    # carrier
    "carrier_name": "carrier", "trucking_company": "carrier", "provider": "carrier",
    # facility
    "warehouse": "facility", "location": "facility",
    "dc": "facility", "distribution_center": "facility",
    # weight
    "weight": "weight_lbs", "lbs": "weight_lbs",
    "weight_lb": "weight_lbs", "shipment_weight": "weight_lbs",
    # miles
    "distance": "miles", "distance_miles": "miles",
    "route_miles": "miles", "trip_miles": "miles",
    # freight
    "freight_cost": "base_freight_usd", "base_rate": "base_freight_usd",
    "freight": "base_freight_usd", "base_cost": "base_freight_usd",
    # accessorial / detention / surcharge (all map to the same target)
    "extra_charges": "accessorial_charge_usd", "accessorial_cost": "accessorial_charge_usd",
    "extra_cost": "accessorial_charge_usd", "accessorials": "accessorial_charge_usd",
    "extra_fees": "accessorial_charge_usd",
    "detention_fee": "accessorial_charge_usd", "detention": "accessorial_charge_usd",
    "surcharge": "accessorial_charge_usd", "surcharges": "accessorial_charge_usd",
    "penalty_fee": "accessorial_charge_usd", "penalty": "accessorial_charge_usd",
    "late_delivery_penalty": "accessorial_charge_usd", "late_fee": "accessorial_charge_usd",
    "demurrage_fee": "accessorial_charge_usd", "demurrage": "accessorial_charge_usd",
    "additional_cost": "accessorial_charge_usd", "additional_charges": "accessorial_charge_usd",
    "service_failure_cost": "accessorial_charge_usd", "overage_fee": "accessorial_charge_usd",
}

EXPECTED_COLUMNS = list(PACE_FIELDS.keys())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and underscore-normalize all column names."""
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and remap known aliases to PACE schema."""
    df = normalize_columns(df)
    mapped = {old: new for old, new in _RENAME_MAP.items()
              if old in df.columns and new not in df.columns}
    df = df.rename(columns=_RENAME_MAP)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df.attrs["column_mapping"] = mapped
    return df[[c for c in EXPECTED_COLUMNS if c in df.columns] +
               [c for c in df.columns if c not in EXPECTED_COLUMNS]]


# ── Streamlit-compatible entry point (accepts file-like objects) ───────────────
def parse_uploaded_document(file_obj, filename: str) -> pd.DataFrame:
    """
    Accepts a Streamlit UploadedFile (or any file-like object) and filename.
    Delegates to parse_document() after reading the bytes.
    This is the function imported by pages/2_Upload.py.
    """
    file_bytes = file_obj.read()
    return parse_document(file_bytes, filename)
