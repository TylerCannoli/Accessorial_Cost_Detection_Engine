"""
utils/database.py
Azure SQL database connection for PACE.
Reads credentials from .env (local) or st.secrets (Streamlit Cloud).
"""
import os
import hashlib
from typing import Optional, Tuple       # ← added from teammate (used by several functions)

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pymssql
    PYMSSQL_AVAILABLE = True
except ImportError:
    PYMSSQL_AVAILABLE = False


def _get_secret(key: str) -> str:
    """Read a secret from st.secrets (cloud) or os.environ (local)."""
    try:
        return st.secrets["azure_sql"][key]
    except Exception:                    # ← teammate's broader except (safer than KeyError only)
        return os.getenv(key, "")


# ── Added from teammate ───────────────────────────────────────────────────────
def get_db_config_status() -> Tuple[bool, str]:
    """Check whether all required DB env vars are present."""
    if not PYMSSQL_AVAILABLE:
        return False, "pymssql not installed"

    required = ["DB_SERVER", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD"]
    missing = [k for k in required if not _get_secret(k)]

    if missing:
        return False, f"Missing: {', '.join(missing)}"

    return True, "ok"


@st.cache_resource
def get_connection():
    """
    Return a cached pymssql connection to Azure SQL.
    Retries up to 3 times with a 2-second delay before giving up.
    Returns None if all attempts fail.

    NOTE: Uses @st.cache_resource so the connection is shared across
    all sessions and reruns — avoids reconnecting on every page load.
    The connection is tested with SELECT 1 before being returned.
    """
    import time

    if not PYMSSQL_AVAILABLE:
        return None

    server   = _get_secret("DB_SERVER")
    database = _get_secret("DB_DATABASE")
    username = _get_secret("DB_USERNAME")
    password = _get_secret("DB_PASSWORD")

    missing = [k for k, v in {"DB_SERVER": server, "DB_DATABASE": database,
                               "DB_USERNAME": username, "DB_PASSWORD": password}.items() if not v]
    if missing:
        return None

    last_err = None
    for attempt in range(1, 4):
        try:
            conn = pymssql.connect(
                server=server,
                user=username,
                password=password,
                database=database,
                port=1433,
                login_timeout=30,
                tds_version="7.4",
            )
            # Smoke-test: ensure the connection is usable
            conn.cursor().execute("SELECT 1")
            return conn
        except Exception as e:
            last_err = e
            if attempt < 3:
                # Azure SQL serverless can take 20-60 s to wake from idle;
                # 12 s between retries gives it enough time across 3 attempts.
                time.sleep(12)

    # All retries exhausted — fail silently, fallback to demo data
    return None


def get_connection_safe():
    """
    Wrapper around get_connection() that returns None if the cached
    connection has gone stale (e.g. Azure SQL idle timeout) rather than
    raising an error. Clears the cache and retries once if the connection
    fails the liveness check.
    """
    conn = get_connection()
    if conn is None:
        return None
    try:
        conn.cursor().execute("SELECT 1")
        return conn
    except Exception:
        # Connection went stale — clear cache and get a fresh one
        get_connection.clear()
        return get_connection()


# ── Added from teammate ───────────────────────────────────────────────────────
def test_connection() -> Tuple[bool, str]:
    """Create a fresh (uncached) connection to verify credentials are working."""
    ok, reason = get_db_config_status()
    if not ok:
        return False, reason

    try:
        conn = pymssql.connect(
            server=_get_secret("DB_SERVER"),
            user=_get_secret("DB_USERNAME"),
            password=_get_secret("DB_PASSWORD"),
            database=_get_secret("DB_DATABASE"),
            port=1433,
            login_timeout=30,   # Azure serverless needs 20-60 s to wake
            timeout=45,
            tds_version="7.4",
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
        return True, "Connected successfully."
    except Exception as e:
        return False, f"Database connection failed: {e}"


# ── Added from teammate ───────────────────────────────────────────────────────
def clear_db_cache():
    """Clear all Streamlit data and resource caches (used after user mutations)."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass


@st.cache_data
def get_tables(_conn) -> list[str]:
    """Return list of user table names from the database."""
    if _conn is None:
        return []
    query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
    """
    try:
        return pd.read_sql(query, _conn)["TABLE_NAME"].tolist()
    except Exception:
        return []


@st.cache_data
def get_table_data(_conn, table_name: str, row_limit: int = 500) -> pd.DataFrame:
    """Fetch up to row_limit rows from a table."""
    if _conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(f"SELECT TOP {row_limit} * FROM [{table_name}]", _conn)  # nosec B608
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_shipments(_conn) -> pd.DataFrame:
    """Fetch all shipment records joined with carrier name."""
    if _conn is None:
        return pd.DataFrame()
    query = """
        SELECT
            s.ShipmentId       AS shipment_id,
            s.ShipDate         AS ship_date,
            c.carrier_name     AS carrier,
            s.FacilityType     AS facility,
            s.weight_lbs,
            s.DistanceMiles    AS miles,
            s.LinehaulCost     AS base_freight_usd,
            s.AccessorialCost  AS accessorial_charge_usd,
            s.OriginRegion,
            s.DestRegion,
            s.Revenue,
            s.AccessorialFlag,
            s.risk_score,
            s.risk_tier,
            s.AppointmentType,
            c.dot_number
        FROM Shipments s
        LEFT JOIN Carriers c ON s.CarrierId = c.carrier_id
        ORDER BY s.ShipDate DESC
    """
    try:
        df = pd.read_sql(query, _conn)
        df["base_freight_usd"]        = pd.to_numeric(df["base_freight_usd"],        errors="coerce").fillna(0)
        df["accessorial_charge_usd"]  = pd.to_numeric(df["accessorial_charge_usd"],  errors="coerce").fillna(0)
        df["miles"]                   = pd.to_numeric(df["miles"],                   errors="coerce").fillna(0)
        df["total_cost_usd"]          = df["base_freight_usd"] + df["accessorial_charge_usd"]
        df["cost_per_mile"]           = (
            df["base_freight_usd"] / df["miles"].replace(0, float("nan"))
        ).fillna(0)
        df["lane"]             = df["OriginRegion"].fillna("?") + " → " + df["DestRegion"].fillna("?")
        df["origin_city"]      = df["OriginRegion"]
        df["destination_city"] = df["DestRegion"]
        # Normalize risk_score to 0-100 if stored as 0-1 in the DB
        if "risk_score" in df.columns:
            max_score = df["risk_score"].max()
            if max_score <= 1.0:
                df["risk_score"] = (df["risk_score"] * 100).round(2)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_accessorial_charges(_conn, row_limit: int = 2000) -> pd.DataFrame:
    """Fetch accessorial charge records."""
    if _conn is None:
        return pd.DataFrame()
    query = f"""
        SELECT TOP {row_limit}
            ac.charge_id,
            ac.shipment_id,
            ac.charge_type,
            ac.amount,
            ac.risk_flag,
            ac.invoice_date,
            ac.disputed,
            ac.dispute_resolved,
            ac.notes
        FROM Accessorial_Charges ac
        ORDER BY ac.invoice_date DESC
    """  # nosec B608
    try:
        return pd.read_sql(query, _conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_shipments_with_charges(_conn, row_limit: int = 2000) -> pd.DataFrame:
    """
    Accessorial Charges joined to Shipment + Carrier for the Accessorial Tracker page.
    Returns one row per charge with carrier/facility context.
    """
    if _conn is None:
        return pd.DataFrame()
    query = f"""
        SELECT TOP {row_limit}
            ac.charge_id,
            ac.shipment_id,
            ac.charge_type        AS accessorial_type,
            ac.amount             AS accessorial_charge_usd,
            ac.risk_flag,
            ac.invoice_date       AS ship_date,
            ac.disputed,
            s.FacilityType        AS facility,
            s.LinehaulCost        AS base_freight_usd,
            s.risk_score,
            s.risk_tier,
            c.carrier_name        AS carrier
        FROM Accessorial_Charges ac
        LEFT JOIN Shipments s  ON ac.shipment_id = s.ShipmentId
        LEFT JOIN Carriers  c  ON s.CarrierId    = c.carrier_id
        ORDER BY ac.invoice_date DESC
    """  # nosec B608
    try:
        df = pd.read_sql(query, _conn)
        df["base_freight_usd"]       = pd.to_numeric(df["base_freight_usd"],       errors="coerce").fillna(0)
        df["accessorial_charge_usd"] = pd.to_numeric(df["accessorial_charge_usd"], errors="coerce").fillna(0)
        df["total_cost_usd"]         = df["base_freight_usd"] + df["accessorial_charge_usd"]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_carriers(_conn) -> pd.DataFrame:
    """Fetch all carrier records."""
    if _conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM Carriers ORDER BY carrier_name", _conn)
    except Exception:
        return pd.DataFrame()


def update_carrier_dot_number(_conn, carrier_name: str, dot_number: str) -> Tuple[bool, str]:
    """
    Set or clear the dot_number for a carrier row identified by carrier_name.
    dot_number should be a non-empty string like '72011', or None/'' to clear it.
    Returns (success, message).
    """
    if _conn is None:
        return False, "No database connection."
    dot_val = str(dot_number).strip() if dot_number and str(dot_number).strip() else None
    try:
        cursor = _conn.cursor()
        cursor.execute(
            "UPDATE Carriers SET dot_number = %s WHERE carrier_name = %s",
            (dot_val, carrier_name),
        )
        if cursor.rowcount == 0:
            return False, f"No carrier found with name '{carrier_name}'."
        _conn.commit()
        clear_db_cache()
        return True, f"DOT number updated for '{carrier_name}'."
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=300)
def get_facilities(_conn) -> pd.DataFrame:
    """Fetch all facility records."""
    if _conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM Facilities ORDER BY facility_name", _conn)
    except Exception:
        return pd.DataFrame()


def verify_pace_user(_conn, username: str, password: str) -> Optional[str]:
    """
    Verify a username/password against the PaceUsers table.
    Passwords are stored as SHA-256 hex digests.
    Returns the user's role string on success, None on failure.
    """
    if _conn is None:
        return None

    # ← teammate's normalization added: case-insensitive login
    username = username.lower().strip()
    pw_hash  = hashlib.sha256(password.encode()).hexdigest()

    try:
        row = pd.read_sql(
            "SELECT role FROM PaceUsers WHERE LOWER(username) = %s AND password_hash = %s",
            _conn,
            params=(username, pw_hash),
        )
        if not row.empty:
            return str(row.iloc[0]["role"])
    except Exception:
        pass
    return None


def get_pace_users(_conn) -> pd.DataFrame:
    """Return all PaceUsers rows (no password_hash)."""
    if _conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(
            "SELECT username, role, created_at FROM PaceUsers ORDER BY username",
            _conn,
        )
    except Exception:
        try:
            return pd.read_sql("SELECT username, role FROM PaceUsers ORDER BY username", _conn)
        except Exception:
            return pd.DataFrame()


def create_pace_user(_conn, username: str, password: str, role: str) -> Tuple[bool, str]:
    """Insert a new user into PaceUsers. Returns (success, message)."""
    if _conn is None:
        return False, "No database connection."

    # ← teammate: normalize + validate inputs
    username = str(username).strip().lower()
    role     = str(role).strip().lower()

    if not username or not password:
        return False, "Username and password are required."

    if role not in {"user", "admin"}:
        return False, "Invalid role."

    pw_hash = hashlib.sha256(password.encode()).hexdigest()

    try:
        # ← teammate: duplicate-user check before insert
        existing = pd.read_sql(
            "SELECT username FROM PaceUsers WHERE LOWER(username) = %s",
            _conn,
            params=(username,),
        )
        if not existing.empty:
            return False, f"User '{username}' already exists."

        cursor = _conn.cursor()
        cursor.execute(
            "INSERT INTO PaceUsers (username, password_hash, role) VALUES (%s, %s, %s)",
            (username, pw_hash, role),
        )
        _conn.commit()
        clear_db_cache()                 # ← teammate: refresh cache after mutation
        return True, f"User '{username}' created successfully."
    except Exception as e:
        return False, str(e)


def delete_pace_user(
    _conn,
    username: str,
    current_username: Optional[str] = None,   # ← teammate: self-delete guard
) -> Tuple[bool, str]:
    """Delete a user from PaceUsers by username."""
    if _conn is None:
        return False, "No database connection."

    username         = str(username).strip().lower()
    current_username = (current_username or "").strip().lower()

    # ← teammate: prevent admin from deleting their own account
    if username == current_username:
        return False, "You cannot delete the currently logged-in admin."

    try:
        cursor = _conn.cursor()
        cursor.execute("DELETE FROM PaceUsers WHERE LOWER(username) = %s", (username,))
        _conn.commit()
        clear_db_cache()                 # ← teammate: refresh cache after mutation
        return True, f"User '{username}' deleted."
    except Exception as e:
        return False, str(e)


# =============================================================================
# FALLBACK / LOAD HELPERS
# =============================================================================

def load_shipments_from_teradata(row_limit: int = 50000) -> pd.DataFrame:
    """
    Pull records from the Teradata training view and reshape them into the
    dashboard schema (same column names as get_shipments()).
    """
    try:
        td_host = os.getenv("TD_HOST", "")
        td_user = os.getenv("TD_USERNAME", "")
        td_pass = os.getenv("TD_PASSWORD", "")
        td_db   = os.getenv("TD_DATABASE", "CTGAN")
        td_view = os.getenv("TD_VIEW", "pace_synthetic_v")

        if not td_host:
            return pd.DataFrame()

        import teradatasql
        conn = teradatasql.connect(
            host=td_host, user=td_user,
            password=td_pass, database=td_db,
        )

        query = f"""
            SELECT TOP {row_limit}
                dot_number,
                insp_year,
                insp_month,
                carrier_phy_state,
                sms_nbr_power_unit,
                carrier_mcs150_mileage,
                carrier_power_units,
                carrier_total_drivers,
                carrier_carrier_operation,
                carrier_status_code,
                carrier_fleetsize,
                accessorial_risk_score,
                accessorial_type
            FROM {td_db}.{td_view}
            WHERE accessorial_risk_score IS NOT NULL
            ORDER BY insp_year DESC, insp_month DESC
        """  # nosec B608
        raw = pd.read_sql(query, conn)
        conn.close()

        if raw.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df["shipment_id"] = raw["dot_number"].astype(str)
        df["carrier"]     = "DOT-" + raw["dot_number"].astype(str)

        year  = pd.to_numeric(raw["insp_year"],  errors="coerce").fillna(2024).astype(int)
        month = pd.to_numeric(raw["insp_month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
        df["ship_date"] = pd.to_datetime(
            year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )

        state = raw["carrier_phy_state"].fillna("Unknown")
        df["facility"]         = state
        df["OriginRegion"]     = state
        df["DestRegion"]       = "Unknown"
        df["origin_city"]      = state
        df["destination_city"] = "Unknown"
        df["lane"]             = state + " → Unknown"

        df["weight_lbs"] = pd.to_numeric(raw["sms_nbr_power_unit"],    errors="coerce").fillna(0) * 15
        df["miles"]      = pd.to_numeric(raw["carrier_mcs150_mileage"], errors="coerce").fillna(0).clip(0, 500000)

        raw_score = pd.to_numeric(raw["accessorial_risk_score"], errors="coerce").fillna(0)
        score_max = raw_score.max() if raw_score.max() > 0 else 1
        df["risk_score"] = (raw_score / score_max * 100).clip(0, 100).round(1)

        def _tier(s):
            if s >= 75: return "High"
            if s >= 40: return "Medium"
            return "Low"
        df["risk_tier"] = df["risk_score"].apply(_tier)

        df["accessorial_charge_usd"] = df["risk_score"] * 12
        df["base_freight_usd"]       = df["miles"].clip(100, 5000) * 2.1
        df["total_cost_usd"]         = df["base_freight_usd"] + df["accessorial_charge_usd"]
        df["cost_per_mile"]          = (
            df["base_freight_usd"] / df["miles"].replace(0, float("nan"))
        ).fillna(0)
        df["AppointmentType"] = "Unknown"
        df["Revenue"]         = df["base_freight_usd"]
        df["AccessorialFlag"] = (df["risk_score"] >= 40).astype(int)

        return df

    except Exception:
        return pd.DataFrame()


def load_teradata_for_inference(row_limit: int = 5000) -> pd.DataFrame:
    """
    Pull full-schema PACE records from Teradata for live inference without requiring
    a file upload.  Returns all columns from pace_training_v so the inference engine
    can score them directly via predict_dataframe().

    Falls back to an empty DataFrame (and logs nothing) if Teradata is unavailable,
    so the caller can handle the fallback gracefully.
    """
    try:
        td_host = os.getenv("TD_HOST", "")
        td_user = os.getenv("TD_USERNAME", "")
        td_pass = os.getenv("TD_PASSWORD", "")
        td_db   = os.getenv("TD_DATABASE", "CTGAN")

        if not td_host:
            return pd.DataFrame()

        import teradatasql  # cluster-only dependency; imported lazily
        conn = teradatasql.connect(
            host=td_host, user=td_user,
            password=td_pass, database=td_db,
        )

        # pace_training_v contains all PACE schema columns + computed labels
        query = f"""
            SELECT TOP {row_limit} *
            FROM {td_db}.pace_training_v
            ORDER BY insp_year DESC, insp_month DESC
        """  # nosec B608
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    except Exception:
        return pd.DataFrame()


def load_shipments_with_fallback(n_mock: int = 300) -> pd.DataFrame:
    """
    Load shipments with a three-tier fallback:
      1. Azure SQL  (live data)
      2. Teradata   (pace_synthetic_v training view)
      3. Mock data  (generated, n_mock rows)
    """
    # ── Tier 1: Azure SQL ─────────────────────────────────────────────
    conn = get_connection_safe()
    df = get_shipments(conn) if conn is not None else pd.DataFrame()
    # If Azure is waking from idle/serverless pause, first query can return empty.
    # Retry once after cache clear before falling back.
    if conn is not None and df.empty:
        clear_db_cache()
        conn = get_connection_safe()
        df = get_shipments(conn) if conn is not None else pd.DataFrame()
    if not df.empty:
        return df

    # ── Tier 2: Teradata ──────────────────────────────────────────────
    df = load_shipments_from_teradata()
    if not df.empty:
        st.info(
            f"Azure SQL unavailable — showing {len(df):,} records from Teradata.",
            icon="🗄️",
        )
        return df

    # ── Tier 3: Mock data ─────────────────────────────────────────────
    from utils.mock_data import generate_mock_shipments
    df = generate_mock_shipments(n_mock)
    # Mock data outputs risk_score in 0-1 range; normalize to 0-100 to match
    # the DB schema so dashboard metrics display correctly (e.g. "60%" not "0.6%")
    if "risk_score" in df.columns and df["risk_score"].max() <= 1.0:
        df["risk_score"] = (df["risk_score"] * 100).round(1)
    st.info("Live database unavailable — showing demo data.", icon="ℹ️")
    return df


# ── Added from teammate ───────────────────────────────────────────────────────
def load_accessorial_with_fallback(row_limit: int = 2000) -> pd.DataFrame:
    """
    Load accessorial charges with fallback to mock data if Azure is unavailable.
    """
    conn = get_connection_safe()

    if conn is not None:
        df = get_shipments_with_charges(conn, row_limit=row_limit)
        if df.empty:
            clear_db_cache()
            conn = get_connection_safe()
            df = get_shipments_with_charges(conn, row_limit=row_limit) if conn is not None else pd.DataFrame()
        if not df.empty:
            return df

    try:
        from utils.mock_data import generate_mock_shipments
        df = generate_mock_shipments(min(row_limit, 300)).copy()

        if "accessorial_type" not in df.columns:
            if "accessorial_charge_usd" in df.columns:
                df["accessorial_type"] = df["accessorial_charge_usd"].apply(
                    lambda x: "Accessorial" if float(x or 0) > 0 else "None"
                )
            else:
                df["accessorial_type"] = "None"

        if "total_cost_usd" not in df.columns:
            base = df["base_freight_usd"]       if "base_freight_usd"       in df.columns else 0
            acc  = df["accessorial_charge_usd"] if "accessorial_charge_usd" in df.columns else 0
            df["total_cost_usd"] = base + acc

        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# MODEL RESULTS
# =============================================================================

def ensure_model_results_table(_conn) -> None:
    """
    Create the ModelResults table if it does not already exist.

    Schema:
        result_id          INT IDENTITY PRIMARY KEY
        dot_number         VARCHAR(20)    — USDOT number (nullable for manual/batch inputs)
        run_timestamp      DATETIME       — UTC time of inference
        risk_score         FLOAT          — predicted risk score (0–100)
        risk_label         VARCHAR(20)    — Low / Medium / High / Critical / None
        charge_type        VARCHAR(50)    — predicted charge type label
        probabilities_json NVARCHAR(MAX)  — JSON object of per-class probabilities
        input_source       VARCHAR(50)    — live_fmcsa / manual_input / csv_batch / etc.
    """
    if _conn is None:
        return
    ddl = """
        IF NOT EXISTS (
            SELECT 1 FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = 'ModelResults'
        )
        BEGIN
            CREATE TABLE ModelResults (
                result_id          INT IDENTITY(1,1) PRIMARY KEY,
                dot_number         VARCHAR(20)    NULL,
                run_timestamp      DATETIME       NOT NULL,
                risk_score         FLOAT          NOT NULL,
                risk_label         VARCHAR(20)    NOT NULL,
                charge_type        VARCHAR(50)    NOT NULL,
                probabilities_json NVARCHAR(MAX)  NULL,
                input_source       VARCHAR(50)    NOT NULL
            )
        END
    """
    try:
        cursor = _conn.cursor()
        cursor.execute(ddl)
        _conn.commit()
    except Exception:
        pass


def write_model_result(
    _conn,
    dot_number: str,
    run_timestamp,
    risk_score: float,
    risk_label: str,
    charge_type: str,
    probabilities_json: str,
    input_source: str,
) -> bool:
    """
    Insert one inference result into ModelResults.
    Silently returns False on failure so scoring is never blocked by a DB error.
    """
    if _conn is None:
        return False
    ensure_model_results_table(_conn)
    try:
        cursor = _conn.cursor()
        cursor.execute(
            """
            INSERT INTO ModelResults
                (dot_number, run_timestamp, risk_score, risk_label,
                 charge_type, probabilities_json, input_source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (dot_number, run_timestamp, risk_score, risk_label,
             charge_type, probabilities_json, input_source),
        )
        _conn.commit()
        return True
    except Exception:
        return False


@st.cache_data(ttl=300)
def get_model_results(_conn, row_limit: int = 2000) -> pd.DataFrame:
    """Return recent ModelResults rows ordered by run_timestamp descending."""
    if _conn is None:
        return pd.DataFrame()
    query = f"""
        SELECT TOP {row_limit}
            result_id, dot_number, run_timestamp, risk_score,
            risk_label, charge_type, probabilities_json, input_source
        FROM ModelResults
        ORDER BY run_timestamp DESC
    """  # nosec B608
    try:
        return pd.read_sql(query, _conn)
    except Exception:
        return pd.DataFrame()