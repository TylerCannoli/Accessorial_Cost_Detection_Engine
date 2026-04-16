"""
upload_enrichments_to_teradata.py
==================================
Uploads all Phase 2 enrichment CSV files to Teradata as lookup tables,
then creates the updated pace_synthetic_v training view.

Run this script ON THE UNIVERSITY VM (or any machine with Teradata access).
The enrichment CSVs come from the PACE GitHub repo — just git pull first.

PREREQUISITES
-------------
1. The university VM has teradatasql installed:
       pip install teradatasql
2. Teradata credentials are set in your .env or environment variables:
       TD_HOST, TD_USERNAME, TD_PASSWORD, TD_DATABASE
3. You have CREATE TABLE and CREATE VIEW permissions in the CTGAN database.
   If not, ask the university DBA for:
       GRANT CREATE TABLE ON CTGAN TO <your_username>;
       GRANT CREATE VIEW  ON CTGAN TO <your_username>;

USAGE
-----
    # From the PACE repo root on the university VM:
    git pull
    python scripts/upload_enrichments_to_teradata.py

    # Dry run (show what would happen, no DB writes):
    python scripts/upload_enrichments_to_teradata.py --dry-run

    # Upload only specific tables:
    python scripts/upload_enrichments_to_teradata.py --tables carrier_history sms_percentiles

    # Skip recreating the view (upload tables only):
    python scripts/upload_enrichments_to_teradata.py --skip-view
"""

import os
import sys
import argparse
import math
import pandas as pd
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Teradata credentials ───────────────────────────────────────────────
TD_HOST     = os.getenv("TD_HOST", "")
TD_USERNAME = os.getenv("TD_USERNAME", "")
TD_PASSWORD = os.getenv("TD_PASSWORD", "")
TD_DATABASE = os.getenv("TD_DATABASE", "CTGAN")

REPO_ROOT  = Path(__file__).parents[1]
ENRICH_DIR = REPO_ROOT / "outputs" / "enrichment"
SQL_FILE   = REPO_ROOT / "scripts" / "create_teradata_view_v2.sql"

CHUNK_SIZE = 5_000   # rows per INSERT batch


# ── Table definitions ──────────────────────────────────────────────────
TABLES = {
    "carrier_history": {
        "csv":        ENRICH_DIR / "carrier_oos_history.csv",
        "table":      f"{TD_DATABASE}.pace_carrier_history",
        "key":        "dot_number",
        "ddl_types": {
            "dot_number":                    "VARCHAR(20)",
            "carrier_hist_insp_count":       "INTEGER",
            "carrier_hist_oos_rate":         "FLOAT",
            "carrier_hist_driver_oos_rate":  "FLOAT",
            "carrier_hist_vehicle_oos_rate": "FLOAT",
            "carrier_hist_hazmat_oos_rate":  "FLOAT",
            "carrier_hist_avg_viol_per_insp":"FLOAT",
            "carrier_hist_driver_viol_rate": "FLOAT",
            "carrier_hist_vehicle_viol_rate":"FLOAT",
            "carrier_hist_hazmat_viol_rate": "FLOAT",
            "carrier_hist_post_acc_rate":    "FLOAT",
        },
    },
    "sms_percentiles": {
        "csv":    ENRICH_DIR / "sms_percentile_features.csv",
        "table":  f"{TD_DATABASE}.pace_sms_percentiles",
        "key":    "dot_number",
        "ddl_types": {
            "dot_number":               "VARCHAR(20)",
            "sms_insp_total":           "INTEGER",
            "sms_driver_insp_total":    "INTEGER",
            "sms_driver_oos_insp_total":"INTEGER",
            "sms_vehicle_insp_total":   "INTEGER",
            "sms_vehicle_oos_insp_total":"INTEGER",
            "sms_unsafe_driv_measure":  "FLOAT",
            "sms_unsafe_driv_alert":    "INTEGER",
            "sms_hos_measure":          "FLOAT",
            "sms_hos_alert":            "INTEGER",
            "sms_driv_fit_measure":     "FLOAT",
            "sms_driv_fit_alert":       "INTEGER",
            "sms_contr_subst_measure":  "FLOAT",
            "sms_contr_subst_alert":    "INTEGER",
            "sms_veh_maint_measure":    "FLOAT",
            "sms_veh_maint_alert":      "INTEGER",
            "sms_alert_count":          "INTEGER",
            "sms_max_measure":          "FLOAT",
        },
    },
    "parking_features": {
        "csv":    ENRICH_DIR / "state_parking_features.csv",
        "table":  f"{TD_DATABASE}.pace_parking_features",
        "key":    "state_abbr",
        "ddl_types": {
            "state_abbr":                     "CHAR(2)",
            "state_parking_spots_total":      "FLOAT",
            "state_parking_spots_per_100mi":  "FLOAT",
            "state_parking_scarcity_index":   "FLOAT",
        },
    },
    "fatal_crash_features": {
        "csv":    ENRICH_DIR / "state_fatal_crash_features.csv",
        "table":  f"{TD_DATABASE}.pace_fatal_crash_features",
        "key":    "state_abbr",
        "ddl_types": {
            "state_abbr":                        "CHAR(2)",
            "state_fatal_crashes_2023":          "INTEGER",
            "state_fatal_fatalities_2023":       "INTEGER",
            "state_avg_fatals_per_crash":        "FLOAT",
            "state_crashes_involving_trucks":    "FLOAT",
            "state_fatal_crash_index":           "FLOAT",
        },
    },
    "atri_features": {
        "csv":    ENRICH_DIR / "atri_detention_features.csv",
        "table":  f"{TD_DATABASE}.pace_atri_features",
        "key":    "state_abbr",
        "ddl_types": {
            "state_abbr":                        "CHAR(2)",
            "state_atri_detention_propensity":   "FLOAT",
            "state_atri_severity_index":         "FLOAT",
        },
    },
    "freight_rates": {
        "csv":    ENRICH_DIR / "freight_rate_features.csv",
        "table":  f"{TD_DATABASE}.pace_freight_rates",
        "key":    ["year", "month"],
        "ddl_types": {
            "year":                          "INTEGER",
            "month":                         "INTEGER",
            "ftsi_index":                    "FLOAT",
            "ftsi_yoy_pct":                  "FLOAT",
            "ppi_trucking_mom_pct":          "FLOAT",
            "lmi_composite_index":           "FLOAT",
            "lmi_transportation_capacity":   "FLOAT",
            "lmi_transportation_utilization":"FLOAT",
            "lmi_transportation_prices":     "FLOAT",
            "freight_tightness_score":       "FLOAT",
        },
    },
    "faf_features": {
        "csv":    ENRICH_DIR / "faf_freight_features.csv",
        "table":  f"{TD_DATABASE}.pace_faf_features",
        "key":    "state_abbr",
        "ddl_types": {
            "state_abbr":                   "CHAR(2)",
            "faf_state_freight_intensity":  "FLOAT",
            "faf_state_hazmat_share":       "FLOAT",
            "faf_state_reefer_share_faf":   "FLOAT",
        },
    },
}


def get_connection():
    import teradatasql
    if not TD_HOST:
        raise ValueError(
            "TD_HOST is not set. Set environment variables: "
            "TD_HOST, TD_USERNAME, TD_PASSWORD, TD_DATABASE"
        )
    return teradatasql.connect(
        host=TD_HOST, user=TD_USERNAME,
        password=TD_PASSWORD, database=TD_DATABASE
    )


def table_exists(cursor, table_name: str) -> bool:
    db, tbl = table_name.split(".")
    try:
        cursor.execute(
            f"SELECT COUNT(*) FROM DBC.TablesV WHERE DatabaseName='{db}' AND TableName='{tbl}'"
        )
        return cursor.fetchone()[0] > 0
    except Exception:
        return False


def drop_table_if_exists(cursor, table_name: str):
    if table_exists(cursor, table_name):
        cursor.execute(f"DROP TABLE {table_name}")
        print(f"    Dropped existing {table_name}")


def create_table(cursor, table_name: str, ddl_types: dict):
    col_defs = ", ".join(f"{col} {dtype}" for col, dtype in ddl_types.items())
    cursor.execute(f"CREATE TABLE {table_name} ({col_defs})")
    print(f"    Created {table_name}")


def upload_table(conn, spec: dict, dry_run: bool = False) -> bool:
    csv_path = spec["csv"]
    table    = spec["table"]
    types    = spec["ddl_types"]

    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found — run enrichment script first")
        return False

    df = pd.read_csv(csv_path, low_memory=False)
    # Keep only columns defined in DDL
    keep = [c for c in types.keys() if c in df.columns]
    df   = df[keep].copy()

    # Type coercions
    for col, dtype in types.items():
        if col not in df.columns:
            continue
        if "INT" in dtype:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        elif "FLOAT" in dtype:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = df[col].astype(str).str.strip()

    print(f"  {table}: {len(df):,} rows × {len(df.columns)} columns")

    if dry_run:
        print(f"    [DRY RUN] Would upload {len(df):,} rows")
        return True

    cursor = conn.cursor()
    drop_table_if_exists(cursor, table)
    create_table(cursor, table, {k: v for k, v in types.items() if k in df.columns})

    # Insert in chunks
    placeholders = "(" + ",".join(["?"] * len(df.columns)) + ")"
    insert_sql   = f"INSERT INTO {table} VALUES {placeholders}"
    total_chunks = math.ceil(len(df) / CHUNK_SIZE)

    for i, start in enumerate(range(0, len(df), CHUNK_SIZE)):
        chunk = df.iloc[start:start + CHUNK_SIZE]
        cursor.executemany(insert_sql, chunk.values.tolist())
        if (i + 1) % 20 == 0 or (i + 1) == total_chunks:
            print(f"    Inserted chunk {i+1}/{total_chunks}")

    conn.commit()
    print(f"    Done — {len(df):,} rows uploaded to {table}")
    return True


def run_view_sql(conn, sql_file: Path, dry_run: bool = False):
    """Execute the CREATE VIEW SQL."""
    if not sql_file.exists():
        print(f"  SKIP: {sql_file} not found")
        return

    with open(sql_file, encoding="utf-8") as f:
        sql = f.read()

    # Split on semicolons (Teradata doesn't support multi-statement execution)
    statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]

    if dry_run:
        print(f"  [DRY RUN] Would execute {len(statements)} SQL statements")
        return

    cursor = conn.cursor()
    for i, stmt in enumerate(statements):
        if not stmt:
            continue
        try:
            cursor.execute(stmt)
            print(f"  SQL {i+1}/{len(statements)}: OK")
        except Exception as e:
            # DROP VIEW/TABLE may fail if doesn't exist — that's OK
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                print(f"  SQL {i+1}: OK (object did not exist, DROP skipped)")
            else:
                print(f"  SQL {i+1}: ERROR — {e}")
                raise
    conn.commit()
    print("  View created successfully")


def main():
    parser = argparse.ArgumentParser(description="Upload enrichment tables to Teradata")
    parser.add_argument("--dry-run",    action="store_true", help="Show what would happen, no DB writes")
    parser.add_argument("--skip-view",  action="store_true", help="Upload tables but skip creating view")
    parser.add_argument("--tables",     nargs="+", choices=list(TABLES.keys()),
                        default=list(TABLES.keys()), help="Which tables to upload")
    args = parser.parse_args()

    print(f"PACE Phase 2 — Teradata Enrichment Upload")
    print(f"Database: {TD_DATABASE} @ {TD_HOST or '(TD_HOST not set)'}")
    print(f"Tables:   {args.tables}")
    print(f"Dry run:  {args.dry_run}")
    print()

    if not args.dry_run:
        try:
            conn = get_connection()
            print("Connected to Teradata\n")
        except Exception as e:
            print(f"Connection failed: {e}")
            print()
            print("Check that these environment variables are set:")
            print("  TD_HOST, TD_USERNAME, TD_PASSWORD, TD_DATABASE")
            sys.exit(1)
    else:
        conn = None

    # Upload each table
    success = []
    for name in args.tables:
        spec = TABLES[name]
        print(f"Uploading: {name}")
        try:
            ok = upload_table(conn, spec, dry_run=args.dry_run)
            if ok:
                success.append(name)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nUploaded {len(success)}/{len(args.tables)} tables: {success}")

    # Create view
    if not args.skip_view:
        print(f"\nCreating pace_synthetic_v view ...")
        try:
            run_view_sql(conn, SQL_FILE, dry_run=args.dry_run)
        except Exception as e:
            print(f"  View creation failed: {e}")
            print("  You can run create_teradata_view_v2.sql manually in Teradata Studio")

    if conn and not args.dry_run:
        conn.close()

    print("\nDone. Next step: retrain the FT-Transformer")
    print("  ssh ubuntu@131.186.6.189")
    print("  cd /home/ubuntu/PACE && git pull")
    print("  # pipeline/pace_transformer.py already imports CONTINUOUS_COLUMNS_V2 as CONTINUOUS_COLUMNS")
    print("  python pipeline/pace_transformer.py")


if __name__ == "__main__":
    main()
