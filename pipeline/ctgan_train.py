import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import teradatasql
from ctgan import CTGAN
from pipeline.config import (
    TD_HOST, TD_USERNAME, TD_PASSWORD, TD_DATABASE,
    TD_SOURCE_TABLE, TD_SYNTHETIC_TABLE,
    CTGAN_TRAIN_ROWS, CTGAN_SYNTHETIC_ROWS,
    CTGAN_EPOCHS, CTGAN_BATCH_SIZE,
    CTGAN_MODEL_PATH, SYNTHETIC_CSV_PATH,
    CTGAN_DISCRETE_COLUMNS,
)
from pipeline.data_pipeline import BOOL_COLS

warnings.filterwarnings("ignore")


def get_connection():
    """Return connection."""
    return teradatasql.connect(
        host=TD_HOST, user=TD_USERNAME,
        password=TD_PASSWORD, database=TD_DATABASE
    )


def load_sample() -> pd.DataFrame:
    print(f"[1/5] Loading {CTGAN_TRAIN_ROWS:,} rows from {TD_SOURCE_TABLE}...")
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT * FROM {TD_DATABASE}.{TD_SOURCE_TABLE} SAMPLE {CTGAN_TRAIN_ROWS}",  # nosec B608
        conn
    )
    conn.close()
    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Preprocessing...")

    # Fix boolean-as-string columns
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].map({"TRUE": "Y", "FALSE": "N"}).fillna("N")

    # Cast all discrete columns to string
    for col in CTGAN_DISCRETE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")

    # Cast numeric columns
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)

    # Drop rows with nulls in key columns
    df = df.dropna()

    print(f"  After cleaning: {df.shape[0]:,} rows")
    return df


def train_ctgan(df: pd.DataFrame) -> CTGAN:
    print("[3/5] Training CTGAN...")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {CTGAN_EPOCHS} | Batch size: {CTGAN_BATCH_SIZE}")
    discrete_cols = [c for c in CTGAN_DISCRETE_COLUMNS if c in df.columns]
    model = CTGAN(
        epochs=CTGAN_EPOCHS,
        batch_size=CTGAN_BATCH_SIZE,
        verbose=True,
        cuda=torch.cuda.is_available(),
    )
    model.fit(df, discrete_columns=discrete_cols)
    os.makedirs(os.path.dirname(CTGAN_MODEL_PATH), exist_ok=True)
    with open(CTGAN_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to {CTGAN_MODEL_PATH}")
    return model


def generate_synthetic(model: CTGAN) -> pd.DataFrame:
    print(f"[4/5] Generating {CTGAN_SYNTHETIC_ROWS:,} synthetic rows...")
    synthetic_df = model.sample(CTGAN_SYNTHETIC_ROWS)
    os.makedirs(os.path.dirname(SYNTHETIC_CSV_PATH), exist_ok=True)
    synthetic_df.to_csv(SYNTHETIC_CSV_PATH, index=False)
    print(f"  CSV backup saved to {SYNTHETIC_CSV_PATH}")
    return synthetic_df


def write_to_teradata(synthetic_df: pd.DataFrame):
    print(f"[5/5] Writing to Teradata table {TD_SYNTHETIC_TABLE}...")
    conn = get_connection()
    cursor = conn.cursor()

    # Drop if exists — Teradata has no "DROP TABLE IF EXISTS" syntax,
    # so we catch the error raised when the table is absent.
    try:
        cursor.execute(f"DROP TABLE {TD_DATABASE}.{TD_SYNTHETIC_TABLE}")
        print(f"  Dropped existing {TD_SYNTHETIC_TABLE}")
    except Exception:
        pass  # Table not found — safe to proceed with CREATE

    # Build CREATE TABLE
    col_defs = []
    discrete_set = set(CTGAN_DISCRETE_COLUMNS)
    for col in synthetic_df.columns:
        if col in discrete_set:
            col_defs.append(f"{col} VARCHAR(100)")
        elif synthetic_df[col].dtype in [np.float32, np.float64]:
            col_defs.append(f"{col} FLOAT")
        else:
            col_defs.append(f"{col} INTEGER")

    cursor.execute(
        f"CREATE TABLE {TD_DATABASE}.{TD_SYNTHETIC_TABLE} ({', '.join(col_defs)})"
    )
    print(f"  Created table {TD_SYNTHETIC_TABLE}")

    # Insert in chunks
    chunk_size = 10000
    total = len(synthetic_df)
    n_chunks = total // chunk_size + 1
    placeholders = "(" + ", ".join(["?"] * len(synthetic_df.columns)) + ")"
    insert_sql = f"INSERT INTO {TD_DATABASE}.{TD_SYNTHETIC_TABLE} VALUES {placeholders}"  # nosec B608

    for i, start in enumerate(range(0, total, chunk_size)):
        chunk = synthetic_df.iloc[start:start + chunk_size]
        cursor.executemany(insert_sql, chunk.values.tolist())
        if (i + 1) % 10 == 0:
            print(f"  Inserted chunk {i+1}/{n_chunks}")

    conn.commit()
    conn.close()
    print(f"  Done — {total:,} rows written to {TD_DATABASE}.{TD_SYNTHETIC_TABLE}")


if __name__ == "__main__":
    df = load_sample()
    df = preprocess(df)
    model = train_ctgan(df)
    synthetic_df = generate_synthetic(model)
    write_to_teradata(synthetic_df)
    print("\nCTGAN pipeline complete!")
