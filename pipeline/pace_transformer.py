import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import functools
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

from pipeline.config import (
    TD_HOST, TD_USERNAME, TD_PASSWORD, TD_DATABASE, TD_VIEW,
    ID_COLUMN, DATE_COLUMN, REGRESSION_TARGET, MULTICLASS_TARGET,
    N_CLASSES, CONTINUOUS_COLUMNS_V2 as CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS,
    LTL_DATE_COLUMN, LTL_REGRESSION_TARGET, LTL_CONTINUOUS_COLUMNS, LTL_CATEGORICAL_COLUMNS,
    MODEL_WEIGHTS_PATH, MODEL_ARTIFACTS_PATH, RESULTS_DIR, CHUNK_SIZE, NUM_THREADS,
    CHARGE_TYPE_LABELS,
)


# ── Hyperparameters ───────────────────────────────────────────────
class HyperParameters:
    n_layers: int = 3
    n_heads: int = 8
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.2
    ffn_multiplier: float = 4 / 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 50
    early_stopping_patience: int = 12
    embed_base_factor: float = 1.6
    embed_exponent: float = 0.56
    embed_max_dim: int = 64
    embed_min_dim: int = 8
    token_dim: int = 192
    random_state: int = 42

    def compute_embedding_dim(self, cardinality: int) -> int:
        dim = int(round(self.embed_base_factor * (cardinality ** self.embed_exponent)))
        return max(self.embed_min_dim, min(self.embed_max_dim, dim))


# ── Data loading ──────────────────────────────────────────────────
def get_connection():
    """Return connection."""
    import teradatasql  # cluster-only dependency; imported lazily so web deployments don't crash
    return teradatasql.connect(
        host=TD_HOST, user=TD_USERNAME,
        password=TD_PASSWORD, database=TD_DATABASE
    )


def get_row_count() -> int:
    """Return row count."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {TD_DATABASE}.{TD_VIEW}")  # nosec B608
        count = cur.fetchone()[0]
        print(f"Total rows: {count:,}")
        return count


def fetch_chunk(offset: int, chunk_size: int) -> pd.DataFrame:
    query = f"""
    SELECT * FROM {TD_DATABASE}.{TD_VIEW}
    QUALIFY ROW_NUMBER() OVER (ORDER BY {ID_COLUMN} ASC)
    BETWEEN {offset + 1} AND {offset + chunk_size}
    """  # nosec B608
    with get_connection() as conn:
        return pd.read_sql(query, conn)


def load_data() -> pd.DataFrame:
    total_rows = get_row_count()
    offsets = list(range(0, total_rows, CHUNK_SIZE))
    print(f"Loading {len(offsets)} chunks using {NUM_THREADS} threads...")
    chunk_dict = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_offset = {
            executor.submit(fetch_chunk, offset, CHUNK_SIZE): offset
            for offset in offsets
        }
        for future in as_completed(future_to_offset):
            offset = future_to_offset[future]
            chunk_dict[offset] = future.result()
            completed += 1
            print(f"  Chunk {completed}/{len(offsets)}")
    ordered = [chunk_dict[o] for o in sorted(chunk_dict.keys())]
    df = pd.concat(ordered, ignore_index=True)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def _compute_ltl_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive accessorial_risk_score and accessorial_type from C/ charge columns.

    Used when loading enriched_ltl_training.csv, which has binary C/ flags
    instead of FMCSA OOS counts.
    """
    charge_weights = {
        "C/HAZARDOUS MATERIALS":    30.0,
        "C/Detention":              25.0,
        "C/Limited Access Delivery":20.0,
        "C/Inside Delivery":        15.0,
        "C/Lift Gate Delivery":     15.0,
        "C/Excessive Length Charge":15.0,
        "C/Lumper Fee":             15.0,
        "C/Delivery Appointment":   10.0,
        "C/Residential Delivery":   10.0,
        "C/Single Shipment":        10.0,
        "C/SORT/SEGREGATING FEE":   10.0,
        "C/Re Weigh Fee":            5.0,
        "C/Inspection":              5.0,
    }
    score = pd.Series(0.0, index=df.index)
    for col, w in charge_weights.items():
        if col in df.columns:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0) * w
    df["accessorial_risk_score"] = np.minimum(100.0, score)

    hazmat    = pd.to_numeric(df.get("C/HAZARDOUS MATERIALS",   0), errors="coerce").fillna(0)
    detention = pd.to_numeric(df.get("C/Detention",             0), errors="coerce").fillna(0)
    vehicle   = (
        pd.to_numeric(df.get("C/Lift Gate Delivery",       0), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("C/Limited Access Delivery",  0), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("C/Excessive Length Charge",  0), errors="coerce").fillna(0)
    ).clip(upper=1)
    compliance = (
        pd.to_numeric(df.get("C/Re Weigh Fee",    0), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("C/Inspection",      0), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("C/Single Shipment", 0), errors="coerce").fillna(0)
    ).clip(upper=1)
    n_charges = sum(
        pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).clip(upper=1)
        for c in charge_weights
    )
    # Priority order matters: detention always wins so it isn't buried by multi-charge rule.
    # n_charges >= 2 catches non-detention multi-charge shipments as High Risk / Multiple.
    conditions = [
        detention > 0,   # class 1 — Detention (critical; wins even when combined with others)
        n_charges >= 2,  # class 5 — High Risk / Multiple (non-detention multi-charge)
        hazmat > 0,      # class 4 — Hazmat Fee
        vehicle > 0,     # class 2 — Safety Surcharge
        compliance > 0,  # class 3 — Compliance Fee (re-weigh, inspection, single shipment)
    ]
    choices = [1, 5, 4, 2, 3]
    df["accessorial_type"] = np.select(conditions, choices, default=0).astype(int)

    dist = df["accessorial_type"].value_counts().sort_index()
    print(f"  LTL targets derived — type dist: { {k: int(v) for k, v in dist.items()} }")

    if "pickup_dt" in df.columns and LTL_DATE_COLUMN not in df.columns:
        df[LTL_DATE_COLUMN] = pd.to_datetime(df["pickup_dt"], errors="coerce").dt.year.fillna(0).astype(int)

    return df


def load_data_from_csv(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """Load training data from a local CSV file instead of Teradata."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run: python scripts/enrich_data/build_enriched_training_set.py --source fmcsa"
        )
    size_mb = path.stat().st_size / 1e6
    print(f"Loading from CSV: {csv_path} ({size_mb:.0f} MB)")
    chunks = []
    total  = 0
    for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  Loaded {total:,} rows")
        if max_rows and total >= max_rows:
            break
    df = pd.concat(chunks, ignore_index=True)
    if max_rows:
        df = df.iloc[:max_rows]
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    if REGRESSION_TARGET not in df.columns:
        print("  accessorial_risk_score missing — deriving from C/ columns (LTL mode)")
        df = _compute_ltl_targets(df)
    return df


# ── Categorical encoder ───────────────────────────────────────────
class CategoricalEncoder:
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.cardinalities: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]) -> "CategoricalEncoder":
        for col in cat_cols:
            le = LabelEncoder()
            vals = df[col].astype(str).fillna("__missing__")
            le.fit(list(vals.unique()) + ["__unseen__"])
            self.encoders[col] = le
            self.cardinalities[col] = len(le.classes_)
        return self

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
        encoded = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
        for i, col in enumerate(cat_cols):
            le = self.encoders[col]
            vals = df[col].astype(str).fillna("__missing__")
            unseen_idx = np.where(le.classes_ == "__unseen__")[0][0]
            encoded[:, i] = np.array([
                le.transform([v])[0] if v in le.classes_ else unseen_idx
                for v in vals
            ])
        return encoded


# ── Dataset ───────────────────────────────────────────────────────
class PACEDataset(Dataset):
    def __init__(self, cat_data, cont_data, reg_targets, cls_targets):
        self.cat = torch.tensor(cat_data, dtype=torch.long)
        self.cont = torch.tensor(cont_data, dtype=torch.float32)
        self.reg = torch.tensor(reg_targets, dtype=torch.float32)
        self.cls = torch.tensor(cls_targets, dtype=torch.long)

    def __len__(self):
        return len(self.reg)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.reg[idx], self.cls[idx]


# ── Model ─────────────────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    def __init__(self, cat_cardinalities, cat_embed_dims, n_continuous, token_dim):
        super().__init__()
        self.cat_embeddings = nn.ModuleList()
        self.cat_projections = nn.ModuleList()
        for card, edim in zip(cat_cardinalities, cat_embed_dims):
            self.cat_embeddings.append(nn.Embedding(card, edim))
            self.cat_projections.append(nn.Linear(edim, token_dim))
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, token_dim) for _ in range(n_continuous)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))

    def forward(self, x_cat, x_cont):
        batch_size = x_cat.size(0)
        tokens = []
        for i, (emb, proj) in enumerate(zip(self.cat_embeddings, self.cat_projections)):
            tokens.append(proj(emb(x_cat[:, i])))
        for i, proj in enumerate(self.cont_projections):
            tokens.append(proj(x_cont[:, i:i+1]))
        tokens = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls, tokens], dim=1)


class PACETransformer(nn.Module):
    def __init__(self, cat_cardinalities, cat_embed_dims, n_continuous,
                 token_dim, n_layers, n_heads, ffn_multiplier,
                 attn_dropout, ffn_dropout, n_classes):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            cat_cardinalities, cat_embed_dims, n_continuous, token_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=int(token_dim * ffn_multiplier),
            dropout=ffn_dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, n_classes),
        )

    def forward(self, x_cat, x_cont):
        tokens = self.tokenizer(x_cat, x_cont)
        tokens = self.attn_dropout(tokens)
        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]
        return self.reg_head(cls_out).squeeze(-1), self.cls_head(cls_out)


# ── Device + build ────────────────────────────────────────────────
def get_device():
    """Return device."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"  CUDA: {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda"), n_gpus
    print("  No CUDA — using CPU")
    return torch.device("cpu"), 0


def build_model(hp, cat_encoder, cat_cols, device, n_gpus, n_continuous=None):
    cardinalities = [cat_encoder.cardinalities[c] for c in cat_cols]
    embed_dims = [hp.compute_embedding_dim(c) for c in cardinalities]
    token_dim = hp.token_dim
    token_dim = ((token_dim + hp.n_heads - 1) // hp.n_heads) * hp.n_heads
    if n_continuous is None:
        n_continuous = len(CONTINUOUS_COLUMNS)
    model = PACETransformer(
        cat_cardinalities=cardinalities,
        cat_embed_dims=embed_dims,
        n_continuous=n_continuous,
        token_dim=token_dim,
        n_layers=hp.n_layers,
        n_heads=hp.n_heads,
        ffn_multiplier=hp.ffn_multiplier,
        attn_dropout=hp.attn_dropout,
        ffn_dropout=hp.ffn_dropout,
        n_classes=N_CLASSES,
    )
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── Training loop ─────────────────────────────────────────────────
def train_one_epoch(model, loader, reg_crit, cls_crit, optimizer, device, reg_loss_scale=1.0):
    model.train()
    total_loss, n = 0, 0
    for cat, cont, reg_t, cls_t in loader:
        cat, cont = cat.to(device), cont.to(device)
        reg_t, cls_t = reg_t.to(device), cls_t.to(device)
        optimizer.zero_grad()
        reg_out, cls_out = model(cat, cont)
        loss = reg_loss_scale * reg_crit(reg_out, reg_t) + cls_crit(cls_out, cls_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if n % 100 == 0:
            print(f"  Batch {n}/{len(loader)} | Loss: {total_loss/n:.4f}")
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, reg_crit, cls_crit, device, reg_loss_scale=1.0):
    model.eval()
    total_loss, n = 0, 0
    reg_preds, reg_trues, cls_preds, cls_trues = [], [], [], []
    for cat, cont, reg_t, cls_t in loader:
        cat, cont = cat.to(device), cont.to(device)
        reg_t, cls_t = reg_t.to(device), cls_t.to(device)
        reg_out, cls_out = model(cat, cont)
        loss = reg_loss_scale * reg_crit(reg_out, reg_t) + cls_crit(cls_out, cls_t)
        total_loss += loss.item()
        n += 1
        reg_preds.append(reg_out.cpu().numpy())
        reg_trues.append(reg_t.cpu().numpy())
        cls_preds.append(cls_out.argmax(dim=1).cpu().numpy())
        cls_trues.append(cls_t.cpu().numpy())
    return (
        total_loss / n,
        np.concatenate(reg_preds), np.concatenate(reg_trues),
        np.concatenate(cls_preds), np.concatenate(cls_trues),
    )


# ── Main pipeline ─────────────────────────────────────────────────
def run_pipeline(csv_path: str = None, max_rows: int = None):
    hp = HyperParameters()

    print("=" * 60)
    print("PACE FT-TRANSFORMER TRAINING PIPELINE")
    print("=" * 60)

    device, n_gpus = get_device()
    torch.manual_seed(hp.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hp.random_state)

    print("\n[1/6] Loading data...")
    if csv_path:
        print(f"  Source: CSV ({csv_path})")
        df = load_data_from_csv(csv_path, max_rows=max_rows)
    else:
        print(f"  Source: Teradata ({TD_DATABASE}.{TD_VIEW})")
        df = load_data()

    # Detect LTL mode (enriched_ltl_training.csv has C/ columns, no insp_year)
    ltl_mode = DATE_COLUMN not in df.columns
    if ltl_mode:
        # 24k rows can't support a 791k-param model; shrink to ~100k to prevent overfitting
        hp.n_layers      = 2
        hp.n_heads       = 4
        hp.token_dim     = 64
        hp.attn_dropout  = 0.3
        hp.ffn_dropout   = 0.4
        hp.weight_decay  = 1e-3
        hp.batch_size    = 512
        print("  LTL mode: reduced architecture (2 layers, dim=64) for 24k-row dataset")
    if ltl_mode:
        print("  LTL mode: using LTL column lists and pickup_year for split")
        active_date_col  = LTL_DATE_COLUMN
        active_cat_list  = LTL_CATEGORICAL_COLUMNS
        active_cont_list = LTL_CONTINUOUS_COLUMNS
    else:
        active_date_col  = DATE_COLUMN
        active_cat_list  = CATEGORICAL_COLUMNS
        active_cont_list = CONTINUOUS_COLUMNS
    active_reg_target = REGRESSION_TARGET  # accessorial_risk_score in both modes

    print("\n[2/6] Normalizing regression target...")
    max_score = df[active_reg_target].max()
    df[active_reg_target] = (df[active_reg_target] / max_score * 100).astype(np.float32)

    print("\n[3/6] Train/test split...")
    max_year = df[active_date_col].max()
    df_train = df[df[active_date_col] < max_year].reset_index(drop=True)
    df_test  = df[df[active_date_col] == max_year].reset_index(drop=True)
    if len(df_train) < 1000:
        # All rows are from the same year — fall back to random 80/20 split
        print(f"  Time-based split produced empty train set — using random 80/20 split")
        df = df.sample(frac=1, random_state=hp.random_state).reset_index(drop=True)
        split = int(len(df) * 0.8)
        df_train = df.iloc[:split].reset_index(drop=True)
        df_test  = df.iloc[split:].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

    cat_cols = [c for c in active_cat_list if c in df.columns]
    cont_cols = [c for c in active_cont_list if c in df.columns]
    cat_encoder = CategoricalEncoder().fit(df_train, cat_cols)

    print("\n[4/6] Building datasets...")
    scaler = StandardScaler()
    # Compute training-set medians for imputation; fillna(0) is wrong for columns like
    # LMI (~58), FRED indices (~113-180), and terminal distance (~80 mi).
    cont_medians = df_train[cont_cols].median()

    def make_dataset(frame, fit_scaler=False):
        cat_data  = cat_encoder.transform(frame, cat_cols)
        cont_data = frame[cont_cols].fillna(cont_medians).values.astype(np.float32)
        if fit_scaler:
            cont_data = scaler.fit_transform(cont_data)
        else:
            cont_data = scaler.transform(cont_data)
        reg_t = frame[active_reg_target].values.astype(np.float32)
        cls_t = frame[MULTICLASS_TARGET].values.astype(np.int64)
        return PACEDataset(cat_data, cont_data, reg_t, cls_t)

    train_ds = make_dataset(df_train, fit_scaler=True)
    test_ds  = make_dataset(df_test,  fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=hp.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    print("\n[5/6] Training...")
    model    = build_model(hp, cat_encoder, cat_cols, device, n_gpus, n_continuous=len(cont_cols))
    reg_crit = nn.MSELoss()
    if ltl_mode:
        reg_loss_scale = 0.0
        # Sqrt-inverse-frequency weights capped at 6x to boost rare classes without collapse.
        # Detention (0.8%) and Compliance Fee (0.66%) need the most help; majority classes ~1x.
        ltl_class_weights = torch.tensor(
            [1.0, 4.6, 1.4, 3.5, 2.6, 1.2], dtype=torch.float32
        ).to(device)
        cls_crit = nn.CrossEntropyLoss(weight=ltl_class_weights)
    else:
        reg_loss_scale = 1.0
        cls_crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate,
                                  weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(1, hp.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, reg_crit, cls_crit,
                                     optimizer, device, reg_loss_scale)
        val_loss, rp, rt, cp, ct = evaluate(model, test_loader, reg_crit,
                                             cls_crit, device, reg_loss_scale)
        scheduler.step(val_loss)
        rmse   = np.sqrt(mean_squared_error(rt, rp))
        r2     = r2_score(rt, rp)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{hp.epochs} | train={train_loss:.4f} "
              f"val={val_loss:.4f} RMSE={rmse:.2f} R2={r2:.4f} "
              f"lr={lr_now:.1e} | {time.time()-t0:.1f}s")
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            raw        = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= hp.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print("\n[6/6] Saving results...")
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(exist_ok=True)

    # Save model weights FIRST before any plots can crash
    os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
    torch.save(best_state, MODEL_WEIGHTS_PATH)
    print(f"  Model saved to {MODEL_WEIGHTS_PATH}")

    # Save preprocessing artifacts for inference.py
    artifacts = {
        "cat_encoder":    cat_encoder,
        "scaler":         scaler,
        "cat_cols":       cat_cols,
        "cont_cols":      cont_cols,
        "cont_medians":   cont_medians,
        "risk_score_max": float(max_score),
        "ltl_mode":       ltl_mode,
    }
    with open(MODEL_ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Preprocessing artifacts saved to {MODEL_ARTIFACTS_PATH}")

    print(f"  Regression RMSE: {np.sqrt(mean_squared_error(rt, rp)):.4f}")
    print(f"  Regression R2:   {r2_score(rt, rp):.4f}")
    print("\n  Classification Report:")
    present_labels = sorted(set(ct) | set(cp))
    present_names  = [CHARGE_TYPE_LABELS[i] for i in present_labels if i < len(CHARGE_TYPE_LABELS)]
    print(classification_report(ct, cp, labels=present_labels, target_names=present_names, digits=4))

    # Save predictions
    id_cols = [c for c in [ID_COLUMN, active_date_col] if c in df_test.columns]
    preds_df = df_test[id_cols].copy() if id_cols else pd.DataFrame(index=df_test.index)
    preds_df["risk_score_true"]  = rt
    preds_df["risk_score_pred"]  = rp
    preds_df["charge_type_true"] = ct
    preds_df["charge_type_pred"] = cp
    preds_df.to_csv(output_dir / "predictions.csv", index=False)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    present_labels = sorted(set(ct) | set(cp))
    present_names  = [CHARGE_TYPE_LABELS[i] for i in present_labels if i < len(CHARGE_TYPE_LABELS)]
    ConfusionMatrixDisplay(
        confusion_matrix(ct, cp, labels=present_labels), display_labels=present_names
    ).plot(ax=ax)
    ax.set_title("PACE Accessorial Type Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Regression scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rt[:5000], rp[:5000], alpha=0.3, s=5)
    ax.plot([0, 100], [0, 100], "r--")
    reg_label = "Cost (normalized)" if ltl_mode else "Risk Score"
    ax.set(xlabel=f"True {reg_label}", ylabel=f"Predicted {reg_label}",
           title=f"PACE {reg_label}: Predicted vs Actual")
    plt.tight_layout()
    fig.savefig(output_dir / "regression_scatter.png", dpi=150)
    plt.close(fig)

    print(f"  Results saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PACE FT-Transformer Training")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to enriched CSV file to train from instead of Teradata. "
             "Generate with: python scripts/enrich_data/build_enriched_training_set.py --source fmcsa"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Cap rows loaded from CSV (e.g. 2000000 for faster training)"
    )
    args = parser.parse_args()
    run_pipeline(csv_path=args.csv, max_rows=args.max_rows)
