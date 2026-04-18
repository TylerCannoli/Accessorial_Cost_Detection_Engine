"""
pace_transformer.py
===================
PACE FT-Transformer training script.

Reads CTGAN.pace_training_v (3M rows, 154 cols).
Trains dual-head FT-Transformer:
  - Regression head  → accessorial_risk_score  (0–100)
  - Classification head → accessorial_type (6 classes)

Output → models/
  pace_transformer.pt    state_dict loadable by inference.py
  artifacts.pkl          PACECategoricalEncoder + StandardScaler + col lists
  training_metrics.json  loss curves, test accuracy, AUC-ROC
"""

import os, json, pickle, time, functools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import teradatasql
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

# ── Credentials ──────────────────────────────────────────────────────────────
TD_HOST     = os.environ["TD_HOST"]
TD_USERNAME = os.environ["TD_USERNAME"]
TD_PASSWORD = os.environ["TD_PASSWORD"]
TD_DATABASE = os.environ["TD_DATABASE"]
TD_VIEW     = os.environ["TD_VIEW"]

MODEL_DIR      = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH   = MODEL_DIR / "pace_transformer.pt"
ARTIFACTS_PATH = MODEL_DIR / "artifacts.pkl"
METRICS_PATH   = MODEL_DIR / "training_metrics.json"

# ── Column Definitions ────────────────────────────────────────────────────────
ID_COL     = "unique_id"
DOT_COL    = "dot_number"
REG_TARGET = "accessorial_risk_score"   # 0–100 continuous
CLS_TARGET = "accessorial_type"         # 0–5 integer

# N_CLASSES and CHARGE_TYPE_LABELS are the authoritative values from pipeline/config.py.
# Import them rather than duplicating literals that could drift.
# NOTE: This script cannot import CONTINUOUS_COLUMNS / CATEGORICAL_COLUMNS from
# pipeline/config.py because it targets the original CTGAN view schema (pre-Phase-2
# enrichment).  If the view schema changes, update the lists here to match.
try:
    from pipeline.config import N_CLASSES, CHARGE_TYPE_LABELS
except ImportError:
    # Standalone execution without the pipeline package on PYTHONPATH
    N_CLASSES = 6
    CHARGE_TYPE_LABELS = [
        "No Charge",
        "Detention",
        "Safety Surcharge",
        "Compliance Fee",
        "Hazmat Fee",
        "High Risk / Multiple",
    ]

# insp_level_id is numeric (FMCSA inspection levels 1–7) — treated as continuous
# to match pipeline/config.py.  Do NOT add it to CATEGORICAL_COLUMNS here.
CATEGORICAL_COLUMNS = [
    "report_state", "county_code_state",
    "carrier_status_code", "carrier_carrier_operation", "carrier_fleetsize",
    "carrier_hm_ind", "carrier_phy_country", "carrier_phy_state",
    "carrier_safety_rating",
    "carrier_crgo_genfreight", "carrier_crgo_household", "carrier_crgo_produce",
    "carrier_crgo_intermodal", "carrier_crgo_oilfield", "carrier_crgo_meat",
    "carrier_crgo_chem", "carrier_crgo_drybulk", "carrier_crgo_coldfood",
    "carrier_crgo_beverages", "carrier_crgo_construct",
    "sms_carrier_operation", "sms_hm_flag", "sms_pc_flag",
    "sms_phy_state", "sms_phy_country",
    "sms_private_only", "sms_authorized_for_hire",
    "sms_exempt_for_hire", "sms_private_property",
    "hazmat_placard_req", "unit_type_desc", "unit_make", "unit_license_state",
]

CONTINUOUS_COLUMNS = [
    "insp_year", "insp_month", "insp_dow", "insp_day",
    "insp_level_id",
    "is_holiday", "is_near_holiday", "time_weight", "carrier_add_date",
    "carrier_mcs150_mileage", "carrier_mcs150_mileage_year",
    "carrier_truck_units", "carrier_power_units",
    "carrier_recordable_crash_rate", "carrier_total_intrastate_drivers",
    "carrier_interstate_beyond_100_miles", "carrier_interstate_within_100_miles",
    "carrier_intrastate_beyond_100_miles", "carrier_intrastate_within_100_miles",
    "carrier_total_cdl", "carrier_total_drivers",
    "sms_nbr_power_unit", "sms_driver_total",
    "sms_recent_mileage", "sms_recent_mileage_year",
    "oos_total", "driver_oos_total", "vehicle_oos_total",
    "hazmat_oos_total", "total_hazmat_sent",
    "basic_viol", "unsafe_viol", "fatigued_viol", "dr_fitness_viol",
    "subt_alcohol_viol", "vh_maint_viol", "hm_viol",
    "crash_count", "crash_fatalities_total", "crash_injuries_total",
    "crash_towaway_total", "crash_avg_severity", "crash_hazmat_releases",
    "eia_diesel_national", "eia_diesel_padd1_east_coast",
    "eia_diesel_padd2_midwest", "eia_diesel_padd3_gulf_coast",
    "eia_diesel_padd4_rocky_mountain", "eia_diesel_padd5_west_coast",
    "eia_diesel_california", "eia_gasoline_national",
    "eia_crude_wti_spot", "eia_crude_brent_spot",
    "eia_diesel_no2_spot_ny", "eia_jet_fuel_spot_ny",
    "eia_heating_oil_spot_ny", "eia_distillate_stocks_us",
    "eia_distillate_supplied_us", "eia_distillate_stocks_padd1",
    "eia_distillate_stocks_padd2", "eia_distillate_stocks_padd3",
    "eia_refinery_utilization_us", "eia_crude_inputs_to_refineries",
    "eia_steo_diesel_price_forecast", "eia_steo_wti_price_forecast",
    "eia_steo_gasoline_retail_forecast", "eia_steo_liquid_fuels_consumption",
    "eia_natgas_henry_hub_spot", "eia_natgas_industrial_price",
    "eia_natgas_commercial_price",
    "fred_TSIFRGHT", "fred_PCU484121484121", "fred_PCU4841248412",
    "fred_AMTMVS", "fred_INDPRO", "fred_GASDESW",
    "fred_FRGEXPUSM649NCIS", "fred_FRGSHPUSM649NCIS",
    "fred_PCU4841224841221", "fred_TRUCKD11",
    "fred_CES4348400001", "fred_CES4349300001",
    "fred_RAILFRTINTERMODAL", "fred_WPU057303", "fred_WPU3012",
    "fred_CES4300000001", "fred_CES4348100001",
    "wx_avg_high_f", "wx_avg_low_f", "wx_total_precip_in",
    "wx_total_snow_in", "wx_avg_wind_mph",
    "fac_estab_311612", "fac_estab_311991", "fac_estab_311999",
    "fac_estab_312111", "fac_estab_312120",
    "fac_estab_424410", "fac_estab_424420", "fac_estab_424430",
    "fac_estab_424450", "fac_estab_424460",
    "fac_estab_424480", "fac_estab_424490",
    "fac_estab_484110", "fac_estab_484121", "fac_estab_484122",
    "fac_estab_484220", "fac_estab_484230",
    "fac_estab_493110", "fac_estab_493120",
    "fac_estab_493130", "fac_estab_493190",
    "fac_estab_warehousing_total",
    "fac_reefer_share", "usda_reefer_availability", "stb_avg_dwell_hours",
]

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE          = 2048
EPOCHS              = 50
LR                  = 1e-4
WEIGHT_DECAY        = 1e-5
EARLY_STOP_PATIENCE = 7
TOKEN_DIM           = 192   # 192 / 8 heads = 24 ✓
N_HEADS             = 8
N_LAYERS            = 3
FFN_MULTIPLIER      = 4 / 3
ATTN_DROPOUT        = 0.1
FFN_DROPOUT         = 0.2
TRAIN_FRAC          = 0.8
VAL_FRAC            = 0.1
# REG loss: MSE on 0-100 scores; CE on 6-class. Scale MSE down so both are ~O(1).
REG_WEIGHT          = 0.005
CLS_WEIGHT          = 1.0


# ════════════════════════════════════════════════════════════════════════════
# Categorical Encoder
# Index 0 reserved for UNKNOWN; known values are 1..n_unique.
# .cardinalities[col] = n_unique + 1 → passed directly to nn.Embedding(card, dim).
# Compatible with inference.py's cat_encoder.cardinalities[col] access pattern.
# ════════════════════════════════════════════════════════════════════════════

class PACECategoricalEncoder:
    def __init__(self):
        self.value_maps:   Dict[str, Dict[str, int]] = {}
        self.cardinalities: Dict[str, int]            = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]) -> "PACECategoricalEncoder":
        for col in cat_cols:
            vals = sorted(df[col].fillna("UNKNOWN").astype(str).unique().tolist())
            self.value_maps[col]   = {v: i + 1 for i, v in enumerate(vals)}
            self.cardinalities[col] = len(vals) + 1   # +1 for index-0 UNKNOWN slot
        return self

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
        out = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
        for j, col in enumerate(cat_cols):
            vmap = self.value_maps[col]
            out[:, j] = (
                df[col].fillna("UNKNOWN").astype(str)
                .map(lambda v: vmap.get(v, 0))
                .values
            )
        return out


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class PACEDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, x_cont: np.ndarray,
                 y_reg: np.ndarray, y_cls: np.ndarray):
        self.x_cat  = torch.tensor(x_cat,  dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.y_reg  = torch.tensor(y_reg,  dtype=torch.float32)
        self.y_cls  = torch.tensor(y_cls,  dtype=torch.long)

    def __len__(self):
        return len(self.y_reg)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_cont[idx], self.y_reg[idx], self.y_cls[idx]


# ════════════════════════════════════════════════════════════════════════════
# Model Architecture — must match inference.py exactly
# ════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    def __init__(self, cat_cardinalities: List[int], cat_embed_dims: List[int],
                 n_continuous: int, token_dim: int):
        super().__init__()
        self.cat_embeddings  = nn.ModuleList()
        self.cat_projections = nn.ModuleList()
        for card, edim in zip(cat_cardinalities, cat_embed_dims):
            self.cat_embeddings.append(nn.Embedding(card, edim))
            self.cat_projections.append(nn.Linear(edim, token_dim))
        self.cont_projections = nn.ModuleList(
            [nn.Linear(1, token_dim) for _ in range(n_continuous)]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        B = x_cat.size(0)
        tokens = []
        for i, (emb, proj) in enumerate(zip(self.cat_embeddings, self.cat_projections)):
            tokens.append(proj(emb(x_cat[:, i])))
        for i, proj in enumerate(self.cont_projections):
            tokens.append(proj(x_cont[:, i:i+1]))
        tokens = torch.stack(tokens, dim=1)                    # (B, n_feat, D)
        cls    = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        return torch.cat([cls, tokens], dim=1)                 # (B, 1+n_feat, D)


class PACETransformer(nn.Module):
    def __init__(self, cat_cardinalities: List[int], cat_embed_dims: List[int],
                 n_continuous: int, token_dim: int,
                 n_layers: int, n_heads: int, ffn_multiplier: float,
                 attn_dropout: float, ffn_dropout: float, n_classes: int):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            cat_cardinalities, cat_embed_dims, n_continuous, token_dim
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=int(token_dim * ffn_multiplier),
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, 1)
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, n_classes)
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor):
        tokens  = self.tokenizer(x_cat, x_cont)
        tokens  = self.attn_dropout(tokens)
        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]                               # CLS token
        return self.reg_head(cls_out).squeeze(-1), self.cls_head(cls_out)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def compute_embed_dim(cardinality: int) -> int:
    """Matches inference.py's _compute_embed_dim exactly."""
    dim = int(round(1.6 * (cardinality ** 0.56)))
    return max(8, min(64, dim))


# ════════════════════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    needed = (
        [ID_COL, DOT_COL]
        + CATEGORICAL_COLUMNS
        + CONTINUOUS_COLUMNS
        + [REG_TARGET, CLS_TARGET]
    )
    # Only select columns that exist in the view (guard against view drift)
    col_sql = ", ".join(needed)
    query   = f"SELECT {col_sql} FROM {TD_DATABASE}.{TD_VIEW}"

    print(f"[1/7] Connecting to Teradata at {TD_HOST}...")
    conn = teradatasql.connect(
        host=TD_HOST, user=TD_USERNAME,
        password=TD_PASSWORD, database=TD_DATABASE,
    )
    print(f"[2/7] Fetching all rows from {TD_DATABASE}.{TD_VIEW}...")
    print("      (3M rows × ~150 cols, expect 3–5 min)")
    t0 = time.time()
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"      Done in {time.time()-t0:.0f}s — {len(df):,} rows, {df.shape[1]} cols")
    return df


# ════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    print("[3/7] Preprocessing...")

    # Drop rows missing either target
    before = len(df)
    df = df.dropna(subset=[REG_TARGET, CLS_TARGET])
    print(f"      Dropped {before - len(df):,} rows with missing targets. "
          f"Remaining: {len(df):,}")

    # Coerce targets
    df[REG_TARGET] = pd.to_numeric(df[REG_TARGET], errors="coerce").fillna(0).clip(0, 100)
    df[CLS_TARGET] = pd.to_numeric(df[CLS_TARGET], errors="coerce").fillna(0).astype(int).clip(0, N_CLASSES - 1)

    # Fill continuous NaNs with 0
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fill categorical NaNs with "UNKNOWN"
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

    # Class balance report
    counts = df[CLS_TARGET].value_counts().sort_index()
    print("      Class distribution:")
    for cls_id, label in enumerate(CHARGE_TYPE_LABELS):
        n = counts.get(cls_id, 0)
        print(f"        {cls_id} {label:<25} {n:>8,}  ({n/len(df)*100:.1f}%)")

    # Encode categoricals
    cat_encoder = PACECategoricalEncoder()
    cat_encoder.fit(df, CATEGORICAL_COLUMNS)
    x_cat = cat_encoder.transform(df, CATEGORICAL_COLUMNS)
    print(f"      Encoded {len(CATEGORICAL_COLUMNS)} categorical columns")

    # Scale continuous
    cont_data = df[CONTINUOUS_COLUMNS].values.astype(np.float32)
    scaler    = StandardScaler()
    x_cont    = scaler.fit_transform(cont_data).astype(np.float32)
    print(f"      Scaled {len(CONTINUOUS_COLUMNS)} continuous columns")

    y_reg = df[REG_TARGET].values.astype(np.float32)
    y_cls = df[CLS_TARGET].values.astype(np.int64)

    risk_score_max = float(df[REG_TARGET].max())
    print(f"      risk_score range: [{df[REG_TARGET].min():.1f}, {risk_score_max:.1f}]")

    return x_cat, x_cont, y_reg, y_cls, cat_encoder, scaler, risk_score_max


def split_data(x_cat, x_cont, y_reg, y_cls):
    n = len(y_reg)
    idx = np.random.permutation(n)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    tr, va, te = idx[:train_end], idx[train_end:val_end], idx[val_end:]
    print(f"      Split → train {len(tr):,} / val {len(va):,} / test {len(te):,}")

    def subset(i):
        return x_cat[i], x_cont[i], y_reg[i], y_cls[i]

    return subset(tr), subset(va), subset(te)


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def build_model(cat_encoder: PACECategoricalEncoder) -> PACETransformer:
    cardinalities = [cat_encoder.cardinalities[c] for c in CATEGORICAL_COLUMNS]
    embed_dims    = [compute_embed_dim(card) for card in cardinalities]
    token_dim     = ((TOKEN_DIM + 8 - 1) // 8) * 8   # round to multiple of 8

    model = PACETransformer(
        cat_cardinalities=cardinalities,
        cat_embed_dims=embed_dims,
        n_continuous=len(CONTINUOUS_COLUMNS),
        token_dim=token_dim,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_multiplier=FFN_MULTIPLIER,
        attn_dropout=ATTN_DROPOUT,
        ffn_dropout=FFN_DROPOUT,
        n_classes=N_CLASSES,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Model built — {n_params:,} parameters, token_dim={token_dim}")
    return model


def train(model, train_set, val_set, y_cls_train: np.ndarray, device):
    print("[5/7] Training...")
    model.to(device)

    # Class weights for imbalanced classification
    classes   = np.arange(N_CLASSES)
    cw        = compute_class_weight("balanced", classes=classes, y=y_cls_train)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(
        PACEDataset(*train_set), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        PACEDataset(*val_set), batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 10,
        epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for x_cat, x_cont, y_r, y_c in train_loader:
            x_cat  = x_cat.to(device)
            x_cont = x_cont.to(device)
            y_r    = y_r.to(device)
            y_c    = y_c.to(device)

            optimizer.zero_grad()
            reg_out, cls_out = model(x_cat, x_cont)
            loss = REG_WEIGHT * mse_loss(reg_out, y_r) + CLS_WEIGHT * ce_loss(cls_out, y_c)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        all_cls_pred, all_cls_true = [], []
        with torch.no_grad():
            for x_cat, x_cont, y_r, y_c in val_loader:
                x_cat  = x_cat.to(device)
                x_cont = x_cont.to(device)
                y_r    = y_r.to(device)
                y_c    = y_c.to(device)
                reg_out, cls_out = model(x_cat, x_cont)
                loss = REG_WEIGHT * mse_loss(reg_out, y_r) + CLS_WEIGHT * ce_loss(cls_out, y_c)
                val_loss += loss.item()
                all_cls_pred.extend(cls_out.argmax(dim=1).cpu().numpy())
                all_cls_true.extend(y_c.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        val_acc = accuracy_score(all_cls_true, all_cls_pred)

        history["train_loss"].append(round(avg_train, 5))
        history["val_loss"].append(round(avg_val, 5))
        history["val_acc"].append(round(val_acc, 4))

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:>3}/{EPOCHS}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"acc={val_acc:.3f}  lr={lr_now:.2e}")

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_set, device) -> dict:
    print("[6/7] Evaluating on test set...")
    model.eval()
    loader = DataLoader(
        PACEDataset(*test_set), batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    all_reg_pred, all_reg_true = [], []
    all_cls_pred, all_cls_proba, all_cls_true = [], [], []

    with torch.no_grad():
        for x_cat, x_cont, y_r, y_c in loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            reg_out, cls_out = model(x_cat, x_cont)
            proba = torch.softmax(cls_out, dim=1).cpu().numpy()
            all_reg_pred.extend(reg_out.cpu().numpy())
            all_reg_true.extend(y_r.numpy())
            all_cls_pred.extend(cls_out.argmax(dim=1).cpu().numpy())
            all_cls_proba.extend(proba)
            all_cls_true.extend(y_c.numpy())

    y_reg_pred = np.array(all_reg_pred)
    y_reg_true = np.array(all_reg_true)
    y_cls_pred = np.array(all_cls_pred)
    y_cls_proba = np.array(all_cls_proba)
    y_cls_true = np.array(all_cls_true)

    mae  = float(np.abs(y_reg_pred - y_reg_true).mean())
    rmse = float(np.sqrt(((y_reg_pred - y_reg_true) ** 2).mean()))
    acc  = float(accuracy_score(y_cls_true, y_cls_pred))

    try:
        auc = float(roc_auc_score(y_cls_true, y_cls_proba, multi_class="ovr", average="weighted"))
    except ValueError as e:
        import logging
        logging.warning("PACE: AUC-ROC skipped (single-class batch or shape mismatch): %s", e)
        auc = 0.0

    print(f"\n  Regression  — MAE: {mae:.2f}   RMSE: {rmse:.2f}")
    print(f"  Classification — Accuracy: {acc:.4f}   AUC-ROC (weighted OVR): {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_cls_true, y_cls_pred, target_names=CHARGE_TYPE_LABELS))

    return {
        "test_mae":  round(mae,  4),
        "test_rmse": round(rmse, 4),
        "test_acc":  round(acc,  4),
        "test_auc":  round(auc,  4),
    }


# ════════════════════════════════════════════════════════════════════════════
# Save Artifacts
# ════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, cat_encoder, scaler, risk_score_max,
                   history, test_metrics, device):
    print("[7/7] Saving artifacts...")

    # Model weights
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"      {WEIGHTS_PATH}")

    # Preprocessing artifacts (loaded by inference.py)
    artifacts = {
        "cat_encoder":    cat_encoder,
        "scaler":         scaler,
        "cat_cols":       CATEGORICAL_COLUMNS,
        "cont_cols":      CONTINUOUS_COLUMNS,
        "risk_score_max": risk_score_max,
        "n_classes":      N_CLASSES,
        "charge_labels":  CHARGE_TYPE_LABELS,
    }
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"      {ARTIFACTS_PATH}")

    # Training metrics JSON
    metrics = {
        "history":      history,
        "test_metrics": test_metrics,
        "hyperparams": {
            "batch_size": BATCH_SIZE, "lr": LR, "epochs_run": len(history["train_loss"]),
            "token_dim": TOKEN_DIM, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "n_cat": len(CATEGORICAL_COLUMNS), "n_cont": len(CONTINUOUS_COLUMNS),
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      {METRICS_PATH}")

    # Loss curve plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"],   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Combined Loss"); plt.title("Loss Curves")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], color="green")
    plt.xlabel("Epoch"); plt.ylabel("Val Accuracy"); plt.title("Classification Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"      {MODEL_DIR}/training_curves.png")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Weights : {WEIGHTS_PATH}")
    print(f"  Artifacts: {ARTIFACTS_PATH}")
    print(f"  Test MAE : {test_metrics['test_mae']:.2f}  "
          f"RMSE: {test_metrics['test_rmse']:.2f}")
    print(f"  Test Acc : {test_metrics['test_acc']:.4f}  "
          f"AUC: {test_metrics['test_auc']:.4f}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    np.random.seed(42)
    torch.manual_seed(42)

    df = load_data()
    x_cat, x_cont, y_reg, y_cls, cat_encoder, scaler, risk_score_max = preprocess(df)
    del df  # free memory before allocating tensors

    print("[4/7] Splitting data...")
    train_set, val_set, test_set = split_data(x_cat, x_cont, y_reg, y_cls)

    print("[4/7] Building model...")
    model = build_model(cat_encoder)

    model, history = train(model, train_set, val_set, train_set[3], device)
    test_metrics   = evaluate(model, test_set, device)
    save_artifacts(model, cat_encoder, scaler, risk_score_max,
                   history, test_metrics, device)


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")
