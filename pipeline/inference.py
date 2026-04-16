from pipeline.pace_transformer import CategoricalEncoder  # noqa
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pipeline.pace_transformer import CategoricalEncoder  # noqa
from pipeline.config import (
    CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, N_CLASSES,
    MODEL_WEIGHTS_PATH, MODEL_ARTIFACTS_PATH, CHARGE_TYPE_LABELS, DOT_COLUMN,
    TD_HOST, TD_USERNAME, TD_PASSWORD, TD_DATABASE,
)

# ── Environment flag ──────────────────────────────────────────────
# Set PACE_ENV=production on Oracle Cloud (daxori) to enable API enrichment.
# On the GPU cluster (aimlsrv) this stays False — no internet access.
API_ENRICHMENT_ENABLED = os.environ.get("PACE_ENV", "").lower() == "production"

if API_ENRICHMENT_ENABLED:
    try:
        from pipeline.api_integration import get_enricher
        print("API enrichment enabled (production mode)")
    except ImportError:
        API_ENRICHMENT_ENABLED = False
        print("Warning: api_integration not found — enrichment disabled")


# ══════════════════════════════════════════════════════════════════
# Model Architecture (must match pace_transformer.py exactly)
# ══════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """Represent the FeatureTokenizer component."""
    def __init__(self, cat_cardinalities, cat_embed_dims, n_continuous, token_dim):
        """Handle init."""
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
        """Handle forward."""
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
    """Represent the PACETransformer component."""
    def __init__(self, cat_cardinalities, cat_embed_dims, n_continuous,
                 token_dim, n_layers, n_heads, ffn_multiplier,
                 attn_dropout, ffn_dropout, n_classes):
        """Handle init."""
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
            nn.LayerNorm(token_dim), nn.Linear(token_dim, 1)
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, n_classes)
        )

    def forward(self, x_cat, x_cont):
        """Handle forward."""
        tokens = self.tokenizer(x_cat, x_cont)
        tokens = self.attn_dropout(tokens)
        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]
        return self.reg_head(cls_out).squeeze(-1), self.cls_head(cls_out)


# ══════════════════════════════════════════════════════════════════
# PACE Inference Engine
# ══════════════════════════════════════════════════════════════════

class PACEInference:
    """
    Loads trained model weights and preprocessing artifacts,
    then exposes three prediction methods:

        predict_dot(dot_number)         — DOT number lookup
        predict_single(row)             — Manual shipment input
        predict_csv(filepath)           — Batch CSV upload

    On Oracle Cloud (PACE_ENV=production), each method is enriched
    with live data from FMCSA, FRED, EIA, NWS, and OWM before
    being passed to the model.
    """

    def __init__(self, weights_path: str = MODEL_WEIGHTS_PATH,
                 artifacts_path: str = MODEL_ARTIFACTS_PATH):
        """Handle init."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = None
        self.cat_encoder = None
        self.scaler      = None
        self.cat_cols    = None
        self.cont_cols   = None
        self.risk_score_max = 1000.0
        self._load(weights_path, artifacts_path)

    # ── Model loading ─────────────────────────────────────────────

    def _load(self, weights_path: str, artifacts_path: str):
        """Handle load."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Run pace_transformer.py first."
            )
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(
                f"Preprocessing artifacts not found at {artifacts_path}. "
                "Run pace_transformer.py first."
            )

        with open(artifacts_path, "rb") as f:
            import __main__
            __main__.CategoricalEncoder = CategoricalEncoder
            artifacts = pickle.load(f)  # nosec B301

        self.cat_encoder    = artifacts["cat_encoder"]
        self.scaler         = artifacts["scaler"]
        self.cat_cols       = artifacts["cat_cols"]
        self.cont_cols      = artifacts["cont_cols"]
        self.risk_score_max = artifacts.get("risk_score_max", 1000.0)

        cardinalities = [self.cat_encoder.cardinalities[c] for c in self.cat_cols]
        embed_dims    = [self._compute_embed_dim(c) for c in cardinalities]
        token_dim     = 192
        token_dim     = ((token_dim + 8 - 1) // 8) * 8

        self.model = PACETransformer(
            cat_cardinalities=cardinalities,
            cat_embed_dims=embed_dims,
            n_continuous=len(self.cont_cols),
            token_dim=token_dim,
            n_layers=3, n_heads=8, ffn_multiplier=4/3,
            attn_dropout=0.1, ffn_dropout=0.2,
            n_classes=N_CLASSES,
        )
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)  # nosec B614
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"PACE model loaded on {self.device}")

    def _compute_embed_dim(self, cardinality: int) -> int:
        """Handle compute embed dim."""
        dim = int(round(1.6 * (cardinality ** 0.56)))
        return max(8, min(64, dim))

    # ── Preprocessing ─────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame):
        """Normalize and encode a DataFrame for model input."""
        df = df.copy()

        # Fix boolean-as-string columns
        bool_cols = [
            "sms_hm_flag", "sms_pc_flag", "sms_private_only",
            "sms_authorized_for_hire", "sms_exempt_for_hire", "sms_private_property",
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].map({"TRUE": "Y", "FALSE": "N"}).fillna("N")

        # Fill missing columns with 0 or "UNKNOWN"
        for col in self.cont_cols:
            if col not in df.columns:
                df[col] = 0.0
        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "UNKNOWN"

        cat_data  = self.cat_encoder.transform(df, self.cat_cols)
        cont_data = df[self.cont_cols].fillna(0).values.astype(np.float32)
        cont_data = self.scaler.transform(cont_data)

        return (
            torch.tensor(cat_data,  dtype=torch.long).to(self.device),
            torch.tensor(cont_data, dtype=torch.float32).to(self.device),
        )

    # ── Core prediction ───────────────────────────────────────────

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a DataFrame. Returns enriched results."""
        x_cat, x_cont    = self._preprocess(df)
        reg_out, cls_out = self.model(x_cat, x_cont)

        risk_scores    = reg_out.cpu().numpy()
        charge_classes = cls_out.argmax(dim=1).cpu().numpy()
        charge_probs   = torch.softmax(cls_out, dim=1).cpu().numpy()

        results = pd.DataFrame({
            "risk_score":     risk_scores,
            "risk_score_pct": np.clip(risk_scores, 0, 100),
            "charge_type_id": charge_classes,
            "charge_type":    [CHARGE_TYPE_LABELS[i] for i in charge_classes],
            "risk_label":     [self._risk_label(s) for s in risk_scores],
        })

        for i, label in enumerate(CHARGE_TYPE_LABELS):
            safe_label = label.lower().replace(" ", "_").replace("/", "_")
            results[f"prob_{safe_label}"] = charge_probs[:, i]

        return results

    def _risk_label(self, score: float) -> str:
        """Convert numeric risk score to human-readable label."""
        if score >= 75:  return "Critical"
        if score >= 50:  return "High"
        if score >= 25:  return "Medium"
        if score > 0:    return "Low"
        return "None"

    @torch.no_grad()
    def predict_single(self, row: Dict) -> Dict:
        """Predict from a single feature dict."""
        df      = pd.DataFrame([row])
        results = self.predict(df)
        r       = results.iloc[0]
        safe    = lambda l: l.lower().replace(" ", "_").replace("/", "_")
        result = {
            "risk_score":    float(r["risk_score_pct"]),
            "risk_label":    r["risk_label"],
            "charge_type":   r["charge_type"],
            "charge_type_id":int(r["charge_type_id"]),
            "probabilities": {
                label: float(results[f"prob_{safe(label)}"].iloc[0])
                for label in CHARGE_TYPE_LABELS
            },
        }

        # In production mode, persist every inference result to ModelResults
        if API_ENRICHMENT_ENABLED:
            try:
                import json
                from datetime import datetime, timezone
                from utils.database import get_connection, write_model_result
                conn = get_connection()
                write_model_result(
                    _conn=conn,
                    dot_number=str(row.get("dot_number", "")) or None,
                    run_timestamp=datetime.now(timezone.utc),
                    risk_score=result["risk_score"],
                    risk_label=result["risk_label"],
                    charge_type=result["charge_type"],
                    probabilities_json=json.dumps(result["probabilities"]),
                    input_source=str(row.get("_input_source", "predict_single")),
                )
            except Exception:
                pass  # Never block scoring due to a DB write failure

        return result

    # ── Input method 1: DOT Number Lookup ─────────────────────────

    def predict_dot(self, dot_number: int,
                    origin_lat: float = None,
                    origin_lon: float = None,
                    origin_state: str = None) -> Dict:
        """
        Look up a carrier by DOT number and predict accessorial risk.

        On production (Oracle Cloud):
            - Pulls live FMCSA carrier profile, SMS scores,
              recent inspections, violations, and crash history
            - Enriches with live FRED + EIA economic indicators
            - Enriches with live OWM/NWS weather if location provided

        On cluster (aimlsrv):
            - Falls back to Teradata historical data for the DOT
        """
        if API_ENRICHMENT_ENABLED:
            # Production path — live FMCSA + economic enrichment
            enricher = get_enricher()
            features = enricher.enrich_dot(
                dot_number=dot_number,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                origin_state=origin_state,
            )
            if "error" in features:
                return features

            result = self.predict_single(features)
            result["dot_number"]   = dot_number
            result["carrier_name"] = features.get("carrier_name", "Unknown")
            result["data_source"]  = "live_fmcsa"
            return result

        else:
            # Cluster path — Teradata historical lookup
            return self._predict_dot_teradata(dot_number)

    def _predict_dot_teradata(self, dot_number: int) -> Dict:
        """Fallback: pull DOT record from Teradata and predict."""
        if not TD_HOST:
            import logging
            logging.warning(
                "PACE: TD_HOST is not configured — Teradata fallback disabled. "
                "Set TD_HOST in .env or environment to enable cluster lookup."
            )
            return {
                "error": (
                    "Teradata host not configured. "
                    "Set TD_HOST to enable cluster lookup, or use PACE_ENV=production "
                    "for live FMCSA enrichment instead."
                )
            }
        try:
            import teradatasql
            conn = teradatasql.connect(
                host=TD_HOST, user=TD_USERNAME,
                password=TD_PASSWORD, database=TD_DATABASE,
            )
            df = pd.read_sql(
                f"SELECT * FROM {TD_DATABASE}.pace_training_v "  # nosec B608
                f"WHERE dot_number = {dot_number} SAMPLE 1",
                conn,
            )
            conn.close()
            if df.empty:
                return {"error": f"DOT {dot_number} not found in database"}
            result = self.predict_single(df.iloc[0].to_dict())
            result["dot_number"]  = dot_number
            result["data_source"] = "teradata_historical"
            return result
        except Exception as e:
            return {"error": str(e)}

    # ── Input method 2: Manual Shipment Input ─────────────────────

    def predict_manual(self, user_inputs: Dict,
                       origin_lat: float = None,
                       origin_lon: float = None,
                       origin_city: str = None,
                       origin_state: str = None) -> Dict:
        """
        Predict from manually entered shipment/carrier details.

        On production: enriches with live FRED, EIA, weather,
        and FMCSA data (if DOT number provided).
        On cluster: uses inputs as-is.
        """
        if API_ENRICHMENT_ENABLED:
            try:
                enricher = get_enricher()
                features = enricher.enrich_manual(
                    user_inputs=user_inputs,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    origin_city=origin_city,
                    origin_state=origin_state,
                )
            except Exception as e:
                import logging
                logging.warning(f"PACE: Manual enrichment failed, using raw inputs: {e}")
                features = dict(user_inputs)
        else:
            features = dict(user_inputs)

        result = self.predict_single(features)
        result["data_source"] = "manual_input"
        return result

    # ── Input method 3: CSV Batch Upload ──────────────────────────

    def predict_csv(self, filepath: str,
                    output_path: str = None) -> pd.DataFrame:
        """
        Run batch inference on an uploaded CSV file.

        On production: enriches all rows with current FRED + EIA signals.
        On cluster: uses CSV data as-is.

        Saves results CSV if output_path provided.
        """
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} rows from {filepath}")

        if API_ENRICHMENT_ENABLED:
            try:
                enricher = get_enricher()
                df = enricher.enrich_dataframe(df)
            except Exception as e:
                import logging
                logging.warning(f"PACE: CSV enrichment failed, scoring without live signals: {e}")

        results = self.predict(df)

        # Prepend identifier columns
        id_cols = [c for c in [DOT_COLUMN, "unique_id", "carrier_name"]
                   if c in df.columns]
        df_out = pd.concat([df[id_cols], results], axis=1) if id_cols else results

        df_out["data_source"] = "csv_batch"

        if output_path:
            df_out.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        return df_out

    # ── Input method 4: Carrier-Agnostic Normalized Input ─────────

    def predict_carrier_data(
        self,
        df: pd.DataFrame,
        carrier_id: str = None,
        schema=None,
        fill_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Accept raw data from any carrier and normalize it to the PACE schema
        before running inference.

        Args:
            df:           Raw carrier DataFrame (any column names/formats)
            carrier_id:   Carrier identifier — loads configs/carriers/<id>.yaml
            schema:       Pre-loaded CarrierSchema instance (alternative to carrier_id)
            fill_missing: If True, auto-fill missing features using enrichment
                          lookups and training-data medians (recommended).

        Returns:
            DataFrame with risk_score, charge_type, and probability columns.

        Example:
            engine = get_inference_engine()

            # Using a saved carrier config
            results = engine.predict_carrier_data(raw_df, carrier_id="acme_logistics")

            # Using a passthrough schema (data already in PACE format)
            results = engine.predict_carrier_data(normalized_df)
        """
        from pipeline.ingest.carrier_schema import normalize_carrier_data
        from pipeline.ingest.normalizer import DataNormalizer

        # Step 1 — map carrier field names to PACE names
        normalized = normalize_carrier_data(df, carrier_id=carrier_id, schema=schema)

        # Step 2 — fill missing features from enrichment lookups + medians
        if fill_missing:
            normalizer = DataNormalizer()
            normalized = normalizer.fill(normalized)

        # Step 3 — enrich with live API signals if in production mode
        if API_ENRICHMENT_ENABLED:
            try:
                enricher = get_enricher()
                normalized = enricher.enrich_dataframe(normalized)
            except Exception as e:
                import logging
                logging.warning(f"PACE: API enrichment failed: {e}")

        # Step 4 — run inference
        results = self.predict(normalized)

        # Prepend identifier columns from the original data
        id_cols = [c for c in ["dot_number", "unique_id", "carrier_name"]
                   if c in normalized.columns]
        if id_cols:
            results = pd.concat([normalized[id_cols].reset_index(drop=True),
                                  results.reset_index(drop=True)], axis=1)

        results["carrier_id"] = carrier_id or "unknown"
        return results

    # ── Batch predict from DataFrame directly ─────────────────────

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict directly from a DataFrame (used internally by
        Streamlit pages after preprocessing).
        """
        if API_ENRICHMENT_ENABLED:
            try:
                enricher = get_enricher()
                df = enricher.enrich_dataframe(df)
            except Exception as e:
                import logging
                logging.warning(f"PACE: API enrichment failed, scoring without live signals: {e}")
        return self.predict(df)

    # ── Model info ────────────────────────────────────────────────

    def model_info(self) -> Dict:
        """Return model metadata for display in the UI."""
        return {
            "device":           str(self.device),
            "api_enrichment":   API_ENRICHMENT_ENABLED,
            "cat_features":     len(self.cat_cols),
            "cont_features":    len(self.cont_cols),
            "n_classes":        N_CLASSES,
            "charge_types":     CHARGE_TYPE_LABELS,
            "parameters":       sum(p.numel() for p in self.model.parameters()),
        }


# ══════════════════════════════════════════════════════════════════
# Singleton loader
# ══════════════════════════════════════════════════════════════════

_engine: Optional[PACEInference] = None

def get_inference_engine() -> PACEInference:
    """
    Returns a singleton PACEInference instance.
    Call this from Streamlit pages — model loads once and is reused.
    """
    global _engine
    if _engine is None:
        _engine = PACEInference()
    return _engine
