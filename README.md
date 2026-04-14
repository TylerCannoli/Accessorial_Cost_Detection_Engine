# PACE — Predictive Accessorial Cost Engine

![CI](https://github.com/Spring-ISYS-2026-Alpha-Team/Accessorial_Cost_Detection_Engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-FT--Transformer-EE4C2C?logo=pytorch&logoColor=white)
![Azure SQL](https://img.shields.io/badge/Azure%20SQL-Connected-0078D4?logo=microsoftazure&logoColor=white)

> An end-to-end ML system that predicts accessorial charge risk for freight shipments — before they're executed. Built with a custom FT-Transformer, live FMCSA/EIA/FRED data enrichment, and a full-stack Streamlit operations dashboard.

**[Live Demo →](https://accessorialcostdetectionengine.streamlit.app)**

---

## What It Does

Accessorial charges — detention fees, safety surcharges, compliance penalties — are not random. They are the downstream result of identifiable, repeatable patterns: carriers with poor SMS violation histories, facilities that routinely cause delays, lanes with elevated economic risk. The problem is that without a system to surface these patterns at the right time, logistics teams absorb costs that were entirely predictable.

PACE shifts that point of insight from invoice reconciliation to pre-dispatch decision-making. Given a USDOT number or a batch of shipment records, PACE outputs:

- A **risk score (0–100%)** quantifying the likelihood of an accessorial charge
- A **predicted charge type** (Detention, Safety Surcharge, Compliance Fee, Hazmat Fee, High Risk / Multiple, or No Charge)
- **Per-class probabilities** for every charge category
- A **risk label** (Low / Medium / High / Critical) for fast triage

---

## ML Architecture

The core model is a **Feature Tokenizer Transformer (FT-Transformer)** — the same architecture introduced in [Revisiting Deep Learning Models for Tabular Data (Gorishniy et al., 2021)](https://arxiv.org/abs/2106.11959) — trained on ~500K synthetic carrier inspection records generated via **CTGAN** from a university Teradata cluster.

```
Input (152 features)
  ├── Categorical: carrier_phy_state, carrier_safety_rating,
  │               sms_hm_flag, unit_type_desc, ... (34 features)
  └── Continuous:  oos_total, crash_count, eia_diesel_national,
                  fred_TSIFRGHT, wx_avg_high_f, ... (118 features)
         ↓
FeatureTokenizer  →  Embedding + Linear projection per feature → token sequence
         ↓
TransformerEncoder  →  3 layers, 8 heads, GELU, d_model=192
         ↓
CLS token
  ├── Regression head  →  risk_score (0–100)
  └── Classification head  →  charge_type (6 classes)
```

**Training data pipeline:**
- Raw FMCSA carrier inspection + SMS violation data pulled from Teradata (`pace_training_v`)
- Synthetic expansion via CTGAN (300M-epoch, 3M synthetic rows)
- Live enrichment with FMCSA API, FRED economic indicators, EIA diesel prices, NWS/OWM weather

**Inference:**
- Single DOT lookup: enriched with live FMCSA + FRED + EIA + weather in production
- Batch CSV: full dataframe inference via `predict_dataframe()`
- Results persisted to Azure SQL `ModelResults` table on every production inference

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | PyTorch FT-Transformer (custom), CTGAN synthetic data generation |
| **Data Enrichment** | FMCSA API, FRED (Federal Reserve), EIA (Energy Information Administration), OpenWeatherMap, NWS |
| **UI** | Streamlit 1.31+, Plotly, custom dark glass CSS |
| **Data Processing** | pandas, NumPy, scikit-learn |
| **Legacy Models** | LightGBM (risk classifier), RandomForest (cost estimator) |
| **Database** | Azure SQL via pymssql; Teradata via teradatasql (training/fallback) |
| **Auth** | SHA-256 password hashing, Streamlit session state, role-based routing |
| **Column Mapping** | sentence-transformers (semantic similarity) + Ollama (local LLM) |
| **Document Parsing** | pdfplumber, pytesseract, Pillow |
| **Geospatial** | geopy, openrouteservice |
| **CI/CD** | GitHub Actions (lint, security scan, test) |
| **Config** | python-dotenv / Streamlit secrets |

---

## Application Pages

| Page | Description |
|---|---|
| **Landing** | Pre-login hero with feature overview and Sign In CTA |
| **Home** | KPI cards, weekly shipment trend, cost/mile by carrier, risk distribution |
| **Risk Dashboard** | Risk tier breakdown, score histogram, carrier benchmarks, searchable shipment table |
| **Upload & Score** | CSV/Excel batch upload → validation → FT-Transformer scoring → downloadable results |
| **Shipments** | Paginated explorer with per-shipment detail: risk gauge, factor breakdown, recommended actions |
| **Cost Estimator** | RandomForest cost prediction with 95% CI, feature importance, historical comparison |
| **Route Analysis** | Lane-level cost and risk metrics, miles vs cost scatter |
| **Carrier Benchmarking** | Side-by-side benchmarking with radar chart; single-carrier mode shows DOT-style deep details |
| **Accessorial Tracker** | Charge trends over time, donut by type, carrier/facility breakdowns |
| **Admin** | User management (CRUD), model training (incremental/full retrain), version rollback, risk threshold tuning |

---

## Data Architecture

```
Teradata (University HPC Cluster)
  └── pace_training_v  →  CTGAN training  →  3M synthetic rows
                                                    ↓
                                          FT-Transformer training
                                                    ↓
                                    models/pace_transformer_weights.pt
                                    models/artifacts.pkl

Production (PACE_ENV=production)
  └── FMCSA API ──┐
      FRED API ───┤  →  enrich_dot() / enrich_dataframe()  →  predict_single()
      EIA API ────┤                                                  ↓
      OWM API ────┘                                       Azure SQL: ModelResults

Dashboard Data
  └── Azure SQL: Shipments, Carriers, Facilities, Accessorial_Charges, PaceUsers
        ↑ fallback ↓
      Teradata: pace_synthetic_v  (7K+ records)
        ↑ fallback ↓
      Mock data generator
```

---

## Project Structure

```
Accessorial_Cost_Detection_Engine/
├── app.py                        # Entry / landing (pre-login hero); run: streamlit run app.py
├── auth_utils.py                 # Session auth, logout, require_auth guard
├── requirements.txt
│
├── pages/
│   ├── _Login.py                 # Login (underscore = hidden from sidebar nav)
│   ├── _loading.py               # Post-login cache pre-warm + weight download
│   ├── 0_Home.py                 # Home KPIs + charts
│   ├── 2_Upload.py               # Batch upload & FT-Transformer scoring
│   ├── 3_Shipments.py            # Shipment explorer + detail view
│   ├── 4_Cost_Estimate.py        # RandomForest cost predictor
│   ├── 5_Route_Analysis.py       # Lane/route metrics
│   ├── 6_Carrier_Benchmarking.py # Carrier benchmarking
│   ├── 7_Accessorial_Tracker.py  # Accessorial charge analytics
│   ├── 8_Admin.py                # Admin panel (role-gated)
│
├── pipeline/
│   ├── config.py                 # Column definitions, model paths, env config
│   ├── inference.py              # PACEInference engine (singleton, GPU/CPU)
│   ├── pace_transformer.py       # FT-Transformer architecture + training loop
│   ├── data_pipeline.py          # Schema detection, validation, cleaning
│   ├── api_integration.py        # Live FMCSA / FRED / EIA / OWM enrichment
│   └── ctgan_train.py            # CTGAN synthetic data generation
│
├── utils/
│   ├── database.py               # Azure SQL + Teradata fallback + ModelResults
│   ├── column_mapper.py          # AI column alias mapping (semantic + Ollama)
│   ├── doc_parser.py             # CSV/Excel/PDF/image parser
│   ├── validation.py             # Row-level validation rules
│   ├── mock_data.py              # Synthetic shipment generator (demo fallback)
│   ├── styling.py                # Dark glass theme CSS, color tokens, top nav
│   ├── geo.py                    # Geospatial routing
│   └── legacy/                   # Deprecated LightGBM + RandomForest models
│
├── models/                       # Gitignored — downloaded at startup
│   ├── pace_transformer_weights.pt
│   └── artifacts.pkl
│
├── scripts/
│   ├── download_weights.py       # Bootstrap: pulls weights from GitHub Release
│   └── deploy_weights.py         # SCP weights to/from HPC cluster
│
└── .github/workflows/ci.yml      # Lint, security scan, test pipeline
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & install

```bash
git clone https://github.com/Spring-ISYS-2026-Alpha-Team/Accessorial_Cost_Detection_Engine.git
cd Accessorial_Cost_Detection_Engine
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env  # then fill in Azure SQL credentials
```

> **No database?** Skip this entirely. PACE runs in demo mode with Teradata fallback data or generated mock data — every page is fully functional without any credentials.

### 3. Download model weights

```bash
python scripts/download_weights.py
```

This pulls `pace_transformer_weights.pt` and `artifacts.pkl` from the GitHub Release automatically. The loading screen also does this on first startup.

### 4. Run

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

**Default credentials:**
| Username | Password | Role |
|---|---|---|
| `admin` | `admin` | Admin |
| `user` | `user` | User |

---

## Model Weights

Weights are gitignored (3.5 MB + 27 KB) and distributed via GitHub Releases.

| File | Size | Description |
|---|---|---|
| `pace_transformer_weights.pt` | 3.5 MB | Trained FT-Transformer state dict |
| `artifacts.pkl` | 27 KB | Categorical encoder, StandardScaler, feature column lists |

**[Download from Releases →](https://github.com/Spring-ISYS-2026-Alpha-Team/Accessorial_Cost_Detection_Engine/releases/tag/v1.0-weights)**

To retrain: run `pipeline/pace_transformer.py` on the HPC cluster, then upload new weights as a new release and update `RELEASE_TAG` in `scripts/download_weights.py`.

---

## CI/CD

GitHub Actions runs on every push to `main` and on all pull requests:

- **Lint** — flake8 (PEP8 compliance)
- **Security scan** — bandit (OWASP top 10, SQL injection, pickle safety)
- **Tests** — pytest covering upload validation and data pipeline

---

## Team

**Team Alpha — University of Arkansas, ISYS 43603, Spring 2026**

| Name | Role |
|---|---|
| [Clayton Josef](https://github.com/clayton-josef) | Scrum Master — ML pipeline, inference engine, database layer, deployment |
| Tyler Connolly | Product Owner — UI/UX, dashboard pages, theme system |
| Bui Vu | Bui Vu — ML Engineer — Cost estimator, data integration, context-aware filtering |
| Anna Diggs | Developer |
| Kirsten Capangpangan | Developer |

---

## Academic Context

Built as a capstone project for ISYS 43603 at the University of Arkansas (Spring 2026). The system addresses a real operational problem in freight logistics: the inability to anticipate accessorial charges before dispatch. The ML approach — FT-Transformer on heterogeneous tabular data with live economic and regulatory enrichment — reflects current industry-grade techniques for structured prediction tasks.
