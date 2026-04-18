# DEPRECATED — no active callers. Superseded by pipeline/inference.py (PACE FT-Transformer).
# Safe to delete once utils/legacy/cost_model.py is also removed.
"""
utils/ml.py
LightGBM cost prediction pipeline for PACE.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pace_model.joblib")

CAT_FEATURES = ["CarrierId", "FacilityType", "AppointmentType"]
NUM_FEATURES = ["DistanceMiles", "weight_lbs", "avg_dwell_time_hrs", "month", "day_of_week"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "total_cost"


def load_training_data(conn) -> pd.DataFrame:
    """Pull all shipment rows with facility dwell time joined."""
    query = """
        SELECT
            s.CarrierId,
            s.FacilityType,
            s.AppointmentType,
            s.DistanceMiles,
            s.weight_lbs,
            COALESCE(f.avg_dwell_time_hrs, 4.0) AS avg_dwell_time_hrs,
            s.ShipDate,
            s.LinehaulCost + s.AccessorialCost AS total_cost
        FROM Shipments s
        LEFT JOIN Facilities f ON s.facility_id = f.facility_id
    """
    df = pd.read_sql(query, conn)
    df["month"] = pd.to_datetime(df["ShipDate"]).dt.month
    df["day_of_week"] = pd.to_datetime(df["ShipDate"]).dt.dayofweek
    df = df.dropna(subset=ALL_FEATURES + [TARGET])
    return df


def _to_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")
    return df


def train(conn):
    """Train on all available shipment data. Save model to disk. Return (model, metrics)."""
    df = load_training_data(conn)
    X = _to_categoricals(df[ALL_FEATURES])
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CAT_FEATURES)

    y_pred = model.predict(X_test)
    metrics = {
        "mae":     float(mean_absolute_error(y_test, y_pred)),
        "rmse":    float(mean_squared_error(y_test, y_pred) ** 0.5),
        "r2":      float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }

    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    return model, metrics


def load_model():
    """Load saved model from disk. Returns (model, metrics) or (None, None)."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    return data["model"], data["metrics"]


def predict(model, *, carrier_id, facility_type, appointment_type,
            distance, weight, dwell_time, month, day_of_week) -> float:
    """Run a single prediction. Returns predicted total cost in dollars."""
    X = pd.DataFrame([{
        "CarrierId":         carrier_id,
        "FacilityType":      facility_type,
        "AppointmentType":   appointment_type,
        "DistanceMiles":     float(distance),
        "weight_lbs":        float(weight),
        "avg_dwell_time_hrs": float(dwell_time),
        "month":             int(month),
        "day_of_week":       int(day_of_week),
    }])
    X = _to_categoricals(X)
    return float(model.predict(X)[0])


def get_feature_importance(model) -> pd.DataFrame:
    """Return feature importances sorted descending."""
    labels = [
        "Carrier", "Facility Type", "Appointment Type",
        "Distance (mi)", "Weight (lbs)", "Avg Dwell Time (hrs)",
        "Month", "Day of Week",
    ]
    return pd.DataFrame({
        "Feature":    labels,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)
