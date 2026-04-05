"""Encode raw inputs and run inference (same column order as training)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def _encode_single_safe(le: LabelEncoder, value: str) -> int:
    v = str(value).strip()
    if v in le.classes_:
        return int(le.transform([v])[0])
    return 0


def load_artifacts():
    model = joblib.load(MODEL_DIR / "xgb_pricedrive.pkl")
    encoders: dict[str, LabelEncoder] = joblib.load(MODEL_DIR / "label_encoders.pkl")
    feature_columns: list[str] = joblib.load(MODEL_DIR / "feature_columns.pkl")
    return model, encoders, feature_columns


def build_feature_row(
    encoders: dict[str, LabelEncoder],
    feature_columns: list[str],
    *,
    year: float,
    make: str,
    model: str,
    trim: str,
    body: str,
    transmission: str,
    condition: float,
    odometer: float,
    vehicle_age: float,
) -> pd.DataFrame:
    row: dict[str, Any] = {
        "year": year,
        "make": _encode_single_safe(encoders["make"], make),
        "model": _encode_single_safe(encoders["model"], model),
        "trim": _encode_single_safe(encoders["trim"], trim),
        "body": _encode_single_safe(encoders["body"], body),
        "transmission": _encode_single_safe(encoders["transmission"], transmission),
        "condition": condition,
        "odometer": odometer,
        "vehicle_age": vehicle_age,
    }
    return pd.DataFrame([row], columns=feature_columns)


def predict_price(
    model,
    encoders: dict[str, LabelEncoder],
    feature_columns: list[str],
    **kwargs: Any,
) -> float:
    X = build_feature_row(encoders, feature_columns, **kwargs)
    return float(model.predict(X)[0])
