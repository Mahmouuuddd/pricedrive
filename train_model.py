"""
Train XGBoost regressor and label encoders to match project.ipynb:
- Data: dataset/cleaned_dataset.csv (post-preprocessing tabular features)
- Categoricals: make, model, trim, body, transmission (LabelEncoder)
- Target: sellingprice
- Model: XGBRegressor with tuned hyperparameters from RandomizedSearchCV
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "cleaned_dataset.csv"
MODEL_DIR = ROOT / "models"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    cat_cols = ["make", "model", "trim", "body", "transmission"]
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop("sellingprice", axis=1)
    y = df["sellingprice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        subsample=0.8,
        n_estimators=700,
        min_child_weight=3,
        max_depth=10,
        learning_rate=0.03,
        gamma=0.3,
        colsample_bytree=0.7,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    feature_columns = X_train.columns.tolist()
    joblib.dump(model, MODEL_DIR / "xgb_pricedrive.pkl")
    joblib.dump(encoders, MODEL_DIR / "label_encoders.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")
    print("Saved:", MODEL_DIR / "xgb_pricedrive.pkl")
    print("Feature order:", feature_columns)


if __name__ == "__main__":
    main()
