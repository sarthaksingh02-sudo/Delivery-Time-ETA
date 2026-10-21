"""Last-mile delivery ETA training and inference utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Delivery_Time"
ALIASES = {
    "time_taken(min)": TARGET, "Time_taken (min)": TARGET,
    "Delivery_Time (min)": TARGET, "delivery_time": TARGET,
    "Agent_Age": "Agent_Age", "Delivery_person_Age": "Agent_Age",
    "Agent_Rating": "Agent_Rating", "Delivery_person_Ratings": "Agent_Rating",
    "Store_Latitude": "Pickup_Latitude", "Store_Longitude": "Pickup_Longitude",
    "Restaurant_latitude": "Pickup_Latitude", "Restaurant_longitude": "Pickup_Longitude",
    "Drop_Latitude": "Drop_Latitude", "Drop_Longitude": "Drop_Longitude",
    "Road_traffic_density": "Traffic", "Weatherconditions": "Weather",
    "Weather": "Weather", "Vehicle": "Vehicle", "Type_of_vehicle": "Vehicle",
    "Area": "Area", "multiple_deliveries": "Multiple_Deliveries",
    "Festival": "Festival", "City": "City", "Category": "Category", "Order_Date": "Order_Date",
    "Time_Orderd": "Order_Time", "Time_Order_picked": "Pickup_Time",
}
NUMERIC = ["Agent_Age", "Agent_Rating", "Distance_km", "Multiple_Deliveries"]
CATEGORICAL = ["Weather", "Traffic", "Vehicle", "Area", "Festival", "City", "Hour", "Category"]
FEATURES = NUMERIC + CATEGORICAL


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    renames = {c: ALIASES[c] for c in df.columns if c in ALIASES}
    df = df.rename(columns=renames)
    # Kaggle exports sometimes include spaces, leading/trailing underscores or BOMs.
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    return df


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize(data)
    for c in ["Agent_Age", "Agent_Rating", "Multiple_Deliveries", "Pickup_Latitude", "Pickup_Longitude", "Drop_Latitude", "Drop_Longitude"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace("_", "", regex=False), errors="coerce")
    if all(c in df for c in ["Pickup_Latitude", "Pickup_Longitude", "Drop_Latitude", "Drop_Longitude"]):
        lat1, lon1, lat2, lon2 = (np.radians(df[c].to_numpy(dtype=float)) for c in ["Pickup_Latitude", "Pickup_Longitude", "Drop_Latitude", "Drop_Longitude"])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df["Distance_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    elif "Distance_km" not in df:
        df["Distance_km"] = np.nan
    # Order time is available at dispatch; pickup time often occurs after dispatch and is excluded.
    if "Order_Time" in df:
        parsed = pd.to_datetime(df["Order_Time"].replace({"NaN": np.nan, "nan": np.nan}), format="mixed", errors="coerce")
        df["Hour"] = parsed.dt.hour.astype(float)
    if "Order_Date" in df:
        dt = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)
        if "Hour" not in df:
            df["Hour"] = np.nan
        df["Weekday"] = dt.dt.dayofweek
    for c in NUMERIC:
        if c not in df: df[c] = np.nan
    for c in CATEGORICAL:
        if c not in df: df[c] = np.nan
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    return df


def _pipeline(model: Any, numeric_features: list[str] | None = None, categorical_features: list[str] | None = None) -> Pipeline:
    numeric_features = NUMERIC if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL if categorical_features is None else categorical_features
    transformers = []
    if numeric_features:
        numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
        transformers.append(("numeric", numeric, numeric_features))
    if categorical_features:
        categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("categorical", categorical, categorical_features))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("features", prep), ("model", model)])


def train(df: pd.DataFrame, output_dir: str | Path = "models", random_state: int = 42) -> dict[str, Any]:
    data = add_features(df)
    if TARGET not in data:
        raise ValueError("Target column not found. Expected Delivery_Time or Kaggle's time_taken(min).")
    y = pd.to_numeric(data[TARGET].astype(str).str.extract(r"([-+]?\d*\.?\d+)", expand=False), errors="coerce")
    valid = y.notna() & y.between(1, 300)
    numeric_features = [c for c in NUMERIC if data[c].notna().any()]
    categorical_features = [c for c in CATEGORICAL if data[c].notna().any()]
    active_features = numeric_features + categorical_features
    if not active_features:
        raise ValueError("No usable model features were found in this CSV.")
    X, y = data.loc[valid, active_features], y.loc[valid]
    if len(X) < 40:
        raise ValueError(f"Need at least 40 usable delivery rows; found {len(X)}.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    candidates = {
        "Ridge baseline": _pipeline(Ridge(alpha=10), numeric_features, categorical_features),
        "Random Forest": _pipeline(RandomForestRegressor(n_estimators=100, min_samples_leaf=4, max_leaf_nodes=1400, max_features=0.9, n_jobs=-1, random_state=random_state), numeric_features, categorical_features),
        "Extra Trees": _pipeline(ExtraTreesRegressor(n_estimators=110, min_samples_leaf=4, max_leaf_nodes=1400, max_features=0.9, n_jobs=-1, random_state=random_state), numeric_features, categorical_features),
    }
    scores = []
    fitted = {}
    # Keep CV responsive for the interactive classroom app while preserving a
    # separate, untouched held-out test split for the final comparison.
    cv_n = min(len(X_train), 12000)
    cv_idx = X_train.sample(n=cv_n, random_state=random_state).index
    X_cv, y_cv = X_train.loc[cv_idx], y_train.loc[cv_idx]
    for name, pipe in candidates.items():
        cv_result = cross_validate(
            pipe, X_cv, y_cv, cv=KFold(n_splits=3, shuffle=True, random_state=random_state),
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            n_jobs=1, return_train_score=False,
        )
        pipe.fit(X_train, y_train)
        pred = np.clip(pipe.predict(X_test), 1, None)
        mape = float(np.mean(np.abs((y_test - pred) / np.maximum(np.abs(y_test), 1))) * 100)
        scores.append({"Model": name, "CV MAE (min)": float(-cv_result["test_mae"].mean()), "CV RMSE (min)": float(-cv_result["test_rmse"].mean()), "MAE (min)": float(mean_absolute_error(y_test, pred)), "RMSE (min)": float(np.sqrt(mean_squared_error(y_test, pred))), "MAPE (%)": mape, "R²": float(r2_score(y_test, pred))})
        fitted[name] = pipe
    score_df = pd.DataFrame(scores).sort_values("MAE (min)")
    winner = str(score_df.iloc[0]["Model"])
    # Final model is refit on all eligible training records after the held-out comparison.
    final_model = candidates[winner]
    final_model.fit(X, y)
    report = {"winner": winner, "metrics": score_df.to_dict(orient="records"), "rows_used": int(len(X)), "rows_dropped": int(len(data) - len(X)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "cv_rows": int(cv_n), "cv_folds": 3, "features": active_features, "unused_all_missing_features": [c for c in FEATURES if c not in active_features], "target_min": float(y.min()), "target_median": float(y.median()), "target_max": float(y.max()), "note": "Three-fold shuffled CV on up to 12,000 training rows; model choice and reported test metrics use an untouched random 20% holdout. Use time/grouped validation before a production claim."}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_model, "features": active_features, "numeric_features": numeric_features, "categorical_features": categorical_features, "winner": winner, "training_rows": int(len(X)), "mode": "real_dataset"}, out / "eta_model.joblib")
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    score_df.to_csv(out / "model_comparison.csv", index=False)
    return report


def demo_model() -> dict[str, Any]:
    """Create a clearly synthetic illustrative model for a live demo without the CSV."""
    rng = np.random.default_rng(23); n = 1800
    dist = rng.gamma(2.2, 2.1, n).clip(0.4, 22)
    age = rng.normal(29, 6, n).clip(18, 55); rating = rng.normal(4.7, .25, n).clip(2.5, 5)
    traffic = rng.choice(["Low", "Medium", "High", "Jam"], n, p=[.2, .35, .3, .15])
    weather = rng.choice(["Sunny", "Cloudy", "Rainy", "Fog", "Stormy"], n, p=[.3, .25, .25, .12, .08])
    X = pd.DataFrame({"Agent_Age": age, "Agent_Rating": rating, "Distance_km": dist, "Multiple_Deliveries": rng.choice([0,1,2], n, p=[.65,.28,.07]), "Weather": weather, "Traffic": traffic, "Vehicle": rng.choice(["motorcycle", "scooter", "bicycle"], n), "Area": rng.choice(["Urban", "Metropolitan", "Other"], n), "Festival": rng.choice(["No", "Yes"], n, p=[.92,.08]), "City": rng.choice(["Urban", "Metropolitan", "Semi-Urban"], n), "Hour": rng.integers(8, 23, n), "Category": rng.choice(["Clothing", "Electronics", "Sports", "Cosmetics", "Toys"], n)})
    traffic_effect = X.Traffic.map({"Low": 0, "Medium": 7, "High": 15, "Jam": 23}).to_numpy()
    weather_effect = X.Weather.map({"Sunny": 0, "Cloudy": 2, "Rainy": 7, "Fog": 5, "Stormy": 12}).to_numpy()
    y = 8 + dist * 4.0 + traffic_effect + weather_effect + X.Multiple_Deliveries.to_numpy() * 5 - (rating - 4.5) * 5 + rng.normal(0, 5, n)
    pipe = _pipeline(ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2, n_jobs=-1, random_state=23)).fit(X, y.clip(5, 180))
    return {"pipeline": pipe, "features": FEATURES, "winner": "Extra Trees (synthetic demo)", "training_rows": n, "mode": "synthetic_demo"}


def predict(model_bundle: dict[str, Any], raw: pd.DataFrame) -> np.ndarray:
    x = add_features(raw)
    # Build exactly the training columns with single-row safe categorical values.
    return np.clip(model_bundle["pipeline"].predict(x[model_bundle["features"]]), 1, None)
