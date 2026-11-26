"""Core data preparation and machine-learning routines for Beijing PM2.5."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

TARGET = "pm2.5"
BASE_FEATURES = ["year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
ENGINEERED_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
NUMERIC_FEATURES = [c for c in FEATURES if c != "cbwd"]
CATEGORICAL_FEATURES = ["cbwd"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column spelling/case variants without losing meaning."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower_map = {str(c).lower(): c for c in out.columns}
    aliases = {
        "pm25": "pm2.5", "pm_2.5": "pm2.5", "pm2_5": "pm2.5",
        "dewp": "DEWP", "temp": "TEMP", "pres": "PRES",
        "iws": "Iws", "is": "Is", "ir": "Ir", "cbwd": "cbwd",
        "year": "year", "month": "month", "day": "day", "hour": "hour",
    }
    rename = {}
    for lower_name, canonical in aliases.items():
        if lower_name in lower_map and lower_map[lower_name] != canonical:
            rename[lower_map[lower_name]] = canonical
    if rename:
        out = out.rename(columns=rename)
    return out


def prepare_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a supervised-learning frame and a transparent cleaning report."""
    if raw_df is None or raw_df.empty:
        raise ValueError("Dataset kosong. Unggah CSV yang memiliki data.")

    raw = normalize_columns(raw_df)
    required = ["year", "month", "day", "hour", TARGET, "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing_cols))

    before_rows = len(raw)
    missing_before = raw.isna().sum().to_dict()
    data = raw.copy()

    numeric_cols = ["year", "month", "day", "hour", TARGET, "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # The time fields are required to preserve the temporal order of the dataset.
    data["timestamp"] = pd.to_datetime(
        {
            "year": data["year"], "month": data["month"],
            "day": data["day"], "hour": data["hour"],
        }, errors="coerce"
    )
    invalid_timestamps = int(data["timestamp"].isna().sum())
    data = data.dropna(subset=["timestamp"])

    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates()
    missing_target_rows = int(data[TARGET].isna().sum())
    data = data.dropna(subset=[TARGET])
    data["cbwd"] = data["cbwd"].astype(object)
    data["cbwd"] = data["cbwd"].apply(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    data["cbwd"] = data["cbwd"].replace({"": np.nan, "nan": np.nan, "<NA>": np.nan})

    # Chronological ordering is intentional: later records must not leak into training.
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)
    data["dayofweek"] = data["timestamp"].dt.dayofweek

    report = {
        "rows_before": before_rows,
        "rows_after": len(data),
        "missing_target_removed": missing_target_rows,
        "invalid_timestamp_removed": invalid_timestamps,
        "duplicate_rows_removed": duplicate_rows,
        "missing_before": missing_before,
        "missing_after_predictors": data[BASE_FEATURES].isna().sum().to_dict(),
        "columns": list(raw.columns),
    }
    return data, report


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Impute missing features, encode wind direction, and optionally scale."""
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_regression_estimators() -> dict[str, tuple[Any, bool]]:
    """Regression counterparts of the module's supervised algorithms."""
    return {
        "Linear Regression": (LinearRegression(), True),
        "Support Vector Regression (LinearSVR)": (
            LinearSVR(C=0.12, epsilon=0.1, max_iter=12000, random_state=42), True
        ),
        "Decision Tree Regressor": (
            DecisionTreeRegressor(max_depth=20, min_samples_leaf=3, random_state=42), False
        ),
        "Random Forest Regressor": (
            RandomForestRegressor(
                n_estimators=120, max_depth=24, min_samples_leaf=2,
                max_features=0.9, n_jobs=-1, random_state=42
            ), False
        ),
        "Extra Trees Regressor": (
            ExtraTreesRegressor(
                n_estimators=120, max_depth=24, min_samples_leaf=2,
                max_features=0.9, n_jobs=-1, random_state=42
            ), False
        ),
        "Gradient Boosting Regressor": (
            HistGradientBoostingRegressor(
                max_iter=160, learning_rate=0.07, max_leaf_nodes=25,
                l2_regularization=1.0, random_state=42
            ), False
        ),
    }


def chronological_split(data: pd.DataFrame, test_fraction: float = 0.2):
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("Proporsi data test harus berada pada rentang 5% sampai 50%.")
    X = data[FEATURES].copy()
    y = data[TARGET].astype(float).copy()
    split_at = int(len(data) * (1.0 - test_fraction))
    if split_at < 100 or len(data) - split_at < 20:
        raise ValueError("Jumlah data tidak cukup untuk pembagian train/test tersebut.")
    return X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:], split_at


def train_regression_models(
    data: pd.DataFrame,
    model_names: list[str],
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, split_at = chronological_split(data, test_fraction)
    estimators = make_regression_estimators()
    results: dict[str, Any] = {}

    for name in model_names:
        if name not in estimators:
            continue
        estimator, scale = estimators[name]
        pipeline = Pipeline(
            steps=[("preprocessor", make_preprocessor(scale_numeric=scale)), ("regressor", estimator)]
        )
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        predictions = pipeline.predict(X_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R²": float(r2_score(y_test, predictions)),
        }
        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "predictions": np.asarray(predictions),
            "actual": np.asarray(y_test),
            "test_timestamps": data["timestamp"].iloc[split_at:].reset_index(drop=True),
            "training_seconds": elapsed,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
    if not results:
        raise ValueError("Pilih sedikitnya satu algoritma untuk dilatih.")
    best_name = min(results, key=lambda key: results[key]["metrics"]["RMSE"])
    return {
        "task": "regression",
        "results": results,
        "best_name": best_name,
        "feature_columns": FEATURES.copy(),
        "test_fraction": test_fraction,
        "split_at": split_at,
        "dataset_rows": len(data),
    }


def make_classification_estimators() -> dict[str, tuple[Any, bool]]:
    return {
        "Support Vector Classifier (LinearSVC)": (
            LinearSVC(C=0.12, max_iter=12000, random_state=42), True
        ),
        "Decision Tree Classifier": (
            DecisionTreeClassifier(max_depth=18, min_samples_leaf=4, random_state=42), False
        ),
        "Random Forest Classifier": (
            RandomForestClassifier(
                n_estimators=100, max_depth=22, min_samples_leaf=3,
                n_jobs=-1, random_state=42, class_weight="balanced_subsample"
            ), False
        ),
        "Gaussian Naive Bayes": (GaussianNB(), True),
    }


def train_classification_models(
    data: pd.DataFrame,
    model_names: list[str],
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Optional classification view; classes are dataset-relative tertiles, not AQI standards."""
    X_train, X_test, y_train, y_test, split_at = chronological_split(data, test_fraction)
    q1, q2 = y_train.quantile([1 / 3, 2 / 3]).tolist()
    if q1 >= q2:
        raise ValueError("Batas kuantil PM2.5 tidak dapat membentuk tiga kelas yang berbeda.")
    labels = ["Relatif rendah", "Relatif sedang", "Relatif tinggi"]
    bins = [-np.inf, q1, q2, np.inf]
    y_train_class = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
    y_test_class = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

    estimators = make_classification_estimators()
    results: dict[str, Any] = {}
    for name in model_names:
        if name not in estimators:
            continue
        estimator, scale = estimators[name]
        pipeline = Pipeline(
            steps=[("preprocessor", make_preprocessor(scale_numeric=scale)), ("classifier", estimator)]
        )
        start = time.perf_counter()
        pipeline.fit(X_train, y_train_class)
        elapsed = time.perf_counter() - start
        predictions = pipeline.predict(X_test)
        metrics = {
            "Accuracy": float(accuracy_score(y_test_class, predictions)),
            "Precision macro": float(precision_score(y_test_class, predictions, average="macro", zero_division=0)),
            "Recall macro": float(recall_score(y_test_class, predictions, average="macro", zero_division=0)),
            "F1 macro": float(f1_score(y_test_class, predictions, average="macro", zero_division=0)),
        }
        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "predictions": np.asarray(predictions, dtype=object),
            "actual": np.asarray(y_test_class, dtype=object),
            "confusion_matrix": confusion_matrix(y_test_class, predictions, labels=labels),
            "report": classification_report(y_test_class, predictions, labels=labels, output_dict=True, zero_division=0),
            "training_seconds": elapsed,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
    if not results:
        raise ValueError("Pilih sedikitnya satu algoritma klasifikasi.")
    best_name = max(results, key=lambda key: results[key]["metrics"]["F1 macro"])
    return {
        "task": "classification",
        "results": results,
        "best_name": best_name,
        "feature_columns": FEATURES.copy(),
        "test_fraction": test_fraction,
        "split_at": split_at,
        "dataset_rows": len(data),
        "class_labels": labels,
        "class_thresholds": [float(q1), float(q2)],
        "class_definition": "Tiga kelas kuantil dari data train (bukan kategori baku ISPU/AQI).",
    }


def get_feature_importance(pipeline: Pipeline, task: str = "regression") -> pd.DataFrame:
    step_name = "regressor" if task == "regression" else "classifier"
    estimator = pipeline.named_steps[step_name]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["Fitur", "Importance"])
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    values = estimator.feature_importances_
    out = pd.DataFrame({"Fitur": feature_names, "Importance": values})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def engineer_prediction_features(input_df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(input_df)
    data = data.copy()
    # Accept inputs using the base feature schema. Date-derived fields are generated here.
    for col in ["year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
        if col not in data.columns:
            raise ValueError(f"Kolom prediktor '{col}' belum tersedia.")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if "cbwd" not in data.columns:
        raise ValueError("Kolom 'cbwd' belum tersedia.")
    data["cbwd"] = data["cbwd"].astype(object)
    data["cbwd"] = data["cbwd"].apply(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    data["cbwd"] = data["cbwd"].replace({"": np.nan, "nan": np.nan, "<NA>": np.nan})
    ts = pd.to_datetime({"year": data["year"], "month": data["month"], "day": data["day"], "hour": data["hour"]}, errors="coerce")
    if ts.isna().any():
        raise ValueError("Tanggal/jam pada data prediksi tidak valid.")
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)
    data["dayofweek"] = ts.dt.dayofweek
    return data[FEATURES].copy()


def run_kmeans_analysis(
    data: pd.DataFrame,
    selected_k: int = 3,
    feature_columns: list[str] | None = None,
    sample_size: int = 5000,
) -> dict[str, Any]:
    """Evaluate k=2..7 by silhouette on a bounded sample, then fit selected k."""
    from sklearn.pipeline import make_pipeline

    candidates = feature_columns or [TARGET, "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    candidates = [c for c in candidates if c in data.columns]
    if len(candidates) < 2:
        raise ValueError("Pilih sedikitnya dua fitur numerik untuk clustering.")
    matrix = data[candidates].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median()).replace([np.inf, -np.inf], np.nan).fillna(0)
    rng = np.random.RandomState(42)
    if len(matrix) > sample_size:
        sample_idx = rng.choice(len(matrix), size=sample_size, replace=False)
        sample = matrix.iloc[np.sort(sample_idx)]
    else:
        sample = matrix
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample)
    scores = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(sample_scaled)
        score = float(silhouette_score(sample_scaled, labels))
        scores.append({"K": k, "Silhouette": score})
    selected_k = int(np.clip(selected_k, 2, 7))
    # Fit scaler and KMeans on all available observations for final cluster profiles.
    full_scaled = scaler.fit_transform(matrix)
    model = KMeans(n_clusters=selected_k, n_init=10, random_state=42)
    labels = model.fit_predict(full_scaled)
    clustered = data.copy()
    clustered["Cluster"] = labels
    profile = clustered.groupby("Cluster")[candidates].mean().round(2)
    if len(full_scaled) <= 12000:
        selected_silhouette = silhouette_score(full_scaled, labels)
    else:
        silhouette_idx = rng.choice(len(full_scaled), 12000, replace=False)
        selected_silhouette = silhouette_score(full_scaled[silhouette_idx], labels[silhouette_idx])
    return {
        "model": model,
        "scaler": scaler,
        "scores": pd.DataFrame(scores),
        "clustered": clustered,
        "profile": profile,
        "features": candidates,
        "selected_k": selected_k,
        "silhouette_selected": float(selected_silhouette),
    }
