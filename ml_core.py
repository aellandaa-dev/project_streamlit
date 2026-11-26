"""Core workflow for the Beijing PM2.5 Streamlit learning lab.

Supervised task: derive a teaching label from pm2.5, then predict it from
weather/time features only (pm2.5 is never included in X).
Unsupervised task: K-Means on standardized numeric weather/time features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

NUMERIC_FEATURES = [
    "year", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
CATEGORICAL_FEATURES = ["cbwd"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLUSTER_FEATURES = [
    "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
MODEL_LABELS = {
    "SVM": "Support Vector Machine",
    "Decision Tree": "Decision Tree (Entropy)",
    "Random Forest": "Random Forest",
    "Naive Bayes": "Gaussian Naive Bayes",
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the original Beijing hourly PM2.5 dataset."""
    df = pd.read_csv(csv_path)
    required = {
        "year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP",
        "PRES", "cbwd", "Iws", "Is", "Ir",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclic encodings for hour and month, retaining year as trend."""
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"].astype(float) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"].astype(float) / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"].astype(float) - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"].astype(float) - 1) / 12.0)
    return out


def make_target(pm25: pd.Series, low_threshold: float = 35.0,
                high_threshold: float = 75.0) -> pd.Series:
    """Derive three classroom classes from numeric PM2.5 thresholds."""
    if low_threshold >= high_threshold:
        raise ValueError("Ambang Rendah harus lebih kecil daripada ambang Tinggi.")
    labels = ["Rendah", "Sedang", "Tinggi"]
    # right=True means: <= low, (low, high], > high.
    return pd.cut(
        pm25,
        bins=[-np.inf, low_threshold, high_threshold, np.inf],
        labels=labels,
        include_lowest=True,
        right=True,
    )


def build_supervised_frame(raw: pd.DataFrame, low_threshold: float = 35.0,
                           high_threshold: float = 75.0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and cleaned labeled rows. Rows without PM2.5 are excluded.

    The original PM2.5 target is intentionally omitted from the feature matrix.
    """
    cleaned = raw.dropna(subset=["pm2.5"]).copy()
    engineered = add_time_features(cleaned)
    y = make_target(engineered["pm2.5"], low_threshold, high_threshold).astype(str)
    X = engineered[FEATURES].copy()
    X["cbwd"] = X["cbwd"].astype(str)
    return X, y, engineered


def _preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_classifier(name: str, rf_trees: int = 100,
                     tree_depth: int = 12, svm_c: float = 1.0) -> Pipeline:
    if name == "SVM":
        estimator = SVC(
            kernel="linear", C=svm_c, class_weight="balanced",
            probability=True, random_state=42,
        )
    elif name == "Decision Tree":
        estimator = DecisionTreeClassifier(
            criterion="entropy", max_depth=tree_depth,
            class_weight="balanced", random_state=42,
        )
    elif name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=rf_trees, max_depth=tree_depth,
            min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )
    elif name == "Naive Bayes":
        estimator = GaussianNB()
    else:
        raise ValueError(f"Algoritma tidak dikenal: {name}")
    return Pipeline([
        ("preprocess", _preprocessor()),
        ("model", estimator),
    ])


def _sample_for_training(X: pd.DataFrame, y: pd.Series,
                         sample_size: int, random_state: int = 42):
    if sample_size and sample_size < len(X):
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_size, random_state=random_state,
            stratify=y,
        )
        return X_sample, y_sample
    return X, y


def train_classifier(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    sample_size: int = 8000,
    rf_trees: int = 100,
    tree_depth: int = 12,
    svm_c: float = 1.0,
) -> dict[str, Any]:
    """Train one model and return its pipeline and held-out test diagnostics."""
    X_sample, y_sample = _sample_for_training(X, y, sample_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=test_size, random_state=42,
        stratify=y_sample,
    )
    pipeline = build_classifier(name, rf_trees, tree_depth, svm_c)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    labels = ["Rendah", "Sedang", "Tinggi"]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "sample_rows": int(len(X_sample)),
    }
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {
        "name": name,
        "label": MODEL_LABELS[name],
        "pipeline": pipeline,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importance": get_feature_importance(pipeline),
    }


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = estimator.feature_importances_
    result = pd.DataFrame({"feature": names, "importance": values})
    result["feature"] = result["feature"].str.replace("numeric__", "", regex=False)
    result["feature"] = result["feature"].str.replace("categorical__cbwd_", "cbwd = ", regex=False)
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def prepare_prediction_input(values: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Convert a row or batch of raw time/weather values to model features."""
    frame = values.copy() if isinstance(values, pd.DataFrame) else pd.DataFrame([values])
    required = ["year", "month", "hour", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError("Kolom input belum lengkap: " + ", ".join(missing))
    engineered = add_time_features(frame)
    engineered["cbwd"] = engineered["cbwd"].astype(str)
    return engineered[FEATURES].copy()


def run_kmeans(
    raw: pd.DataFrame,
    k: int = 4,
    sample_size: int = 3500,
    max_k: int = 8,
) -> dict[str, Any]:
    """Fit K-Means on standardized weather/time fields and score k=2..max_k."""
    # Clustering is unsupervised: retain rows even when pm2.5 is missing.
    # PM2.5 is used only for descriptive post-cluster profiling, never as X.
    valid = raw.copy()
    engineered = add_time_features(valid)
    if sample_size and sample_size < len(engineered):
        engineered = engineered.sample(n=sample_size, random_state=42)
    X = engineered[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow_rows = []
    for candidate_k in range(2, max_k + 1):
        candidate = KMeans(n_clusters=candidate_k, n_init=10, random_state=42)
        labels = candidate.fit_predict(X_scaled)
        # Silhouette is O(n^2); use a capped sample for practical UI latency.
        score = silhouette_score(
            X_scaled, labels,
            sample_size=min(1800, len(X_scaled)),
            random_state=42,
        )
        elbow_rows.append({"k": candidate_k, "inertia": float(candidate.inertia_), "silhouette": float(score)})

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    scored = engineered.copy()
    scored["cluster"] = labels.astype(int)
    scored["timestamp"] = pd.to_datetime(
        dict(year=scored["year"], month=scored["month"], day=scored["day"], hour=scored["hour"]),
        errors="coerce",
    )
    cluster_profile = scored.groupby("cluster")[CLUSTER_FEATURES + ["pm2.5"]].mean(numeric_only=True).reset_index()
    selected_silhouette = next(row["silhouette"] for row in elbow_rows if row["k"] == k)
    return {
        "model": model,
        "scaler": scaler,
        "data": scored,
        "elbow": pd.DataFrame(elbow_rows),
        "silhouette": float(selected_silhouette),
        "inertia": float(model.inertia_),
        "profile": cluster_profile,
        "sample_rows": int(len(scored)),
        "features": CLUSTER_FEATURES,
    }
