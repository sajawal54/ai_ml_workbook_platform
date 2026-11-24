from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


_MISSING_VALUES = {"", "na", "nan", "null", "none"}


def _is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(
        _MISSING_VALUES
    )


def _numeric_param(
    params: dict[str, Any], key: str, default: int | float, *, integer: bool = False
) -> int | float:
    raw = params.get(key, default)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Parameter '{key}' must be a valid number.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Parameter '{key}' must be finite.")
    if integer:
        if not numeric.is_integer():
            raise ValueError(f"Parameter '{key}' must be a whole number.")
        return int(numeric)
    return numeric


def _format_numeric_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _extract_numeric_features(
    df: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    if not feature_columns:
        raise ValueError("Select at least one numeric feature column.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature columns must be unique.")

    feature_frames: list[pd.Series] = []
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' was not found in the dataset.")
        raw = df[col]
        present = raw[~_is_missing(raw)]
        if present.empty:
            raise ValueError(
                f"Feature column '{col}' has no numeric values. Remove it or provide data."
            )
        parsed = pd.to_numeric(present, errors="coerce")
        numeric_values = parsed.to_numpy(dtype=float)
        invalid = parsed.isna().to_numpy() | ~np.isfinite(numeric_values)
        if invalid.any():
            samples = (
                present.iloc[np.flatnonzero(invalid)]
                .astype(str)
                .unique()[:3]
                .tolist()
            )
            raise ValueError(
                f"Feature column '{col}' has non-numeric or non-finite values ({samples}). Use finite numeric values."
            )
        numeric_col = pd.to_numeric(raw.mask(_is_missing(raw)), errors="coerce")
        feature_frames.append(numeric_col)

    X_df = pd.concat(feature_frames, axis=1)
    X_df.columns = feature_columns
    return X_df.to_numpy(dtype=float)


def _project_2d(X: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    if X.shape[0] == 0:
        return []
    if X.shape[1] >= 2 and X.shape[0] >= 2:
        if np.allclose(X, X[0]):
            coords = np.zeros((X.shape[0], 2), dtype=float)
        else:
            coords = PCA(n_components=2, random_state=42).fit_transform(X)
    elif X.shape[1] == 1:
        coords = np.column_stack([X[:, 0], np.zeros(X.shape[0])])
    else:
        coords = np.column_stack(
            [np.arange(X.shape[0], dtype=float), np.zeros(X.shape[0])]
        )

    points: list[dict[str, Any]] = []
    for i in range(len(coords)):
        x_val = float(coords[i, 0]) if np.isfinite(coords[i, 0]) else 0.0
        y_val = float(coords[i, 1]) if np.isfinite(coords[i, 1]) else 0.0
        points.append(
            {
                "x": round(x_val, 4),
                "y": round(y_val, 4),
                "label": str(labels[i]),
                "row_index": i + 1,
            }
        )
    return points


def _resolve_task(
    y_series: pd.Series, requested_task: str, algorithm: str
) -> tuple[str, np.ndarray, list[str]]:
    numeric_try = pd.to_numeric(y_series, errors="coerce")
    is_numeric = bool(numeric_try.notna().all())
    numeric_values = numeric_try.to_numpy(dtype=float) if is_numeric else None
    if numeric_values is not None and not np.isfinite(numeric_values).all():
        raise ValueError("Target values must be finite numbers.")

    if (
        algorithm == "linear_ridge_regression"
        and requested_task == "auto"
        and numeric_values is not None
    ):
        return "regression", numeric_values, []

    def encode_numeric_classes(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
        unique_vals = np.unique(values)
        encoded = np.searchsorted(unique_vals, values)
        labels = [_format_numeric_label(float(value)) for value in unique_vals]
        return encoded.astype(int), labels

    if requested_task == "regression":
        if numeric_values is None:
            raise ValueError("Regression requires a numeric target column.")
        return "regression", numeric_values, []

    if requested_task == "classification":
        if numeric_values is not None:
            if len(np.unique(numeric_values)) < 2:
                raise ValueError(
                    "Target column needs at least 2 distinct classes for classification."
                )
            y_encoded, labels = encode_numeric_classes(numeric_values)
            return "classification", y_encoded, labels
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y_series.astype(str))
        if len(encoder.classes_) < 2:
            raise ValueError(
                "Target column needs at least 2 distinct classes for classification."
            )
        return "classification", y_enc, [str(c) for c in encoder.classes_]

    # requested_task == "auto"
    if numeric_values is not None:
        unique_vals = np.unique(numeric_values)
        is_discrete_ints = (
            len(unique_vals) <= 10
            and np.all(np.equal(unique_vals, np.floor(unique_vals)))
        )
        if is_discrete_ints and len(unique_vals) >= 2:
            y_encoded, labels = encode_numeric_classes(numeric_values)
            return "classification", y_encoded, labels
        return "regression", numeric_values, []

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_series.astype(str))
    if len(encoder.classes_) < 2:
        raise ValueError("Target column needs at least 2 distinct categories.")
    return "classification", y_enc, [str(c) for c in encoder.classes_]


def _compute_roc_details(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> tuple[float | None, dict[str, Any]]:
    scores: np.ndarray | None = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(X_test)
            if num_classes == 2 and probs.ndim == 2 and probs.shape[1] >= 2:
                scores = probs[:, 1]
            elif probs.ndim == 2 and probs.shape[1] == num_classes:
                auc_val = float(
                    roc_auc_score(
                        y_test, probs, multi_class="ovr", average="macro"
                    )
                )
                return round(auc_val, 4), {}
        except Exception:
            scores = None

    if scores is None and hasattr(pipeline, "decision_function"):
        try:
            df_scores = pipeline.decision_function(X_test)
            if num_classes == 2 and np.ndim(df_scores) == 1:
                scores = np.asarray(df_scores, dtype=float)
        except Exception:
            scores = None

    if scores is not None and len(np.unique(y_test)) == 2:
        try:
            auc_val = float(roc_auc_score(y_test, scores))
            fpr, tpr, _ = roc_curve(y_test, scores)
            curve_points = [
                {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                for f, t in zip(fpr, tpr)
            ]
            return round(auc_val, 4), {
                "roc_curve": {
                    "auc": round(auc_val, 4),
                    "points": curve_points,
                }
            }
        except Exception:
            return None, {}

    return None, {}
