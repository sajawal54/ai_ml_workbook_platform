from typing import Any
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


__all__ = [
    "_build_feature_preprocessor",
    "_compute_roc_details",
    "_display_feature_values",
    "_extract_feature_frame",
    "_format_feature_row",
    "_is_missing",
    "_numeric_param",
    "_project_2d",
    "_resolve_task",
]

_MISSING_VALUES = {"", "na", "nan", "null", "none"}
_MAX_CATEGORICAL_VALUES = 10
_MAX_CATEGORY_TEXT_LENGTH = 80
_TEXT_COLUMN_HINTS = {
    "address",
    "comment",
    "description",
    "email",
    "free_text",
    "name",
    "note",
    "phone",
    "review",
    "street",
    "text",
    "url",
}


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


def _is_identifier_column(column: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")
    return (
        normalized in {"id", "identifier", "uuid"}
        or normalized.endswith(("_id", "_identifier", "_uuid"))
    )


def _is_free_text_column(column: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")
    return bool(set(normalized.split("_")) & _TEXT_COLUMN_HINTS)


def _extract_feature_frame(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not feature_columns:
        raise ValueError("Select at least one numeric or categorical feature.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature columns must be unique.")

    feature_frame = pd.DataFrame(index=df.index)
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' was not found in the dataset.")
        raw = df[col]
        present = raw[~_is_missing(raw)]
        if present.empty:
            raise ValueError(
                f"Feature column '{col}' has no usable values. Remove it or provide data."
            )
        parsed = pd.to_numeric(present, errors="coerce")
        parsed_values = parsed.dropna().to_numpy(dtype=float)
        if not np.isfinite(parsed_values).all():
            raise ValueError(
                f"Feature column '{col}' contains non-finite numbers. Use finite numeric values."
            )
        if parsed.notna().all():
            if _is_identifier_column(col):
                raise ValueError(
                    f"Feature column '{col}' appears to be an identifier. Remove ID columns from Input Features."
                )
            feature_frame[col] = pd.to_numeric(
                raw.mask(_is_missing(raw)), errors="coerce"
            )
            numeric_columns.append(col)
            continue

        categories = present.astype(str).str.strip()
        unique_count = int(categories.nunique())
        too_many_values = unique_count > _MAX_CATEGORICAL_VALUES
        mostly_unique = (
            unique_count >= 5 and unique_count / len(categories) > 0.8
        )
        long_text = any(
            len(value) > _MAX_CATEGORY_TEXT_LENGTH for value in categories
        )
        if (
            too_many_values
            or mostly_unique
            or long_text
            or _is_identifier_column(col)
            or _is_free_text_column(col)
        ):
            raise ValueError(
                f"Feature column '{col}' looks like high-cardinality text or an ID and cannot be safely encoded. "
                "Remove it or use numeric features and categorical columns with 10 or fewer short values."
            )

        feature_frame[col] = raw.mask(_is_missing(raw)).map(
            lambda value: str(value).strip() if pd.notna(value) else np.nan
        )
        categorical_columns.append(col)

    return feature_frame.loc[:, feature_columns], numeric_columns, categorical_columns


def _build_feature_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_features: bool,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_features:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric_columns)
        )
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _display_feature_values(
    feature_frame: pd.DataFrame,
    preprocessor: ColumnTransformer,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
) -> np.ndarray:
    display_values = np.empty(
        (len(feature_frame), len(feature_columns)), dtype=object
    )
    column_positions = {
        column: index for index, column in enumerate(feature_columns)
    }

    if numeric_columns:
        numeric_pipeline = preprocessor.named_transformers_["numeric"]
        if scale_numeric:
            numeric_values = numeric_pipeline.transform(
                feature_frame.loc[:, numeric_columns]
            )
        else:
            numeric_values = numeric_pipeline.named_steps["imputer"].transform(
                feature_frame.loc[:, numeric_columns]
            )
        for index, column in enumerate(numeric_columns):
            display_values[:, column_positions[column]] = numeric_values[:, index]

    if categorical_columns:
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        categorical_values = categorical_pipeline.named_steps["imputer"].transform(
            feature_frame.loc[:, categorical_columns]
        )
        for index, column in enumerate(categorical_columns):
            display_values[:, column_positions[column]] = categorical_values[:, index]

    return display_values


def _format_feature_row(values: np.ndarray) -> list[Any]:
    return [
        round(float(value), 4)
        if isinstance(value, (int, float, np.number))
        else str(value)
        for value in values
    ]


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
