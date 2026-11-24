from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from app.services.ml_common import (
    _build_feature_preprocessor,
    _compute_roc_details,
    _display_feature_values,
    _extract_feature_frame,
    _format_feature_row,
    _is_missing,
    _numeric_param,
    _project_2d,
    _resolve_task,
)


def run_supervised(
    algorithm: str,
    columns: list[str],
    rows: list[list[Any]],
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    test_size: float,
    scale_features: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    if target_column in feature_columns:
        raise ValueError(
            f"Target column '{target_column}' is also selected as a feature. Unselect it from Input Features first."
        )

    df = pd.DataFrame(rows, columns=columns)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    X_raw, numeric_columns, categorical_columns = _extract_feature_frame(
        df, feature_columns
    )
    y_raw_series = df[target_column]

    valid_mask = ~_is_missing(y_raw_series)
    X_valid = X_raw.loc[valid_mask].reset_index(drop=True)
    y_valid = y_raw_series.loc[valid_mask].reset_index(drop=True)

    if len(y_valid) < 5:
        raise ValueError(
            "At least 5 valid rows with non-empty target values are required."
        )

    resolved_task, y_arr, class_labels = _resolve_task(
        y_valid, task_type, algorithm
    )
    if algorithm == "logistic_regression" and resolved_task != "classification":
        raise ValueError(
            "Logistic Regression supports classification only; this target resolves to regression."
        )
    if (
        algorithm == "linear_ridge_regression"
        and resolved_task != "regression"
    ):
        raise ValueError(
            "Linear / Ridge Regression supports regression only; this target resolves to classification."
        )

    stratify = None
    if resolved_task == "classification":
        _, class_counts = np.unique(y_arr, return_counts=True)
        if np.min(class_counts) < 2:
            raise ValueError(
                "Classification requires at least 2 target rows in every class."
            )
        test_rows = int(np.ceil(len(y_arr) * test_size))
        train_rows = len(y_arr) - test_rows
        if test_rows >= len(class_counts) and train_rows >= len(class_counts):
            stratify = y_arr.astype(int)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid,
            y_arr,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create the train/test split: {exc}") from exc

    if resolved_task == "classification" and len(np.unique(y_train)) != len(
        class_labels
    ):
        raise ValueError(
            "The training split must contain every class. Add rows or increase the test split."
        )

    if algorithm == "logistic_regression":
        c_val = float(_numeric_param(params, "C", 1.0))
        max_iter = int(_numeric_param(params, "max_iter", 2000, integer=True))
        if c_val <= 0 or max_iter <= 0:
            raise ValueError("C and max_iter must be greater than zero.")
        model = LogisticRegression(C=c_val, max_iter=max_iter, random_state=42)
    elif algorithm == "random_forest_supervised":
        n_est = int(
            _numeric_param(params, "n_estimators", 150, integer=True)
        )
        if n_est <= 0:
            raise ValueError("n_estimators must be greater than zero.")
        depth_raw = params.get("max_depth", None)
        if depth_raw is None or (
            isinstance(depth_raw, str)
            and depth_raw.strip().lower() in {"", "none"}
        ):
            max_depth = None
        else:
            max_depth = int(
                _numeric_param(
                    {"max_depth": depth_raw}, "max_depth", 1, integer=True
                )
            )
            if max_depth <= 0:
                raise ValueError("max_depth must be greater than zero.")
        model = (
            RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth, random_state=42
            )
            if resolved_task == "classification"
            else RandomForestRegressor(
                n_estimators=n_est, max_depth=max_depth, random_state=42
            )
        )
    elif algorithm == "svm_supervised":
        c_val = float(_numeric_param(params, "C", 1.0))
        if c_val <= 0:
            raise ValueError("C must be greater than zero.")
        kernel = str(params.get("kernel", "rbf"))
        model = (
            SVC(C=c_val, kernel=kernel, random_state=42)
            if resolved_task == "classification"
            else SVR(C=c_val, kernel=kernel)
        )
    elif algorithm == "knn_supervised":
        k_val = int(
            _numeric_param(
                params,
                "n_neighbors",
                min(5, max(1, len(X_train) - 1)),
                integer=True,
            )
        )
        if k_val <= 0:
            raise ValueError("n_neighbors must be greater than zero.")
        k_val = max(1, min(k_val, len(X_train)))
        weights = str(params.get("weights", "uniform"))
        model = (
            KNeighborsClassifier(n_neighbors=k_val, weights=weights)
            if resolved_task == "classification"
            else KNeighborsRegressor(n_neighbors=k_val, weights=weights)
        )
    elif algorithm == "linear_ridge_regression":
        alpha = float(_numeric_param(params, "alpha", 1.0))
        if alpha < 0:
            raise ValueError("alpha must be zero or greater.")
        use_ridge = bool(params.get("use_ridge", True))
        model = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    else:
        raise ValueError(f"Unsupported supervised algorithm: {algorithm}")

    preprocessor = _build_feature_preprocessor(
        numeric_columns, categorical_columns, scale_features
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("estimator", model)]
    )
    pipeline.fit(X_train, y_train)
    test_preds = pipeline.predict(X_test)
    all_preds = pipeline.predict(X_valid)

    X_processed = pipeline.named_steps["preprocessor"].transform(X_valid)
    X_clean_display = _display_feature_values(
        X_valid,
        pipeline.named_steps["preprocessor"],
        feature_columns,
        numeric_columns,
        categorical_columns,
        scale_numeric=scale_features,
    )
    extra_chart_series: dict[str, Any] = {}

    if resolved_task == "classification":
        y_test_int = y_test.astype(int)
        test_preds_int = np.rint(test_preds).astype(int)
        all_preds_int = np.rint(all_preds).astype(int)
        num_classes = len(np.unique(y_arr))
        avg = "binary" if num_classes == 2 else "macro"

        roc_auc, roc_extra = _compute_roc_details(
            pipeline, X_test, y_test_int, num_classes
        )
        extra_chart_series.update(roc_extra)

        metrics: dict[str, Any] = {
            "accuracy": round(float(accuracy_score(y_test_int, test_preds_int)), 4),
            "precision": round(
                float(
                    precision_score(
                        y_test_int, test_preds_int, average=avg, zero_division=0
                    )
                ),
                4,
            ),
            "recall": round(
                float(
                    recall_score(
                        y_test_int, test_preds_int, average=avg, zero_division=0
                    )
                ),
                4,
            ),
            "f1_score": round(
                float(
                    f1_score(
                        y_test_int, test_preds_int, average=avg, zero_division=0
                    )
                ),
                4,
            ),
            "roc_auc": roc_auc if roc_auc is not None else "N/A",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

        def fmt_cls(val: int) -> str:
            if 0 <= val < len(class_labels):
                return class_labels[val]
            return str(val)

        table_headers = [
            "Row #",
            *feature_columns,
            f"Actual ({target_column})",
            "Predicted",
        ]
        table_rows = [
            [
                i + 1,
                *_format_feature_row(X_clean_display[i]),
                fmt_cls(int(y_arr[i])),
                fmt_cls(int(all_preds_int[i])),
            ]
            for i in range(len(X_valid))
        ]
        chart_points = _project_2d(
            X_processed, [f"Pred {fmt_cls(int(p))}" for p in all_preds_int]
        )
    else:
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, test_preds)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, test_preds)), 4),
            "r2_score": (
                round(float(r2_score(y_test, test_preds)), 4)
                if len(y_test) > 1
                else "N/A"
            ),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        table_headers = [
            "Row #",
            *feature_columns,
            f"Actual ({target_column})",
            "Predicted",
            "Error",
        ]
        table_rows = [
            [
                i + 1,
                *_format_feature_row(X_clean_display[i]),
                round(float(y_arr[i]), 4),
                round(float(all_preds[i]), 4),
                round(float(y_arr[i] - all_preds[i]), 4),
            ]
            for i in range(len(X_valid))
        ]
        chart_points = _project_2d(
            X_processed,
            ["Rows"] * len(y_arr),
        )

    return {
        "algorithm": algorithm,
        "category": "supervised",
        "task_resolved": resolved_task,
        "summary": (
            f"{algorithm} completed ({resolved_task}) on {len(X_valid)} rows "
            f"({len(X_train)} train / {len(X_test)} test). Metrics use the "
            "held-out test split; the table shows predictions for all valid rows."
        ),
        "metrics": metrics,
        "table_headers": table_headers,
        "table_rows": table_rows,
        "chart_points": chart_points,
        "extra_chart_series": extra_chart_series,
    }
