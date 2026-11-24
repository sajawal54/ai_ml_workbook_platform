from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR


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

    X_raw = _extract_numeric_features(df, feature_columns)
    y_raw_series = df[target_column]

    valid_mask = ~_is_missing(y_raw_series)
    X_valid = X_raw[valid_mask.to_numpy()]
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

    prep_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_features:
        prep_steps.append(("scaler", StandardScaler()))

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

    pipeline = Pipeline(prep_steps + [("estimator", model)])
    pipeline.fit(X_train, y_train)
    test_preds = pipeline.predict(X_test)
    all_preds = pipeline.predict(X_valid)

    X_clean_display = Pipeline(prep_steps).fit_transform(X_valid)
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
                *[round(float(v), 4) for v in X_clean_display[i]],
                fmt_cls(int(y_arr[i])),
                fmt_cls(int(all_preds_int[i])),
            ]
            for i in range(len(X_valid))
        ]
        chart_points = _project_2d(
            X_clean_display, [f"Pred {fmt_cls(int(p))}" for p in all_preds_int]
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
                *[round(float(v), 4) for v in X_clean_display[i]],
                round(float(y_arr[i]), 4),
                round(float(all_preds[i]), 4),
                round(float(y_arr[i] - all_preds[i]), 4),
            ]
            for i in range(len(X_valid))
        ]
        chart_points = _project_2d(
            X_clean_display,
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


def run_unsupervised(
    algorithm: str,
    columns: list[str],
    rows: list[list[Any]],
    feature_columns: list[str],
    scale_features: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    df = pd.DataFrame(rows, columns=columns)
    X_raw = _extract_numeric_features(df, feature_columns)

    if len(X_raw) < 4:
        raise ValueError("At least 4 rows are required for unsupervised analysis.")

    prep_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_features:
        prep_steps.append(("scaler", StandardScaler()))

    X_proc = Pipeline(prep_steps).fit_transform(X_raw)
    X_imp = SimpleImputer(strategy="median").fit_transform(X_raw)

    metrics: dict[str, Any] = {"rows_analyzed": int(len(X_proc))}
    labels_for_chart: list[str] = []
    table_headers: list[str] = ["Row #", *feature_columns]
    table_rows: list[list[Any]] = []
    task_resolved = algorithm

    if algorithm == "kmeans":
        requested_clusters = int(
            _numeric_param(params, "n_clusters", 3, integer=True)
        )
        if requested_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        distinct_rows = np.unique(X_proc, axis=0).shape[0]
        if distinct_rows < 2:
            raise ValueError("K-Means requires at least 2 distinct data rows.")
        n_clusters = min(requested_clusters, len(X_proc) - 1, distinct_rows)
        max_iter = int(_numeric_param(params, "max_iter", 300, integer=True))
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")
        km = KMeans(
            n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init=10
        )
        clusters = km.fit_predict(X_proc)
        metrics["clusters"] = int(len(np.unique(clusters)))
        metrics["inertia"] = round(float(km.inertia_), 4)
        if 1 < len(np.unique(clusters)) < len(X_proc):
            metrics["silhouette_score"] = round(
                float(silhouette_score(X_proc, clusters)), 4
            )
        labels_for_chart = [f"Cluster {c}" for c in clusters]
        table_headers.append("Cluster")
        table_rows = [
            [i + 1, *[round(float(v), 4) for v in X_imp[i]], int(clusters[i])]
            for i in range(len(X_imp))
        ]
    elif algorithm == "dbscan":
        eps = float(_numeric_param(params, "eps", 0.9))
        min_samples = int(
            _numeric_param(params, "min_samples", 3, integer=True)
        )
        if eps <= 0:
            raise ValueError("eps must be greater than zero.")
        if min_samples <= 0:
            raise ValueError("min_samples must be greater than zero.")
        db = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = db.fit_predict(X_proc)
        unique_clusters = [int(c) for c in np.unique(clusters) if c != -1]
        noise_count = int(np.sum(clusters == -1))
        metrics["clusters_found"] = int(len(unique_clusters))
        metrics["noise_points"] = noise_count
        if 1 < len(np.unique(clusters)) < len(X_proc):
            metrics["silhouette_score"] = round(
                float(silhouette_score(X_proc, clusters)), 4
            )
        labels_for_chart = [
            "Noise (-1)" if c == -1 else f"Cluster {c}" for c in clusters
        ]
        table_headers.append("Cluster (-1 = Noise)")
        table_rows = [
            [i + 1, *[round(float(v), 4) for v in X_imp[i]], int(clusters[i])]
            for i in range(len(X_imp))
        ]
    elif algorithm == "hierarchical":
        requested_clusters = int(
            _numeric_param(params, "n_clusters", 3, integer=True)
        )
        if requested_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        n_clusters = min(requested_clusters, len(X_proc) - 1)
        linkage = str(params.get("linkage", "ward"))
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clusters = hc.fit_predict(X_proc)
        metrics["clusters"] = int(n_clusters)
        metrics["linkage"] = linkage
        if 1 < len(np.unique(clusters)) < len(X_proc):
            metrics["silhouette_score"] = round(
                float(silhouette_score(X_proc, clusters)), 4
            )
        labels_for_chart = [f"Cluster {c}" for c in clusters]
        table_headers.append("Cluster")
        table_rows = [
            [i + 1, *[round(float(v), 4) for v in X_imp[i]], int(clusters[i])]
            for i in range(len(X_imp))
        ]
    elif algorithm == "pca":
        max_comp = min(len(X_proc), X_proc.shape[1])
        n_comp = int(
            _numeric_param(params, "n_components", 2, integer=True)
        )
        if n_comp < 1:
            raise ValueError("n_components must be at least 1.")
        n_comp = min(n_comp, max_comp)
        if np.allclose(X_proc, X_proc[0]):
            projected = np.zeros((len(X_proc), n_comp), dtype=float)
            explained = np.zeros(n_comp, dtype=float)
        else:
            pca = PCA(n_components=n_comp, random_state=42)
            projected = pca.fit_transform(X_proc)
            explained = pca.explained_variance_ratio_
        metrics["components"] = int(n_comp)
        metrics["total_explained_variance"] = round(
            float(np.sum(explained)), 4
        )
        for idx, ratio in enumerate(explained):
            metrics[f"pc{idx + 1}_variance"] = round(float(ratio), 4)
        pc_headers = [f"PC{i + 1}" for i in range(n_comp)]
        table_headers = ["Row #", *feature_columns, *pc_headers]
        table_rows = [
            [
                i + 1,
                *[round(float(v), 4) for v in X_imp[i]],
                *[round(float(c), 4) for c in projected[i]],
            ]
            for i in range(len(X_imp))
        ]
        labels_for_chart = [
            f"PC1={round(float(projected[i, 0]), 2)}" for i in range(len(X_imp))
        ]
    elif algorithm == "isolation_forest":
        n_est = int(
            _numeric_param(params, "n_estimators", 100, integer=True)
        )
        if n_est <= 0:
            raise ValueError("n_estimators must be greater than zero.")
        contam_raw = params.get("contamination", "auto")
        if contam_raw is None or (
            isinstance(contam_raw, str) and not contam_raw.strip()
        ):
            contam_raw = "auto"
        if isinstance(contam_raw, str) and contam_raw.strip().lower() == "auto":
            contamination: Any = "auto"
        else:
            contamination = float(
                _numeric_param(
                    {"contamination": contam_raw}, "contamination", 0.1
                )
            )
            if not 0 < contamination <= 0.5:
                raise ValueError("contamination must be greater than 0 and at most 0.5.")
        iso = IsolationForest(
            n_estimators=n_est, contamination=contamination, random_state=42
        )
        preds = iso.fit_predict(X_proc)
        scores = iso.decision_function(X_proc)
        anomalies = int(np.sum(preds == -1))
        metrics["anomalies_detected"] = anomalies
        metrics["normal_points"] = int(np.sum(preds == 1))
        table_headers.extend(["Status", "Anomaly Score"])
        table_rows = [
            [
                i + 1,
                *[round(float(v), 4) for v in X_imp[i]],
                "Anomaly" if preds[i] == -1 else "Normal",
                round(float(scores[i]), 4),
            ]
            for i in range(len(X_imp))
        ]
        labels_for_chart = [
            "Anomaly" if p == -1 else "Normal" for p in preds
        ]
    else:
        raise ValueError(f"Unsupported unsupervised algorithm: {algorithm}")

    return {
        "algorithm": algorithm,
        "category": "unsupervised",
        "task_resolved": task_resolved,
        "summary": f"{algorithm} completed on {len(X_proc)} rows across {len(feature_columns)} features.",
        "metrics": metrics,
        "table_headers": table_headers,
        "table_rows": table_rows,
        "chart_points": _project_2d(X_proc, labels_for_chart),
        "extra_chart_series": {},
    }