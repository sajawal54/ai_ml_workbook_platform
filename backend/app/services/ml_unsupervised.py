from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from app.services.ml_common import (
    _build_feature_preprocessor,
    _display_feature_values,
    _extract_feature_frame,
    _format_feature_row,
    _numeric_param,
    _project_2d,
)


def run_unsupervised(
    algorithm: str,
    columns: list[str],
    rows: list[list[Any]],
    feature_columns: list[str],
    scale_features: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    df = pd.DataFrame(rows, columns=columns)
    X_raw, numeric_columns, categorical_columns = _extract_feature_frame(
        df, feature_columns
    )

    if len(X_raw) < 4:
        raise ValueError("At least 4 rows are required for unsupervised analysis.")

    preprocessor = _build_feature_preprocessor(
        numeric_columns, categorical_columns, scale_features
    )
    X_proc = preprocessor.fit_transform(X_raw)
    X_display = _display_feature_values(
        X_raw,
        preprocessor,
        feature_columns,
        numeric_columns,
        categorical_columns,
        scale_numeric=False,
    )

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
            [i + 1, *_format_feature_row(X_display[i]), int(clusters[i])]
            for i in range(len(X_display))
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
            [i + 1, *_format_feature_row(X_display[i]), int(clusters[i])]
            for i in range(len(X_display))
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
            [i + 1, *_format_feature_row(X_display[i]), int(clusters[i])]
            for i in range(len(X_display))
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
                *_format_feature_row(X_display[i]),
                *[round(float(c), 4) for c in projected[i]],
            ]
            for i in range(len(X_display))
        ]
        labels_for_chart = [
            f"PC1={round(float(projected[i, 0]), 2)}"
            for i in range(len(X_display))
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
                *_format_feature_row(X_display[i]),
                "Anomaly" if preds[i] == -1 else "Normal",
                round(float(scores[i]), 4),
            ]
            for i in range(len(X_display))
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
