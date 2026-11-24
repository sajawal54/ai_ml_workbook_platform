from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetPayload(BaseModel):
    columns: list[str] = Field(..., min_length=1)
    rows: list[list[Any]] = Field(..., min_length=4)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns: list[str]) -> list[str]:
        normalized = [column.strip() for column in columns]
        if any(not column for column in normalized):
            raise ValueError("Column names cannot be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Column names must be unique.")
        return normalized

    @model_validator(mode="after")
    def validate_rows(self) -> "DatasetPayload":
        expected_len = len(self.columns)
        for idx, row in enumerate(self.rows):
            if len(row) != expected_len:
                raise ValueError(
                    f"Row {idx + 1} has {len(row)} values, expected {expected_len}."
                )
        return self


class SupervisedRunRequest(BaseModel):
    algorithm: Literal[
        "logistic_regression",
        "random_forest_supervised",
        "svm_supervised",
        "knn_supervised",
        "linear_ridge_regression",
    ]
    dataset: DatasetPayload
    feature_columns: list[str] = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    task_type: Literal["auto", "classification", "regression"] = "auto"
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)
    scale_features: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_columns(self) -> "SupervisedRunRequest":
        if self.target_column not in self.dataset.columns:
            raise ValueError(
                f"Target column '{self.target_column}' was not found in the dataset."
            )
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Feature columns must be unique.")
        missing = [c for c in self.feature_columns if c not in self.dataset.columns]
        if missing:
            raise ValueError(f"Feature column(s) not found: {', '.join(missing)}.")
        if self.target_column in self.feature_columns:
            raise ValueError("The target column cannot also be selected as a feature.")
        return self


class UnsupervisedRunRequest(BaseModel):
    algorithm: Literal[
        "kmeans",
        "dbscan",
        "hierarchical",
        "pca",
        "isolation_forest",
    ]
    dataset: DatasetPayload
    feature_columns: list[str] = Field(..., min_length=1)
    scale_features: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_columns(self) -> "UnsupervisedRunRequest":
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Feature columns must be unique.")
        missing = [c for c in self.feature_columns if c not in self.dataset.columns]
        if missing:
            raise ValueError(f"Feature column(s) not found: {', '.join(missing)}.")
        return self


class ChartPoint(BaseModel):
    x: float
    y: float
    label: str
    row_index: int


class AlgorithmRunResponse(BaseModel):
    algorithm: str
    category: Literal["supervised", "unsupervised"]
    task_resolved: str
    summary: str
    metrics: dict[str, Any]
    table_headers: list[str]
    table_rows: list[list[Any]]
    chart_points: list[ChartPoint]
    extra_chart_series: dict[str, Any] = Field(default_factory=dict)