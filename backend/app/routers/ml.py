import logging

from fastapi import APIRouter, HTTPException
from app.schemas.ml import (
    AlgorithmRunResponse,
    SupervisedRunRequest,
    UnsupervisedRunRequest,
)
from app.services.ml_engine import run_supervised, run_unsupervised

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])
logger = logging.getLogger(__name__)


@router.post("/supervised", response_model=AlgorithmRunResponse)
def execute_supervised(req: SupervisedRunRequest) -> AlgorithmRunResponse:
    try:
        result = run_supervised(
            algorithm=req.algorithm,
            columns=req.dataset.columns,
            rows=req.dataset.rows,
            feature_columns=req.feature_columns,
            target_column=req.target_column,
            task_type=req.task_type,
            test_size=req.test_size,
            scale_features=req.scale_features,
            params=req.params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected supervised model execution failure.")
        raise HTTPException(
            status_code=500,
            detail="Model execution failed unexpectedly. Please retry or check server logs.",
        ) from None
    return AlgorithmRunResponse(**result)


@router.post("/unsupervised", response_model=AlgorithmRunResponse)
def execute_unsupervised(req: UnsupervisedRunRequest) -> AlgorithmRunResponse:
    try:
        result = run_unsupervised(
            algorithm=req.algorithm,
            columns=req.dataset.columns,
            rows=req.dataset.rows,
            feature_columns=req.feature_columns,
            scale_features=req.scale_features,
            params=req.params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected unsupervised model execution failure.")
        raise HTTPException(
            status_code=500,
            detail="Model execution failed unexpectedly. Please retry or check server logs.",
        ) from None
    return AlgorithmRunResponse(**result)