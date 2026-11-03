"""
Schema Pydantic untuk Evaluasi dan Pengujian Massal (Batch Evaluation / Benchmark Regresi).
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class BatchIterationResult(BaseModel):
    """
    Hasil evaluasi untuk satu estimator pada satu iterasi random seed.
    """
    estimator_type: str
    random_state: int
    test_size: float
    total_samples: int
    fit_intercept: bool
    intercept: Optional[float] = None
    coefficients: Dict[str, float]
    scale: Optional[float] = None
    iterations: int
    converged: bool
    test_rmse: float
    test_mae: float
    train_rmse: float
    train_mae: float


class BatchSummaryRow(BaseModel):
    """
    Baris rangkuman statistik agregat rata-rata & standar deviasi RMSE dan MAE per estimator.
    """
    estimator: str
    test_rmse_mean: float
    test_rmse_std: float
    test_rmse_display: str
    test_mae_mean: float
    test_mae_std: float
    test_mae_display: str
    train_rmse_mean: float
    train_rmse_std: float
    train_rmse_display: str
    train_mae_mean: float
    train_mae_std: float
    train_mae_display: str


class BatchBenchmarkResult(BaseModel):
    """
    Hasil lengkap komputasi pengujian massal (batch evaluation).
    """
    records: List[BatchIterationResult]
    summary_rows: List[BatchSummaryRow]
    seeds: List[int] = []


class BatchIterationResponse(BaseModel):
    id: int
    session_id: int
    estimator_type: str
    random_state: Optional[int] = None
    intercept: Optional[float] = None
    coefficients: Dict[str, float]
    scale: Optional[float] = None
    iterations: Optional[int] = None
    converged: Optional[bool] = None
    test_rmse: float
    test_mae: float
    train_rmse: Optional[float] = None
    train_mae: Optional[float] = None
    created_at: Optional[Union[datetime, str]] = None

    model_config = ConfigDict(from_attributes=True)


class BatchSessionResponse(BaseModel):
    id: int
    session_name: Optional[str] = None
    dataset_name: str
    test_size: float
    total_seeds: int
    total_samples: Optional[int] = None
    fit_intercept: bool
    summary_metrics: Optional[List[Dict]] = None
    seeds: Optional[List[int]] = None
    best_iteration: Optional[BatchIterationResponse] = None
    created_at: Optional[Union[datetime, str]] = None

    model_config = ConfigDict(from_attributes=True)


class PaginatedIterationsResponse(BaseModel):
    items: List[BatchIterationResponse]
    total: int
    page: int
    limit: int
    total_pages: int


class BatchRunResponse(BaseModel):
    session: BatchSessionResponse
    summary_rows: List[BatchSummaryRow]
    top_iterations: List[BatchIterationResponse]
    seeds: List[int] = []
    has_interpolated: bool = False
    interpolated_count: int = 0
