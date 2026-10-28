"""
Schema Pydantic untuk Evaluasi Validasi Silang Monte Carlo (Benchmark Regresi).
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class MCCVIterationResult(BaseModel):
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
    test_r2: float
    train_rmse: float
    train_mae: float
    train_r2: float
    f_statistic: Optional[float] = None
    f_table: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None


class BenchmarkSummaryRow(BaseModel):
    """
    Baris rangkuman statistik agregat rata-rata & standar deviasi per estimator.
    """
    estimator: str
    test_r2_mean: float
    test_r2_std: float
    test_r2_display: str
    test_rmse_mean: float
    test_rmse_std: float
    test_rmse_display: str
    test_mae_mean: float
    test_mae_std: float
    test_mae_display: str
    train_r2_mean: float
    train_r2_std: float
    train_r2_display: str
    train_rmse_mean: float
    train_rmse_std: float
    train_rmse_display: str
    train_mae_mean: float
    train_mae_std: float
    train_mae_display: str
    f_statistic_mean: Optional[float] = None
    f_statistic_display: Optional[str] = None
    f_table: Optional[float] = None
    is_significant: Optional[bool] = None
    significance_rate: Optional[float] = None


class MCCVBenchmarkResult(BaseModel):
    """
    Hasil lengkap komputasi benchmark Monte Carlo Cross-Validation.
    """
    records: List[MCCVIterationResult]
    summary_rows: List[BenchmarkSummaryRow]
    seeds: List[int] = []


class BenchmarkSessionResponse(BaseModel):
    id: int
    session_name: Optional[str] = None
    dataset_name: str
    test_size: float
    total_seeds: int
    total_samples: Optional[int] = None
    fit_intercept: bool
    summary_metrics: Optional[List[Dict]] = None
    created_at: Optional[Union[datetime, str]] = None

    model_config = ConfigDict(from_attributes=True)


class MCCVIterationResponse(BaseModel):
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
    test_r2: float
    train_rmse: Optional[float] = None
    train_mae: Optional[float] = None
    train_r2: Optional[float] = None
    f_statistic: Optional[float] = None
    f_table: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    created_at: Optional[Union[datetime, str]] = None

    model_config = ConfigDict(from_attributes=True)


class PaginatedIterationsResponse(BaseModel):
    items: List[MCCVIterationResponse]
    total: int
    page: int
    limit: int
    total_pages: int


class BenchmarkRunResponse(BaseModel):
    session: BenchmarkSessionResponse
    summary_rows: List[BenchmarkSummaryRow]
    top_iterations: List[MCCVIterationResponse]
    seeds: List[int] = []
