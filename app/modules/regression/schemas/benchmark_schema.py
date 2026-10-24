"""
Schema Pydantic untuk Evaluasi Validasi Silang Monte Carlo (Benchmark Regresi).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


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


class MCCVBenchmarkResult(BaseModel):
    """
    Hasil lengkap komputasi benchmark Monte Carlo Cross-Validation.
    """
    records: List[MCCVIterationResult]
    summary_rows: List[BenchmarkSummaryRow]
    seeds: List[int] = []
