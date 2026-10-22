"""
Schema Pydantic untuk Pelatihan (Training) dan Pengujian (Testing) Model Regresi.
"""

from typing import Optional, List
from pydantic import BaseModel


class RegressionPoint(BaseModel):
    index: int
    actual: float
    predicted: float


class TrainModelResult(BaseModel):
    estimator_type: str
    equation: str
    intercept: float = 0.0
    coef_co: float
    coef_no2: float
    scale: Optional[float] = None
    rmse: float
    mae: float
    r2: float
    chart_data: List[RegressionPoint]
    total_samples: int


class TestModelResult(BaseModel):
    model_id: Optional[int] = None
    model_name: str
    estimator_type: str
    equation: str
    rmse: float
    mae: float
    r2: float
    chart_data: List[RegressionPoint]
    total_samples: int
