"""
Schema Pydantic untuk Modul Estimasi O3 dan Klasifikasi ISPU.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class SingleEstimateRequest(BaseModel):
    co: float = Field(..., description="Konsentrasi CO dari Sentinel-5P (mol/m^2)")
    no2: float = Field(..., description="Konsentrasi NO2 dari Sentinel-5P (mol/m^2)")
    model_id: Optional[int] = Field(None, description="ID model tersimpan (opsional, jika None pakai default)")


class SingleEstimateResponse(BaseModel):
    model_id: Optional[int]
    model_name: str
    estimator_type: str
    equation: Optional[str] = None
    estimated_o3: float
    ispu_value: float
    ispu_category: str


class BatchEstimatePoint(BaseModel):
    index: int
    label: str
    co: float
    no2: float
    actual_o3: Optional[float] = None
    estimated_o3: float
    ispu_value: float
    ispu_category: str


class BatchEstimateSummary(BaseModel):
    total_samples: int
    avg_estimated_o3: float
    min_estimated_o3: float
    max_estimated_o3: float
    dominant_ispu_category: str
    ispu_distribution: Dict[str, int]


class BatchEstimateResponse(BaseModel):
    model_id: int
    model_name: str
    estimator_type: str
    equation: str
    chart_data: List[BatchEstimatePoint]
    summary: BatchEstimateSummary
