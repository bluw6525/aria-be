"""
Schema Pydantic untuk Modul Estimasi Nilai ISPU dan Klasifikasi Kategori.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ForecastItem(BaseModel):
    day_offset: int
    date: str
    label: str
    co: float
    no2: float
    estimated_value: float
    ispu_category: str
    delta: float


class BatchEstimatePoint(BaseModel):
    index: int
    label: str
    co: float
    no2: float
    actual_value: Optional[float] = None
    estimated_value: Optional[float] = None
    ispu_category: Optional[str] = None
    is_forecast: bool = False
    forecast_value: Optional[float] = None


class BatchEstimateSummary(BaseModel):
    total_samples: int
    avg_estimated: float
    min_estimated: float
    max_estimated: float


class BatchEstimateResponse(BaseModel):
    model_id: int
    model_name: str
    estimator_type: str
    chart_data: List[BatchEstimatePoint]
    summary: BatchEstimateSummary
    forecast_days: Optional[List[ForecastItem]] = None


class SatelliteEstimateRequest(BaseModel):
    region: str = Field(..., description="Kode wilayah (misal: 'jakarta_barat')")
    start_date: str = Field(..., description="Tanggal awal (YYYY-MM-DD)")
    end_date: str = Field(..., description="Tanggal akhir (YYYY-MM-DD)")


class MapOverviewItem(BaseModel):
    region: str
    region_name: str
    estimated_ispu: Optional[float] = None
    ispu_category: Optional[str] = None
    observation_date: Optional[str] = None
    is_fallback: bool = False


class MapOverviewResponse(BaseModel):
    start_date: str
    end_date: str
    regions: List[MapOverviewItem]
