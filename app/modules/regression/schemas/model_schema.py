"""
Schema Pydantic untuk Operasi Model Tersimpan (TrainedModel).
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class TrainedModelBase(BaseModel):
    name: str = Field(..., description="Nama unik untuk model")
    estimator_type: str = Field(..., description="Tipe estimator: MM, M, S, atau OLS")
    is_default: bool = Field(False, description="Apakah model ini model bawaan sistem")
    intercept: float = Field(..., description="Nilai konstanta beta_0")
    coef_co: float = Field(..., description="Koefisien CO Sentinel-5P (beta_1)")
    coef_no2: float = Field(..., description="Koefisien NO2 Sentinel-5P (beta_2)")
    scale: Optional[float] = Field(None, description="Estimasi skala residual sigma")
    train_rmse: float = Field(..., description="RMSE pada data latih")
    train_mae: float = Field(..., description="MAE pada data latih")
    train_r2: float = Field(..., description="R^2 (%) pada data latih")


class TrainedModelCreate(TrainedModelBase):
    pass


class TrainedModelResponse(TrainedModelBase):
    id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
