"""
Schema Pydantic untuk Operasi Model Tersimpan (TrainedModel) dan Promosi Model.
"""

from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainedModelBase(BaseModel):
    name: str = Field(..., description="Nama unik untuk model")
    estimator_type: str = Field(..., description="Tipe estimator: MM, M, S, atau OLS")
    is_default: bool = Field(False, description="Apakah model ini model bawaan sistem")
    intercept: float = Field(0.0, description="Nilai konstanta beta_0")
    coefficients: Optional[Dict[str, float]] = Field(None, description="Koefisien variabel (JSON)")
    coef_co: Optional[float] = Field(None, description="Koefisien CO Sentinel-5P (beta_1)")
    coef_no2: Optional[float] = Field(None, description="Koefisien NO2 Sentinel-5P (beta_2)")
    scale: Optional[float] = Field(None, description="Estimasi skala residual sigma")
    train_rmse: float = Field(..., description="RMSE pada data latih")
    train_mae: float = Field(..., description="MAE pada data latih")
    train_r2: float = Field(..., description="R^2 (%) pada data latih")

    @model_validator(mode="after")
    def sync_coefficients(self) -> "TrainedModelBase":
        """
        Sinkronisasi dua arah antara coefficients (JSON) dan coef_co / coef_no2
        untuk menjamin 100% kompatibilitas dengan frontend dan sistem lama.
        """
        if self.coefficients:
            if self.coef_co is None:
                self.coef_co = float(self.coefficients.get("CO", 0.0))
            if self.coef_no2 is None:
                self.coef_no2 = float(self.coefficients.get("NO2", 0.0))
        elif self.coef_co is not None or self.coef_no2 is not None:
            self.coefficients = {
                "CO": self.coef_co if self.coef_co is not None else 0.0,
                "NO2": self.coef_no2 if self.coef_no2 is not None else 0.0,
            }
        else:
            self.coefficients = {}
            self.coef_co = 0.0
            self.coef_no2 = 0.0
        return self


class TrainedModelCreate(TrainedModelBase):
    pass


class TrainedModelResponse(TrainedModelBase):
    id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PromoteModelRequest(BaseModel):
    """
    Payload untuk mempromosikan model dari iterasi MCCV ke model definitif (TrainedModel).
    """
    iteration_id: int = Field(..., description="ID dari iterasi model (mccv_iterations) yang ingin dipromosikan")
    model_name: str = Field(..., min_length=2, max_length=100, description="Nama unik model tersimpan")
    is_default: bool = Field(False, description="Apakah model ini menjadi model default untuk inferensi")
