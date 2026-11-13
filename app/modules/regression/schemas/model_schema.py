"""
Schema Pydantic untuk Operasi Model Publik (PublicModel) dan Promosi Model.
"""

from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PublicModelBase(BaseModel):
    name: str = Field(..., description="Nama unik untuk model")
    estimator_type: str = Field(..., description="Tipe estimator: MM, M, S, atau OLS")
    intercept: float = Field(0.0, description="Nilai konstanta beta_0")
    coefficients: Optional[Dict[str, float]] = Field(None, description="Koefisien variabel (JSON)")
    coef_co: Optional[float] = Field(None, description="Koefisien CO Sentinel-5P (beta_1)")
    coef_no2: Optional[float] = Field(None, description="Koefisien NO2 Sentinel-5P (beta_2)")
    scale: Optional[float] = Field(None, description="Estimasi skala residual sigma")
    train_rmse: float = Field(..., description="RMSE pada data latih")
    train_mae: float = Field(..., description="MAE pada data latih")
    is_active: bool = Field(False, description="Status apakah model aktif untuk estimasi ISPU default")

    @model_validator(mode="after")
    def sync_coefficients(self) -> "PublicModelBase":
        """
        Sinkronisasi dua arah antara coefficients (JSON) dan coef_co / coef_no2.
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


class PublicModelCreate(PublicModelBase):
    pass


class PublicModelResponse(PublicModelBase):
    id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PromoteModelRequest(BaseModel):
    """
    Payload untuk mempromosikan model dari iterasi batch ke model publik definitif (PublicModel).
    """
    iteration_id: int = Field(..., description="ID dari iterasi model (batch_iterations) yang ingin dipromosikan")
    model_name: str = Field(..., min_length=2, max_length=100, description="Nama unik model tersimpan")
