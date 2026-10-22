"""
Schema Pydantic untuk Deteksi dan Diagnostik Pencilan (Outlier).
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class OutlierCategorySummary(BaseModel):
    threshold: Optional[float] = None
    outliers_count: int = 0
    outlier_indices: List[int] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["OutlierCategorySummary"]:
        if not data:
            return None
        return cls(
            threshold=data.get("threshold"),
            outliers_count=data.get("outliers_count", 0),
            outlier_indices=data.get("outlier_indices", []),
        )


class OutlierDetectionResponse(BaseModel):
    total_outliers: int
    leverage: Optional[OutlierCategorySummary] = None
    dffits: Optional[OutlierCategorySummary] = None
    both: Optional[OutlierCategorySummary] = None

    @classmethod
    def from_detector_dict(cls, res_dict: Dict[str, Any]) -> "OutlierDetectionResponse":
        return cls(
            total_outliers=res_dict.get("total_outliers", 0),
            leverage=OutlierCategorySummary.from_dict(res_dict.get("leverage")),
            dffits=OutlierCategorySummary.from_dict(res_dict.get("dffits")),
            both=OutlierCategorySummary.from_dict(res_dict.get("both")),
        )
