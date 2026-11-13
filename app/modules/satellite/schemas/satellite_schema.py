"""
Skema Pydantic untuk request dan response sinkronisasi data satelit Sentinel-5P.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SatelliteSyncRequest(BaseModel):
    """Request untuk menarik data dari GEE dan menyimpannya ke tabel database."""
    region: Optional[str] = Field(None, description="Kode wilayah (opsional, jika kosong akan sync semua 5 wilayah)")
    start_date: str = Field(..., description="Tanggal awal (YYYY-MM-DD)")
    end_date: str = Field(..., description="Tanggal akhir (YYYY-MM-DD)")


class SatelliteSyncResponse(BaseModel):
    """Response hasil sinkronisasi data GEE ke database."""
    synced_regions: List[str]
    total_inserted: int
    message: str
