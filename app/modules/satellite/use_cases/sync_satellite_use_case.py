"""
Use Case untuk sinkronisasi massal data satelit Sentinel-5P dari Google Earth Engine ke basis data lokal.
Bisa dijalankan manual oleh admin atau via background task.
"""

import logging
from typing import List, Set
from datetime import datetime, date
from injector import inject

from app.core.geometry import SUPPORTED_REGIONS
from app.core.dates import validate_date_range
from app.modules.satellite.core.extractor import Sentinel5PExtractor
from app.modules.satellite.repositories.satellite_repository import SatelliteRepository
from app.modules.satellite.schemas.satellite_schema import (
    SatelliteSyncRequest,
    SatelliteSyncResponse,
)

logger = logging.getLogger(__name__)


class SyncSatelliteUseCase:
    """
    Use case untuk sinkronisasi data GEE ke tabel `sentinel_measurements`.
    Mengekstrak data Sentinel-5P dari Google Earth Engine dan menyimpannya ke basis data lokal.
    """

    @inject
    def __init__(
        self,
        repository: SatelliteRepository,
        extractor: Sentinel5PExtractor,
    ):
        self.repository = repository
        self.extractor = extractor

    async def sync(self, payload: SatelliteSyncRequest) -> SatelliteSyncResponse:
        """
        Menarik data Sentinel-5P dari GEE dan menyimpannya ke database lokal.
        Jika seluruh tanggal yang diminta sudah berstatus OFFL lengkap, penarikan GEE dilewati.
        """
        validate_date_range(payload.start_date, payload.end_date)

        target_regions: List[str] = []
        if payload.region:
            if payload.region not in SUPPORTED_REGIONS:
                valid_keys = ", ".join(SUPPORTED_REGIONS.keys())
                raise ValueError(f"Wilayah '{payload.region}' tidak valid. Pilihan: {valid_keys}")
            target_regions = [payload.region]
        else:
            target_regions = list(SUPPORTED_REGIONS.keys())

        d_start = datetime.strptime(payload.start_date, "%Y-%m-%d").date()
        d_end = datetime.strptime(payload.end_date, "%Y-%m-%d").date()
        total_days_requested = (d_end - d_start).days + 1

        total_saved = 0
        synced: List[str] = []

        for reg in target_regions:
            # 1. Cek data yang sudah ada di basis data lokal
            existing = await self.repository.get_by_date_range(
                region=reg,
                start_date=payload.start_date,
                end_date=payload.end_date,
                valid_only=False,
            )

            # 2. Identifikasi tanggal yang sudah final OFFL (tidak akan berubah lagi di server GEE)
            final_offl_dates: Set[str] = {
                r.date for r in existing
                if r.product_type == "OFFL" and r.co is not None and r.no2 is not None
            }

            # Jika seluruh tanggal pada rentang sudah mutakhir OFFL, lewati penarikan GEE
            if len(final_offl_dates) >= total_days_requested:
                logger.info(
                    "Data satelit untuk %s (%s s/d %s) sudah final OFFL. Melewati penarikan GEE.",
                    reg, payload.start_date, payload.end_date
                )
                synced.append(reg)
                continue

            # 3. Tarik data dari GEE dengan fallback OFFL -> NRTI
            logger.info("Sinkronisasi data GEE untuk wilayah: %s (%s s/d %s)...", reg, payload.start_date, payload.end_date)
            try:
                extracted = self.extractor.extract(
                    region_key=reg,
                    start_date_str=payload.start_date,
                    end_date_str=payload.end_date,
                )
                if extracted:
                    count = await self.repository.bulk_upsert(extracted)
                    total_saved += count
                synced.append(reg)
            except Exception as err:
                logger.error("Gagal sinkronisasi wilayah %s: %s", reg, err)
                raise RuntimeError(f"Gagal sinkronisasi wilayah {reg}: {err}") from err

        return SatelliteSyncResponse(
            synced_regions=synced,
            total_inserted=total_saved,
            message=f"Berhasil menyinkronkan {len(synced)} wilayah. Total baris data tersimpan/diperbarui: {total_saved}.",
        )
