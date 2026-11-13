"""
CLI Command untuk mengunduh dan menyinkronkan (seeding) data satelit Sentinel-5P
langsung dari Google Earth Engine (GEE) ke basis data lokal (tabel `sentinel_measurements`).

Secara default menarik data untuk seluruh 5 wilayah DKI Jakarta
dengan rentang waktu start_date dan end_date yang otomatis di-chunk (maks 30 hari/chunk).
"""

import argparse
import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple

# Pastikan root project terdaftar di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import AsyncSessionLocal
from app.core.gee import GEEClient
from app.core.geometry import SUPPORTED_REGIONS
from app.modules.satellite.core.extractor import Sentinel5PExtractor
from app.modules.satellite.repositories.satellite_repository import SatelliteRepository
from app.modules.satellite.schemas.satellite_schema import SatelliteSyncRequest
from app.modules.satellite.use_cases.sync_satellite_use_case import SyncSatelliteUseCase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_date_chunks(
    start_date_str: str,
    end_date_str: str,
    max_chunk_days: int = 30
) -> List[Tuple[str, str]]:
    """
    Memecah rentang tanggal yang panjang menjadi potongan-potongan kecil (chunks)
    maksimal 30 hari agar aman dari batas maksimal validasi GEE dan tidak membebani memori.
    """
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError as err:
        raise ValueError("Format tanggal harus YYYY-MM-DD.") from err

    if start_dt > end_dt:
        raise ValueError("start_date tidak boleh lebih besar dari end_date.")

    chunks = []
    cur_start = start_dt
    while cur_start <= end_dt:
        cur_end = min(cur_start + timedelta(days=max_chunk_days - 1), end_dt)
        chunks.append((cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
        cur_start = cur_end + timedelta(days=1)

    return chunks


async def seed_satellite_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    region: Optional[str] = None
) -> int:
    """
    Menarik data Sentinel-5P (CO & NO2) langsung dari Google Earth Engine
    untuk seluruh DKI Jakarta (5 wilayah) atau wilayah tertentu.

    Parameters:
        start_date: Tanggal awal format YYYY-MM-DD (default: H-8 dari hari ini)
        end_date: Tanggal akhir format YYYY-MM-DD (default: hari ini)
        region: Wilayah spesifik opsional (default: None untuk seluruh 5 wilayah Jakarta)

    Returns:
        Total baris data yang berhasil disimpan/diperbarui.
    """
    today = datetime.now().date()
    if not end_date:
        end_date = today.strftime("%Y-%m-%d")
    if not start_date:
        start_date = (today - timedelta(days=8)).strftime("%Y-%m-%d")

    chunks = generate_date_chunks(start_date, end_date, max_chunk_days=30)
    target_label = (
        SUPPORTED_REGIONS.get(region, region)
        if region
        else "Seluruh DKI Jakarta (5 Wilayah)"
    )

    print("\n" + "=" * 65)
    print("🛰️   SEEDING DATA SATELIT SENTINEL-5P DARI GOOGLE EARTH ENGINE")
    print("=" * 65)
    print(f"Target Wilayah : {target_label}")
    print(f"Rentang Tanggal: {start_date} s/d {end_date}")
    print(f"Jumlah Chunk   : {len(chunks)} bagian (maks 30 hari/chunk)")
    print("=" * 65 + "\n")

    gee_client = GEEClient()
    extractor = Sentinel5PExtractor(gee_client=gee_client)

    total_inserted = 0

    for idx, (c_start, c_end) in enumerate(chunks, 1):
        print(f"[{idx}/{len(chunks)}] Menarik data GEE: {c_start} s/d {c_end}...")
        async with AsyncSessionLocal() as session:
            repository = SatelliteRepository(session=session)
            use_case = SyncSatelliteUseCase(
                repository=repository,
                extractor=extractor,
            )

            req = SatelliteSyncRequest(
                region=region,
                start_date=c_start,
                end_date=c_end,
            )
            resp = await use_case.sync(req)
            total_inserted += resp.total_inserted
            print(f"       ✅ Selesai: {resp.total_inserted} data tersimpan untuk wilayah: {', '.join(resp.synced_regions)}")

    print("\n" + "=" * 65)
    print(f"🎉 SUKSES! Total {total_inserted} baris data satelit berhasil disimpan.")
    print("=" * 65 + "\n")
    return total_inserted


# Alias agar tetap backward-compatible jika dipanggil sebagai seed_from_gee
seed_from_gee = seed_satellite_data


def main():
    today = datetime.now().date()
    default_end = today.strftime("%Y-%m-%d")
    default_start = (today - timedelta(days=8)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Seeding data satelit Sentinel-5P dari Google Earth Engine (GEE) ke basis data lokal."
    )
    parser.add_argument(
        "--start-date",
        "-s",
        type=str,
        default=default_start,
        help=f"Tanggal awal (YYYY-MM-DD). Default: H-8 ({default_start})",
    )
    parser.add_argument(
        "--end-date",
        "-e",
        type=str,
        default=default_end,
        help=f"Tanggal akhir (YYYY-MM-DD). Default: hari ini ({default_end})",
    )
    parser.add_argument(
        "--region",
        "-r",
        type=str,
        default=None,
        help="Wilayah spesifik (opsional, contoh: jakarta_utara). Default: seluruh Jakarta (5 wilayah)",
    )

    args = parser.parse_args()

    asyncio.run(seed_satellite_data(
        start_date=args.start_date,
        end_date=args.end_date,
        region=args.region,
    ))


if __name__ == "__main__":
    main()
