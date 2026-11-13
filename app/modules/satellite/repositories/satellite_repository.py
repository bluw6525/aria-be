"""
Repository untuk data satelit Sentinel-5P (SentinelMeasurement).
Mengelola operasi pembacaan, filter rentang tanggal, dan penyimpanan massal (bulk upsert).
"""

from typing import List, Dict, Any, Set, Optional
from datetime import datetime
from injector import inject
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from app.modules.satellite.core.models import SentinelMeasurement


class SatelliteRepository:
    """
    Data Access Object (DAO) untuk tabel `sentinel_measurements`.
    """

    @inject
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_date_range(
        self,
        region: str,
        start_date: str,
        end_date: str,
        valid_only: bool = False,
    ) -> List[SentinelMeasurement]:
        """
        Mengambil daftar record satelit berdasarkan wilayah dan rentang tanggal (inklusif).
        Jika valid_only=True, hanya mengembalikan record yang kedua nilai CO dan NO2 tidak null.
        Diurutkan berdasarkan tanggal secara ascending.
        """
        conditions = [
            SentinelMeasurement.region == region,
            SentinelMeasurement.date >= start_date,
            SentinelMeasurement.date <= end_date,
        ]
        if valid_only:
            conditions.extend([
                SentinelMeasurement.co.is_not(None),
                SentinelMeasurement.no2.is_not(None),
            ])

        stmt = (
            select(SentinelMeasurement)
            .where(and_(*conditions))
            .order_by(SentinelMeasurement.date.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent_measurements(
        self,
        region: str,
        up_to_date: str,
        limit: int = 3
    ) -> List[SentinelMeasurement]:
        """
        Mengambil hingga limit pengukuran satelit valid terakhir untuk suatu wilayah
        sampai tanggal tertentu (inklusif), diurutkan secara kronologis ascending.
        """
        stmt = (
            select(SentinelMeasurement)
            .where(
                and_(
                    SentinelMeasurement.region == region,
                    SentinelMeasurement.date <= up_to_date,
                    SentinelMeasurement.co.is_not(None),
                    SentinelMeasurement.no2.is_not(None),
                )
            )
            .order_by(SentinelMeasurement.date.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        rows.reverse()
        return rows

    async def bulk_upsert(self, records: List[Dict[str, Any]]) -> int:
        """
        Menyimpan atau memperbarui daftar pengukuran satelit secara efisien.
        Mengambil data yang sudah ada dalam 1 kali query (batch), lalu memperbarui atau menyisipkan.
        Mendukung nilai co & no2 null (tutupan awan), serta menjaga data lengkap yang valid
        agar tidak tertimpa null/parsial.
        """
        if not records:
            return 0

        # Ambil daftar unik region dan tanggal untuk batch query
        regions: Set[str] = {r["region"] for r in records}
        dates: Set[str] = {r["date"] for r in records}

        stmt = select(SentinelMeasurement).where(
            and_(
                SentinelMeasurement.region.in_(regions),
                SentinelMeasurement.date.in_(dates),
            )
        )
        existing_rows = (await self.session.execute(stmt)).scalars().all()
        existing_map = {(row.region, row.date): row for row in existing_rows}

        now_utc = datetime.utcnow()
        inserted_count = 0

        for rec in records:
            key = (rec["region"], rec["date"])
            co_val = float(rec["co"]) if rec.get("co") is not None else None
            no2_val = float(rec["no2"]) if rec.get("no2") is not None else None
            has_data = (co_val is not None) or (no2_val is not None)
            prod_type = rec.get("product_type") if has_data else None

            if key in existing_map:
                row = existing_map[key]
                # Aturan perlindungan data dan upgrade produk:
                # 1. Jika data lokal sudah OFFL lengkap, jangan pernah di-downgrade ke NRTI atau null
                if row.product_type == "OFFL" and row.co is not None and row.no2 is not None:
                    row.updated_at = now_utc
                # 2. Jika data lokal sudah lengkap dan incoming tidak lengkap, pertahankan data lama
                elif (row.co is not None and row.no2 is not None) and not (co_val is not None and no2_val is not None):
                    row.updated_at = now_utc
                # 3. Selain itu, perbarui data (misal upgrade dari NRTI ke OFFL, atau dari null ke NRTI/OFFL)
                else:
                    row.co = co_val
                    row.no2 = no2_val
                    row.product_type = prod_type
                    row.updated_at = now_utc
            else:
                new_row = SentinelMeasurement(
                    region=rec["region"],
                    date=rec["date"],
                    co=co_val,
                    no2=no2_val,
                    product_type=prod_type,
                    updated_at=now_utc,
                )
                self.session.add(new_row)
                inserted_count += 1

        await self.session.commit()
        return inserted_count
