"""
Repository untuk entitas PublicModel.
Menangani seluruh operasi akses data basis data PostgreSQL secara asinkron.
"""

from typing import List, Optional
from injector import inject
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.regression.core.models import PublicModel


class PublicModelRepository:
    """
    Data Access Layer untuk tabel public_models.
    """

    @inject
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self, offset: int = 0, limit: int = 100) -> List[PublicModel]:
        """Mengambil seluruh daftar model publik yang tersimpan diurutkan dari yang terbaru."""
        stmt = (
            select(PublicModel)
            .order_by(PublicModel.is_active.desc(), PublicModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_active(self) -> Optional[PublicModel]:
        """Mengambil model publik yang sedang aktif (is_active=True)."""
        stmt = (
            select(PublicModel)
            .filter(PublicModel.is_active == True)
            .order_by(PublicModel.created_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def set_active(self, model_id: int) -> Optional[PublicModel]:
        """Mengaktifkan satu model spesifik dan menonaktifkan model publik lainnya."""
        model = await self.get_by_id(model_id)
        if not model:
            return None
        await self.db.execute(update(PublicModel).values(is_active=False))
        model.is_active = True
        await self.db.commit()
        await self.db.refresh(model)
        return model

    async def get_by_id(self, model_id: int) -> Optional[PublicModel]:
        """Mengambil model berdasarkan ID."""
        stmt = select(PublicModel).filter(PublicModel.id == model_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_name(self, name: str) -> Optional[PublicModel]:
        """Mengambil model berdasarkan nama unik."""
        stmt = select(PublicModel).filter(PublicModel.name == name)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def create(self, model_data: dict) -> PublicModel:
        """
        Menyimpan model regresi publik baru ke basis data.
        Otomatis menonaktifkan model lama (is_active=False) dan menjadikan model baru ini aktif (is_active=True).
        """
        data = dict(model_data)

        # Pastikan coefficients JSON terisi dan key non-kolom dibersihkan
        if "coefficients" not in data or not data["coefficients"]:
            data["coefficients"] = {
                "CO": data.pop("coef_co", 0.0),
                "NO2": data.pop("coef_no2", 0.0),
            }
        else:
            data.pop("coef_co", None)
            data.pop("coef_no2", None)

        # Nonaktifkan seluruh model lama sebelum menyimpan model baru sebagai aktif
        await self.db.execute(update(PublicModel).values(is_active=False))
        data["is_active"] = True

        # Filter hanya kolom yang sah terdaftar pada model PublicModel
        allowed_columns = {c.name for c in PublicModel.__table__.columns}
        clean_data = {k: v for k, v in data.items() if k in allowed_columns}

        db_model = PublicModel(**clean_data)
        self.db.add(db_model)
        await self.db.commit()
        await self.db.refresh(db_model)
        return db_model

    async def delete(self, model: PublicModel) -> None:
        """Menghapus model dari basis data."""
        was_active = model.is_active
        await self.db.delete(model)
        await self.db.commit()

        # Jika model yang dihapus adalah model aktif, aktifkan model terbaru yang tersisa
        if was_active:
            latest = (
                await self.db.execute(
                    select(PublicModel).order_by(PublicModel.created_at.desc()).limit(1)
                )
            ).scalars().first()
            if latest:
                latest.is_active = True
                await self.db.commit()
