"""
Repository untuk entitas TrainedModel.
Menangani seluruh operasi akses data basis data PostgreSQL secara asinkron.
"""

from typing import List, Optional
from injector import inject
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.regression.core.models import TrainedModel


class TrainedModelRepository:
    """
    Data Access Layer untuk tabel trained_models.
    """

    @inject
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self, offset: int = 0, limit: int = 100) -> List[TrainedModel]:
        """Mengambil seluruh daftar model yang tersimpan."""
        stmt = (
            select(TrainedModel)
            .order_by(TrainedModel.is_default.desc(), TrainedModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, model_id: int) -> Optional[TrainedModel]:
        """Mengambil model berdasarkan ID."""
        stmt = select(TrainedModel).filter(TrainedModel.id == model_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_name(self, name: str) -> Optional[TrainedModel]:
        """Mengambil model berdasarkan nama unik."""
        stmt = select(TrainedModel).filter(TrainedModel.name == name)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_default(self) -> Optional[TrainedModel]:
        """Mengambil model yang ditandai sebagai default."""
        stmt = select(TrainedModel).filter(TrainedModel.is_default == True)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def create(self, model_data: dict) -> TrainedModel:
        """Menyimpan model regresi baru ke basis data."""
        if model_data.get("is_default", False):
            await self.db.execute(
                update(TrainedModel).values(is_default=False)
            )

        db_model = TrainedModel(**model_data)
        self.db.add(db_model)
        await self.db.commit()
        await self.db.refresh(db_model)
        return db_model

    async def delete(self, model: TrainedModel) -> None:
        """Menghapus model dari basis data."""
        await self.db.delete(model)
        await self.db.commit()
