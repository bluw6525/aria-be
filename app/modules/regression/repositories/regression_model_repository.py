"""
Repository untuk entitas RegressionModel (riwayat hasil evaluasi regresi batch/monte carlo).
Menangani operasi akses basis data PostgreSQL secara asinkron.
"""

from typing import List
from injector import inject
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.regression.core.models import RegressionModel


class RegressionModelRepository:
    """
    Data Access Layer untuk tabel regression_models.
    """

    @inject
    def __init__(self, db: AsyncSession):
        self.db = db

    async def bulk_create(self, records: List[RegressionModel]) -> int:
        """
        Menyimpan banyak record hasil evaluasi regresi sekaligus (bulk insert).
        """
        if not records:
            return 0
        self.db.add_all(records)
        await self.db.commit()
        return len(records)

    async def delete_by_dataset_and_test_size(
        self, dataset_name: str, test_size: float
    ) -> int:
        """
        Menghapus riwayat evaluasi berdasarkan nama dataset dan rasio test_size.
        """
        stmt = delete(RegressionModel).where(
            RegressionModel.dataset_name == dataset_name,
            RegressionModel.test_size == test_size,
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount or 0
