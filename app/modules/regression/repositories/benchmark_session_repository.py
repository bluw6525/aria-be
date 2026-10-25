"""
Repository untuk entitas BenchmarkSession dan MCCVIterationModel.
Menangani persistensi sesi benchmark Monte Carlo Cross-Validation dan riwayat iterasinya.
"""

from typing import List, Optional
from injector import inject
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.regression.core.models import BenchmarkSession, MCCVIterationModel


class BenchmarkSessionRepository:
    """
    Data Access Layer untuk tabel benchmark_sessions dan mccv_iterations.
    """

    @inject
    def __init__(self, db: AsyncSession):
        self.db = db

    async def save_session(
        self,
        session: BenchmarkSession,
        iterations: List[MCCVIterationModel],
    ) -> BenchmarkSession:
        """
        Menyimpan sesi benchmark beserta seluruh record iterasinya dalam satu transaksi atomik.
        """
        self.db.add(session)
        await self.db.flush()  # Dapatkan session.id yang ter-generate

        for item in iterations:
            item.session_id = session.id
        
        self.db.add_all(iterations)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_session_by_id(self, session_id: int) -> Optional[BenchmarkSession]:
        """
        Mengambil sesi benchmark berdasarkan ID.
        """
        stmt = select(BenchmarkSession).filter(BenchmarkSession.id == session_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def list_sessions(
        self, offset: int = 0, limit: int = 50
    ) -> List[BenchmarkSession]:
        """
        Mengambil daftar seluruh sesi benchmark diurutkan dari yang terbaru.
        """
        stmt = (
            select(BenchmarkSession)
            .order_by(BenchmarkSession.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_iterations_by_session_id(
        self, session_id: int, estimator_type: Optional[str] = None
    ) -> List[MCCVIterationModel]:
        """
        Mengambil seluruh record iterasi dalam satu sesi tertentu.
        """
        stmt = select(MCCVIterationModel).filter(MCCVIterationModel.session_id == session_id)
        if estimator_type:
            stmt = stmt.filter(MCCVIterationModel.estimator_type == estimator_type)
        stmt = stmt.order_by(MCCVIterationModel.id.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_iteration_by_id(self, iteration_id: int) -> Optional[MCCVIterationModel]:
        """
        Mengambil satu record iterasi tertentu berdasarkan ID.
        """
        stmt = select(MCCVIterationModel).filter(MCCVIterationModel.id == iteration_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def delete_session(self, session_id: int) -> bool:
        """
        Menghapus sesi benchmark beserta iterasinya (otomatis ON DELETE CASCADE).
        """
        stmt = delete(BenchmarkSession).where(BenchmarkSession.id == session_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return (result.rowcount or 0) > 0
