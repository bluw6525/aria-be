"""
Repository untuk entitas BenchmarkSession dan MCCVIterationModel.
Menangani persistensi sesi benchmark Monte Carlo Cross-Validation dan riwayat iterasinya.
"""

from typing import List, Optional, Tuple
from injector import inject
from sqlalchemy import select, delete, func, insert
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

        if iterations:
            col_names = [c.name for c in MCCVIterationModel.__table__.columns if c.name != "id"]
            rows = []
            for item in iterations:
                item.session_id = session.id
                rows.append({col: getattr(item, col) for col in col_names})
            chunk_size = 2000
            for i in range(0, len(rows), chunk_size):
                await self.db.execute(insert(MCCVIterationModel), rows[i : i + chunk_size])

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

    async def get_iterations_paginated(
        self,
        session_id: int,
        offset: int = 0,
        limit: int = 20,
        estimator_type: Optional[str] = None,
        sort_by: str = "test_r2",
        sort_order: str = "desc",
    ) -> Tuple[List[MCCVIterationModel], int]:
        """
        Mengambil record iterasi dalam satu sesi dengan paginasi dan pengurutan (default: test_r2 DESC).
        """
        base_filter = [MCCVIterationModel.session_id == session_id]
        if estimator_type:
            base_filter.append(MCCVIterationModel.estimator_type == estimator_type)

        count_stmt = select(func.count(MCCVIterationModel.id)).filter(*base_filter)
        count_res = await self.db.execute(count_stmt)
        total_count = count_res.scalar() or 0

        sort_col = getattr(MCCVIterationModel, sort_by, MCCVIterationModel.test_r2)
        order_expr = sort_col.desc() if sort_order.lower() == "desc" else sort_col.asc()

        stmt = (
            select(MCCVIterationModel)
            .filter(*base_filter)
            .order_by(order_expr, MCCVIterationModel.id.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all()), total_count

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
