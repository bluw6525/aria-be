"""
Repository untuk entitas BatchSession dan BatchIteration.
Menangani persistensi sesi pengujian massal (batch evaluation) dan riwayat iterasinya.
"""

from typing import List, Optional, Tuple
from injector import inject
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.regression.core.models import BatchSession, BatchIteration


class BatchSessionRepository:
    """
    Data Access Layer untuk tabel batch_sessions dan batch_iterations.
    """

    @inject
    def __init__(self, db: AsyncSession):
        self.db = db

    async def save_session(
        self,
        session: BatchSession,
        iterations: List[BatchIteration],
    ) -> BatchSession:
        """
        Menyimpan sesi batch beserta seluruh record iterasinya dalam satu transaksi atomik.
        """
        self.db.add(session)
        await self.db.flush()  # Dapatkan session.id yang ter-generate

        for item in iterations:
            item.session_id = session.id
        
        self.db.add_all(iterations)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_session_by_id(self, session_id: int) -> Optional[BatchSession]:
        """
        Mengambil sesi batch berdasarkan ID.
        """
        stmt = select(BatchSession).filter(BatchSession.id == session_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def list_sessions(
        self, offset: int = 0, limit: int = 50
    ) -> List[BatchSession]:
        """
        Mengambil daftar seluruh sesi batch diurutkan dari yang terbaru.
        """
        stmt = (
            select(BatchSession)
            .order_by(BatchSession.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_iterations_by_session_id(
        self, session_id: int, estimator_type: Optional[str] = None
    ) -> List[BatchIteration]:
        """
        Mengambil seluruh record iterasi dalam satu sesi tertentu.
        """
        stmt = select(BatchIteration).filter(BatchIteration.session_id == session_id)
        if estimator_type:
            stmt = stmt.filter(BatchIteration.estimator_type == estimator_type)
        stmt = stmt.order_by(BatchIteration.id.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_iterations_paginated(
        self,
        session_id: int,
        offset: int = 0,
        limit: int = 20,
        estimator_type: Optional[str] = None,
        sort_by: str = "test_rmse",
        sort_order: str = "asc",
    ) -> Tuple[List[BatchIteration], int]:
        """
        Mengambil record iterasi dalam satu sesi dengan paginasi dan pengurutan (default: test_rmse ASC).
        """
        base_filter = [BatchIteration.session_id == session_id]
        if estimator_type:
            base_filter.append(BatchIteration.estimator_type == estimator_type)

        count_stmt = select(func.count(BatchIteration.id)).filter(*base_filter)
        count_res = await self.db.execute(count_stmt)
        total_count = count_res.scalar() or 0

        sort_col = getattr(BatchIteration, sort_by, BatchIteration.test_rmse)
        order_expr = sort_col.desc() if sort_order.lower() == "desc" else sort_col.asc()

        stmt = (
            select(BatchIteration)
            .filter(*base_filter)
            .order_by(order_expr, BatchIteration.id.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all()), total_count

    async def get_iteration_by_id(self, iteration_id: int) -> Optional[BatchIteration]:
        """
        Mengambil satu record iterasi tertentu berdasarkan ID.
        """
        stmt = select(BatchIteration).filter(BatchIteration.id == iteration_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_seeds_by_session_id(self, session_id: int) -> List[int]:
        """
        Mengambil daftar seed (random_state) unik dari batch_iterations untuk sesi tertentu.
        Bermanfaat sebagai fallback jika sesi lama belum menyimpan kolom seeds di batch_sessions.
        """
        stmt = (
            select(BatchIteration.random_state)
            .filter(
                BatchIteration.session_id == session_id,
                BatchIteration.random_state.isnot(None),
            )
            .order_by(BatchIteration.id.asc())
        )
        result = await self.db.execute(stmt)
        raw_seeds = result.scalars().all()
        seen = set()
        unique_seeds: List[int] = []
        for s in raw_seeds:
            if s is not None and s not in seen:
                seen.add(s)
                unique_seeds.append(int(s))
        return unique_seeds

    async def get_best_iteration_by_session_id(self, session_id: int) -> Optional[BatchIteration]:
        """
        Mengambil satu record iterasi terbaik (test_rmse terendah) dalam satu sesi.
        """
        stmt = (
            select(BatchIteration)
            .filter(
                BatchIteration.session_id == session_id,
                BatchIteration.test_rmse.isnot(None),
            )
            .order_by(BatchIteration.test_rmse.asc(), BatchIteration.id.asc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def delete_session(self, session_id: int) -> bool:
        """
        Menghapus sesi batch beserta iterasinya (otomatis ON DELETE CASCADE).
        """
        stmt = delete(BatchSession).where(BatchSession.id == session_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return (result.rowcount or 0) > 0


# Backward compatibility alias
BenchmarkSessionRepository = BatchSessionRepository
