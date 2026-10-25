"""
Repository untuk entitas User.
Menangani query data akun pengguna ke database PostgreSQL secara asinkron.
"""

from typing import Optional
from injector import inject
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.core.models import User


class UserRepository:
    """Data Access Layer untuk tabel users."""

    @inject
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Mengambil data pengguna berdasarkan ID."""
        stmt = select(User).filter(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_username(self, username: str) -> Optional[User]:
        """Mengambil data pengguna berdasarkan username unik."""
        stmt = select(User).filter(User.username == username)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def create(self, user_data: dict) -> User:
        """Membuat pengguna baru dan menyimpannya ke database."""
        db_user = User(**user_data)
        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user)
        return db_user
