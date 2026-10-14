"""
Modul Dependency Injection Container menggunakan pustaka 'injector' & 'fastapi-injector'.
Menyediakan integrasi murni antara Injector (IoC Container) dan FastAPI.
"""

from fastapi_injector import Injected, request_scope
from injector import Module, provider, singleton
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from app.core.database import AsyncSessionLocal


class DatabaseModule(Module):
    """
    Modul injector untuk menyediakan AsyncSession berbasis request_scope.
    """

    # Menyediakan instance sessionmaker secara singleton (hanya dibuat 1 kali saat start)
    @provider
    @singleton
    def provide_session_maker(self) -> async_sessionmaker[AsyncSession]:
        # Mengembalikan instance pabrik sesi database
        return AsyncSessionLocal

    # Menyediakan AsyncSession unik per request (request_scope)
    @provider
    @request_scope
    def provide_async_session(
        self,
        # session_maker di-inject secara otomatis dari provide_session_maker di atas
        session_maker: async_sessionmaker[AsyncSession],
    ) -> AsyncSession:
        # Membuka sesi baru untuk request HTTP saat ini
        return session_maker()


