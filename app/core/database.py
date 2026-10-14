"""
Modul Konfigurasi Database Asinkron (Async SQLAlchemy).
Mengatur pembuatan Async Engine, Connection Pooling, dan Session Maker.
"""

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession
)
from sqlalchemy.orm import declarative_base
from app.core.config import settings


# Engine bertugas mengelola koneksi fisik dan connection pool ke database server
engine = create_async_engine(
    # Connection string database (contoh: postgresql+asyncpg://user:pass@host:5432/db)
    settings.DATABASE_URL,
    # Menguji koneksi (SELECT 1) sebelum dipakai, memastikan koneksi mati langsung dibuang
    pool_pre_ping=True,
    # Jumlah koneksi minimum yang selalu disiapkan dan dipertahankan dalam pool
    pool_size=10,
    # Jumlah koneksi ekstra maksimum yang boleh dibuka saat request sedang ramai
    max_overflow=20
)


# Factory yang bertugas membuat instance AsyncSession baru saat dibutuhkan
AsyncSessionLocal = async_sessionmaker(
    # Mengikat setiap sesi ke Async Engine di atas
    bind=engine,
    # Menentukan class sesi bertipe asinkron (AsyncSession)
    class_=AsyncSession,
    # Mematikan autocommit agar transaksi tetap aman dan dieksekusi secara manual
    autocommit=False,
    # Mematikan autoflush otomatis sebelum query dieksekusi
    autoflush=False,
    # Mencegah atribut model kedaluwarsa setelah commit (sangat krusial untuk async)
    expire_on_commit=False
)


# Base class yang harus diwarisi (inherit) oleh semua model/tabel ORM di folder modules
# Sekaligus bertindak sebagai metadata registry untuk Alembic
Base = declarative_base()


