"""
Modul Konfigurasi Database Asinkron (Async SQLAlchemy).
Mengatur pembuatan Async Engine untuk SQLite, Event Listener PRAGMA, dan Session Maker.
"""

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession
)
from sqlalchemy.orm import declarative_base
from app.core.config import settings


# Engine bertugas mengelola koneksi fisik dan eksekusi query ke berkas database SQLite
engine = create_async_engine(
    # Connection string database (contoh: sqlite+aiosqlite:///./aria.db)
    settings.DATABASE_URL,
    # Argumen koneksi tambahan khusus driver (menonaktifkan pembatasan thread bawaan SQLite)
    connect_args={"check_same_thread": False},
    # Menguji koneksi (SELECT 1) sebelum dipakai, memastikan koneksi siap digunakan
    pool_pre_ping=True,
)


# Konfigurasi PRAGMA khusus SQLite untuk memastikan integritas dan performa optimal
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    # Mengaktifkan penegakan Foreign Key secara tegas (default SQLite adalah OFF)
    cursor.execute("PRAGMA foreign_keys=ON")
    # Mengaktifkan Write-Ahead Logging (WAL) agar read dan write dapat berjalan bersamaan tanpa saling lock
    cursor.execute("PRAGMA journal_mode=WAL")
    # Menentukan batas waktu tunggu (10000 ms) saat database sibuk sebelum melempar error lock
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.close()


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
    expire_on_commit=False,
)


# Base class yang harus diwarisi (inherit) oleh semua model/tabel ORM di folder modules
# Sekaligus bertindak sebagai metadata registry untuk Alembic
Base = declarative_base()
