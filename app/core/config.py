import os
from dotenv import load_dotenv

# Memuat isi file .env ke dalam environment variables sistem saat aplikasi dimulai
load_dotenv()


class Settings:
    # Nama aplikasi/project, default: 'Aria API'
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Aria API")

    # Prefix URL dasar untuk seluruh endpoint API versi 1, default: '/api/v1'
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")

    # Connection string database SQLite asinkron (menggunakan driver aiosqlite)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./aria.db"
    )

    # Konfigurasi Keamanan & Token JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "aria-super-secure-production-jwt-secret-key-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 7)))  # 7 hari


# Instance tunggal Settings yang siap diimpor dan digunakan di seluruh bagian aplikasi
settings = Settings()
