import os
from dotenv import load_dotenv

# Memuat isi file .env ke dalam environment variables sistem saat aplikasi dimulai
load_dotenv()


class Settings:
    # Nama aplikasi/project, default: 'Aria API'
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Aria API")

    # Prefix URL dasar untuk seluruh endpoint API versi 1, default: '/api/v1'
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")

    # Connection string database PostgreSQL asinkron (menggunakan driver asyncpg)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/aria_db"
    )


# Instance tunggal Settings yang siap diimpor dan digunakan di seluruh bagian aplikasi
settings = Settings()

