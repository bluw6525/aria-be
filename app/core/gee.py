"""
Modul Konfigurasi dan Koneksi Google Earth Engine (GEE).
Sejajar dengan database.py, bertugas mengelola koneksi fisik dan autentikasi ke Google Earth Engine.
"""

import json
import logging
from pathlib import Path
import ee
from injector import inject, singleton

from app.core.config import settings

logger = logging.getLogger(__name__)


@singleton
class GEEClient:
    """
    Objek klien singleton untuk koneksi Google Earth Engine.
    Dikelola oleh Dependency Injection Container.
    """

    @inject
    def __init__(self):
        self.is_connected: bool = self.connect()

    def connect(self) -> bool:
        """
        Menghubungkan ke Google Earth Engine.
        Prioritas:
        1. Service Account Key File (GEE_KEY_FILE di .env)
        2. Persistent Credentials lokal / ADC (GEE_PROJECT_ID di .env)
        """
        project_id = settings.GEE_PROJECT_ID
        key_file = settings.GEE_KEY_FILE

        try:
            if key_file and Path(key_file).is_file():
                with open(key_file, "r", encoding="utf-8") as f:
                    sa_email = json.load(f).get("client_email")
                logger.info("Menginisialisasi GEE via Service Account: %s (Project: %s)", sa_email, project_id)
                credentials = ee.ServiceAccountCredentials(sa_email, key_file)
                ee.Initialize(credentials=credentials, project=project_id)
            else:
                logger.info("Menginisialisasi GEE via default credentials (Project: %s)", project_id)
                ee.Initialize(project=project_id)

            logger.info("Koneksi Google Earth Engine (GEE) berhasil tersambung.")
            return True
        except Exception as exc:
            logger.warning("Inisialisasi koneksi GEE gagal: %s", exc)
            return False

    def ensure_connected(self) -> None:
        """Memastikan koneksi GEE aktif atau mencoba menghubungkan ulang jika sebelumnya belum."""
        if not self.is_connected:
            self.is_connected = self.connect()
        if not self.is_connected:
            raise RuntimeError(
                "Koneksi ke Google Earth Engine belum siap. "
                "Pastikan kredensial GEE telah disiapkan di file .env."
            )
