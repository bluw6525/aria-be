"""
Modul Factory untuk Dependency Injection Container.
Menyediakan registrasi modul aplikasi secara terpusat untuk Injector container.
"""

from typing import Iterable, List
from injector import Injector, Module
from app.core.container import DatabaseModule, GEEModule
from app.modules.regression.container import RegressionContainer
from app.modules.auth.container import AuthContainer
from app.modules.satellite.container import SatelliteContainer


def _create_app_modules() -> Iterable[Module]:
    """
    Fungsi internal untuk mendaftarkan seluruh modul domain/fitur bisnis ke container.
    Jika membuat fitur baru (misal User, Order), tambahkan containernya di sini.
    """
    return [RegressionContainer(), AuthContainer(), SatelliteContainer()]


def create_injector() -> Injector:
    """
    Membuat dan mengembalikan instance Injector global dengan menggabungkan
    DatabaseModule, GEEModule, dan seluruh modul aplikasi yang terdaftar.
    """
    # Menggabungkan modul fondasi infrastruktur inti (Database, GEE) dan seluruh modul domain bisnis
    modules: List[Module] = [DatabaseModule(), GEEModule(), *_create_app_modules()]

    # Menginisialisasi IoC Container Injector dengan seluruh modul yang sudah dikumpulkan
    return Injector(modules)


# Instance global Injector aplikasi yang siap dihubungkan ke FastAPI lewat attach_injector
app_injector = create_injector()
