"""
Container Injector untuk Modul Satellite.
Mengatur binding dependensi untuk repository dan use case data satelit.
"""

from injector import Module, Binder, singleton
from app.modules.satellite.core.extractor import Sentinel5PExtractor
from app.modules.satellite.repositories import SatelliteRepository
from app.modules.satellite.use_cases import SyncSatelliteUseCase


class SatelliteContainer(Module):
    """
    Module Injector untuk modul Satelit.
    Didaftarkan ke Core Factory.
    """

    def configure(self, binder: Binder) -> None:
        # Extractor (Singleton - mengelola cache geometri dan koneksi GEE)
        binder.bind(Sentinel5PExtractor, to=Sentinel5PExtractor, scope=singleton)

        # Repository
        binder.bind(SatelliteRepository)

        # Use Cases
        binder.bind(SyncSatelliteUseCase)
