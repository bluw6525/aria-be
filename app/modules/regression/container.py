"""
Container / Module Injector untuk Modul Regresi Robust.
Mengatur binding dependensi untuk repository dan masing-masing Use Case modular.
"""

from injector import Module, Binder
from app.modules.regression.repositories import (
    TrainedModelRepository,
    RegressionModelRepository,
)
from app.modules.regression.use_cases import (
    DetectOutliersUseCase,
    TrainModelUseCase,
    TestModelUseCase,
    ModelUseCase,
    EstimateUseCase,
    BatchEstimateUseCase,
)


class RegressionContainer(Module):
    """
    Module Injector untuk modul Regresi.
    Didaftarkan ke Core Factory.
    """

    def configure(self, binder: Binder) -> None:
        # Data Access Repository
        binder.bind(TrainedModelRepository)
        binder.bind(RegressionModelRepository)

        # Modular Business Logic Use Cases
        binder.bind(DetectOutliersUseCase)
        binder.bind(TrainModelUseCase)
        binder.bind(TestModelUseCase)
        binder.bind(ModelUseCase)
        binder.bind(EstimateUseCase)
        binder.bind(BatchEstimateUseCase)

