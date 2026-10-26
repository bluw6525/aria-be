"""
Use Case untuk Operasi Model Tersimpan (CRUD TrainedModel di PostgreSQL).
"""

from typing import List
from injector import inject

from app.core.pagination import calculate_offset
from app.modules.regression.repositories import (
    TrainedModelRepository,
    BenchmarkSessionRepository,
)
from app.modules.regression.schemas import (
    TrainedModelCreate,
    TrainedModelResponse,
    PromoteModelRequest,
)


class ModelUseCase:
    """
    Use case untuk operasi basis data model tersimpan (List, Detail, Save, Delete, Promote).
    """

    @inject
    def __init__(
        self,
        repository: TrainedModelRepository,
        benchmark_repo: BenchmarkSessionRepository,
    ):
        self.repository = repository
        self.benchmark_repo = benchmark_repo

    async def list_models(self, page: int = 1, limit: int = 100) -> List[TrainedModelResponse]:
        """Mengambil seluruh daftar model tersimpan dengan pagination berbasis nomor halaman."""
        offset = calculate_offset(page=page, limit=limit)
        models = await self.repository.get_all(offset=offset, limit=limit)
        return [TrainedModelResponse.model_validate(m) for m in models]

    async def get_model(self, model_id: int) -> TrainedModelResponse:
        """Mengambil detail satu model berdasarkan ID."""
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model dengan ID {model_id} tidak ditemukan.")
        return TrainedModelResponse.model_validate(model)

    async def save_model(self, payload: TrainedModelCreate) -> TrainedModelResponse:
        """Menyimpan model pelatihan baru ke database."""
        existing = await self.repository.get_by_name(payload.name)
        if existing:
            raise ValueError(f"Model dengan nama '{payload.name}' sudah ada. Gunakan nama yang berbeda.")

        model = await self.repository.create(payload.model_dump())
        return TrainedModelResponse.model_validate(model)

    async def delete_model(self, model_id: int) -> None:
        """Menghapus model tersimpan dari database."""
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model dengan ID {model_id} tidak ditemukan.")
        await self.repository.delete(model)

    async def promote_model(self, payload: PromoteModelRequest) -> TrainedModelResponse:
        """
        Mempromosikan hasil iterasi MCCV menjadi TrainedModel di PostgreSQL.
        """
        existing = await self.repository.get_by_name(payload.model_name)
        if existing:
            raise ValueError(f"Model dengan nama '{payload.model_name}' sudah ada. Gunakan nama yang berbeda.")

        iteration = await self.benchmark_repo.get_iteration_by_id(payload.iteration_id)
        if not iteration:
            raise ValueError(f"Iterasi MCCV dengan ID {payload.iteration_id} tidak ditemukan.")

        # Buat data TrainedModel dari parameter iterasi terpilih
        model_data = {
            "name": payload.model_name,
            "estimator_type": iteration.estimator_type,
            "is_default": payload.is_default,
            "intercept": iteration.intercept if iteration.intercept is not None else 0.0,
            "coefficients": iteration.coefficients or {},
            "scale": iteration.scale,
            "train_rmse": iteration.train_rmse or iteration.test_rmse,
            "train_mae": iteration.train_mae or iteration.test_mae,
            "train_r2": iteration.train_r2 or iteration.test_r2,
        }
        saved_model = await self.repository.create(model_data)
        return TrainedModelResponse.model_validate(saved_model)
