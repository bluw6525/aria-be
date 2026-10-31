"""
Use Case untuk Operasi Model Publik (CRUD PublicModel di PostgreSQL).
"""

from typing import List
from injector import inject

from app.core.pagination import calculate_offset
from app.modules.regression.repositories import (
    PublicModelRepository,
    BatchSessionRepository,
)
from app.modules.regression.schemas import (
    PublicModelCreate,
    PublicModelResponse,
    PromoteModelRequest,
)


class ModelUseCase:
    """
    Use case untuk operasi basis data model publik tersimpan (List, Detail, Save, Delete, Promote).
    """

    @inject
    def __init__(
        self,
        repository: PublicModelRepository,
        benchmark_repo: BatchSessionRepository,
    ):
        self.repository = repository
        self.benchmark_repo = benchmark_repo

    async def list_models(self, page: int = 1, limit: int = 100) -> List[PublicModelResponse]:
        """Mengambil seluruh daftar model publik tersimpan dengan pagination berbasis nomor halaman."""
        offset = calculate_offset(page=page, limit=limit)
        models = await self.repository.get_all(offset=offset, limit=limit)
        return [PublicModelResponse.model_validate(m) for m in models]

    async def get_model(self, model_id: int) -> PublicModelResponse:
        """Mengambil detail satu model publik berdasarkan ID."""
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model dengan ID {model_id} tidak ditemukan.")
        return PublicModelResponse.model_validate(model)

    async def save_model(self, payload: PublicModelCreate) -> PublicModelResponse:
        """Menyimpan model pelatihan baru ke database."""
        existing = await self.repository.get_by_name(payload.name)
        if existing:
            raise ValueError(f"Model dengan nama '{payload.name}' sudah ada. Gunakan nama yang berbeda.")

        model = await self.repository.create(payload.model_dump())
        return PublicModelResponse.model_validate(model)

    async def delete_model(self, model_id: int) -> None:
        """Menghapus model tersimpan dari database."""
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model dengan ID {model_id} tidak ditemukan.")
        await self.repository.delete(model)

    async def promote_model(self, payload: PromoteModelRequest) -> PublicModelResponse:
        """
        Mempromosikan hasil iterasi batch menjadi PublicModel di PostgreSQL.
        """
        existing = await self.repository.get_by_name(payload.model_name)
        if existing:
            raise ValueError(f"Model dengan nama '{payload.model_name}' sudah ada. Gunakan nama yang berbeda.")

        iteration = await self.benchmark_repo.get_iteration_by_id(payload.iteration_id)
        if not iteration:
            raise ValueError(f"Iterasi batch dengan ID {payload.iteration_id} tidak ditemukan.")

        # Buat data PublicModel dari parameter iterasi terpilih dengan koefisien terstandarisasi
        coefs = iteration.coefficients or {}
        val_co = float(coefs.get("CO", 0.0))
        val_no2 = float(coefs.get("NO2", 0.0))
        if val_co == 0.0 and len(coefs) >= 1:
            val_co = float(list(coefs.values())[0])
        if val_no2 == 0.0 and len(coefs) >= 2:
            val_no2 = float(list(coefs.values())[1])

        model_data = {
            "name": payload.model_name,
            "estimator_type": iteration.estimator_type,
            "intercept": iteration.intercept if iteration.intercept is not None else 0.0,
            "coefficients": {
                "CO": val_co,
                "NO2": val_no2,
            },
            "scale": iteration.scale,
            "train_rmse": iteration.train_rmse or iteration.test_rmse,
            "train_mae": iteration.train_mae or iteration.test_mae,
        }
        saved_model = await self.repository.create(model_data)
        return PublicModelResponse.model_validate(saved_model)
