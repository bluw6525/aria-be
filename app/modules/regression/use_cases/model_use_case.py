"""
Use Case untuk Operasi Model Tersimpan (CRUD TrainedModel di PostgreSQL).
"""

from typing import List
from injector import inject

from app.core.pagination import calculate_offset
from app.modules.regression.repositories import TrainedModelRepository
from app.modules.regression.schemas import (
    TrainedModelCreate,
    TrainedModelResponse,
)


class ModelUseCase:
    """
    Use case untuk operasi basis data model tersimpan (List, Detail, Save, Delete).
    """

    @inject
    def __init__(self, repository: TrainedModelRepository):
        self.repository = repository

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
