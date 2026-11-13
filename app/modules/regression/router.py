"""
Router FastAPI untuk Modul Regresi Robust (Latih, Uji, Deteksi Outlier, Estimasi, dan Pengujian Massal).
Menggunakan Dependency Injection terpisah per Use Case melalui fastapi_injector.
"""

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Query, Depends
from fastapi_injector import Injected

from app.modules.auth.core.models import User
from app.modules.auth.dependencies import require_admin
from app.modules.regression.schemas import (
    PublicModelResponse,
    PublicModelCreate,
    PromoteModelRequest,
    OutlierDetectionResponse,
    TrainModelResult,
    TestModelResult,
    BatchEstimateResponse,
    SatelliteEstimateRequest,
    MapOverviewResponse,
    BatchSessionResponse,
    BatchIterationResponse,
    PaginatedIterationsResponse,
    BatchRunResponse,
)
from app.modules.regression.use_cases import (
    DetectOutliersUseCase,
    TrainModelUseCase,
    TestModelUseCase,
    ModelUseCase,
    EstimateUseCase,
    BatchEstimateUseCase,
)

router = APIRouter(prefix="/regression", tags=["Regression & Models"])


# ============================================================================
# 1. Endpoints Manajemen Model Publik (CRUD & Promote)
# ============================================================================

@router.get("/models", response_model=List[PublicModelResponse])
async def get_models(
    page: int = Query(1, ge=1, description="Nomor halaman (mulai dari 1)"),
    limit: int = Query(100, ge=1, le=100, description="Jumlah data per halaman"),
    usecase: ModelUseCase = Injected(ModelUseCase)
):
    """Mengambil daftar model publik yang tersimpan dengan pagination (page & limit)."""
    return await usecase.list_models(page=page, limit=limit)


@router.post("/models", response_model=PublicModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model_endpoint(
    payload: PublicModelCreate,
    admin: User = Depends(require_admin),
    usecase: ModelUseCase = Injected(ModelUseCase)
):
    """Menyimpan parameter model hasil pelatihan ke basis data (RESTful Create)."""
    try:
        return await usecase.save_model(payload)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.get("/models/{model_id}", response_model=PublicModelResponse)
async def get_model(
    model_id: int,
    usecase: ModelUseCase = Injected(ModelUseCase)
):
    """Mengambil detail satu model publik berdasarkan ID."""
    try:
        return await usecase.get_model(model_id)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    admin: User = Depends(require_admin),
    usecase: ModelUseCase = Injected(ModelUseCase)
):
    """Menghapus model dari basis data."""
    try:
        await usecase.delete_model(model_id)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


@router.post("/models/{model_id}/activate", response_model=PublicModelResponse)
async def activate_model_endpoint(
    model_id: int,
    admin: User = Depends(require_admin),
    usecase: ModelUseCase = Injected(ModelUseCase),
):
    """Mengaktifkan satu model spesifik dan menonaktifkan model publik lainnya."""
    try:
        return await usecase.activate_model(model_id)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


@router.post("/models/promote", response_model=PublicModelResponse, status_code=status.HTTP_201_CREATED)
async def promote_model_endpoint(
    payload: PromoteModelRequest,
    admin: User = Depends(require_admin),
    usecase: ModelUseCase = Injected(ModelUseCase),
):
    """Mempromosikan hasil iterasi batch menjadi Model Publik definitif."""
    try:
        return await usecase.promote_model(payload)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 2. Endpoint Deteksi Outlier Robust (/outliers)
# ============================================================================

@router.post("/outliers", response_model=OutlierDetectionResponse)
async def detect_outliers_endpoint(
    file: UploadFile = File(..., description="Berkas data excel (.xlsx) atau .csv"),
    fit_intercept: bool = Form(True, description="Apakah memperhitungkan intercept"),
    admin: User = Depends(require_admin),
    usecase: DetectOutliersUseCase = Injected(DetectOutliersUseCase)
):
    """Mendeteksi pencilan multivariat & residual pada dataset spasial (Memerlukan Hak Admin)."""
    try:
        return await usecase.detect_outliers(file=file, fit_intercept=fit_intercept)
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 3. Endpoint Pelatihan Model (/trainings)
# ============================================================================

@router.post("/trainings", response_model=TrainModelResult)
async def train_model_endpoint(
    file: UploadFile = File(..., description="Berkas dataset latih (.xlsx atau .csv)"),
    estimator_type: str = Form("MM", description="Algoritma: MM, M, S, atau OLS"),
    fit_intercept: bool = Form(True, description="Apakah menggunakan konstanta (intercept)"),
    admin: User = Depends(require_admin),
    usecase: TrainModelUseCase = Injected(TrainModelUseCase)
):
    """Melatih algoritma regresi robust pada data latih baru."""
    try:
        return await usecase.train_model(
            file=file,
            estimator_type=estimator_type,
            fit_intercept=fit_intercept
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 4. Endpoint Pengujian Model (/models/{id}/tests & /tests/custom)
# ============================================================================

@router.post("/models/{model_id}/tests", response_model=TestModelResult)
async def test_saved_model_by_id_endpoint(
    model_id: int,
    file: UploadFile = File(..., description="Berkas dataset uji (.xlsx atau .csv)"),
    usecase: TestModelUseCase = Injected(TestModelUseCase)
):
    """Menguji akurasi model publik tersimpan terhadap data uji baru via path parameter."""
    try:
        return await usecase.test_model(model_id=model_id, file=file)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.post("/tests/custom", response_model=TestModelResult)
async def test_custom_parameters_endpoint(
    file: UploadFile = File(..., description="Berkas dataset uji (.xlsx atau .csv)"),
    coef_co: float = Form(..., description="Koefisien variabel CO Sentinel-5P"),
    coef_no2: float = Form(..., description="Koefisien variabel NO2 Sentinel-5P"),
    intercept: float = Form(0.0, description="Konstanta intercept (beta 0)"),
    estimator_type: str = Form("MM", description="Nama algoritma/metode"),
    model_name: str = Form("Model Baru Dilatih", description="Label nama model"),
    model_id: Optional[int] = Form(None, description="ID model opsional jika tersimpan"),
    usecase: TestModelUseCase = Injected(TestModelUseCase)
):
    """Menguji parameter koefisien kustom terhadap data uji."""
    try:
        return await usecase.test_custom_model(
            coef_co=coef_co,
            coef_no2=coef_no2,
            file=file,
            intercept=intercept,
            estimator_type=estimator_type,
            model_name=model_name,
            model_id=model_id,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 5. Endpoint Inferensi Estimasi (/estimates)
# ============================================================================

@router.post("/estimates/satellite", response_model=BatchEstimateResponse)
async def estimate_satellite_endpoint(
    payload: SatelliteEstimateRequest,
    usecase: EstimateUseCase = Injected(EstimateUseCase)
):
    """Melakukan estimasi O3 dan kategori ISPU langsung dari data satelit Sentinel-5P."""
    try:
        return await usecase.estimate_from_satellite(payload)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))


@router.get("/estimates/map-overview", response_model=MapOverviewResponse)
async def get_map_overview_endpoint(
    start_date: str = Query(..., description="Tanggal awal (YYYY-MM-DD)"),
    end_date: str = Query(..., description="Tanggal akhir (YYYY-MM-DD)"),
    usecase: EstimateUseCase = Injected(EstimateUseCase)
):
    """Mengambil rangkuman estimasi ISPU dan kategori untuk 5 wilayah untuk peta interaktif."""
    try:
        return await usecase.get_map_overview(start_date=start_date, end_date=end_date)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))



# ============================================================================
# 6. Endpoints Evaluasi & Pengujian Massal (/batch-sessions)
# ============================================================================

@router.post("/batch-sessions", response_model=BatchRunResponse, status_code=status.HTTP_201_CREATED)
async def run_batch_endpoint(
    file: UploadFile = File(..., description="Berkas dataset latih (.xlsx, .csv)"),
    session_name: Optional[str] = Form(None, description="Nama sesi evaluasi massal opsional"),
    n_splits: int = Form(200, ge=1, le=1000, description="Jumlah iterasi seed pengacakan"),
    seeds: Optional[str] = Form(None, description="Daftar seed kustom (pisahkan koma/spasi)"),
    test_size: float = Form(0.2, gt=0.0, lt=1.0, description="Proporsi data uji"),
    fit_intercept: bool = Form(True, description="Apakah estimator menggunakan intercept"),
    estimator_types: Optional[str] = Form(None, description="Daftar estimator dipisahkan koma atau JSON"),
    admin: User = Depends(require_admin),
    usecase: BatchEstimateUseCase = Injected(BatchEstimateUseCase),
):
    """Menjalankan dan membuat sesi evaluasi massal baru (Evaluasi Massal / Batch Cross-Validation)."""
    try:
        return await usecase.batch_estimate(
            file_path=file,
            seeds=seeds,
            n_splits=n_splits,
            session_name=session_name,
            test_size=test_size,
            fit_intercept=fit_intercept,
            estimator_types=estimator_types,
            save_to_db=True,
            verbose=False,
        )
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.get("/batch-sessions", response_model=List[BatchSessionResponse])
async def list_batch_sessions_endpoint(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    usecase: BatchEstimateUseCase = Injected(BatchEstimateUseCase),
):
    """Mengambil daftar seluruh sesi evaluasi massal sebelumnya."""
    return await usecase.list_sessions(page=page, limit=limit)


@router.get("/batch-sessions/{session_id}", response_model=BatchSessionResponse)
async def get_batch_session_endpoint(
    session_id: int,
    usecase: BatchEstimateUseCase = Injected(BatchEstimateUseCase),
):
    """Mengambil detail satu sesi evaluasi massal beserta ringkasan metrik agregatnya."""
    session = await usecase.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sesi batch ID {session_id} tidak ditemukan.")
    return session


@router.get("/batch-sessions/{session_id}/iterations", response_model=PaginatedIterationsResponse)
async def get_batch_iterations_endpoint(
    session_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    estimator_type: Optional[str] = Query(None),
    sort_by: str = Query("test_rmse"),
    sort_order: str = Query("asc"),
    usecase: BatchEstimateUseCase = Injected(BatchEstimateUseCase),
):
    """Mengambil data iterasi dalam satu sesi secara berpaginasi (default diurutkan test_rmse terkecil)."""
    return await usecase.get_iterations_paginated(
        session_id=session_id,
        page=page,
        limit=limit,
        estimator_type=estimator_type,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@router.delete("/batch-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_batch_session_endpoint(
    session_id: int,
    admin: User = Depends(require_admin),
    usecase: BatchEstimateUseCase = Injected(BatchEstimateUseCase),
):
    """Menghapus sesi batch dan seluruh baris iterasinya."""
    deleted = await usecase.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sesi batch ID {session_id} tidak ditemukan.")
