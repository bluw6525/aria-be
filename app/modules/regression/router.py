"""
Router FastAPI untuk Modul Regresi Robust (Latih, Uji, Deteksi Outlier, Estimasi, dan Pengujian Massal).
Menggunakan Dependency Injection terpisah per Use Case melalui fastapi_injector.
"""

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Query, Depends
from fastapi_injector import Injected

from app.core.pagination import calculate_offset
from app.modules.auth.core.models import User
from app.modules.auth.dependencies import require_admin
from app.modules.regression.schemas import (
    PublicModelResponse,
    PublicModelCreate,
    PromoteModelRequest,
    OutlierDetectionResponse,
    TrainModelResult,
    TestModelResult,
    SingleEstimateRequest,
    SingleEstimateResponse,
    BatchEstimateResponse,
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
@router.post("/save", response_model=PublicModelResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
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


@router.post("/models/promote", response_model=PublicModelResponse, status_code=status.HTTP_201_CREATED)
@router.post("/models/promotions", response_model=PublicModelResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
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
@router.post("/outlier-detections", response_model=OutlierDetectionResponse, include_in_schema=False)
@router.post("/detect-outliers", response_model=OutlierDetectionResponse, include_in_schema=False)
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
@router.post("/train", response_model=TrainModelResult, include_in_schema=False)
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
@router.post("/test/{model_id}", response_model=TestModelResult, include_in_schema=False)
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


@router.post("/test", response_model=TestModelResult, include_in_schema=False)
async def test_saved_model_endpoint(
    model_id: int = Form(..., description="ID model publik yang akan diuji"),
    file: UploadFile = File(..., description="Berkas dataset uji (.xlsx atau .csv)"),
    usecase: TestModelUseCase = Injected(TestModelUseCase)
):
    """Menguji akurasi model publik terhadap data uji baru (fallback form parameter)."""
    try:
        return await usecase.test_model(model_id=model_id, file=file)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.post("/tests/custom", response_model=TestModelResult)
@router.post("/test-custom", response_model=TestModelResult, include_in_schema=False)
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

@router.post("/estimates", response_model=SingleEstimateResponse)
@router.post("/estimate", response_model=SingleEstimateResponse, include_in_schema=False)
async def estimate_single_endpoint(
    payload: SingleEstimateRequest,
    usecase: EstimateUseCase = Injected(EstimateUseCase)
):
    """Melakukan estimasi instan nilai O3 dan kategori ISPU dari nilai CO & NO2 tunggal."""
    try:
        return await usecase.estimate_single(payload)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.post("/estimates/batch", response_model=BatchEstimateResponse)
@router.post("/estimate-file", response_model=BatchEstimateResponse, include_in_schema=False)
async def estimate_file_endpoint(
    model_id: int = Form(..., description="ID model tersimpan untuk estimasi"),
    file: UploadFile = File(..., description="Berkas data satelit (.xlsx, .xls, .csv)"),
    usecase: EstimateUseCase = Injected(EstimateUseCase)
):
    """Melakukan estimasi O3 dan kategori ISPU dari berkas satelit secara batch."""
    try:
        return await usecase.estimate_from_file(model_id=model_id, file=file)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 6. Endpoints Evaluasi & Pengujian Massal (/batch-sessions)
# ============================================================================

@router.post("/batch-sessions", response_model=BatchRunResponse, status_code=status.HTTP_201_CREATED)
@router.post("/batch/run", response_model=BatchRunResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
@router.post("/benchmark/run", response_model=BatchRunResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
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
    """Menjalankan dan membuat sesi evaluasi massal baru (Monte Carlo Cross-Validation)."""
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
@router.get("/batch/sessions", response_model=List[BatchSessionResponse], include_in_schema=False)
@router.get("/benchmark/sessions", response_model=List[BatchSessionResponse], include_in_schema=False)
async def list_batch_sessions_endpoint(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    usecase: BatchEstimateUseCase = Injected(BatchEstimateUseCase),
):
    """Mengambil daftar seluruh sesi evaluasi massal sebelumnya."""
    offset = calculate_offset(page=page, limit=limit)
    return await usecase.list_sessions(offset=offset, limit=limit)


@router.get("/batch-sessions/{session_id}", response_model=BatchSessionResponse)
@router.get("/batch/sessions/{session_id}", response_model=BatchSessionResponse, include_in_schema=False)
@router.get("/benchmark/sessions/{session_id}", response_model=BatchSessionResponse, include_in_schema=False)
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
@router.get("/batch/sessions/{session_id}/iterations", response_model=PaginatedIterationsResponse, include_in_schema=False)
@router.get("/benchmark/sessions/{session_id}/iterations", response_model=PaginatedIterationsResponse, include_in_schema=False)
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
    offset = calculate_offset(page=page, limit=limit)
    return await usecase.get_iterations_paginated(
        session_id=session_id,
        offset=offset,
        limit=limit,
        page=page,
        estimator_type=estimator_type,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@router.delete("/batch-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
@router.delete("/batch/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
@router.delete("/benchmark/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
async def delete_batch_session_endpoint(
    session_id: int,
    admin: User = Depends(require_admin),
    usecase: BatchEstimateUseCase = Injected(BatchEstimateUseCase),
):
    """Menghapus sesi batch dan seluruh baris iterasinya."""
    deleted = await usecase.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sesi batch ID {session_id} tidak ditemukan.")
