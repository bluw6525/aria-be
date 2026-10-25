"""
Router FastAPI untuk Modul Regresi Robust (Latih, Uji, Deteksi Outlier, dan Estimasi).
Menggunakan Dependency Injection terpisah per Use Case melalui fastapi_injector.
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Query, Depends
from fastapi_injector import Injected

from app.modules.auth.core.models import User
from app.modules.auth.dependencies import require_admin
from app.modules.regression.schemas import (
    TrainedModelResponse,
    TrainedModelCreate,
    OutlierDetectionResponse,
    TrainModelResult,
    TestModelResult,
    SingleEstimateRequest,
    SingleEstimateResponse,
    BatchEstimateResponse,
)
from app.modules.regression.use_cases import (
    DetectOutliersUseCase,
    TrainModelUseCase,
    TestModelUseCase,
    ModelUseCase,
    EstimateUseCase,
)

router = APIRouter(prefix="/regression", tags=["Regression & Models"])


# ============================================================================
# 1. Endpoints Manajemen Model Tersimpan (CRUD)
# ============================================================================

@router.get("/models", response_model=List[TrainedModelResponse])
async def get_models(
    page: int = Query(1, ge=1, description="Nomor halaman (mulai dari 1)"),
    limit: int = Query(100, ge=1, le=100, description="Jumlah data per halaman"),
    usecase: ModelUseCase = Injected(ModelUseCase)
):
    """Mengambil daftar model yang tersimpan dengan pagination (page & limit)."""
    return await usecase.list_models(page=page, limit=limit)


@router.get("/models/{model_id}", response_model=TrainedModelResponse)
async def get_model(
    model_id: int,
    usecase: ModelUseCase = Injected(ModelUseCase)
):
    """Mengambil detail satu model tersimpan berdasarkan ID."""
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


@router.post("/save", response_model=TrainedModelResponse, status_code=status.HTTP_201_CREATED)
async def save_model_endpoint(
    payload: TrainedModelCreate,
    admin: User = Depends(require_admin),
    usecase: ModelUseCase = Injected(ModelUseCase)
):
    """Menyimpan parameter model hasil pelatihan ke basis data."""
    try:
        return await usecase.save_model(payload)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 2. Endpoint Deteksi Outlier
# ============================================================================

@router.post("/detect-outliers", response_model=OutlierDetectionResponse)
async def detect_outliers_endpoint(
    file: UploadFile = File(..., description="Berkas data latih (.xlsx, .xls, .csv)"),
    fit_intercept: bool = Form(True),
    admin: User = Depends(require_admin),
    usecase: DetectOutliersUseCase = Injected(DetectOutliersUseCase)
):
    """Mendeteksi outlier pada berkas data latih menggunakan Leverage dan DFFITS."""
    try:
        return await usecase.detect_outliers(file=file, fit_intercept=fit_intercept)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 3. Endpoint Pelatihan Model (Training)
# ============================================================================

@router.post("/train", response_model=TrainModelResult)
async def train_model_endpoint(
    file: UploadFile = File(..., description="Berkas data latih (.xlsx, .xls, .csv)"),
    estimator_type: str = Form("MM", description="Tipe estimator: MM, MM-Low, MM-High, M, S, atau OLS"),
    fit_intercept: bool = Form(True),
    admin: User = Depends(require_admin),
    usecase: TrainModelUseCase = Injected(TrainModelUseCase)
):
    """Melatih model regresi robust pada data latih dan mengembalikan metrik evaluasi serta grafik."""
    try:
        return await usecase.train_model(
            file=file,
            estimator_type=estimator_type,
            fit_intercept=fit_intercept
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 4. Endpoint Pengujian Model Tersimpan (Testing)
# ============================================================================

@router.post("/test/{model_id}", response_model=TestModelResult)
async def test_model_endpoint(
    model_id: int,
    file: UploadFile = File(..., description="Berkas data uji (.xlsx, .xls, .csv)"),
    admin: User = Depends(require_admin),
    usecase: TestModelUseCase = Injected(TestModelUseCase)
):
    """Menguji model tersimpan di database dengan berkas data uji baru."""
    try:
        return await usecase.test_model(model_id=model_id, file=file)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.post("/test-custom", response_model=TestModelResult)
async def test_custom_model_endpoint(
    intercept: float = Form(0.0, description="Nilai intercept (w0)"),
    coef_co: float = Form(..., description="Koefisien CO (w1)"),
    coef_no2: float = Form(..., description="Koefisien NO2 (w2)"),
    estimator_type: str = Form("MM", description="Tipe estimator"),
    model_name: str = Form("Model Baru Dilatih", description="Label nama model"),
    file: UploadFile = File(..., description="Berkas data uji (.xlsx, .xls, .csv)"),
    admin: User = Depends(require_admin),
    usecase: TestModelUseCase = Injected(TestModelUseCase)
):
    """Menguji parameter model langsung (misal model yang baru dilatih tanpa disimpan ke database)."""
    try:
        return await usecase.test_custom_model(
            intercept=intercept,
            coef_co=coef_co,
            coef_no2=coef_no2,
            file=file,
            estimator_type=estimator_type,
            model_name=model_name,
            model_id=None,
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


# ============================================================================
# 5. Endpoint Estimasi O3 & ISPU (Halaman /estimate)
# ============================================================================

@router.post("/estimate", response_model=SingleEstimateResponse)
async def estimate_endpoint(
    payload: SingleEstimateRequest,
    usecase: EstimateUseCase = Injected(EstimateUseCase)
):
    """Melakukan estimasi O3 dan kategori ISPU dari nilai CO dan NO2 tunggal."""
    try:
        return await usecase.estimate_single(payload)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.post("/estimate-file", response_model=BatchEstimateResponse)
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
