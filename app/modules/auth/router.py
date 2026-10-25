"""
Router FastAPI untuk Modul Autentikasi Pengguna (/auth).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_injector import Injected

from app.modules.auth.core.models import User
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.schemas.auth_schema import (
    LoginRequest,
    TokenResponse,
    UserResponse,
)
from app.modules.auth.use_cases.auth_use_case import AuthUseCase

router = APIRouter(prefix="/auth", tags=["Authentication & Profile"])


@router.post("/login", response_model=TokenResponse)
async def login_endpoint(
    payload: LoginRequest,
    usecase: AuthUseCase = Injected(AuthUseCase),
):
    """
    Otentikasi login pengguna.
    Mengembalikan token JWT dan objek profil pengguna beserta rolenya.
    """
    try:
        return await usecase.login(payload)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(err),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
def get_me_endpoint(
    current_user: User = Depends(get_current_user),
):
    """
    Mengambil data profil pengguna yang sedang aktif (sesuai token Bearer).
    """
    return UserResponse.model_validate(current_user)
