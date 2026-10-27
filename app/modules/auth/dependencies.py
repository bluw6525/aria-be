"""
Dependensi FastAPI untuk validasi autentikasi dan otorisasi role (Admin Guard).
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_injector import Injected

from app.modules.auth.core.models import User
from app.modules.auth.core.security import decode_access_token
from app.modules.auth.repositories.user_repository import UserRepository

security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    repo: UserRepository = Injected(UserRepository),
) -> User:
    """
    Mengekstrak token Bearer dari Header HTTP, memvalidasinya,
    dan mengambil entitas User dari database melalui Dependency Injection container.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token autentikasi tidak ditemukan. Silakan login terlebih dahulu.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token kedaluwarsa atau tidak valid. Silakan login kembali.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Klaim token tidak valid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await repo.get_by_id(int(user_id_str))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pengguna tidak aktif atau tidak ditemukan.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Guard untuk membatasi endpoint hanya bagi pengguna dengan role 'admin'.
    Fungsi ini bersifat sinkron (def) karena hanya memeriksa atribut role in-memory.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Akses ditolak: Operasi ini membutuhkan hak akses Administrator.",
        )
    return current_user
