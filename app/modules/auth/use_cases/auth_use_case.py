"""
Use Case untuk alur proses Autentikasi Pengguna.
"""

from injector import inject

from app.modules.auth.core.models import User
from app.modules.auth.core.security import (
    hash_password,
    verify_password,
    create_access_token,
)
from app.modules.auth.repositories.user_repository import UserRepository
from app.modules.auth.schemas.auth_schema import (
    LoginRequest,
    TokenResponse,
    UserResponse,
)


class AuthUseCase:
    """Business logic untuk login dan verifikasi pengguna."""

    @inject
    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def login(self, payload: LoginRequest) -> TokenResponse:
        """Memverifikasi kredensial pengguna dan menerbitkan JWT token."""
        user = await self.repository.get_by_username(payload.username)
        if not user or not user.is_active:
            raise ValueError("Username atau password salah.")

        if not verify_password(payload.password, user.hashed_password):
            raise ValueError("Username atau password salah.")

        token_data = {
            "sub": str(user.id),
            "username": user.username,
            "role": user.role,
        }
        access_token = create_access_token(token_data)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse.model_validate(user),
        )

    async def reset_password(self, username: str, new_password: str) -> User:
        """
        Logika bisnis untuk mereset kata sandi pengguna berdasarkan username.
        Hanya digunakan untuk operasi internal / CLI (tidak disambungkan ke router publik).
        """
        if not new_password or len(new_password) < 6:
            raise ValueError("Password minimal harus terdiri dari 6 karakter.")

        user = await self.repository.get_by_username(username)
        if not user:
            raise ValueError(f"Pengguna dengan username '{username}' tidak ditemukan.")

        hashed = hash_password(new_password)
        return await self.repository.update_password(user, hashed)
