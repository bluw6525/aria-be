"""
Schema Pydantic untuk permintaan dan respon domain Autentikasi.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    """Payload permintaan login."""
    username: str = Field(..., min_length=1, max_length=50, description="Username akun")
    password: str = Field(..., min_length=1, description="Password akun")


class UserResponse(BaseModel):
    """Data profil pengguna publik."""
    id: int
    username: str
    role: str
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """Respon berhasil login dengan token akses JWT."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserCreate(BaseModel):
    """Payload pendaftaran pengguna baru / seeder."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field("user", description="'admin' atau 'user'")
