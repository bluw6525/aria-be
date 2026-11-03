"""
Utilitas keamanan untuk Autentikasi: Password hashing (Bcrypt) dan JWT Token.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import bcrypt
import jwt

from app.core.config import settings


def hash_password(password: str) -> str:
    """Mengenkripsi password teks polos menjadi hash Bcrypt."""
    # Menghasilkan salt acak dengan work factor 12 (2^12 iterasi hashing)
    salt = bcrypt.gensalt(rounds=12)
    # Bcrypt hanya menerima data biner (bytes), sehingga password teks di-encode ke UTF-8
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    # Decode bytes hasil hash kembali ke string teks agar bisa disimpan di kolom database
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Mencocokkan password teks polos dengan hash Bcrypt di basis data."""
    try:
        # Mencocokkan password teks polos dengan hash tersimpan (keduanya wajib di-encode ke bytes UTF-8)
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        # Mengembalikan False jika string hash rusak atau format Bcrypt tidak valid
        return False


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Menghasilkan token JWT bertanda tangan digital."""
    # Salin payload agar tidak mengubah dictionary asli dari pemanggil
    to_encode = data.copy()
    # Mengambil waktu saat ini dalam zona waktu UTC
    now = datetime.now(timezone.utc)
    
    # Menghitung waktu kedaluwarsa token (menggunakan durasi kustom jika ada, atau default dari config)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    # Menambahkan klaim resmi JWT: 'exp' (Expiration Time) dan 'iat' (Issued At / waktu diterbitkan)
    to_encode.update({"exp": expire, "iat": now})
    # Menandatangani payload secara digital dengan SECRET_KEY menggunakan algoritma HS256
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    # Mengembalikan string token JWT yang sudah bertanda tangan digital
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Memverifikasi dan mendekode JWT token."""
    try:
        # Memvalidasi tanda tangan digital dan mengecek apakah token sudah kedaluwarsa
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        # Mengembalikan isi data payload jika token valid
        return payload
    except jwt.PyJWTError:
        # Mengembalikan None jika signature tidak cocok atau token sudah kedaluwarsa
        return None
