"""
Script CLI untuk melakukan Reset Password Pengguna / Administrator Aria.
Menggunakan AuthUseCase dan UserRepository secara langsung tanpa melalui router HTTP publik.

Perhatian Keamanan:
  Skrip ini HANYA melakukan reset password untuk akun yang SUDAH TERDAFTAR.
  Jika username tidak ditemukan (misal akibat typo), operasi akan langsung ditolak demi keamanan.

Penggunaan:
  1. Interaktif (Input password tertutup / masked):
     uv run python commands/reset_password.py

  2. Berbasis Argumen:
     uv run python commands/reset_password.py -u admin -p "PasswordBaru123"
"""

import argparse
import asyncio
import getpass
from pathlib import Path
import sys

# Pastikan root proyek berada di sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import AsyncSessionLocal
from app.modules.auth.repositories.user_repository import UserRepository
from app.modules.auth.use_cases.auth_use_case import AuthUseCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aria CLI: Reset Password Pengguna/Administrator Terdaftar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-u", "--username",
        type=str,
        default=None,
        help="Username akun terdaftar yang akan direset password-nya",
    )
    parser.add_argument(
        "-p", "--password",
        type=str,
        default=None,
        help="Password baru (opsional; jika tidak diisi akan diminta secara aman / masked)",
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()

    print("=" * 60)
    print("        ARIA SYSTEM - PASSWORD RESET UTILITY (CLI)          ")
    print("=" * 60)

    username = args.username
    if not username:
        username = input("Masukkan Username Akun: ").strip()

    if not username:
        print("[ERROR] Username tidak boleh kosong.")
        sys.exit(1)

    password = args.password

    async with AsyncSessionLocal() as session:
        repo = UserRepository(session)
        use_case = AuthUseCase(repo)

        # Validasi ketat: pastikan akun benar-benar ada di database
        user = await repo.get_by_username(username)
        if user is None:
            print(f"\n[ERROR] Pengguna dengan username '{username}' tidak ditemukan di database.")
            print("[INFO] Operasi dibatalkan. Pastikan tidak ada kesalahan ketik (typo) pada username.")
            sys.exit(1)

        print(f"\n[INFO] Akun terverifikasi: '{user.username}' (Role: {user.role}, Aktif: {user.is_active}).")

        # Jika password belum diisi via argumen CLI, minta input tertutup (masked)
        if not password:
            while True:
                pwd1 = getpass.getpass("Masukkan Password Baru: ")
                if len(pwd1) < 6:
                    print("[ERROR] Password minimal harus 6 karakter. Silakan coba lagi.")
                    continue

                pwd2 = getpass.getpass("Ulangi Password Baru: ")
                if pwd1 != pwd2:
                    print("[ERROR] Konfirmasi password tidak cocok. Silakan coba lagi.\n")
                    continue

                password = pwd1
                break

        try:
            updated_user = await use_case.reset_password(
                username=username,
                new_password=password,
            )
            print(f"[BERHASIL] Password untuk akun '{updated_user.username}' (role: {updated_user.role}) berhasil diperbarui!")

        except ValueError as err:
            print(f"[ERROR] {err}")
            sys.exit(1)


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\nOperasi dibatalkan oleh pengguna.")
        sys.exit(0)


if __name__ == "__main__":
    main()
