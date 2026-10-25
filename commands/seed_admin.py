"""
Script CLI untuk melakukan seeding akun Administrator awal ke basis data Aria.
Usage:
    uv run python commands/seed_admin.py
    uv run python commands/seed_admin.py --username superadmin --password rahasia123
"""

import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import AsyncSessionLocal
from app.modules.auth.core.security import hash_password
from app.modules.auth.repositories.user_repository import UserRepository


async def seed_admin(username: str = "admin", password: str = "admin123", email: str = "admin@aria.id"):
    async with AsyncSessionLocal() as session:
        repo = UserRepository(db=session)
        existing = await repo.get_by_username(username)
        if existing:
            print(f"[INFO] Pengguna '{username}' sudah ada (Role: {existing.role}).")
            # Update password
            existing.hashed_password = hash_password(password)
            existing.role = "admin"
            existing.is_active = True
            await session.commit()
            print(f"[SUKSES] Password untuk admin '{username}' berhasil diperbarui!")
            return

        user_data = {
            "username": username,
            "email": email,
            "hashed_password": hash_password(password),
            "role": "admin",
            "is_active": True,
        }
        user = await repo.create(user_data)
        print(f"[SUKSES] Akun Admin berhasil dibuat!")
        print(f"  - ID       : {user.id}")
        print(f"  - Username : {user.username}")
        print(f"  - Role     : {user.role}")
        print(f"  - Password : {password}")


def main():
    parser = argparse.ArgumentParser(description="Seed akun Administrator Aria")
    parser.add_argument("--username", "-u", default="admin", help="Username admin (default: admin)")
    parser.add_argument("--password", "-p", default="admin123", help="Password admin (default: admin123)")
    parser.add_argument("--email", "-e", default="admin@aria.id", help="Email admin (default: admin@aria.id)")
    args = parser.parse_args()

    asyncio.run(seed_admin(args.username, args.password, args.email))


if __name__ == "__main__":
    main()
