"""
Container / Module Injector untuk Modul Autentikasi.
"""

from injector import Module, Binder
from app.modules.auth.repositories.user_repository import UserRepository
from app.modules.auth.use_cases.auth_use_case import AuthUseCase


class AuthContainer(Module):
    """Module Injector untuk dependensi autentikasi."""

    def configure(self, binder: Binder) -> None:
        binder.bind(UserRepository)
        binder.bind(AuthUseCase)
