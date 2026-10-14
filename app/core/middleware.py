from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_injector import RequestScope
from fastapi_injector.request_scope import _request_id_ctx


class CloseDatabaseSessionsMiddleware(BaseHTTPMiddleware):
    """
    Middleware yang memastikan sesi AsyncSession yang dibuka dalam request_scope
    ditutup secara aman (await session.close()) setelah response selesai dikirim.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            # Meneruskan request ke endpoint/router dan menunggu response selesai dibuat
            response = await call_next(request)
            return response
        finally:
            # Blok finally dijamin selalu jalan, baik sukses maupun error
            try:
                # Mengambil objek RequestScope dari injector container yang terpasang di app
                scope = request.app.state.injector.get(RequestScope)

                # Mengambil ID unik request HTTP saat ini dari contextvar
                request_id = _request_id_ctx.get()

                # Mengambil cache objek-objek yang di-instantiate khusus untuk request_id ini
                cache = scope.cache.get(request_id, {})

                # Jika endpoint ini menggunakan AsyncSession, tutup koneksinya
                if AsyncSession in cache:
                    session: AsyncSession = cache[AsyncSession]
                    # Menutup sesi dan mengembalikan koneksi fisik kembali ke pool
                    await session.close()
            except Exception:
                # Menjaga agar error saat proses penutupan sesi tidak merusak response client
                pass
