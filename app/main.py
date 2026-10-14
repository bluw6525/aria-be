from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi_injector import InjectorMiddleware, attach_injector
from app.core.config import settings
from app.core.database import engine, Base
from app.core.middleware import CloseDatabaseSessionsMiddleware
from app.core.factory import app_injector
from app.modules.item.router import router as item_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# Registrasi middleware
app.add_middleware(CloseDatabaseSessionsMiddleware)
app.add_middleware(InjectorMiddleware, injector=app_injector)

# Lampirkan injector ke FastAPI app instance
attach_injector(app, app_injector)

app.include_router(item_router, prefix=settings.API_V1_STR)
