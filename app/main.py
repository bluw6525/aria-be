from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi_injector import InjectorMiddleware, attach_injector
from app.core.config import settings
from app.core.database import engine
from app.core.middleware import CloseDatabaseSessionsMiddleware
from app.core.factory import app_injector

@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/health", tags=["Health"])
async def health_check():
    """Endpoint pemeriksaan kesehatan server (keep-alive)."""
    return {"status": "alive"}

