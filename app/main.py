from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_injector import InjectorMiddleware, attach_injector
from app.core.config import settings
from app.core.database import engine
from app.core.middleware import CloseDatabaseSessionsMiddleware
from app.core.factory import app_injector
from app.modules.regression.router import router as regression_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# Registrasi middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrasi middleware database & Dependency Injection
app.add_middleware(CloseDatabaseSessionsMiddleware)
app.add_middleware(InjectorMiddleware, injector=app_injector)

# Lampirkan injector ke FastAPI app instance
attach_injector(app, app_injector)

# Registrasi router modul regresi
app.include_router(regression_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health"])
async def health_check():
    """Endpoint pemeriksaan kesehatan server (keep-alive)."""
    return {"status": "alive"}
