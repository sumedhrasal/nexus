"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.storage.postgres import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Nexus API",
    description="AI Context Retrieval System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "api": "up",
                "database": "up",  # TODO: Add actual checks
                "qdrant": "up",
                "redis": "up",
                "ollama": "up",
            }
        },
        status_code=200
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Nexus API - AI Context Retrieval System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Include API routers
from app.api.routes import collections, search, ingest, sources

app.include_router(collections.router)
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(sources.router)
