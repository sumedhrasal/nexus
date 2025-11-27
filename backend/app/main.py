"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.storage.postgres import init_db, close_db
from app.core.logging import setup_logging, get_logger
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.metrics import MetricsMiddleware

# Setup structured logging
setup_logging(json_logs=settings.json_logs, log_level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("application_starting", version="1.0.0")
    await init_db()
    logger.info("database_initialized")
    yield
    # Shutdown
    logger.info("application_shutting_down")
    await close_db()
    logger.info("database_closed")


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

# Add middlewares (order matters - metrics first, then logging, then CORS)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware)

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
    logger.debug("health_check_requested")
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


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Nexus API - AI Context Retrieval System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# Include API routers
from app.api.routes import collections, search, ingest, sources, auth

app.include_router(auth.router)
app.include_router(collections.router)
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(sources.router)
app.include_router(metrics.router)
