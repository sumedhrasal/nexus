"""Logging middleware for request tracking."""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests with context."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add logging context.

        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint

        Returns:
            Response from endpoint
        """
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Add request context to structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            client_ip=request.client.host if request.client else None,
        )

        # Store request_id in request state for later access
        request.state.request_id = request_id

        start_time = time.time()

        # Log request
        logger.info(
            "request_started",
            url=str(request.url),
            headers=dict(request.headers) if logger.isEnabledFor(10) else None,  # DEBUG level
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Log response
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            # Log error
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "request_failed",
                duration_ms=duration_ms,
                error=str(exc),
                exc_info=True,
            )
            raise
