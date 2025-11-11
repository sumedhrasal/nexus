"""Metrics middleware for Prometheus monitoring."""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.metrics import http_requests_total, http_request_duration_seconds


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics.

        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint

        Returns:
            Response from endpoint
        """
        # Skip metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        method = request.method
        path = request.url.path

        # Normalize path (replace UUIDs with placeholder)
        normalized_path = self._normalize_path(path)

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=normalized_path,
                status=str(status_code)
            ).inc()

            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=normalized_path
            ).observe(duration)

            return response

        except Exception as exc:
            # Record error
            http_requests_total.labels(
                method=method,
                endpoint=normalized_path,
                status="500"
            ).inc()

            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=normalized_path
            ).observe(duration)

            raise

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path by replacing UUIDs with placeholders.

        Args:
            path: Original request path

        Returns:
            Normalized path with UUID placeholders
        """
        parts = path.split('/')
        normalized_parts = []

        for part in parts:
            # Check if part looks like a UUID
            if len(part) == 36 and part.count('-') == 4:
                normalized_parts.append('{uuid}')
            else:
                normalized_parts.append(part)

        return '/'.join(normalized_parts)
