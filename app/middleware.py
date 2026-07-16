import logging
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("app.middleware")

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that generates a unique UUID (Request ID) for each incoming HTTP request,
    attaching it to the request state and returning it in the 'X-Request-ID' response header.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records processing duration, request metadata, and status codes.
    Adds 'X-Process-Time' header to the response.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")
        
        response: Response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.6f}"
        
        # Log request summary
        logger.info(
            f"RID={request_id} | {request.method} {request.url.path} | "
            f"Status={response.status_code} | Duration={process_time:.4f}s"
        )
        return response
