import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.router import api_router, v1_router
from app.services.models_service import model_service
from app.services.carbon_service import carbon_service
from app.middleware import RequestIDMiddleware, RequestLoggingMiddleware

# --- Logging Configuration ---
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

app_log_path = os.path.join(log_dir, "app.log")
error_log_path = os.path.join(log_dir, "errors.log")

# Setup formatter
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Standard log handler (INFO and above)
app_handler = logging.handlers.RotatingFileHandler(app_log_path, maxBytes=10*1024*1024, backupCount=5)
app_handler.setLevel(logging.INFO)
app_handler.setFormatter(formatter)

# Error log handler (ERROR and above)
error_handler = logging.handlers.RotatingFileHandler(error_log_path, maxBytes=10*1024*1024, backupCount=5)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Configure Root Logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(app_handler)
root_logger.addHandler(error_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events manager: loads model pickles and carbon databases
    into memory once at startup.
    """
    logger.info("Initializing VoltIQ application startup lifespans...")
    
    # Pre-load ML models
    model_service.load_models()
    # Pre-load carbon CSVs
    carbon_service.load_data()
    
    logger.info("VoltIQ application startup lifespans initialized successfully.")
    yield
    logger.info("VoltIQ application shutdown complete.")

# --- FastAPI Initialization ---
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="VoltIQ is an enterprise-grade AI-powered fleet electrification and asset intelligence platform.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Custom Timing and Request ID Middleware ---
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# --- Register aggregated API routes ---
# Legacy router for backward compatibility with existing tests
app.include_router(api_router, prefix="/api")

# Version 1.0.0 router
app.include_router(v1_router, prefix="/api/v1")

# --- Global Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for Pydantic validation errors (422)."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"RID={request_id} | Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error: Input data does not match the required schemas.",
            "request_id": request_id,
            "errors": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global handler for unhandled server exceptions (500)."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"RID={request_id} | Unhandled Server Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": f"An unexpected error occurred: {str(exc)}",
            "request_id": request_id
        }
    )

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint serving application metadata.
    """
    logger.info("Root endpoint accessed.")
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "description": "VoltIQ Fleet Electrification & Asset Intelligence API Backend.",
        "status": "online"
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for container probes and status monitoring.
    """
    logger.info("Health check endpoint accessed.")
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version
    }
