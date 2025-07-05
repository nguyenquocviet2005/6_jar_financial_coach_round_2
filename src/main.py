"""Main FastAPI application for MLOps services."""

import structlog
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram
import time

from src.config import get_settings
from src.common.logging import setup_logging
from src.common.dependencies import get_db, get_redis
from src.ai_coach.router import router as ai_coach_router
from src.classification.router import router as classification_router
from src.prediction.router import router as prediction_router
from src.fine_tuning.router import router as fine_tuning_router

# Setup logging
setup_logging()
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Initialize FastAPI app
app = FastAPI(
    title="6-Jar Financial App - MLOps API",
    description="MLOps services for AI coaching, transaction classification, spending prediction, and model fine-tuning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# API Key authentication
settings = get_settings()
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key."""
    if settings.development:
        return api_key  # Skip validation in development
    
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # In production, validate against a proper key store
    valid_keys = ["your-api-key-here"]  # Replace with proper key management
    if api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

# Middleware for metrics and logging
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware for metrics collection and request logging."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration=duration
    )
    
    return response

# Include routers
app.include_router(
    ai_coach_router,
    prefix="/api/v1/ai-coach",
    tags=["AI Coach"],
    dependencies=[Depends(get_api_key)]
)

app.include_router(
    classification_router,
    prefix="/api/v1/classification",
    tags=["Transaction Classification"],
    dependencies=[Depends(get_api_key)]
)

app.include_router(
    prediction_router,
    prefix="/api/v1/prediction",
    tags=["Spending Prediction"],
    dependencies=[Depends(get_api_key)]
)

app.include_router(
    fine_tuning_router,
    prefix="/api/v1/fine-tuning",
    tags=["Model Fine-tuning"],
    dependencies=[Depends(get_api_key)]
)

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mlops-api"}

@app.get("/health/detailed")
async def detailed_health_check(
    db=Depends(get_db),
    redis=Depends(get_redis)
):
    """Detailed health check with dependency checks."""
    health_status = {
        "status": "healthy",
        "service": "mlops-api",
        "checks": {
            "database": "unknown",
            "redis": "unknown",
            "aws": "unknown"
        }
    }
    
    # Check database
    try:
        await db.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        await redis.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check AWS services (basic check)
    try:
        import boto3
        boto3.client('sts').get_caller_identity()
        health_status["checks"]["aws"] = "healthy"
    except Exception as e:
        health_status["checks"]["aws"] = f"unhealthy: {str(e)}"
        # Don't mark as unhealthy for AWS in development
    
    return health_status

# Metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "6-Jar Financial App - MLOps API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("MLOps API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("MLOps API shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.development
    ) 