"""Common Pydantic models for MLOps application."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ModelStatus(str, Enum):
    """Model deployment status."""
    PENDING = "pending"
    TRAINING = "training"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TransactionType(str, Enum):
    """Transaction types for classification."""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"


class JarType(str, Enum):
    """6-jar budget categories."""
    NECESSITIES = "necessities"
    FINANCIAL_FREEDOM = "financial_freedom"
    LONG_TERM_SAVINGS = "long_term_savings"
    EDUCATION = "education"
    PLAY = "play"
    GIVE = "give"


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: str = "Success"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    service: str
    checks: Optional[Dict[str, str]] = None


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    model_name: str
    version: str
    status: ModelStatus
    endpoint_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metrics: Optional[Dict[str, float]] = None


class TrainingJob(BaseModel):
    """Training job information."""
    job_id: str
    job_name: str
    model_name: str
    status: TrainingStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    training_data_uri: str
    hyperparameters: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None


class Transaction(BaseModel):
    """Transaction data model."""
    transaction_id: str
    user_id: str
    amount: float
    description: str
    transaction_type: TransactionType
    category: Optional[str] = None
    jar_type: Optional[JarType] = None
    timestamp: datetime
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class User(BaseModel):
    """User profile model."""
    user_id: str
    income: float
    jar_allocations: Dict[JarType, float]
    spending_preferences: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class Prediction(BaseModel):
    """Spending prediction model."""
    prediction_id: str
    user_id: str
    prediction_type: str
    predicted_amount: float
    confidence_score: float
    prediction_date: datetime
    factors: List[str]
    metadata: Optional[Dict[str, Any]] = None


class CoachingSession(BaseModel):
    """AI coaching session model."""
    session_id: str
    user_id: str
    query: str
    response: str
    context: Dict[str, Any]
    timestamp: datetime
    satisfaction_score: Optional[float] = None 