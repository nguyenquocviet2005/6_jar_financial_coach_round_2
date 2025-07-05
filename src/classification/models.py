"""Pydantic models for Transaction Classification service."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from src.common.models import BaseResponse, TransactionType, JarType


class ClassificationRequest(BaseModel):
    """Request for transaction classification."""
    transaction_id: str
    user_id: str
    amount: float
    description: str
    merchant: Optional[str] = None
    category_hint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ClassificationResponse(BaseResponse):
    """Response from transaction classification."""
    transaction_id: str
    predicted_category: str
    jar_type: JarType
    confidence_score: float = Field(ge=0.0, le=1.0)
    alternative_categories: List[Dict[str, float]] = Field(default_factory=list)
    reasoning: Optional[str] = None
    needs_manual_review: bool = False


class BatchClassificationRequest(BaseModel):
    """Request for batch transaction classification."""
    transactions: List[ClassificationRequest]
    user_id: str
    process_async: bool = False


class BatchClassificationResponse(BaseResponse):
    """Response from batch transaction classification."""
    batch_id: str
    total_transactions: int
    processed_transactions: int
    results: List[ClassificationResponse]
    failed_transactions: List[str] = Field(default_factory=list)


class ManualClassificationRequest(BaseModel):
    """Request for manual classification override."""
    transaction_id: str
    user_id: str
    correct_category: str
    correct_jar_type: JarType
    feedback: Optional[str] = None


class ClassificationFeedback(BaseModel):
    """Feedback for improving classification."""
    transaction_id: str
    user_id: str
    predicted_category: str
    actual_category: str
    predicted_jar_type: JarType
    actual_jar_type: JarType
    confidence_score: float
    feedback_type: str = Field(..., description="correct, incorrect, or partial")
    user_feedback: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingData(BaseModel):
    """Training data for classification model."""
    transaction_id: str
    user_id: str
    amount: float
    description: str
    merchant: Optional[str] = None
    category: str
    jar_type: JarType
    features: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, Dict[str, int]]
    category_performance: Dict[str, Dict[str, float]]
    jar_type_performance: Dict[str, Dict[str, float]]
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)


class ClassificationModelConfig(BaseModel):
    """Configuration for classification model."""
    model_name: str
    version: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    features: List[str]
    target_accuracy: float = 0.85
    min_confidence_threshold: float = 0.7
    manual_review_threshold: float = 0.5 