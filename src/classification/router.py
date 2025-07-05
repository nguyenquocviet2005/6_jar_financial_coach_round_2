"""FastAPI router for Transaction Classification service."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import structlog

from src.classification.service import TransactionClassificationService
from src.classification.models import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationFeedback,
    ManualClassificationRequest,
    ModelPerformanceMetrics,
    TrainingData
)
from src.common.models import BaseResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency
def get_classification_service() -> TransactionClassificationService:
    """Get Transaction Classification service instance."""
    return TransactionClassificationService()


@router.post("/classify", response_model=ClassificationResponse)
async def classify_transaction(
    request: ClassificationRequest,
    service: TransactionClassificationService = Depends(get_classification_service)
):
    """Classify a single transaction."""
    try:
        response = await service.classify_transaction(request)
        return response
    except Exception as e:
        logger.error("Error classifying transaction", error=str(e), transaction_id=request.transaction_id)
        raise HTTPException(status_code=500, detail="Failed to classify transaction")


@router.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(
    request: BatchClassificationRequest,
    background_tasks: BackgroundTasks,
    service: TransactionClassificationService = Depends(get_classification_service)
):
    """Classify multiple transactions in batch."""
    try:
        if request.process_async:
            # Process in background for large batches
            background_tasks.add_task(service.classify_batch, request)
            return BatchClassificationResponse(
                batch_id="async_processing",
                total_transactions=len(request.transactions),
                processed_transactions=0,
                results=[],
                message="Batch processing started in background"
            )
        else:
            response = await service.classify_batch(request)
            return response
    except Exception as e:
        logger.error("Error in batch classification", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process batch classification")


@router.post("/feedback", response_model=BaseResponse)
async def submit_feedback(
    feedback: ClassificationFeedback,
    service: TransactionClassificationService = Depends(get_classification_service)
):
    """Submit feedback for classification improvement."""
    try:
        await service.process_manual_feedback(feedback)
        return BaseResponse(message="Feedback submitted successfully")
    except Exception as e:
        logger.error("Error processing feedback", error=str(e), transaction_id=feedback.transaction_id)
        raise HTTPException(status_code=500, detail="Failed to process feedback")


@router.post("/manual-classify", response_model=BaseResponse)
async def manual_classify(
    request: ManualClassificationRequest,
    service: TransactionClassificationService = Depends(get_classification_service)
):
    """Manually classify a transaction and provide feedback."""
    try:
        # Create feedback from manual classification
        feedback = ClassificationFeedback(
            transaction_id=request.transaction_id,
            user_id=request.user_id,
            predicted_category="unknown",  # Would need to get from original prediction
            actual_category=request.correct_category,
            predicted_jar_type="necessities",  # Would need to get from original prediction
            actual_jar_type=request.correct_jar_type,
            confidence_score=0.0,  # Would need to get from original prediction
            feedback_type="manual",
            user_feedback=request.feedback
        )
        
        await service.process_manual_feedback(feedback)
        return BaseResponse(message="Manual classification recorded successfully")
    except Exception as e:
        logger.error("Error in manual classification", error=str(e), transaction_id=request.transaction_id)
        raise HTTPException(status_code=500, detail="Failed to record manual classification")


@router.post("/retrain", response_model=BaseResponse)
async def retrain_model(
    training_data: List[TrainingData],
    background_tasks: BackgroundTasks,
    service: TransactionClassificationService = Depends(get_classification_service)
):
    """Retrain the classification model with new data."""
    try:
        # Start retraining in background
        background_tasks.add_task(service.retrain_model, training_data)
        return BaseResponse(message="Model retraining started in background")
    except Exception as e:
        logger.error("Error starting model retraining", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start model retraining")


@router.get("/model/performance/{model_version}", response_model=ModelPerformanceMetrics)
async def get_model_performance(
    model_version: str,
    service: TransactionClassificationService = Depends(get_classification_service)
):
    """Get performance metrics for a specific model version."""
    try:
        metrics = await service.get_model_performance(model_version)
        return metrics
    except Exception as e:
        logger.error("Error getting model performance", error=str(e), model_version=model_version)
        raise HTTPException(status_code=500, detail="Failed to get model performance metrics")


@router.get("/categories")
async def get_categories(
    service: TransactionClassificationService = Depends(get_classification_service)
):
    """Get list of available transaction categories."""
    return {
        "categories": list(service.category_to_jar.keys()),
        "jar_types": [jar.value for jar in service.category_to_jar.values()]
    }


@router.get("/health")
async def health_check():
    """Health check for Classification service."""
    return {"status": "healthy", "service": "classification"}


@router.get("/metrics")
async def get_metrics():
    """Get Classification service metrics."""
    # This would typically return real metrics
    return {
        "total_classifications": 10000,
        "accuracy": 0.85,
        "avg_confidence": 0.78,
        "manual_review_rate": 0.15,
        "categories_count": 20
    } 