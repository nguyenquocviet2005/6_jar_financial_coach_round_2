"""Transaction Classification service implementation."""

import json
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import boto3

from src.common.dependencies import get_sagemaker_client, get_s3_client
from src.common.logging import get_logger
from src.common.models import JarType, TransactionType
from src.classification.models import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationFeedback,
    TrainingData,
    ModelPerformanceMetrics
)
from src.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class TransactionClassificationService:
    """Service for classifying transactions into categories and jar types."""
    
    def __init__(self):
        self.sagemaker_client = get_sagemaker_client()
        self.s3_client = get_s3_client()
        self.endpoint_name = settings.classification_model_endpoint
        self.model_bucket = settings.fine_tuning_s3_bucket
        
        # Category mappings
        self.category_to_jar = {
            "groceries": JarType.NECESSITIES,
            "rent": JarType.NECESSITIES,
            "utilities": JarType.NECESSITIES,
            "transportation": JarType.NECESSITIES,
            "insurance": JarType.NECESSITIES,
            "healthcare": JarType.NECESSITIES,
            "dining": JarType.PLAY,
            "entertainment": JarType.PLAY,
            "shopping": JarType.PLAY,
            "travel": JarType.PLAY,
            "education": JarType.EDUCATION,
            "courses": JarType.EDUCATION,
            "books": JarType.EDUCATION,
            "investment": JarType.FINANCIAL_FREEDOM,
            "savings": JarType.LONG_TERM_SAVINGS,
            "retirement": JarType.LONG_TERM_SAVINGS,
            "donation": JarType.GIVE,
            "charity": JarType.GIVE,
            "gifts": JarType.GIVE
        }
        
        # Initialize local model components (for fallback)
        self.text_vectorizer = None
        self.scaler = None
        self.local_model = None
        
    async def classify_transaction(self, request: ClassificationRequest) -> ClassificationResponse:
        """Classify a single transaction."""
        try:
            # Extract features
            features = self._extract_features(request)
            
            # Try SageMaker endpoint first
            try:
                prediction = await self._predict_with_sagemaker(features)
            except Exception as e:
                logger.warning("SageMaker prediction failed, using fallback", error=str(e))
                prediction = await self._predict_with_fallback(features)
            
            # Map category to jar type
            jar_type = self.category_to_jar.get(
                prediction["category"].lower(),
                JarType.NECESSITIES  # Default fallback
            )
            
            # Determine if manual review is needed
            needs_review = prediction["confidence"] < 0.7
            
            return ClassificationResponse(
                transaction_id=request.transaction_id,
                predicted_category=prediction["category"],
                jar_type=jar_type,
                confidence_score=prediction["confidence"],
                alternative_categories=prediction.get("alternatives", []),
                reasoning=prediction.get("reasoning"),
                needs_manual_review=needs_review
            )
            
        except Exception as e:
            logger.error("Error classifying transaction", error=str(e), transaction_id=request.transaction_id)
            raise
    
    async def classify_batch(self, request: BatchClassificationRequest) -> BatchClassificationResponse:
        """Classify multiple transactions in batch."""
        batch_id = str(uuid.uuid4())
        results = []
        failed_transactions = []
        
        try:
            for transaction in request.transactions:
                try:
                    classification = await self.classify_transaction(transaction)
                    results.append(classification)
                except Exception as e:
                    logger.error("Failed to classify transaction", 
                               error=str(e), 
                               transaction_id=transaction.transaction_id)
                    failed_transactions.append(transaction.transaction_id)
            
            return BatchClassificationResponse(
                batch_id=batch_id,
                total_transactions=len(request.transactions),
                processed_transactions=len(results),
                results=results,
                failed_transactions=failed_transactions
            )
            
        except Exception as e:
            logger.error("Error in batch classification", error=str(e), batch_id=batch_id)
            raise
    
    async def process_manual_feedback(self, feedback: ClassificationFeedback):
        """Process manual feedback for model improvement."""
        try:
            # Store feedback for retraining
            feedback_data = {
                "transaction_id": feedback.transaction_id,
                "user_id": feedback.user_id,
                "predicted_category": feedback.predicted_category,
                "actual_category": feedback.actual_category,
                "predicted_jar_type": feedback.predicted_jar_type.value,
                "actual_jar_type": feedback.actual_jar_type.value,
                "confidence_score": feedback.confidence_score,
                "feedback_type": feedback.feedback_type,
                "user_feedback": feedback.user_feedback,
                "timestamp": feedback.timestamp.isoformat()
            }
            
            # Store in S3 for later retraining
            s3_key = f"feedback/{datetime.now().strftime('%Y/%m/%d')}/{feedback.transaction_id}.json"
            
            self.s3_client.put_object(
                Bucket=self.model_bucket,
                Key=s3_key,
                Body=json.dumps(feedback_data),
                ContentType='application/json'
            )
            
            logger.info("Feedback stored", transaction_id=feedback.transaction_id, s3_key=s3_key)
            
        except Exception as e:
            logger.error("Error processing feedback", error=str(e), feedback=feedback.dict())
            raise
    
    def _extract_features(self, request: ClassificationRequest) -> Dict[str, Any]:
        """Extract features from transaction for classification."""
        features = {
            "amount": request.amount,
            "description": request.description.lower(),
            "merchant": (request.merchant or "").lower(),
            "amount_log": np.log1p(request.amount),
            "description_length": len(request.description),
            "has_merchant": 1 if request.merchant else 0,
            "weekend": 0,  # Would need actual date to determine
            "hour_of_day": 12,  # Would need actual timestamp
        }
        
        # Add text features
        description_words = request.description.lower().split()
        features.update({
            "has_food_keywords": 1 if any(word in description_words for word in 
                                        ["restaurant", "food", "grocery", "cafe", "pizza"]) else 0,
            "has_transport_keywords": 1 if any(word in description_words for word in 
                                             ["gas", "uber", "taxi", "metro", "bus"]) else 0,
            "has_shopping_keywords": 1 if any(word in description_words for word in 
                                            ["store", "shop", "mall", "amazon", "walmart"]) else 0,
            "has_entertainment_keywords": 1 if any(word in description_words for word in 
                                                 ["movie", "theater", "game", "netflix", "spotify"]) else 0,
        })
        
        # Add amount-based features
        if request.amount < 10:
            features["amount_category"] = "small"
        elif request.amount < 100:
            features["amount_category"] = "medium"
        else:
            features["amount_category"] = "large"
        
        return features
    
    async def _predict_with_sagemaker(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using SageMaker endpoint."""
        try:
            # Prepare input data
            input_data = {
                "instances": [features]
            }
            
            # Call SageMaker endpoint
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(input_data)
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            # Extract prediction
            prediction = result['predictions'][0]
            
            return {
                "category": prediction["category"],
                "confidence": prediction["confidence"],
                "alternatives": prediction.get("alternatives", []),
                "reasoning": prediction.get("reasoning")
            }
            
        except Exception as e:
            logger.error("SageMaker prediction failed", error=str(e))
            raise
    
    async def _predict_with_fallback(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using fallback local model."""
        try:
            # Simple rule-based fallback
            description = features["description"]
            amount = features["amount"]
            
            # Rule-based classification
            if any(keyword in description for keyword in ["grocery", "food", "restaurant"]):
                if amount < 50:
                    category = "groceries"
                    confidence = 0.8
                else:
                    category = "dining"
                    confidence = 0.7
            elif any(keyword in description for keyword in ["gas", "uber", "taxi"]):
                category = "transportation"
                confidence = 0.75
            elif any(keyword in description for keyword in ["rent", "mortgage"]):
                category = "rent"
                confidence = 0.9
            elif any(keyword in description for keyword in ["electric", "water", "internet"]):
                category = "utilities"
                confidence = 0.85
            elif any(keyword in description for keyword in ["movie", "netflix", "spotify"]):
                category = "entertainment"
                confidence = 0.7
            else:
                category = "other"
                confidence = 0.5
            
            return {
                "category": category,
                "confidence": confidence,
                "alternatives": [],
                "reasoning": f"Rule-based classification based on keywords in '{description}'"
            }
            
        except Exception as e:
            logger.error("Fallback prediction failed", error=str(e))
            # Ultimate fallback
            return {
                "category": "other",
                "confidence": 0.3,
                "alternatives": [],
                "reasoning": "Default classification due to prediction failure"
            }
    
    async def retrain_model(self, training_data: List[TrainingData]) -> str:
        """Retrain the classification model with new data."""
        try:
            job_name = f"classification-training-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Prepare training data
            df = pd.DataFrame([data.dict() for data in training_data])
            
            # Save training data to S3
            training_data_key = f"training-data/{job_name}/training.csv"
            csv_buffer = df.to_csv(index=False)
            
            self.s3_client.put_object(
                Bucket=self.model_bucket,
                Key=training_data_key,
                Body=csv_buffer,
                ContentType='text/csv'
            )
            
            # Start SageMaker training job
            training_job_definition = {
                "TrainingJobName": job_name,
                "AlgorithmSpecification": {
                    "TrainingImage": "382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest",
                    "TrainingInputMode": "File"
                },
                "RoleArn": settings.sagemaker_role_arn,
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": f"s3://{self.model_bucket}/training-data/{job_name}/",
                                "S3DataDistributionType": "FullyReplicated"
                            }
                        },
                        "ContentType": "text/csv",
                        "CompressionType": "None"
                    }
                ],
                "OutputDataConfig": {
                    "S3OutputPath": f"s3://{self.model_bucket}/models/{job_name}/"
                },
                "ResourceConfig": {
                    "InstanceType": "ml.m5.large",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 10
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 3600
                },
                "HyperParameters": {
                    "objective": "multi:softprob",
                    "num_class": "20",
                    "num_round": "100",
                    "eta": "0.1",
                    "max_depth": "6"
                }
            }
            
            self.sagemaker_client.create_training_job(**training_job_definition)
            
            logger.info("Training job started", job_name=job_name)
            return job_name
            
        except Exception as e:
            logger.error("Error starting training job", error=str(e))
            raise
    
    async def get_model_performance(self, model_version: str) -> ModelPerformanceMetrics:
        """Get performance metrics for a model version."""
        try:
            # This would typically load metrics from S3 or database
            # For now, we'll return simulated metrics
            
            return ModelPerformanceMetrics(
                model_version=model_version,
                accuracy=0.85,
                precision=0.83,
                recall=0.82,
                f1_score=0.82,
                confusion_matrix={
                    "groceries": {"groceries": 150, "dining": 5, "other": 2},
                    "dining": {"groceries": 3, "dining": 120, "entertainment": 4},
                    "transportation": {"transportation": 95, "other": 3}
                },
                category_performance={
                    "groceries": {"precision": 0.89, "recall": 0.88, "f1": 0.88},
                    "dining": {"precision": 0.85, "recall": 0.83, "f1": 0.84},
                    "transportation": {"precision": 0.91, "recall": 0.90, "f1": 0.90}
                },
                jar_type_performance={
                    "necessities": {"precision": 0.87, "recall": 0.85, "f1": 0.86},
                    "play": {"precision": 0.82, "recall": 0.80, "f1": 0.81},
                    "education": {"precision": 0.78, "recall": 0.76, "f1": 0.77}
                }
            )
            
        except Exception as e:
            logger.error("Error getting model performance", error=str(e))
            raise 