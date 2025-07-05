"""
Test cases for Transaction Classification service
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

def test_classify_transaction(client, sample_transaction):
    """Test single transaction classification"""
    response = client.post(
        "/classification/classify",
        json=sample_transaction
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "jar_type" in data
    assert "confidence" in data
    assert "explanation" in data

def test_classify_batch_transactions(client, sample_transaction):
    """Test batch transaction classification"""
    transactions = [sample_transaction, sample_transaction]
    
    response = client.post(
        "/classification/batch-classify",
        json={"transactions": transactions}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    
    for result in data["results"]:
        assert "category" in result
        assert "jar_type" in result
        assert "confidence" in result

def test_submit_feedback(client):
    """Test feedback submission"""
    response = client.post(
        "/classification/feedback",
        json={
            "transaction_id": "test-123",
            "user_id": "test-user-123",
            "predicted_category": "NECESSITIES",
            "actual_category": "PLAY",
            "confidence": 0.85,
            "feedback_type": "correction"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "feedback_recorded"

def test_get_model_performance(client):
    """Test model performance metrics"""
    response = client.get("/classification/performance")
    
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data

def test_retrain_model(client):
    """Test model retraining"""
    response = client.post(
        "/classification/retrain",
        json={
            "min_feedback_count": 100,
            "performance_threshold": 0.85
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "training_job_id" in data

def test_classification_invalid_input(client):
    """Test classification with invalid input"""
    response = client.post(
        "/classification/classify",
        json={}  # Empty request
    )
    
    assert response.status_code == 422  # Validation error

def test_classification_service_error(client, sample_transaction):
    """Test classification service error handling"""
    with patch('src.classification.service.ClassificationService.classify_transaction') as mock_classify:
        mock_classify.side_effect = Exception("SageMaker endpoint unavailable")
        
        response = client.post(
            "/classification/classify",
            json=sample_transaction
        )
        
        assert response.status_code == 500 