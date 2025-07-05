"""
Test cases for AI Coach service
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

def test_ai_coach_get_advice(client, sample_user_context):
    """Test AI coach advice endpoint"""
    response = client.post(
        "/ai-coach/advice",
        json={
            "user_id": "test-user-123",
            "message": "How should I manage my spending this month?",
            "context": sample_user_context
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "advice" in data
    assert "confidence" in data
    assert "session_id" in data

def test_ai_coach_chat(client):
    """Test AI coach chat endpoint"""
    response = client.post(
        "/ai-coach/chat",
        json={
            "user_id": "test-user-123",
            "message": "What's my spending pattern?",
            "session_id": "test-session-123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data

def test_ai_coach_proactive_alert(client, sample_user_context):
    """Test proactive alert generation"""
    response = client.post(
        "/ai-coach/proactive-alert",
        json={
            "user_id": "test-user-123",
            "context": sample_user_context,
            "trigger_type": "spending_limit"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "alert" in data
    assert "severity" in data
    assert "recommendations" in data

def test_ai_coach_knowledge_base_search(client):
    """Test knowledge base search"""
    response = client.post(
        "/ai-coach/knowledge-base/search",
        json={
            "query": "budgeting tips",
            "limit": 5
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

def test_ai_coach_invalid_input(client):
    """Test AI coach with invalid input"""
    response = client.post(
        "/ai-coach/advice",
        json={}  # Empty request
    )
    
    assert response.status_code == 422  # Validation error

def test_ai_coach_service_error(client, sample_user_context):
    """Test AI coach service error handling"""
    with patch('src.ai_coach.service.AICoachService.get_advice') as mock_advice:
        mock_advice.side_effect = Exception("Service unavailable")
        
        response = client.post(
            "/ai-coach/advice",
            json={
                "user_id": "test-user-123",
                "message": "How should I manage my spending?",
                "context": sample_user_context
            }
        )
        
        assert response.status_code == 500 