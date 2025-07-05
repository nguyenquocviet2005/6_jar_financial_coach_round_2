"""
Pytest configuration and fixtures for MLOps tests
"""
import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set testing environment
os.environ['TESTING'] = 'true'
os.environ['DATABASE_URL'] = 'sqlite:///./test.db'

from src.main import app
from src.common.dependencies import get_db, get_redis, get_chroma_client, get_sagemaker_client, get_bedrock_client
from src.config import Settings

@pytest.fixture(scope="session")
def test_settings():
    """Test settings fixture"""
    return Settings(
        testing=True,
        database_url="sqlite:///./test.db",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_default_region="us-east-1"
    )

@pytest.fixture(scope="session")
def test_engine():
    """Test database engine"""
    engine = create_engine(
        "sqlite:///./test.db",
        connect_args={"check_same_thread": False}
    )
    yield engine
    # Cleanup
    try:
        os.remove("./test.db")
    except FileNotFoundError:
        pass

@pytest.fixture(scope="function")
def test_session(test_engine):
    """Test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    yield session
    session.close()

@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis client"""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    return mock_redis

@pytest.fixture(scope="function")
def mock_chroma():
    """Mock ChromaDB client"""
    mock_chroma = Mock()
    mock_collection = Mock()
    mock_collection.query.return_value = {
        'documents': [['Sample financial advice']],
        'metadatas': [[{'source': 'test'}]],
        'distances': [[0.5]]
    }
    mock_chroma.get_collection.return_value = mock_collection
    return mock_chroma

@pytest.fixture(scope="function")
def mock_sagemaker():
    """Mock SageMaker client"""
    mock_sagemaker = Mock()
    mock_sagemaker.invoke_endpoint.return_value = {
        'Body': Mock(read=lambda: b'{"category": "NECESSITIES", "confidence": 0.85}')
    }
    return mock_sagemaker

@pytest.fixture(scope="function")
def mock_bedrock():
    """Mock Bedrock client"""
    mock_bedrock = Mock()
    mock_bedrock.invoke_model.return_value = {
        'body': Mock(read=lambda: b'{"completion": "This is financial advice"}')
    }
    return mock_bedrock

@pytest.fixture(scope="function")
def client(test_session, mock_redis, mock_chroma, mock_sagemaker, mock_bedrock):
    """Test client with mocked dependencies"""
    
    def override_get_db():
        yield test_session
    
    def override_get_redis():
        return mock_redis
    
    def override_get_chroma():
        return mock_chroma
    
    def override_get_sagemaker():
        return mock_sagemaker
    
    def override_get_bedrock():
        return mock_bedrock
    
    # Override dependencies
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis
    app.dependency_overrides[get_chroma_client] = override_get_chroma
    app.dependency_overrides[get_sagemaker_client] = override_get_sagemaker
    app.dependency_overrides[get_bedrock_client] = override_get_bedrock
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clear overrides
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def sample_transaction():
    """Sample transaction data"""
    return {
        "user_id": "test-user-123",
        "description": "GROCERY STORE PURCHASE",
        "amount": -85.43,
        "merchant": "SAFEWAY",
        "timestamp": "2024-01-15T10:30:00Z"
    }

@pytest.fixture(scope="function")
def sample_user_context():
    """Sample user context for AI coach"""
    return {
        "user_id": "test-user-123",
        "monthly_income": 5000.0,
        "monthly_expenses": 3500.0,
        "savings_goal": 1000.0,
        "jar_balances": {
            "NECESSITIES": 2000.0,
            "PLAY": 500.0,
            "FINANCIAL_FREEDOM": 300.0,
            "EDUCATION": 200.0,
            "LONG_TERM_SAVINGS": 800.0,
            "GIVE": 100.0
        },
        "recent_transactions": [
            {
                "description": "GROCERY STORE",
                "amount": -85.43,
                "category": "NECESSITIES",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        ]
    }

@pytest.fixture(scope="function")
def temp_directory():
    """Temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def mock_aws_services():
    """Mock all AWS services"""
    with patch('boto3.client') as mock_boto3:
        mock_clients = {
            'sagemaker': Mock(),
            'bedrock-runtime': Mock(),
            's3': Mock(),
            'dynamodb': Mock(),
            'sqs': Mock()
        }
        
        def get_client(service_name, **kwargs):
            return mock_clients.get(service_name, Mock())
        
        mock_boto3.side_effect = get_client
        yield mock_clients

@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment after each test"""
    yield
    # Clean up any test artifacts
    test_files = ['test.db', 'test.db-journal']
    for file in test_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass 