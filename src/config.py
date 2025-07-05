"""Configuration management for the MLOps application."""

import os
from typing import Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    database_url: str = "postgresql://postgres:postgres@localhost:5432/financial_app"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    
    # AWS Configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_endpoint_url: Optional[str] = None  # For LocalStack
    
    # SageMaker Configuration
    sagemaker_role_arn: Optional[str] = None
    sagemaker_execution_role: Optional[str] = None
    
    # Bedrock Configuration
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Vector Database Configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_directory: str = "./data/chroma"
    
    # Model Configuration
    classification_model_endpoint: str = "transaction-classifier-endpoint"
    prediction_model_endpoint: str = "spending-predictor-endpoint"
    fine_tuning_s3_bucket: str = "mlops-fine-tuning-bucket"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Security Configuration
    secret_key: str = "your-secret-key-here"
    jwt_secret: str = "your-jwt-secret"
    api_key_header: str = "X-API-Key"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Development Configuration
    debug: bool = False
    testing: bool = False
    development: bool = True

    @validator('database_url')
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(('postgresql://', 'sqlite://')):
            raise ValueError('Database URL must start with postgresql:// or sqlite://')
        return v

    @validator('redis_url')
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings 