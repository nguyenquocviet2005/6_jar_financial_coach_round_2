"""Dependency injection for common services."""

import boto3
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings

settings = get_settings()

# Database engine
engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Async session maker
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Redis client
redis_client = redis.from_url(settings.redis_url, decode_responses=True)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    return redis_client


def get_s3_client():
    """Get S3 client."""
    return boto3.client(
        's3',
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_endpoint_url
    )


def get_sagemaker_client():
    """Get SageMaker client."""
    return boto3.client(
        'sagemaker',
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_endpoint_url
    )


def get_bedrock_client():
    """Get Bedrock client."""
    return boto3.client(
        'bedrock',
        region_name=settings.bedrock_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_endpoint_url
    )


def get_bedrock_runtime_client():
    """Get Bedrock Runtime client."""
    return boto3.client(
        'bedrock-runtime',
        region_name=settings.bedrock_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_endpoint_url
    )


def get_chroma_client():
    """Get ChromaDB client."""
    return chromadb.Client(
        ChromaSettings(
            chroma_server_host=settings.chroma_host,
            chroma_server_http_port=settings.chroma_port,
            persist_directory=settings.chroma_persist_directory if settings.development else None
        )
    )


def get_sqs_client():
    """Get SQS client."""
    return boto3.client(
        'sqs',
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_endpoint_url
    )


def get_dynamodb_client():
    """Get DynamoDB client."""
    return boto3.client(
        'dynamodb',
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_endpoint_url
    ) 