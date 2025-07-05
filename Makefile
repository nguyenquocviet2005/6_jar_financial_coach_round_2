# 6-Jar Financial App - MLOps Makefile

.PHONY: help install dev test lint format clean docker-build docker-up docker-down aws-setup

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  dev          - Start development server"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean up temporary files"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start Docker services"
	@echo "  docker-down  - Stop Docker services"
	@echo "  aws-setup    - Setup AWS infrastructure"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Start development server
dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Run linting
lint:
	flake8 src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf build/ dist/ *.egg-info/

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# AWS setup
aws-setup:
	python scripts/setup_aws_infrastructure.py

# Model training
train-classification:
	python scripts/train_classification_model.py

# Model deployment
deploy-models:
	python scripts/deploy_models.py

# Database migrations
migrate:
	alembic upgrade head

# Create new migration
migration:
	alembic revision --autogenerate -m "$(name)"

# Start Celery worker
celery-worker:
	celery -A src.common.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery-beat:
	celery -A src.common.celery_app beat --loglevel=info

# Monitor Celery
celery-monitor:
	celery -A src.common.celery_app flower

# Full local setup
setup: install docker-up migrate
	@echo "🚀 Setup complete! Run 'make dev' to start the server."

# Production deployment
deploy: docker-build aws-setup deploy-models
	@echo "🚀 Production deployment complete!"

# Health check
health:
	curl -f http://localhost:8000/health || echo "❌ Service is not healthy"

# Load test data
load-test-data:
	python scripts/load_test_data.py

# Backup database
backup-db:
	python scripts/backup_database.py

# Monitor logs
logs:
	tail -f logs/app.log

# Security scan
security-scan:
	bandit -r src/

# Generate API documentation
docs:
	python scripts/generate_docs.py
	@echo "📚 API documentation generated at docs/api/" 