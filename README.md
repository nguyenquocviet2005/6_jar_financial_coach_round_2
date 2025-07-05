# 6-Jar Financial App - MLOps Components

This repository contains the MLOps infrastructure for the 6-jar financial management application, focusing on AI/ML services including AI coaching, transaction classification, spending prediction, and model fine-tuning.

## 🏗️ Architecture Overview

### MLOps Components
1. **AI Coach Service** - RAG-based financial advisory using SageMaker & Bedrock
2. **Fine-tuning Pipeline** - Automated model training with AWS Bedrock
3. **Transaction Classification** - ML-based transaction categorization
4. **Spending Prediction** - Forecasting and alert generation

### AWS Services Integration
- **SageMaker**: ML model endpoints and training
- **Bedrock**: LLM fine-tuning and inference
- **Pinpoint**: Push notifications
- **S3**: Model artifacts and data storage
- **DynamoDB**: Real-time data storage
- **SQS**: Message queuing
- **Lambda**: Serverless compute

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- AWS CLI configured
- Node.js 18+ (for CDK)

### Local Development Setup

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd round_2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Environment Configuration**:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and configuration
```

3. **Start local services**:
```bash
docker-compose up -d
```

4. **Run database migrations**:
```bash
alembic upgrade head
```

5. **Start the MLOps API**:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### AWS Deployment

1. **Deploy infrastructure**:
```bash
cd infrastructure
cdk deploy --all
```

2. **Deploy ML models**:
```bash
python scripts/deploy_models.py
```

## 📁 Project Structure

```
round_2/
├── src/
│   ├── ai_coach/           # AI coaching service
│   ├── classification/     # Transaction classification
│   ├── prediction/         # Spending prediction
│   ├── fine_tuning/        # Model fine-tuning pipeline
│   ├── common/             # Shared utilities
│   └── main.py            # FastAPI application
├── infrastructure/         # AWS CDK infrastructure
├── models/                # ML model artifacts
├── data/                  # Sample data and schemas
├── tests/                 # Test suites
├── scripts/               # Deployment and utility scripts
└── docker-compose.yml     # Local development services
```

## 🔧 Development Workflow

### Local Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test suite
pytest tests/test_ai_coach.py
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Model Training
```bash
# Train transaction classification model
python scripts/train_classification_model.py

# Fine-tune LLM with Bedrock
python scripts/fine_tune_llm.py

# Deploy models to SageMaker
python scripts/deploy_models.py
```

## 🔄 CI/CD Pipeline

The MLOps pipeline includes:
- **Continuous Integration**: Automated testing and code quality checks
- **Model Training**: Scheduled retraining with new data
- **Model Deployment**: Automated deployment to SageMaker endpoints
- **Monitoring**: Model performance and drift detection

## 📊 Monitoring & Observability

- **Prometheus**: Metrics collection
- **CloudWatch**: AWS service monitoring
- **Model Performance**: Accuracy, latency, and drift metrics
- **Data Quality**: Input validation and anomaly detection

## 🔐 Security

- **IAM Roles**: Least privilege access
- **VPC**: Network isolation
- **Encryption**: Data at rest and in transit
- **API Authentication**: JWT tokens and API keys

## 🤝 Contributing

1. Create a feature branch
2. Make changes with tests
3. Run quality checks
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details 