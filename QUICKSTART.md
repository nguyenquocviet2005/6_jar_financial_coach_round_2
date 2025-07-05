# 🚀 6-Jar Financial App - MLOps Quickstart Guide

This guide will get you up and running with the MLOps components of the 6-jar financial app in under 10 minutes.

## ⚡ Quick Setup

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd round_2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your AWS credentials
nano .env  # or use your preferred editor
```

**Minimum required configurations:**
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# S3 Bucket (must be globally unique)
S3_BUCKET_NAME=your-unique-bucket-name-mlops-financial

# Database
DATABASE_URL=postgresql://mlops:mlops123@localhost:5432/mlops_db

# For local development, you can use:
USE_LOCALSTACK=true
```

### 3. Start Local Services
```bash
# Start all supporting services (PostgreSQL, Redis, ChromaDB, etc.)
docker-compose up -d

# Wait for services to be ready (about 30 seconds)
docker-compose logs -f
```

### 4. Setup AWS Infrastructure (Optional for Local Dev)
```bash
# If using real AWS (not LocalStack)
python scripts/setup_aws_infrastructure.py
```

### 5. Start the MLOps API
```bash
# Start the FastAPI server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 Test the Setup

### Health Check
```bash
curl http://localhost:8000/health
```

### Test AI Coach
```bash
curl -X POST http://localhost:8000/ai-coach/advice \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user-123",
    "message": "How should I manage my spending this month?",
    "context": {
      "monthly_income": 5000,
      "monthly_expenses": 3500,
      "savings_goal": 1000,
      "jar_balances": {
        "NECESSITIES": 2000,
        "PLAY": 500,
        "FINANCIAL_FREEDOM": 300,
        "EDUCATION": 200,
        "LONG_TERM_SAVINGS": 800,
        "GIVE": 100
      }
    }
  }'
```

### Test Transaction Classification
```bash
curl -X POST http://localhost:8000/classification/classify \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user-123",
    "description": "GROCERY STORE PURCHASE",
    "amount": -85.43,
    "merchant": "SAFEWAY",
    "timestamp": "2024-01-15T10:30:00Z"
  }'
```

## 📊 Access Web Interfaces

- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **ChromaDB**: http://localhost:8000 (vector database)

## 🧪 Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ai_coach.py -v
```

## 🔧 Development Commands

```bash
# Using Make (recommended)
make help          # Show all available commands
make setup         # Full setup (install + docker + migrate)
make dev           # Start development server
make test          # Run tests
make lint          # Run linting
make format        # Format code

# Manual commands
uvicorn src.main:app --reload                    # Start API server
celery -A src.common.celery_app worker           # Start background tasks
celery -A src.common.celery_app beat             # Start scheduled tasks
```

## 🚀 Production Deployment

### 1. AWS Infrastructure
```bash
# Setup AWS resources
python scripts/setup_aws_infrastructure.py

# Train and deploy models
python scripts/train_classification_model.py
python scripts/deploy_models.py
```

### 2. Container Deployment
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d
```

## 🔍 Key Features Available

### ✅ AI Coach Service
- **Endpoint**: `/ai-coach/advice`
- **Features**: RAG-based financial advice, context-aware recommendations
- **Integration**: ChromaDB for knowledge base, Bedrock for LLM

### ✅ Transaction Classification
- **Endpoint**: `/classification/classify`
- **Features**: ML-based categorization, jar-type mapping, batch processing
- **Integration**: SageMaker for ML inference

### ✅ Background Tasks
- **Celery Workers**: Automated training, data processing
- **Scheduled Jobs**: Model retraining, performance monitoring

### ✅ Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and alerts
- **Structured Logging**: JSON-formatted logs with correlation IDs

## 🛠️ Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps
   
   # Restart database
   docker-compose restart postgres
   ```

2. **AWS Credentials Not Found**
   ```bash
   # Check AWS configuration
   aws sts get-caller-identity
   
   # Configure AWS CLI
   aws configure
   ```

3. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Kill the process or use different port
   uvicorn src.main:app --reload --port 8001
   ```

4. **Docker Services Not Starting**
   ```bash
   # Check logs
   docker-compose logs
   
   # Restart all services
   docker-compose down && docker-compose up -d
   ```

## 📝 Next Steps

1. **Customize Configuration**: Edit `.env` with your specific AWS resources
2. **Add Real Data**: Replace sample data with your transaction data
3. **Train Models**: Use your data to train classification models
4. **Deploy to Production**: Follow the production deployment guide
5. **Monitor Performance**: Set up alerts and monitoring dashboards

## 🆘 Need Help?

- Check the main `README.md` for detailed documentation
- Review the API documentation at http://localhost:8000/docs
- Check the `tests/` directory for usage examples
- Review the `scripts/` directory for automation examples

Happy coding! 🎉 