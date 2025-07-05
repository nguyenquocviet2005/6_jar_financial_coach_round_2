#!/usr/bin/env python3
"""
Train transaction classification model using SageMaker
"""
import boto3
import json
import time
import logging
from typing import Dict, Any, List
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationModelTrainer:
    def __init__(self):
        self.sagemaker = boto3.client('sagemaker')
        self.s3 = boto3.client('s3')
        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.role_arn = os.getenv('SAGEMAKER_EXECUTION_ROLE_ARN')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME')
        self.training_job_name = f"transaction-classifier-{int(time.time())}"
        
    def prepare_training_data(self, data_path: str = "data/transactions.csv") -> str:
        """Prepare and upload training data to S3"""
        logger.info("Preparing training data...")
        
        # Check if data file exists
        if not os.path.exists(data_path):
            logger.warning(f"Data file {data_path} not found. Creating sample data...")
            self._create_sample_data(data_path)
        
        # Load and preprocess data
        df = pd.read_csv(data_path)
        
        # Basic preprocessing
        df = df.dropna()
        df['amount'] = df['amount'].astype(float)
        df['description'] = df['description'].astype(str)
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Save processed data
        train_path = "data/train.csv"
        test_path = "data/test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Upload to S3
        s3_train_path = f"s3://{self.s3_bucket}/training-data/train.csv"
        s3_test_path = f"s3://{self.s3_bucket}/training-data/test.csv"
        
        self.s3.upload_file(train_path, self.s3_bucket, "training-data/train.csv")
        self.s3.upload_file(test_path, self.s3_bucket, "training-data/test.csv")
        
        logger.info(f"Training data uploaded to {s3_train_path}")
        logger.info(f"Test data uploaded to {s3_test_path}")
        
        return s3_train_path
    
    def _create_sample_data(self, data_path: str):
        """Create sample transaction data for training"""
        logger.info("Creating sample transaction data...")
        
        # Create sample data
        sample_data = {
            'description': [
                'GROCERY STORE PURCHASE', 'RESTAURANT MEAL', 'GAS STATION FUEL',
                'AMAZON PURCHASE', 'UTILITY BILL PAYMENT', 'RENT PAYMENT',
                'SALARY DEPOSIT', 'COFFEE SHOP', 'UBER RIDE', 'NETFLIX SUBSCRIPTION',
                'SUPERMARKET SHOPPING', 'FAST FOOD', 'PHARMACY PURCHASE',
                'ONLINE SHOPPING', 'ELECTRICITY BILL', 'MORTGAGE PAYMENT',
                'BONUS PAYMENT', 'STARBUCKS', 'TAXI FARE', 'SPOTIFY SUBSCRIPTION',
                'WALMART PURCHASE', 'PIZZA ORDER', 'CLOTHING STORE',
                'ELECTRONICS STORE', 'WATER BILL', 'INSURANCE PAYMENT',
                'FREELANCE INCOME', 'CAFE PURCHASE', 'BUS TICKET', 'GAMING SUBSCRIPTION'
            ],
            'amount': [
                -85.43, -24.99, -45.67, -129.99, -78.50, -1200.00,
                2500.00, -4.95, -12.30, -12.99, -67.80, -8.75, -34.50,
                -89.99, -156.78, -1850.00, 500.00, -5.25, -18.40, -9.99,
                -112.34, -18.99, -89.50, -299.99, -45.67, -234.56,
                800.00, -6.75, -2.50, -14.99
            ],
            'category': [
                'NECESSITIES', 'NECESSITIES', 'NECESSITIES',
                'NECESSITIES', 'NECESSITIES', 'NECESSITIES',
                'INCOME', 'PLAY', 'NECESSITIES', 'PLAY',
                'NECESSITIES', 'PLAY', 'NECESSITIES',
                'NECESSITIES', 'NECESSITIES', 'NECESSITIES',
                'INCOME', 'PLAY', 'NECESSITIES', 'PLAY',
                'NECESSITIES', 'PLAY', 'NECESSITIES',
                'NECESSITIES', 'NECESSITIES', 'NECESSITIES',
                'INCOME', 'PLAY', 'NECESSITIES', 'PLAY'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(data_path, index=False)
        logger.info(f"Sample data created at {data_path}")
    
    def start_training_job(self, train_data_path: str) -> str:
        """Start SageMaker training job"""
        logger.info(f"Starting training job: {self.training_job_name}")
        
        # Training job configuration
        training_job_config = {
            'TrainingJobName': self.training_job_name,
            'RoleArn': self.role_arn,
            'AlgorithmSpecification': {
                'TrainingImage': f'{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/transaction-classifier:latest',
                'TrainingInputMode': 'File'
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': train_data_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv',
                    'CompressionType': 'None',
                    'InputMode': 'File'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{self.s3_bucket}/models/classification/'
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600
            },
            'HyperParameters': {
                'epochs': '10',
                'batch_size': '32',
                'learning_rate': '0.001'
            },
            'Environment': {
                'SM_MODEL_DIR': '/opt/ml/model',
                'SM_CHANNELS': '["training"]',
                'SM_CHANNEL_TRAINING': '/opt/ml/input/data/training'
            }
        }
        
        # Create training job
        response = self.sagemaker.create_training_job(**training_job_config)
        
        logger.info(f"Training job started: {self.training_job_name}")
        return self.training_job_name
    
    def wait_for_training_completion(self, training_job_name: str):
        """Wait for training job to complete"""
        logger.info(f"Waiting for training job {training_job_name} to complete...")
        
        waiter = self.sagemaker.get_waiter('training_job_completed_or_stopped')
        waiter.wait(
            TrainingJobName=training_job_name,
            WaiterConfig={
                'Delay': 30,
                'MaxAttempts': 120
            }
        )
        
        # Check final status
        response = self.sagemaker.describe_training_job(TrainingJobName=training_job_name)
        status = response['TrainingJobStatus']
        
        if status == 'Completed':
            logger.info(f"Training job {training_job_name} completed successfully!")
        else:
            logger.error(f"Training job {training_job_name} failed with status: {status}")
            if 'FailureReason' in response:
                logger.error(f"Failure reason: {response['FailureReason']}")
    
    def get_training_job_metrics(self, training_job_name: str) -> Dict[str, Any]:
        """Get training job metrics"""
        response = self.sagemaker.describe_training_job(TrainingJobName=training_job_name)
        
        metrics = {
            'training_job_name': training_job_name,
            'status': response['TrainingJobStatus'],
            'training_start_time': response.get('TrainingStartTime'),
            'training_end_time': response.get('TrainingEndTime'),
            'model_artifacts': response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
            'final_metric_data_list': response.get('FinalMetricDataList', [])
        }
        
        return metrics

def main():
    """Main training function"""
    trainer = ClassificationModelTrainer()
    
    # Prepare training data
    train_data_path = trainer.prepare_training_data()
    
    # Start training job
    training_job_name = trainer.start_training_job(train_data_path)
    
    # Wait for completion
    trainer.wait_for_training_completion(training_job_name)
    
    # Get metrics
    metrics = trainer.get_training_job_metrics(training_job_name)
    
    logger.info("Training completed!")
    logger.info(f"Model artifacts: {metrics['model_artifacts']}")
    logger.info(f"Final metrics: {metrics['final_metric_data_list']}")

if __name__ == "__main__":
    main() 