#!/usr/bin/env python3
"""
Setup AWS infrastructure for MLOps pipeline
"""
import boto3
import json
import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSInfrastructureSetup:
    def __init__(self):
        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        
        # Initialize AWS clients
        self.s3 = boto3.client('s3')
        self.iam = boto3.client('iam')
        self.dynamodb = boto3.client('dynamodb')
        self.sqs = boto3.client('sqs')
        
    def create_s3_bucket(self):
        """Create S3 bucket for model artifacts and data"""
        logger.info(f"Creating S3 bucket: {self.bucket_name}")
        
        try:
            if self.region == 'us-east-1':
                # us-east-1 doesn't require LocationConstraint
                self.s3.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Create folder structure
            folders = [
                'models/classification/',
                'models/prediction/',
                'data/raw/',
                'data/processed/',
                'training-data/',
                'feature-store/'
            ]
            
            for folder in folders:
                self.s3.put_object(Bucket=self.bucket_name, Key=folder)
            
            logger.info(f"S3 bucket {self.bucket_name} created successfully")
            
        except Exception as e:
            if "BucketAlreadyOwnedByYou" in str(e):
                logger.info(f"S3 bucket {self.bucket_name} already exists")
            else:
                logger.error(f"Error creating S3 bucket: {str(e)}")
    
    def create_iam_roles(self):
        """Create IAM roles for SageMaker and Lambda"""
        logger.info("Creating IAM roles...")
        
        # SageMaker execution role
        sagemaker_role_name = "SageMakerExecutionRole"
        sagemaker_trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            self.iam.create_role(
                RoleName=sagemaker_role_name,
                AssumeRolePolicyDocument=json.dumps(sagemaker_trust_policy),
                Description="Role for SageMaker to access AWS services"
            )
            
            # Attach policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess",
                "arn:aws:iam::aws:policy/AmazonSQSFullAccess"
            ]
            
            for policy in policies:
                self.iam.attach_role_policy(
                    RoleName=sagemaker_role_name,
                    PolicyArn=policy
                )
            
            logger.info(f"SageMaker role {sagemaker_role_name} created")
            
        except Exception as e:
            if "EntityAlreadyExists" in str(e):
                logger.info(f"SageMaker role {sagemaker_role_name} already exists")
            else:
                logger.error(f"Error creating SageMaker role: {str(e)}")
        
        # Lambda execution role
        lambda_role_name = "LambdaExecutionRole"
        lambda_trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            self.iam.create_role(
                RoleName=lambda_role_name,
                AssumeRolePolicyDocument=json.dumps(lambda_trust_policy),
                Description="Role for Lambda to access AWS services"
            )
            
            # Attach policies
            policies = [
                "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess"
            ]
            
            for policy in policies:
                self.iam.attach_role_policy(
                    RoleName=lambda_role_name,
                    PolicyArn=policy
                )
            
            logger.info(f"Lambda role {lambda_role_name} created")
            
        except Exception as e:
            if "EntityAlreadyExists" in str(e):
                logger.info(f"Lambda role {lambda_role_name} already exists")
            else:
                logger.error(f"Error creating Lambda role: {str(e)}")
    
    def create_dynamodb_tables(self):
        """Create DynamoDB tables for application data"""
        logger.info("Creating DynamoDB tables...")
        
        tables = [
            {
                'name': 'mlops-user-sessions',
                'partition_key': 'user_id',
                'sort_key': 'session_id'
            },
            {
                'name': 'mlops-model-metrics',
                'partition_key': 'model_name',
                'sort_key': 'timestamp'
            },
            {
                'name': 'mlops-training-jobs',
                'partition_key': 'job_id',
                'sort_key': None
            },
            {
                'name': 'mlops-predictions',
                'partition_key': 'user_id',
                'sort_key': 'prediction_id'
            }
        ]
        
        for table_config in tables:
            try:
                # Define key schema
                key_schema = [
                    {
                        'AttributeName': table_config['partition_key'],
                        'KeyType': 'HASH'
                    }
                ]
                
                attribute_definitions = [
                    {
                        'AttributeName': table_config['partition_key'],
                        'AttributeType': 'S'
                    }
                ]
                
                if table_config['sort_key']:
                    key_schema.append({
                        'AttributeName': table_config['sort_key'],
                        'KeyType': 'RANGE'
                    })
                    attribute_definitions.append({
                        'AttributeName': table_config['sort_key'],
                        'AttributeType': 'S'
                    })
                
                # Create table
                self.dynamodb.create_table(
                    TableName=table_config['name'],
                    KeySchema=key_schema,
                    AttributeDefinitions=attribute_definitions,
                    BillingMode='PAY_PER_REQUEST'
                )
                
                logger.info(f"DynamoDB table {table_config['name']} created")
                
            except Exception as e:
                if "ResourceInUseException" in str(e):
                    logger.info(f"DynamoDB table {table_config['name']} already exists")
                else:
                    logger.error(f"Error creating DynamoDB table {table_config['name']}: {str(e)}")
    
    def create_sqs_queues(self):
        """Create SQS queues for message processing"""
        logger.info("Creating SQS queues...")
        
        queues = [
            'mlops-training-jobs',
            'mlops-prediction-jobs',
            'mlops-model-updates',
            'mlops-notifications'
        ]
        
        for queue_name in queues:
            try:
                response = self.sqs.create_queue(
                    QueueName=queue_name,
                    Attributes={
                        'VisibilityTimeoutSeconds': '300',
                        'MessageRetentionPeriod': '1209600',  # 14 days
                        'DelaySeconds': '0'
                    }
                )
                
                logger.info(f"SQS queue {queue_name} created")
                
            except Exception as e:
                if "QueueAlreadyExists" in str(e):
                    logger.info(f"SQS queue {queue_name} already exists")
                else:
                    logger.error(f"Error creating SQS queue {queue_name}: {str(e)}")
    
    def setup_all(self):
        """Setup all AWS infrastructure"""
        logger.info("Setting up AWS infrastructure...")
        
        self.create_s3_bucket()
        self.create_iam_roles()
        self.create_dynamodb_tables()
        self.create_sqs_queues()
        
        logger.info("AWS infrastructure setup completed!")
        
        # Print important ARNs
        logger.info("Important ARNs:")
        logger.info(f"SageMaker Role ARN: arn:aws:iam::{self.account_id}:role/SageMakerExecutionRole")
        logger.info(f"Lambda Role ARN: arn:aws:iam::{self.account_id}:role/LambdaExecutionRole")
        logger.info(f"S3 Bucket: s3://{self.bucket_name}")

def main():
    """Main setup function"""
    setup = AWSInfrastructureSetup()
    setup.setup_all()

if __name__ == "__main__":
    main() 