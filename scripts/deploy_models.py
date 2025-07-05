#!/usr/bin/env python3
"""
Deploy ML models to SageMaker endpoints
"""
import boto3
import json
import time
import logging
from typing import Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self):
        self.sagemaker = boto3.client('sagemaker')
        self.account_id = os.getenv('AWS_ACCOUNT_ID')
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.role_arn = os.getenv('SAGEMAKER_EXECUTION_ROLE_ARN')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME')
        
    def deploy_classification_model(self) -> str:
        """Deploy transaction classification model to SageMaker"""
        logger.info("Deploying transaction classification model...")
        
        # Model configuration
        model_name = f"transaction-classifier-{int(time.time())}"
        endpoint_config_name = f"transaction-classifier-config-{int(time.time())}"
        endpoint_name = "transaction-classifier-endpoint"
        
        # Create model
        model_response = self.sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': f'{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/transaction-classifier:latest',
                'ModelDataUrl': f's3://{self.s3_bucket}/models/classification/model.tar.gz',
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                }
            },
            ExecutionRoleArn=self.role_arn
        )
        
        # Create endpoint configuration
        endpoint_config_response = self.sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        # Create or update endpoint
        try:
            # Try to update existing endpoint
            endpoint_response = self.sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Updating existing endpoint: {endpoint_name}")
        except self.sagemaker.exceptions.ClientError:
            # Create new endpoint if it doesn't exist
            endpoint_response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Creating new endpoint: {endpoint_name}")
        
        # Wait for endpoint to be in service
        self._wait_for_endpoint(endpoint_name)
        
        return endpoint_name
    
    def deploy_prediction_model(self) -> str:
        """Deploy spending prediction model to SageMaker"""
        logger.info("Deploying spending prediction model...")
        
        model_name = f"spending-predictor-{int(time.time())}"
        endpoint_config_name = f"spending-predictor-config-{int(time.time())}"
        endpoint_name = "spending-prediction-endpoint"
        
        # Create model
        model_response = self.sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': f'{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/spending-predictor:latest',
                'ModelDataUrl': f's3://{self.s3_bucket}/models/prediction/model.tar.gz',
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                }
            },
            ExecutionRoleArn=self.role_arn
        )
        
        # Create endpoint configuration
        endpoint_config_response = self.sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        # Create or update endpoint
        try:
            endpoint_response = self.sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Updating existing endpoint: {endpoint_name}")
        except self.sagemaker.exceptions.ClientError:
            endpoint_response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Creating new endpoint: {endpoint_name}")
        
        # Wait for endpoint to be in service
        self._wait_for_endpoint(endpoint_name)
        
        return endpoint_name
    
    def _wait_for_endpoint(self, endpoint_name: str):
        """Wait for endpoint to be in service"""
        logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")
        
        waiter = self.sagemaker.get_waiter('endpoint_in_service')
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={
                'Delay': 30,
                'MaxAttempts': 60
            }
        )
        
        logger.info(f"Endpoint {endpoint_name} is now in service!")
    
    def list_endpoints(self) -> Dict[str, Any]:
        """List all active endpoints"""
        response = self.sagemaker.list_endpoints(
            StatusEquals='InService'
        )
        return response['Endpoints']
    
    def delete_endpoint(self, endpoint_name: str):
        """Delete an endpoint"""
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        try:
            self.sagemaker.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting endpoint {endpoint_name}: {str(e)}")

def main():
    """Main deployment function"""
    deployer = ModelDeployer()
    
    # Deploy models
    classification_endpoint = deployer.deploy_classification_model()
    prediction_endpoint = deployer.deploy_prediction_model()
    
    logger.info("Model deployment completed!")
    logger.info(f"Classification endpoint: {classification_endpoint}")
    logger.info(f"Prediction endpoint: {prediction_endpoint}")
    
    # List all endpoints
    endpoints = deployer.list_endpoints()
    logger.info(f"Active endpoints: {len(endpoints)}")
    for endpoint in endpoints:
        logger.info(f"  - {endpoint['EndpointName']}: {endpoint['EndpointStatus']}")

if __name__ == "__main__":
    main() 