import logging
import json
import boto3
import os
import yaml
from io import BytesIO
from typing import Dict

class Saver:
    def __init__(self, config_path: str):
        """
        Initialize the Saver with configuration details.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.config['minio']['endpoint_url'],
            aws_access_key_id=self.config['minio']['access_key'],
            aws_secret_access_key=self.config['minio']['secret_key']
        )
        # Initialize bucket name
        self.bucket_name = self.config['data']['bucket']
        logging.info("Saver initialized with configuration.")

    def _load_config(self, config_path: str) -> dict:
        """Load the YAML configuration file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def save_keras_model(self, model, key: str):
        """
        Save Keras model to MinIO (S3 compatible storage).

        Args:
            model: Trained Keras model.
            key (str): Key (filename) to save the model.
        """
        try:
            with BytesIO() as model_buffer:
                model.save(model_buffer)
                model_buffer.seek(0)
                self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=model_buffer)
            
            logging.info(f"Model saved successfully to {self.bucket_name}/{key}.")
        
        except Exception as e:
            logging.error(f"Failed to save model to MinIO: {e}")
            raise

    def save_metrics(self, metrics: Dict, key: str):
        """
        Save metrics to MinIO (S3 compatible storage).

        Args:
            metrics: Metrics dictionary.
            key (str): Key (filename) to save the metrics.
        """
        try:
            metrics_json = json.dumps(metrics, indent=4)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=metrics_json)
            
            logging.info(f"Metrics saved successfully to {self.bucket_name}/{key}.")
        
        except Exception as e:
            logging.error(f"Failed to save metrics to MinIO: {e}")
            raise
