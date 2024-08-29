import logging
import pandas as pd
from io import StringIO
import boto3
import yaml

class Loader:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.config['minio']['endpoint_url'],
            aws_access_key_id=self.config['minio']['access_key'],
            aws_secret_access_key=self.config['minio']['secret_key']
        )
        self.bucket_name = self.config['data']['bucket']
        self.file_name = self.config['data']['path']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from a YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration file: {e}")
            raise

    def load_data_from_minio(self) -> pd.DataFrame:
        """Load data from MinIO and return a DataFrame."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.file_name)
            data = response['Body'].read().decode('utf-8')
            return pd.read_csv(StringIO(data))
        except Exception as e:
            logging.error(f"Error loading data from MinIO: {e}")
            raise

