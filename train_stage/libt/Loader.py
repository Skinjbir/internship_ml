import logging
import pandas as pd
from io import StringIO
import boto3
import yaml


class Loader:
    def __init__(self, config_path: str):
        """
        Initialize the Loader with configuration details.
        
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
        # Initialize bucket name and file name
        self.bucket_name = self.config['data']['bucket']
        self.file_name = self.config['data']['original_data']
        logging.info("Loader initialized with configuration.")

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        
        Returns:
            dict: Configuration details.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def load_data_from_minio(self) -> pd.DataFrame:
        """
        Load data from MinIO and return a DataFrame.
        
        Returns:
            pd.DataFrame: Data loaded from MinIO.
        
        Raises:
            Exception: If there is an error while loading data from MinIO.
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.file_name)
            data = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(data))
            logging.info("Data successfully loaded from MinIO.")
            return df
        except Exception as e:
            logging.error(f"Error loading data from MinIO: {e}")
            raise
