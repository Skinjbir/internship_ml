import boto3
import logging
import yaml
import json
import io
import os
from libt.split import DataSplitter
from libt.train_model import ModelTrainer
from libt.create_model import create_autoencoder
from libt.evaluate_model import detect_anomalies, evaluate_anomalies
import time

# Wait for the signal file


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize S3 client using boto3 (compatible with MinIO)
s3_client = boto3.client('s3', endpoint_url='http://10.17.0.243:9000', 
                         aws_access_key_id='minioadmin', 
                         aws_secret_access_key='minioadmin')

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_file_from_s3(bucket_name, object_key, local_path):
    """Download a file from S3/MinIO."""
    try:
        s3_client.download_file(bucket_name, object_key, local_path)
        logging.info(f"File downloaded successfully from S3: {bucket_name}/{object_key}")
    except Exception as e:
        logging.error(f"Failed to download file from S3: {e}")
        raise

def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

import tempfile

def save_keras_model_to_s3(model, bucket_name, file_name='model.keras'):
    """Save a Keras model to S3/MinIO."""
    try:
        # Create a temporary file with the desired extension
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
            temp_file_name = temp_file.name
            
        # Save the Keras model to the temporary file
        model.save(temp_file_name)
        
        # Upload the temporary file to S3/MinIO
        with open(temp_file_name, 'rb') as model_file:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=f'models/{file_name}',
                Body=model_file,
                ContentType='application/octet-stream'
            )
        
        logging.info(f"Model saved to S3 bucket {bucket_name}/models/{file_name}")
        
    except Exception as e:
        logging.error(f"Error saving model to S3: {e}")
    finally:
        # Clean up by deleting the temporary file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def main():
    try:
        

        # Load configuration from YAML
        config = load_config('/app/train_stage/train_config.yaml')

        # Extract configuration values
        bucket_name = config['dataset']['bucket_name']
        object_key = config['dataset']['object_key']
        local_path = config['dataset']['local_path']
        batch_size = config['training']['batch_size']
        epochs = config['training']['epochs']
        percentile = config['anomalies']['percentile']
        metric = config['anomalies']['metric']
        model_save_path = config['model']['save_path']
        metrics_save_path = 'metrics/metrics.json'

        # Ensure the metrics directory exists
        ensure_directory_exists(os.path.dirname(metrics_save_path))

        # Download the cleaned data from S3/MinIO
        download_file_from_s3(bucket_name, object_key, local_path)

        # Split Data
        splitter = DataSplitter(local_path)
        X_train, X_test, y_train, y_test = splitter.split_data_v2()

        logging.info("Data split successfully.")
        logging.debug(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        # Create Autoencoder Model
        input_dim = X_train.shape[1]
        autoencoder = create_autoencoder(input_dim=input_dim)

        # Train Autoencoder Model
        trainer = ModelTrainer(training_set=X_train, validation_set=X_test, autoencoder=autoencoder)
        trainer.train_model(batch_size=batch_size, epochs=epochs)

        trained_autoencoder = trainer.get_trained_model()
        logging.info("Autoencoder model trained successfully.")

        save_keras_model_to_s3(trained_autoencoder, bucket_name, 'model.keras')
        
        # Detect Anomalies
        outliers, errors, threshold = detect_anomalies(trained_autoencoder, X_test, percentile=percentile, metric=metric)

        # Evaluate Model Performance
        metrics = evaluate_anomalies(y_test, outliers, errors)
        logging.info(f"Evaluation Metrics: {metrics}")

        # Save metrics to file
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Metrics saved successfully at {metrics_save_path}")


    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
