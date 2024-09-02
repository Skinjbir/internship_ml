import boto3
import logging
import yaml
import json
import io
import os
from libt.split import DataSplitter
from libt.Loader import Loader
from libt.train_model import ModelTrainer
from libt.create_model import create_autoencoder
from libt.evaluate_model import detect_anomalies, evaluate_anomalies

def save_keras_model_to_s3(s3_client, model, bucket_name, key):
    """
    Save Keras model to MinIO (S3 compatible storage).

    Args:
        s3_client: Boto3 S3 client object.
        model: Trained Keras model.
        bucket_name (str): Name of the bucket to save the model.
        key (str): Key (filename) to save the model.
    """
    try:
        # Save the model locally with a .keras extension
        local_model_path = 'temp_model.keras'
        model.save(local_model_path)

        # Upload the local file to MinIO
        with open(local_model_path, 'rb') as model_file:
            s3_client.put_object(Bucket=bucket_name, Key=key, Body=model_file)
        
        logging.info(f"Model saved successfully to {bucket_name}/{key}.")
        
        # Remove the local file after upload
        os.remove(local_model_path)
        
    except Exception as e:
        logging.error(f"Failed to save model to MinIO: {e}")
        raise

def save_metrics_to_minio(s3_client, metrics, bucket_name, key):
    """
    Save metrics to MinIO (S3 compatible storage).

    Args:
        s3_client: Boto3 S3 client object.
        metrics: Metrics dictionary.
        bucket_name (str): Name of the bucket to save the metrics.
        key (str): Key (filename) to save the metrics.
    """
    try:
        # Convert metrics to JSON string
        metrics_json = json.dumps(metrics, indent=4)
        
        # Upload the JSON string to MinIO
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=metrics_json)
        
        logging.info(f"Metrics saved successfully to {bucket_name}/{key}.")
        
    except Exception as e:
        logging.error(f"Failed to save metrics to MinIO: {e}")
        raise

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load the configuration from the config file using the Loader class
        loader = Loader(config_path='./train_config.yaml')
        config = loader.config

        # Initialize the s3_client
        s3_client = boto3.client(
            's3',
            endpoint_url=config['minio']['endpoint_url'],
            aws_access_key_id=config['minio']['access_key'],
            aws_secret_access_key=config['minio']['secret_key']
        )

        logging.info("MinIO S3 client initialized successfully.")
        
        # Load data from MinIO
        df = loader.load_data_from_minio()
        
        # Split Data
        splitter = DataSplitter(df)
        X_train, X_test, y_train, y_test = splitter.split_data_v2()

        logging.info("Data split successfully.")
        logging.debug(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Create Autoencoder Model
        autoencoder = create_autoencoder(input_dim=X_train.shape[1])
        
        # Train the model
        trainer = ModelTrainer(training_set=X_train, validation_set=X_test, autoencoder=autoencoder)
        trainer.train_model(batch_size=config['training']['batch_size'], epochs=config['training']['epochs'])
        
        logging.info("Model trained successfully.")
                
        trained_autoencoder = trainer.get_trained_model()
        logging.info("Autoencoder model trained successfully.")

        save_keras_model_to_s3(s3_client, trained_autoencoder, config['data']['bucket'], config['model']['key'])
        
        # Detect Anomalies
        outliers, errors, threshold = detect_anomalies(trained_autoencoder, X_test, percentile=config['anomaly_detection']['percentile'], metric=config['anomaly_detection']['metric'])

        # Evaluate Model Performance
        metrics = evaluate_anomalies(y_test, outliers, errors)
        logging.info(f"Evaluation Metrics: {metrics}")

        # Save metrics to MinIO
        save_metrics_to_minio(s3_client, metrics, config['data']['bucket'], config['evaluation']['key'])

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
