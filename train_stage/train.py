import logging
import yaml
from minio import Minio
from libt.split import DataSplitter
from libt.ModelMananger import AutoencoderModel, ModelTrainer
from libt.evaluator import detect_anomalies, evaluate_anomalies


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Path to your configuration file
    config_path = 'train_config.yaml'

    # Load configuration
    config = load_config(config_path)
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    bucket_name = config['model']['bucket_name']

    # Load and split the data
    data_splitter = DataSplitter(config_path)
    X_train, X_test, y_train, y_test = data_splitter.split_data_v2(class_column='Class', train_size=200000)

    # Print shapes of the datasets
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    input_dim = X_train.shape[1]  # Assuming X_train is a 2D array
    autoencoder_model = AutoencoderModel(input_dim)

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(
        training_set=X_train,
        validation_set=X_test,
        autoencoder_model=autoencoder_model
    )

    # Train the model
    try:
        model_trainer.train_model(batch_size=batch_size, epochs=epochs)

        # Save the model to MinIO
        model_save_path = config['model']['save_path']
        autoencoder_model.save_keras_model_to_minio(bucket_name, model_save_path)

        logging.info("Model training complete and saved to MinIO.")
        
        # Detect Anomalies
        outliers, errors, threshold = detect_anomalies(autoencoder_model, X_test, percentile=95, metric='mse')

        # Evaluate Model Performance
        metrics = evaluate_anomalies(y_test, outliers, errors)
        logging.info(f"Evaluation Metrics: {metrics}")

        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
