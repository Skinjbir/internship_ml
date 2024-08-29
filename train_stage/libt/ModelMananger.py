import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from datetime import datetime
import os
import tempfile
from minio import Minio
import io

# Configure logging to include DEBUG level messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MinIO client
minio_client = Minio(
    "localhost:9000",  # MinIO endpoint
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # Change to True if using HTTPS
)

class AutoencoderModel:
    def __init__(self, input_dim: int):
        """
        Initializes the AutoencoderModel with the specified input dimension.

        Parameters:
        - input_dim: int, the dimensionality of the input data.
        """
        self.input_dim = input_dim
        self.autoencoder = self.create_autoencoder()

    def create_autoencoder(self) -> tf.keras.Model:
        """
        Create an autoencoder model.

        Returns:
        - tf.keras.Model: The constructed autoencoder model.
        """
        autoencoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.input_dim, activation='elu', input_shape=(self.input_dim, )),
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dense(12, activation='elu'),
            tf.keras.layers.Dense(8, activation='elu'),
            tf.keras.layers.Dense(4, activation='elu'),
            tf.keras.layers.Dense(2, activation='elu'),
            tf.keras.layers.Dense(4, activation='elu'),
            tf.keras.layers.Dense(8, activation='elu'),
            tf.keras.layers.Dense(12, activation='elu'),
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dense(self.input_dim, activation='elu')
        ])
        autoencoder.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        return autoencoder

    def train_autoencoder(self, X_train, X_validate, batch_size=128, epochs=100):
        """
        Train the autoencoder model.

        Args:
            X_train (np.ndarray): Training data.
            X_validate (np.ndarray): Validation data.
            batch_size (int): The batch size for training.
            epochs (int): The number of epochs to train.

        Returns:
            autoencoder (tf.keras.Model): The trained autoencoder model.
            history (tf.keras.callbacks.History): The training history.
        """
        BATCH_SIZE = batch_size
        EPOCHS = epochs

        yyyymmddHHMM = datetime.now().strftime('%Y%m%d%H%M')
        log_subdir = f'{yyyymmddHHMM}_batch{BATCH_SIZE}_layers{len(self.autoencoder.layers)}'
        log_dir = f'logs/{log_subdir}'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        save_model = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./models/autoencoder_best_weights_{yyyymmddHHMM}.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1,
            mode='min'
        )

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir,
            update_freq='batch'
        )

        cb = [early_stop, save_model, tensorboard]

        try:
            history = self.autoencoder.fit(
                X_train, X_train,
                shuffle=True,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=cb,
                validation_data=(X_validate, X_validate)
            )
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise

        return self.autoencoder, history

    def save_keras_model_to_minio(self, bucket_name: str, file_name='model.keras'):
        """
        Save the Keras model to MinIO.

        Parameters:
        - bucket_name: str, the name of the MinIO bucket.
        - file_name: str, the name of the file in the bucket.

        Raises:
        - Exception: For any errors occurring during the saving process.
        """
        try:
            # Create a temporary file with the desired extension
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
                temp_file_name = temp_file.name
                
            # Save the Keras model to the temporary file
            self.autoencoder.save(temp_file_name)
            
            # Upload the temporary file to MinIO
            with open(temp_file_name, 'rb') as model_file:
                minio_client.put_object(
                    bucket_name,
                    file_name,
                    model_file,
                    length=os.path.getsize(temp_file_name),
                    content_type='application/octet-stream'
                )
            
            logging.info(f"Model saved to MinIO bucket {bucket_name}/{file_name}")
        
        except Exception as e:
            logging.error(f"Error saving model to MinIO: {e}")
        
        finally:
            # Clean up by deleting the temporary file
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

    def predict(self, data):
        """
        Use the autoencoder model to make predictions.

        Parameters:
        - data: np.ndarray, the input data for predictions.

        Returns:
        - np.ndarray: The predicted values.
        """
        return self.autoencoder.predict(data)

class ModelTrainer:
    def __init__(self, training_set: pd.DataFrame, 
                 validation_set: pd.DataFrame, autoencoder_model: AutoencoderModel):
        """
        Initializes the ModelTrainer with training and validation data, and an autoencoder model.

        Parameters:
        - training_set: pd.DataFrame, the training data.
        - validation_set: pd.DataFrame, the validation data.
        - autoencoder_model: AutoencoderModel, the autoencoder model.
        """
        self.training_set = training_set
        self.validation_set = validation_set
        self.autoencoder_model = autoencoder_model

    def get_trained_model(self):
        """
        Return the trained autoencoder model.

        Returns:
        - tf.keras.Model: The trained autoencoder model.
        """
        return self.autoencoder_model.autoencoder

    def train_model(self, batch_size=128, epochs=100):
        """
        Train the autoencoder model with the specified batch size and number of epochs.

        Parameters:
        - batch_size: int, the batch size used for training.
        - epochs: int, the number of epochs for training.

        Raises:
        - ValueError: If training data is not provided.
        - Exception: For any errors occurring during the training process.
        """
        if self.training_set is None:
            raise ValueError("Training data has not been provided.")
        
        logging.info("Starting training of the autoencoder model.")
        
        try:
            # Train the autoencoder using the provided train_autoencoder function
            self.autoencoder_model.train_autoencoder(
                self.training_set, 
                self.validation_set, 
                batch_size=batch_size, 
                epochs=epochs
            )   
            logging.info("Autoencoder model trained successfully.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise
