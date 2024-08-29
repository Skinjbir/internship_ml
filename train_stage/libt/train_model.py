import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from datetime import datetime
import os

import pickle

# Configure logging to include DEBUG level messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    Create an autoencoder model.

    Parameters:
    - input_dim: int, the dimensionality of the input data.

    Returns:
    - tf.keras.Model: The constructed autoencoder model.
    """
    autoencoder = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(12, activation='elu'),
        tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(4, activation='elu'),
        tf.keras.layers.Dense(2, activation='elu'),
        tf.keras.layers.Dense(4, activation='elu'),
        tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(12, activation='elu'),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(input_dim, activation='elu')
    ])
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return autoencoder


def train_autoencoder(autoencoder, X_train, X_validate, batch_size=128, epochs=100):
    """Train the autoencoder model.

    Args:
        autoencoder (tf.keras.Model): The autoencoder model to be trained.
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
    log_subdir = f'{yyyymmddHHMM}_batch{BATCH_SIZE}_layers{len(autoencoder.layers)}'
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
        history = autoencoder.fit(
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

    return autoencoder, history

class ModelTrainer:
    def __init__(self, training_set: pd.DataFrame, 
                 validation_set: pd.DataFrame, autoencoder: tf.keras.Model):
        """
        Initializes the ModelTrainer with training and validation data, and an autoencoder model.

        Parameters:
        - training_set: pd.DataFrame, the training data.
        - validation_set: pd.DataFrame, the validation data.
        - autoencoder: tf.keras.Model, the autoencoder model.
        """
        self.training_set = training_set
        self.validation_set = validation_set
        self.autoencoder = autoencoder
    
    def get_trained_model(self):
        """
        Return the trained autoencoder model.

        Returns:
        - tf.keras.Model: The trained autoencoder model.
        """
        return self.autoencoder


    def train_model(self, batch_size=128, epochs=100, model_path="models/model.plk"):
        """
        Train the autoencoder model with the specified batch size and number of epochs,
        and save the trained model to the given path.

        Parameters:
        - batch_size: int, the batch size used for training.
        - epochs: int, the number of epochs for training.
        - model_path: str, the path where the trained model will be saved.

        Raises:
        - ValueError: If training data is not provided.
        - Exception: For any errors occurring during the training process.
        """
        if self.training_set is None:
            raise ValueError("Training data has not been provided.")
        
        logging.info("Starting training of the autoencoder model.")
        
        try:
            # Train the autoencoder using the provided train_autoencoder function
            self.autoencoder, self.history = train_autoencoder(
                self.autoencoder, 
                self.training_set, 
                self.validation_set, 
                batch_size=batch_size, 
                epochs=epochs
            )   
            logging.info("Autoencoder model trained successfully.")
            
            # Save the trained autoencoder model
            self.save_trained_model(model_path)
            logging.info(f"Autoencoder model saved successfully at {model_path}")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def save_trained_model(self, filepath: str):
        """
        Save the trained autoencoder model to the specified file path.

        Parameters:
        - filepath: str, the path where the model should be saved.

        Raises:
        - ValueError: If the model is not trained or provided.
        - Exception: For any errors occurring during the saving process.
        """
        if self.autoencoder is None:
            raise ValueError("No model to save. Ensure the model is trained before saving.")
        
        try:
            pickle.dump(self.autoencoder, open(filepath, 'wb'))
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error in saving the model: {e}")
            raise
