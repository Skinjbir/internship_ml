import numpy as np
import logging

def detect_anomalies(autoencoder, X_test, threshold, metric):
    logging.info("Starting anomaly detection using autoencoder.")
    reconstructions = autoencoder.predict(X_test)

    if metric == 'mse':
        errors = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        logging.debug("Calculated MSE-based reconstruction errors.")
    elif metric == 'mae':
        errors = np.mean(np.abs(X_test - reconstructions), axis=1)
        logging.debug("Calculated MAE-based reconstruction errors.")
    else:
        raise ValueError("Unsupported metric. Use 'mse' or 'mae'.")

    outliers = errors >= threshold
    logging.debug(f"Number of detected anomalies: {np.sum(outliers)}")

    return outliers, errors, threshold
