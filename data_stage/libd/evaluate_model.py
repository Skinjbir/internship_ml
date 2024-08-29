import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import logging

def detect_anomalies(autoencoder, X_test, percentile=95, metric='mse'):
    """
    Detect anomalies using the autoencoder with a percentile-based threshold.

    Args:
    - autoencoder: The trained autoencoder model.
    - X_test: The test set to evaluate.
    - percentile: The percentile used to determine the anomaly threshold (default: 95).
    - metric: The error metric to use for anomaly detection ('mse' or 'mae').

    Returns:
    - outliers: A boolean array indicating which samples are considered anomalies.
    - errors: The reconstruction errors for each sample.
    - threshold: The anomaly threshold determined by the chosen percentile.
    """

    logging.info("Starting anomaly detection using autoencoder.")

    # Get the autoencoder's reconstructions
    reconstructions = autoencoder.predict(X_test)

    # Choose reconstruction error metric
    if metric == 'mse':
        errors = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        logging.debug("Calculated MSE-based reconstruction errors.")
    elif metric == 'mae':
        errors = np.mean(np.abs(X_test - reconstructions), axis=1)
        logging.debug("Calculated MAE-based reconstruction errors.")
    else:
        raise ValueError("Unsupported metric. Use 'mse' or 'mae'.")

    # Determine the threshold for anomaly detection
    threshold = np.percentile(errors, percentile)
    logging.info(f"Anomaly detection threshold set at the {percentile}th percentile: {threshold:.4f}")

    # Identify outliers
    outliers = errors > threshold
    logging.debug(f"Number of detected anomalies: {np.sum(outliers)}")

    return outliers, errors, threshold

def evaluate_anomalies(y_test, outliers, errors):
    """
    Evaluate the performance of anomaly detection.

    Args:
    - y_test: The true labels for the test set (binary: 0 for normal, 1 for anomaly).
    - outliers: A boolean array indicating which samples are considered anomalies.
    - errors: The reconstruction errors for each sample.

    Returns:
    - metrics: A dictionary containing precision, recall, f1_score, and auc.
    """

    logging.info("Evaluating anomaly detection performance.")

    precision = precision_score(y_test, outliers)
    recall = recall_score(y_test, outliers)
    f1 = f1_score(y_test, outliers)
    auc = roc_auc_score(y_test, errors)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

    logging.info(f"Evaluation Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

    return metrics
