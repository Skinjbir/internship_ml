from flask import Flask, request, jsonify
import logging
import pandas as pd
from utils.config_loader import load_config
from utils.minio_utils import initialize_minio_client, load_model_from_minio
from utils.anomalie import detect_anomalies

app = Flask(__name__)




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load configuration
        config = load_config('server_config.yaml')

        # Initialize MinIO client
        minio_client = initialize_minio_client(config)


        model = load_model_from_minio(minio_client, 'main-bucket', 'models/model.keras')
        
        data = request.get_json()
        logging.info(f"Received data: {data}")

        df = pd.DataFrame(data)

        if df.shape[1] == 1:
            df = df.values.reshape(-1)
        df = df.reshape(1, -1)

        logging.info(f"Reshaped data: {df}")    

        outliers, errors, threshold = detect_anomalies(
            model, df, 
            config['anomaly_detection']['threshold'], 
            config['anomaly_detection']['metric']
        )
        
        return jsonify({'outliers': outliers.tolist(), 'errors': errors.tolist(), 'threshold': threshold})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host='0.0.0.0', port=5000)
