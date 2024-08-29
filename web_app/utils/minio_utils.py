import minio
from tensorflow.keras.models import load_model

def initialize_minio_client(config):
    return minio.Minio(
        endpoint=config['minio']['endpoint'],
        access_key=config['minio']['access_key'],
        secret_key=config['minio']['secret_key'],
        secure=config['minio']['secure']
    )

def load_model_from_minio(minio_client, bucket_name, model_filename):
    model_path = f"/tmp/{model_filename}"
    minio_client.fget_object(bucket_name, model_filename, model_path)
    model = load_model(model_path)
    return model
