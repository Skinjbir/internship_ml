import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from io import StringIO
import yaml
import boto3

# Load the configuration file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Access the MinIO configuration
minio_config = config['minio']


class DataCleaner:
    def __init__(self, data: pd.DataFrame, scale_method: str = 'normalize', bucket_name: str = None) -> None:
            if scale_method not in ['standardize', 'normalize']:
                raise ValueError("Invalid scale_method. Choose 'standardize' or 'normalize'.")
            self.scale_method = scale_method
            self.data = data
            self.cleaned_data = None
            self.bucket_name = bucket_name

            self.s3_client = boto3.client(
            's3',
            endpoint_url=minio_config['endpoint_url'],
            aws_access_key_id=minio_config['aws_access_key_id'],
            aws_secret_access_key=minio_config['aws_secret_access_key']
            )

    def save_cleaned_data(self, file_name: str = 'cleaned_data.csv') -> None:
        try:
            # Convert cleaned data to CSV
            if self.cleaned_data is not None:
                csv_data = self.cleaned_data.to_csv(index=False)
                # Save the CSV data to the specified S3 bucket and folder
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f'data/processed/{file_name}',
                    Body=csv_data
                )
                print(f"Cleaned data saved to {self.bucket_name}/{file_name}")
            else:
                print("No cleaned data to save.")
        except KeyError as e:
            print(f"Error saving cleaned data: {e}")
        
    def encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label encode categorical columns."""
        for col in df.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            logging.info(f"Encoded column '{col}' with labels: {encoder.classes_}")
        return df

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the DataFrame."""
        initial_shape = df.shape
        df = df.drop_duplicates()
        logging.info(f"Removed duplicates: {initial_shape} -> {df.shape}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with column means."""
        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.mean(numeric_only=True))
            logging.info("Filled missing values with column means.")
        return df

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features based on the chosen method."""
        scaler = StandardScaler() if self.scale_method == 'standardize' else MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info(f"Scaled features using {self.scale_method}.")
        return pd.DataFrame(X_scaled, columns=X.columns)

    def clean_data(self, file_name: str, target_column: str = 'Class'):
        """Perform full data cleaning, encoding, and scaling."""
        try:
            df = self.drop_duplicates(self.data)
            df = self.handle_missing_values(df)
            df = self.encode_columns(df)

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_scaled = self.scale_features(X)
            self.cleaned_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
            logging.info("Data cleaning completed successfully.")

                    
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise e
