import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import boto3
import yaml

logging.basicConfig(level=logging.INFO)

class Cleaner:
    def __init__(self, bucket_name: str = None, config_path: str = None) -> None:
        """
        Initialize the Cleaner with configuration details and MinIO client.
        
        Args:
            bucket_name (str, optional): Name of the MinIO bucket.
            config_path (str, optional): Path to the YAML configuration file.
        """
        self.cleaned_data = None
        self.bucket_name = bucket_name
        self.config = self._load_config(config_path)
        
        # Connect to MinIO
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.config['minio']['endpoint_url'],
            aws_access_key_id=self.config['minio']['access_key'],
            aws_secret_access_key=self.config['minio']['secret_key']
        )
        logging.info("Cleaner initialized with configuration and MinIO client.")

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        
        Returns:
            dict: Configuration details.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _save_cleaned_data(self, file_name: str = 'cleaned_data.csv') -> None:
        """
        Save the cleaned data to MinIO.
        
        Args:
            file_name (str): Name of the file to save to MinIO.
        """
        if self.cleaned_data is not None:
            csv_data = self.cleaned_data.to_csv(index=False)
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f'data/processed/{file_name}',
                    Body=csv_data
                )
                logging.info(f"Cleaned data saved to {self.bucket_name}/{file_name}")
            except Exception as e:
                logging.error(f"Error saving cleaned data: {e}")
        else:
            logging.warning("No cleaned data to save.")

    def _encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label encode categorical columns.
        
        Args:
            df (pd.DataFrame): DataFrame to encode.
        
        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        for col in df.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            logging.info(f"Encoded column '{col}' with labels: {encoder.classes_}")
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to clean.
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        initial_shape = df.shape
        df = df.drop_duplicates()
        logging.info(f"Removed duplicates: {initial_shape} -> {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with column means.
        
        Args:
            df (pd.DataFrame): DataFrame with missing values.
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        if df.isnull().sum().sum() > 0:
            df = df.fillna(df.mean(numeric_only=True))
            logging.info("Filled missing values with column means.")
        return df

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using MinMaxScaler.
        
        Args:
            X (pd.DataFrame): DataFrame of features to scale.
        
        Returns:
            pd.DataFrame: Scaled features.
        """
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Scaled features using normalization.")
        return pd.DataFrame(X_scaled, columns=X.columns)

    def split_to_scale(self, df: pd.DataFrame, target_column: str = 'Class'):
        """
        Splits the DataFrame into features and target, scales the features.
        
        Args:
            df (pd.DataFrame): DataFrame to split and scale.
            target_column (str): Name of the target column.
        
        Returns:
            tuple: Scaled features and target column.
        """
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_scaled = self._scale_features(X)
            
            return X_scaled, y
        except Exception as e:
            logging.error(f"Error during feature scaling: {e}")
            raise

    def clean_data(self, data: pd.DataFrame, file_name: str, target_column: str = 'Class') -> None:
        """
        Perform full data cleaning, encoding, and scaling.
        
        Args:
            data (pd.DataFrame): DataFrame to clean.
            file_name (str): Name of the file to save cleaned data.
            target_column (str): Name of the target column.
        """
        try:
            # Perform data cleaning steps in a pipeline
            X_scaled, y = (data
                        .pipe(self._drop_duplicates)
                        .pipe(self._handle_missing_values)
                        .pipe(self._encode_columns)
                        .pipe(self.split_to_scale, target_column))

            # Combine the scaled features with the target
            self.cleaned_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
            
            logging.info("Data cleaning completed successfully.")
            # Save the cleaned data
            self._save_cleaned_data(file_name)
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise
