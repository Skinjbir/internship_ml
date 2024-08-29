import logging
import pandas as pd
from libt.Loader import Loader 

class DataSplitter:
    def __init__(self, config_path: str) -> None:
        self.loader = Loader(config_path)
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """
        Load the processed data using the Loader class.
        """
        try:
            data = self.loader.load_data_from_minio()
            logging.info("Data loaded successfully from MinIO.")
            return data
        except Exception as e:
            logging.error(f"Failed to load data from MinIO: {e}")
            raise

    def check_column_exists(self, column_name: str) -> None:
        """
        Ensure the specified column exists in the DataFrame.
        """
        if column_name not in self.data.columns:
            logging.error(f"Column '{column_name}' not found in the DataFrame.")
            raise KeyError(f"Column '{column_name}' not found in the DataFrame.")
        logging.debug(f"Data columns: {self.data.columns.tolist()}")

    def split_by_class(self, class_column: str) -> tuple:
        """
        Splits the data into normal and fraud cases based on the class column.
        """
        self.check_column_exists(class_column)
        normal = self.data[self.data[class_column] == 0].sample(frac=1).reset_index(drop=True)
        fraud = self.data[self.data[class_column] == 1]
        logging.info(f"Data split by class '{class_column}' into {len(normal)} normal and {len(fraud)} fraud cases.")
        return normal, fraud

    def define_train_test_sets(self, normal: pd.DataFrame, fraud: pd.DataFrame, train_size: int) -> tuple:
        """
        Define training and test sets from the normal and fraud datasets.
        """
        X_train = normal.iloc[:train_size].drop('Class', axis=1)
        y_train = normal.iloc[:train_size]['Class']
        X_test = pd.concat([normal.iloc[train_size:], fraud]).sample(frac=1).reset_index(drop=True)
        y_test = X_test['Class']
        X_test = X_test.drop('Class', axis=1)
        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def split_data_v2(self, class_column: str = 'Class', train_size: int = 200000) -> tuple:
        """
        Custom logic to handle imbalanced datasets with fraud detection use case.
        Splits the data into training and test sets with the target variable.
        """
        try:
            # Split data by class
            normal, fraud = self.split_by_class(class_column)

            # Define training and test sets
            X_train, X_test, y_train, y_test = self.define_train_test_sets(normal, fraud, train_size)

            logging.info(f"Data split into custom train and test sets with {len(X_train)} training samples and {len(X_test)} test samples.")
            return X_train, X_test, y_train, y_test

        except KeyError as ke:
            logging.error(f"Key error: {ke}")
            raise
        except Exception as e:
            logging.error("Failed to split data using custom logic.")
            raise

