import pandas as pd
import logging

class DataSplitter:
    def __init__(self, path: str) -> None:
        self.data = self.load_data(path)

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load the processed data from a CSV file.
        """
        try:
            data = pd.read_csv(path)
            logging.info(f"Data loaded successfully from {path}.")
            return data
        except Exception as e:
            logging.error(f"Failed to load data from {path}: {e}")
            raise e

    def check_column_exists(self, column_name: str) -> None:
        """
        Ensure the specified column exists in the DataFrame.
        """
        if column_name not in self.data.columns:
            raise KeyError(f"Column '{column_name}' not found in the DataFrame.")
        logging.debug(f"Data columns: {self.data.columns.tolist()}")

    def split_by_class(self, class_column: str) -> tuple:
        """
        Splits the data into normal and fraud cases based on the class column.
        """
        normal = self.data[self.data[class_column] == 0].sample(frac=1).reset_index(drop=True)
        fraud = self.data[self.data[class_column] == 1]
        return normal, fraud

    def define_train_test_sets(self, normal: pd.DataFrame, fraud: pd.DataFrame, train_size: int) -> tuple:
        """
        Define training and test sets from the normal and fraud datasets.
        """
        X_Train = normal.iloc[:train_size].drop('Class', axis=1)
        y_Train = normal.iloc[:train_size]['Class']
        X_Test = pd.concat([normal.iloc[train_size:], fraud]).sample(frac=1).reset_index(drop=True)
        y_Test = X_Test['Class']
        X_Test = X_Test.drop('Class', axis=1)
        return X_Train, X_Test, y_Train, y_Test

    def split_data_v2(self) -> tuple:
        """
        Custom logic to handle imbalanced datasets with fraud detection use case.
        Splits the data into training and test sets with the target variable.
        """
        try:
            # Check if 'Class' column exists
            self.check_column_exists('Class')

            # Split data by class
            normal, fraud = self.split_by_class('Class')

            # Define training and test sets
            X_Train, X_Test, y_Train, y_Test = self.define_train_test_sets(normal, fraud, 200000)

            logging.info(f"Data split into custom train and test sets with {len(X_Train)} training samples and {len(X_Test)} test samples.")
            return X_Train, X_Test, y_Train, y_Test

        except KeyError as ke:
            logging.error(f"Key error: {ke}")
            raise ke
        except Exception as e:
            logging.error("Failed to split data using custom logic.")
            raise e
