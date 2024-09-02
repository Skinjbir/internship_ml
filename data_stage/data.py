import logging
from libd.Loader import Loader
from libd.Cleaner import Cleaner

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate data loading and cleaning.
    - Loads data from MinIO using the Loader class.
    - Cleans the data using the Cleaner class.
    - Logs the status and details of the operations.
    """
    try:
        # Initialize Loader with configuration
        loader = Loader(config_path='./data_config.yaml')
        
        # Data ingestion
        df = loader.load_data_from_minio()
       
        # Initialize Cleaner with configuration
        cleaner = Cleaner(
            bucket_name=loader.bucket_name,
            config_path='data_config.yaml'
        )

        # Clean the data
        cleaner.clean_data(data=df, file_name='cleaned_data.csv', target_column='Class')

        # Log cleaned data
        cleaned_data = cleaner.cleaned_data

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
