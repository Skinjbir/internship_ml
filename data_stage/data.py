import logging
from libd.Loader import Loader
from libd.Cleaner import Cleaner

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Initialize Loader with configuration
        loader = Loader(config_path='data_config.yaml')

        # Data ingestion
        df = loader.load_data_from_minio()
        logging.info("Data loaded successfully.")
        logging.debug(f"Data sample:\n{df.head()}")

        # Initialize Cleaner with configuration and data
        scale_method = loader.config['cleaning']['scale_method']
        cleaner = Cleaner(
            data=df,
            scale_method=scale_method,
            bucket_name=loader.bucket_name,
            config_path='data_config.yaml'
        )
        cleaner.clean_data(file_name='cleaned_data.csv')

        # Log cleaned data and save it
        cleaned_data = cleaner.cleaned_data
        logging.debug(f"Cleaned data:\n{cleaned_data.head()}")
        cleaner.save_cleaned_data()

        # Signal completion
        with open('/data/s.signal', 'w') as f:
            f.write('done')
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
