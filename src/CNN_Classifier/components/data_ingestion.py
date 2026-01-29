import os
import zipfile
import gdown
import time
from CNN_Classifier import logger
from CNN_Classifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """Download dataset from Google Drive"""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            # Check if file already exists
            if os.path.exists(zip_download_dir):
                file_size = os.path.getsize(zip_download_dir)
                if file_size > 1000000:  # If file is larger than 1MB, assume it's complete
                    logger.info(f"File already exists at {zip_download_dir}")
                    return
                else:
                    # Remove incomplete file
                    logger.info(f"Removing incomplete file")
                    os.remove(zip_download_dir)

            logger.info(f"Downloading data from {dataset_url}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="

            # Download with fuzzy parameter to handle large files
            gdown.download(
                prefix + file_id,
                str(zip_download_dir),
                quiet=False,
                fuzzy=True
            )

            # Wait a moment for file to be fully written
            time.sleep(2)

            logger.info(f"Downloaded data to {zip_download_dir}")

        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            # Clean up partial files
            try:
                # Remove .part files
                import glob
                part_files = glob.glob(str(os.path.dirname(zip_download_dir)) + "/*.part")
                for part_file in part_files:
                    try:
                        os.remove(part_file)
                        logger.info(f"Cleaned up partial file: {part_file}")
                    except:
                        pass
            except:
                pass
            raise e

    def extract_zip_file(self):
        """Extract zip file"""
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            # Check if already extracted
            expected_folder = os.path.join(unzip_path, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
            if os.path.exists(expected_folder):
                logger.info(f"Data already extracted at {expected_folder}")
                return

            logger.info(f"Extracting {self.config.local_data_file}")

            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extracted zip file into {unzip_path}")

        except Exception as e:
            logger.error(f"Error extracting zip file: {e}")
            raise e