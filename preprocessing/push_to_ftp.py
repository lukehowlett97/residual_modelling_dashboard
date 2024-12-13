import ftplib
import zipfile
from pathlib import Path
import yaml
from config_manager import ConfigManager
from FileLogging.simple_logger import SimpleLogger

class FTPUploader:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.ftp_details = config['ftp_details']
        self.folder_to_zip = Path(config['output_folder'])
        self.zip_path = self.folder_to_zip.with_suffix('.zip')

    def zip_folder(self):
        with zipfile.ZipFile(self.zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in self.folder_to_zip.rglob('*'):
                zipf.write(file, file.relative_to(self.folder_to_zip))

    def push_to_ftp(self):
        # Zip the folder
        self.logger.write_log(f"Zipping folder {self.folder_to_zip} to {self.zip_path}")
        self.zip_folder()

        # Push to FTP
        try:
            self.logger.write_log(f"Connecting to FTP {self.ftp_details['host']}")
            with ftplib.FTP(self.ftp_details['host']) as ftp:
                ftp.login(self.ftp_details['user'], self.ftp_details['passwd'])
                with open(self.zip_path, 'rb') as f:
                    ftp.storbinary(f'STOR {self.zip_path.name}', f)
            self.logger.write_log(f"Successfully pushed {self.zip_path.name} to FTP")
        except Exception as e:
            self.logger.write_log(f"Failed to push {self.zip_path.name} to FTP: {e}")

if __name__ == "__main__":
    # Initialize Logger
    log_file = Path("log_test.log")
    logger = SimpleLogger(log_file, True)
    logger.write_log("Starting FTP push process.")

    # Load Configuration
    config_path = "/home/methodman/Projects/res-mod-dashboard/preprocessing/prep_config.yaml"
    config_manager = ConfigManager(config_path, logger)
    config = config_manager.config

    if config['push_to_ftp']:
        push_to_ftp(config, logger)

    logger.write_log("FTP push process completed.")