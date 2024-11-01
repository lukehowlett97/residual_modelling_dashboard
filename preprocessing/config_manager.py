# config/config_manager.py
import json
import yaml
from pathlib import Path
from FileLogging.simple_logger import SimpleLogger

class ConfigManager:
    def __init__(self, config_path=None, logger = None):
        self.logger = logger 
        
        self.config = self.load_default_config()
        
        if config_path:
            self.load_config_file(config_path)
            
        
    def _log(self, message):
        if self.logger:
            self.logger.write_log(message)
        else:
            pass

    def load_default_config(self):
        return {
            'columns_to_process': ["res_oc1", "reg_trop", "reg_iono", "ppprtk1"],
            'rolling_window': 10,
            'poly_order': 2,
            'input_extension': '.res',
            'file_pattern': '*_???????_intg.res',
            'max_files': None,
            'output_subdir_prefix': 'processed_',
            # Add other default configurations here
        }

    def load_config_file(self, config_path):
        config_path = Path(config_path)
        if not config_path.exists():
            self._log(f"Configuration file {config_path} does not exist.")
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
            else:
                self._log("Unsupported configuration file format.")
                raise ValueError("Unsupported configuration file format.")
            
            self.config.update(file_config)
            self._log(f"Configuration loaded from {config_path}.")
        except Exception as e:
            self._log(f"Error loading configuration file {config_path}: {e}")
            raise

    def get(self, key, default=None):
        return self.config.get(key, default)
