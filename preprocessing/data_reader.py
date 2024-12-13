# data_reading/data_reader.py
from Readers.i2gResRead import ReadI2GRes
from FileLogging.simple_logger import SimpleLogger
import pandas as pd

class DataReader:
    def __init__(self, logger = None):
        self.logger = logger
        
    def _log(self, message):
        if self.logger:
            self.logger.write_log(message)
        else:
            pass

    def read_res_file(self, filepath, columns):
        try:
            self._log(f"Reading file: {filepath}")
            df = ReadI2GRes(filepath).get_fix_s_data()
            df = df[['epoch', 'sys', 'num', *columns]]
            df[['sys', 'num']] = df[['sys', 'num']].astype(int)
            self._log(f"Successfully read and filtered data from {filepath}")
            return df
        except Exception as e:
            self._log(f"Error reading file {filepath}: {e}")
            raise
