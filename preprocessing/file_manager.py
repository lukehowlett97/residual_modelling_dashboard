from FileLogging.simple_logger import SimpleLogger
from pathlib import Path
import json

class FileManager:
    
    def __init__(self, base_output_dir, logger = None):
        self.base_output_dir = Path(base_output_dir)
        self.logger = logger 

    def _log(self, message):
        if self.logger:
            self.logger.write_log(message)
        else:
            pass

    def construct_save_paths(self, config, prn_tag):
        year = str(config['year'])
        doy = f"{config['doy']:03d}"
        station = config['station']
        save_folder = self.base_output_dir / year / doy / station
        save_folder.mkdir(parents=True, exist_ok=True)
        
        save_seg_file = save_folder / f'seg_features_{year}_{doy}_{station}_{prn_tag}.pkl'
        save_sat_stats = save_folder / f'sat_stats_{year}_{doy}_{station}_{prn_tag}.json'
        save_sharp_events = save_folder / f'sharp_events_{year}_{doy}_{station}_{prn_tag}.json'
        group_save_filename = f"proc_res_{year}_{doy}_{station}_{prn_tag}.pkl"
        group_save_path = save_folder / group_save_filename
        
        return save_folder, save_seg_file, save_sat_stats, save_sharp_events, group_save_path

    def save_pickle(self, df, path):
        try:
            df.to_pickle(path)
            self._log(f"Saved pickle file: {path}")
        except Exception as e:
            self._log(f"Error saving pickle file {path}: {e}")
            raise

    def save_json(self, data, path):
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            self._log(f"Saved JSON file: {path}")
        except Exception as e:
            self._log(f"Error saving JSON file {path}: {e}")
            raise
    
    def extract_file_info(self, filename):
        """
        Extract station, year, and day of year from filename.
        Assumes format: CMDN_2024002_intg.res

        Parameters:
        - filename (Path): The Path object of the file.

        Returns:
        - tuple: (station (str), year (int), day_of_year (int))
        """
        base = filename.stem  # e.g., CMDN_2024002_intg
        parts = base.split('_')
        if len(parts) < 2:
            self._log(f"Filename {filename} does not match expected format.")
            raise ValueError(f"Filename {filename} does not match expected format.")
        station = parts[0]
        year_day = parts[1]
        if len(year_day) != 7:
            self._log(f"Year and day in filename {filename} do not match expected format.")
            raise ValueError(f"Year and day in filename {filename} do not match expected format.")
        year = int(year_day[:4])
        day_of_year = int(year_day[4:])
        return station, year, day_of_year