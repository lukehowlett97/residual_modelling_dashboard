# main.py
from config_manager import ConfigManager
from FileLogging.simple_logger import SimpleLogger
from data_reader import DataReader
from rolling_statistics import RollingStatistics
from segmentation import Segmentation
from segment_feature_extraction import FeatureExtractor
from event_detection import EventDetector
from statistics_manager import StatisticsManager
from file_manager import FileManager
from pathlib import Path
from zip_and_push_to_ftp import *
import pandas as pd

def main():
    # Initialize Logger
    log_file = Path("log_test.log")
    logger = SimpleLogger(log_file, True)
    logger.write_log("Starting preprocessing pipeline.")

    # Load Configuration
    # config_path = "prep_config.yaml"  # Path to your configuration file
    config_path = "/home/methodman/Projects/res-mod-dashboard/preprocessing/prep_config.yaml"
    config_manager = ConfigManager(config_path, logger)
    config = config_manager.config
    
    input_folder = Path(config['input_folder'])
    output_folder = Path(config['output_folder']) / f"processed_{input_folder.stem}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize FileManager
    file_manager = FileManager(output_folder)

    # # Initialize DataReader
    data_reader = DataReader(logger = logger)

    # # Initialize other processing classes as needed
    rolling_stats = RollingStatistics(logger = logger)
    segmentation = Segmentation(logger = logger)
    feature_extractor = FeatureExtractor(logger = logger)
    event_detector = EventDetector(logger = logger)
    statistics_manager = StatisticsManager(logger = logger)

    system_map = {1: 'G', 2: 'R', 3: 'C', 4: 'E'}

    file_count = 0
    max_files = config.get('max_files', None)

    for res_file in input_folder.glob(f'*{config["input_extension"]}'):
        if not res_file.is_file():
            continue

        if max_files is not None and file_count >= max_files:
            logger.write_log(f"Reached the maximum number of files to process: {max_files}")
            break

        file_count += 1

        try:
            station, year, day_of_year = file_manager.extract_file_info(res_file)
            logger.write_log(f"Processing file: {res_file.name} | Station: {station}, Year: {year}, Day: {day_of_year}")
        except ValueError as e:
            logger.write_log(f"Skipping file {res_file.name}: {e}")
            continue

        # Read data
        try:
            df = data_reader.read_res_file(res_file, config['columns_to_process'])
        except Exception as e:
            logger.write_log(f"Error reading file {res_file.name}: {e}")
            continue

        # Group by sys and num and process each group hello guys haha boss smells 
        try:
            grouped = df.groupby(['sys', 'num'])

            group_config = config.copy()
            group_config['output_folder'] = output_folder
            group_config['station'] = station
            group_config['year'] = year
            group_config['doy'] = day_of_year

            for (sys, num), group_df in grouped:
                try:
                    group_df = group_df.reset_index(drop=True)

                    # Calculate rolling statistics
                    rolling_stats_df = rolling_stats.calculate(group_df, group_config)
                    # logger.write_log(f"Calculated rolling statistics for group {sys}, {num}")

                    # Segmentation
                    segmented_periods_dict = segmentation.analyse_periods(group_df, group_config)
        
                    # Feature Extraction
                    segmented_features_df = feature_extractor.extract_features_from_segments(group_df, segmented_periods_dict)

                    # Event Detection
                    sharp_events = event_detector.events_detection(rolling_stats_df, group_config)

                    # Statistics Calculation
                    sat_stats = statistics_manager.single_sat_statistics(group_df, segmented_features_df, sharp_events, group_config)

                    # Save results
                    prn_tag = f"{system_map.get(int(sys), 'Unknown')}{int(num):02d}"
                    save_folder, seg_path, stats_path, events_path, group_save_path = file_manager.construct_save_paths(group_config, prn_tag)


                    file_manager.save_pickle(segmented_features_df, seg_path)
                    file_manager.save_json(sat_stats, stats_path)
                    file_manager.save_json(sharp_events, events_path)

                    # Save processed group data
                    file_manager.save_pickle(rolling_stats_df, group_save_path)

                except Exception as e:
                    logger.write_log(f"Error processing group {sys}, {num} in file {res_file.name}: {e}")
                    continue

        except Exception as e:
            logger.write_log(f"Error processing groups in file {res_file.name}: {e}")
            continue

    # Generate dataset statistics
    try:
        statistics_manager.generate_dataset_statistics(output_folder)
    except Exception as e:
        logger.write_log(f"Error generating dataset statistics: {e}")

    if config['push_to_ftp']:
        
        pass


    logger.write_log("Preprocessing pipeline completed.")

if __name__ == "__main__":
    main()
