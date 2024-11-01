# Preprocessing Pipeline Module

This **Preprocessing Package** executes a comprehensive **GNSS data preprocessing pipeline** that includes data ingestion, segmentation, feature extraction, event detection, and statistical analysis. The module supports large-scale data processing, with logging and error handling capabilities.

---

## Workflow Overview

1. **Configuration & Logging**
   - Initializes logging with `SimpleLogger`.
   - Loads configurations from `config.yaml`, covering input/output paths and processing parameters.

2. **Data Reading**
   - Reads GNSS residual data files from a specified input directory via `DataReader`.

3. **Processing Steps**
   - For each data group (by `sys` and `num`):
      - **Rolling Statistics**: Computes rolling statistics for data smoothing and baseline metrics.
      - **Segmentation**: Identifies behavioral segments in the data.
      - **Feature Extraction**: Derives features for each segment.
      - **Event Detection**: Detects sharp changes within rolling statistics.
      - **Statistics Calculation**: Computes single-satellite statistics and merges features and events for analysis.

4. **Output & Saving**
   - Outputs processed files in the structured format defined by `FileManager`.
   - Saves segmented features, satellite statistics, event data, and final processed data.

5. **Dataset Statistics**
   - Generates and saves overall statistics for the entire dataset.

---

## Module Descriptions

### ConfigManager

The **ConfigManager** component centralizes configuration management for the preprocessing pipeline, ensuring consistency across the process.

**Key Features:**
- **Defaults**: Loads default settings, covering columns to process, rolling window size, polynomial order, input file extension, and file patterns.
- **Custom Config Loading**: Reads additional settings from `config.yaml` or JSON configuration files, updating defaults as needed and logging successful loads.
- **Logging**: Uses `SimpleLogger` to report issues during configuration loading.
- **Config Access**: Provides a `get` method for retrieving configuration values, accessing both defaults and custom settings.

---

### FileManager

The **FileManager** module manages file operations, ensuring organized and traceable data storage.

**Key Features:**
- **Path Construction**: The `construct_save_paths` method creates organized output paths based on configuration (e.g., year, day of year, station, and PRN tag) and ensures directories are created prior to saving.
- **Data Saving**:
    - `save_pickle`: Saves DataFrames in pickle format for efficient retrieval.
    - `save_json`: Writes JSON files for serialized data storage, including statistics and event detections.
- **File Info Extraction**: The `extract_file_info` method extracts metadata (e.g., station, year, day of year) from filenames following a specific format (e.g., `CMDN_2024002_intg.res`), raising an error for non-standard formats.

---

### RollingStatistics

The **RollingStatistics** module computes rolling statistics on GNSS residual data to identify trends and variances over a sliding window.

**Key Features:**
- **Rolling Calculations** for each column, including:
    - **Differences (diff)**: Measures changes between consecutive entries.
    - **Rolling Mean**: Calculates average values over a specified window.
    - **Rolling Standard Deviation**: Quantifies variability within the window.
    - **Savitzky-Golay Filter (sg_filter)**: Applies polynomial smoothing to capture underlying trends.
- **Configurable Parameters**: Uses configuration settings for window size and polynomial order.

---

### Segmentation

The **Segmentation** module identifies stable and unstable periods within GNSS time-series data, enabling the detection of behavioral patterns.

**Key Features:**
- **Stable Period Detection**: The `detect_stable_periods` method identifies stable data segments based on a configurable variance threshold. An expanding window approach ensures that each segment meets stability criteria.
- **Unstable Period Detection**: The `get_unstable_periods` method identifies instability by filling gaps between stable segments.
- **Flexible Configuration**: Parameters such as `initial_window_size`, `max_window_size`, `variance_threshold`, and `step_size` allow for fine-tuning based on data characteristics.

---

### FeatureExtractor

The **FeatureExtractor** module derives statistical and temporal features from segmented GNSS data, capturing unique characteristics of both stable and unstable periods.

**Key Features:**
- **Feature Extraction**: Calculates a configurable set of features for each segment, including:
    - **Mean and Standard Deviation**: Provides basic statistical insights.
    - **Length**: Captures duration of each segment.
    - **Slope**: Calculates data trend over time via linear regression.
    - **Kurtosis and Skewness**: Provides higher-order distribution analysis.
- **Segmented Analysis**: Processes both stable and unstable periods, labeling them for easy categorization and analysis.
- **Robust Logging and Error Handling**: Logs each step, skips empty/problematic segments, and reports successes or failures.

---

### StatisticsManager

The **StatisticsManager** module computes and compiles comprehensive statistics for GNSS data, supporting both single-satellite and dataset-wide analysis.

**Key Features:**
- **Single-Satellite Statistics**: The `single_sat_statistics` method computes a range of statistics for individual satellites, including:
    - Range, interquartile range, standard deviation, stability percentages, event counts, mean, median, kurtosis, and skewness.
- **Comprehensive Dataset Statistics**: The `generate_dataset_statistics` method aggregates statistics across all satellite files, normalizes nested data, and reorders columns for clarity.
- **Flexible Configurations**: Specifies which columns and statistics to calculate, adaptable to different datasets.
- **Saving and Logging**: Results are saved as JSON and pickled for efficient access, with detailed logging to monitor progress and identify issues.

---

This module suite offers a robust framework for **automated GNSS data preprocessing**, enabling large-scale data handling, streamlined analysis, and organized storage for efficient processing and subsequent analysis.
