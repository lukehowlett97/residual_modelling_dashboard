# Dashboard Preprocessing Pipeline Documentation

This document provides a detailed overview of the **GNSS Data Preprocessing and Dashboard Application**, which includes initialization, data loading, interactive components, and logging mechanisms.

---

## Setup and Structure

### 1. App Initialization

**File**: `app.py`

- **Purpose**: Initializes the Dash application and integrates Bootstrap for responsive styling across the dashboard.

### 2. Main Application Entry

**Files**: `main.py`, `layout.py`

- **`main.py`**:
  - **Function**: Runs the Dash application.
  - **Responsibilities**: Applies the main layout and registers all callbacks.

- **`layout.py`**:
  - **Function**: Defines the overall layout of the dashboard.
  - **Components**: Customizable quadrants, data selectors, time series visualization, data summary tables, and interactive statistics panels.

---

## Core Components

The dashboard is divided into **four main quadrants**, each with distinct functionalities:

- **Top Left (Data/File Selector and Summary)**
  - **Filtering**: Folder, Year, DOY (Day of Year), Station, and PRNs (satellite identifiers).
  - **Data Summary Tab**: Displays summary statistics with sorting, filtering, and row selection capabilities.

- **Top Right (Time Series Plot)**
  - **Time Series Graphs**: Interactive plots with event overlay capabilities for detailed data analysis.

- **Bottom Left (Data Configuration)**
  - **Settings**: Options for residual types, data formats, and event labels to configure the displayed data.

- **Bottom Right (Statistics Panel)**
  - **Statistics Display**: Real-time dynamic statistics, updating based on user selections.

---

## Modules

### Data Loader (`data_loader.py`)

Handles data loading and navigation through the GNSS time-series data folder structure.

**Key Functions**:
- **Filename Parsing**: `parse_filename` extracts metadata from file names (e.g., system and PRN).
- **Directory Listings**:
  - `list_years`: Lists available years.
  - `list_doys`: Lists DOYs within selected years.
  - `list_stations`: Lists available stations by year and DOY.
- **Data Loading**:
  - `list_pkl_files`: Locates relevant `.pkl` files based on filters.
  - `load_data`: Loads and converts epochs in `.pkl` files for time series compatibility.
- **PRN Organization**: `get_prns_by_system` organizes PRNs by GNSS system (e.g., GPS, GLONASS).

### Dashboard Styling and Configuration

#### Styling (`styles.py`)

Contains custom CSS configurations for the dashboard’s appearance.

**Key Style Elements**:
- **Quadrants**: Defines styles for dashboard sections, including borders, backgrounds, and text colors.
- **Expand Buttons**: Styling for expand buttons on each quadrant.
- **PRN Checklist Label**: Custom styles for PRN selection labels for consistent layout.

#### Constants (`constants.py`)

Defines constants used across the dashboard for consistency.

**Key Constants**:
- **DATA_FOLDER**: Path to the main data directory.
- **residual_types**: Available residual types for analysis.
- **data_formats**: Suffixes for data formats (e.g., rolling mean, diff).
- **event_labels** and **event_colors**: Label and color mappings for time series events.

### Dataset Statistics Generation (`generate_statistics.py`)

Aggregates and normalizes statistical data from satellite JSON files into a structured pandas DataFrame.

**Key Function**:
- **`generate_dataset_statistics`**:
  - Reads satellite JSON files.
  - Flattens nested dictionaries using `pandas.json_normalize`.
  - Converts non-numeric values to NaN.
  - Extracts metadata from file names (year, DOY, station, PRN).
  - Renames and reorders columns for clarity.
  - Saves the consolidated DataFrame as a pickle file for efficient retrieval.

### Core Interactivity (`callbacks.py`)

Manages the callback functions that drive interactivity within the dashboard.

**Key Callbacks**:

- **Data Filtering and Selection**
  - **Year, DOY, and Station Selectors**: Populates dropdowns based on previous selections.
  - **Files Info and PRN Selection Grid**: Updates PRNs and files info based on filters.
  - **Data Summary Table Selection**: Adds selected rows from the data summary table to datasets for analysis.

- **Data Display and Visualization**
  - **Time Series Plot**: Updates the plot based on dataset selection, residual types, and data format.
  - **Statistics Display**: Shows metrics in the statistics panel based on selected datasets.

- **Expand/Collapse Quadrants**:
  - Allows any quadrant to expand to full-screen for focused analysis.

---

## Logging

The application uses a custom **SimpleLogger** for tracking informational messages and errors. Log files are saved in the `logs/` directory, with separate logs for different modules:

- **segmentation.log**
- **feature_extraction.log**
- **dataset_statistics.log**

---

## Usage

### Running the Dashboard

1. **Start the Application**
   python main.py
