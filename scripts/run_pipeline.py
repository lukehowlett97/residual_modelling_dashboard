import yaml
import pandas as pd
from pathlib import Path
from src.data_loader import load_single_file
from src.segmentation_main import apply_segmentation
from src.feature_engineering import feature_engineering_pipeline
from src.statistical_sweep import statistical_sweep
# from src.evaluation import evaluate_model

def load_config(config_file="config/config.yaml"):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def process_file_pipeline(file: Path, config: dict) -> None:
    """
    Process a single file through the entire pipeline.
    """
    print(f"Processing file: {file}")

    # Step 1: Load Data
    df = load_single_file(file,
                          constellations=config['filters'].get('constellations'),
                          excluded_svs=config['filters'].get('excluded_satellites'),
                          residuals=config['filters'].get('residuals'),
                          normalize_data=True)
    
    # Step 2: statistical sweep
    station_name = file.stem[:4]
    outlier_count_dir_path = Path(config['data']['processed_data_path']) / 'outlier_counts'
    # df = statistical_sweep(df, config['filters'].get('residuals'), station_name, output_dir=outlier_count_dir_path)
    
    # Step 2: Apply Segmentation
    segmented_df = apply_segmentation(df, config['segmentation'])
    output_dir = Path(config['data']['processed_data_path'])
    output_segmented_csv_path = output_dir / 'segmented_data'/ file.with_name(file.stem + "_segmented.csv").name
    segmented_df.to_csv(output_segmented_csv_path, index=False)

    # # Step 3: Feature Engineering
    # features_df = feature_engineering_pipeline(segmented_df)
    # features_file = segmented_file.with_name(segmented_file.stem.replace("_segmented", "_features") + ".csv")
    # features_df.to_csv(features_file, index=False)

    # # Step 4: Model Evaluation
    # evaluation_results = evaluate_model(features_df, config['evaluation'])
    # evaluation_file = features_file.with_name(features_file.stem.replace("_features", "_evaluation") + ".csv")
    # evaluation_results.to_csv(evaluation_file, index=False)

def main():
    # Load configuration
    config = load_config()

    # Path to input files
    input_path = Path(config['data']['raw_data_path'])

    # Iterate over each file and process through the pipeline
    for file in input_path.glob(config['data']['file_pattern']):
        try:
            process_file_pipeline(file, config)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    main()
