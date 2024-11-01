import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
import pandas as pd
import numpy as np

# Import the functions to be tested
from src.data_loader import process_file, load_and_process_files, normalize

class TestDataLoader(unittest.TestCase):

    @patch('Readers.ReadI2GRes')  # Mock the ReadI2GRes class
    @patch('pathlib.Path.open', new_callable=mock_open)  # Mock Path.open to prevent FileNotFoundError
    def test_process_file_basic(self, mock_open, MockReadI2GRes):
        # Create a mock DataFrame to be returned by the mocked ReadI2GRes
        mock_df = pd.DataFrame({
            'epoch': pd.date_range('2024-01-01', periods=5, freq='min'),
            'sys': [1, 2, 1, 1, 2],  # Ensure these match the constellation filter
            'num': [10, 20, 10, 10, 20],  # Ensure these match the SV filter
            'reg_iono': np.random.randn(5)
        })
        
        # Set up the mock to return the mock DataFrame
        MockReadI2GRes.return_value.get_fix_s_data.return_value = mock_df

        # Set up a mock for the processing function
        mock_processing_function = MagicMock()

        # Call the function under test
        process_file(
            file=Path('/fake/path/file.res'),
            constellations=[1, 2],  # Filtering criteria that should pass some rows
            svs=[10, 20],  # Filtering criteria that should pass some rows
            residuals=['reg_iono'],
            normalize_columns=['reg_iono'],
            processing_function=mock_processing_function
        )

        # Create the expected DataFrame
        expected_df = mock_df[(mock_df['sys'].isin([1, 2])) & (mock_df['num'].isin([10, 20]))].copy()
        expected_df['reg_iono'] = normalize(expected_df['reg_iono'])
        expected_df.reset_index(drop=True, inplace=True)

        # Debugging statement to print the actual DataFrame being passed to the processing function
        print("Filtered DataFrame:\n", mock_processing_function.call_args[0][0])

        # Ensure the DataFrames have identical labels before comparison
        pd.testing.assert_frame_equal(
            mock_processing_function.call_args[0][0],
            expected_df
        )
        mock_processing_function.assert_called_once_with(expected_df, Path('/fake/path/file.res'))

    @patch('Readers.ReadI2GRes')
    @patch('pathlib.Path.open', new_callable=mock_open)  # Mock Path.open to prevent FileNotFoundError
    def test_process_file_no_processing_function(self, mock_open, MockReadI2GRes):
        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            'epoch': pd.date_range('2024-01-01', periods=5, freq='min'),
            'sys': [1, 2, 3, 1, 2],
            'num': [10, 20, 30, 10, 20],
            'reg_iono': np.random.randn(5)
        })

        # Set up the mock to return the mock DataFrame
        MockReadI2GRes.return_value.get_fix_s_data.return_value = mock_df

        # Mock the to_csv method
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            process_file(
                file=Path('/fake/path/file.res'),
                constellations=[1, 2],
                svs=[10, 20],
                residuals=['reg_iono'],
                normalize_columns=['reg_iono'],
                processing_function=None
            )
            
            # Create the expected DataFrame
            expected_df = mock_df[(mock_df['sys'].isin([1, 2])) & (mock_df['num'].isin([10, 20]))].copy()
            expected_df['reg_iono'] = normalize(expected_df['reg_iono'])
            expected_df.reset_index(drop=True, inplace=True)

            output_file = Path('/fake/path/file.processed.csv')
            mock_to_csv.assert_called_once_with(output_file, index=False)
    
    @patch('src.data_loader.process_file')  # Correctly patch the process_file function
    @patch('pathlib.Path.glob')
    def test_load_and_process_files(self, mock_glob, mock_process_file):
        # Mock the file paths returned by glob
        mock_glob.return_value = [Path('/fake/path/file1.res'), Path('/fake/path/file2.res')]

        # Call the function under test
        load_and_process_files(
            input_folder='/fake/path',
            constellations=[1],
            svs=[10],
            residuals=['reg_iono'],
            normalize_columns=['reg_iono'],
            processing_function=None
        )

        # Assertions
        expected_calls = [
            call(Path('/fake/path/file1.res'), [1], [10], ['reg_iono'], ['reg_iono'], None),
            call(Path('/fake/path/file2.res'), [1], [10], ['reg_iono'], ['reg_iono'], None)
        ]
        mock_process_file.assert_has_calls(expected_calls)

    def test_normalize(self):
        # Test normalization function
        v = pd.Series([3, 4])
        normalized_v = normalize(v)
        expected_v = pd.Series([0.6, 0.8])
        pd.testing.assert_series_equal(normalized_v, expected_v)

        # Test normalization with zero norm
        v = pd.Series([0, 0])
        normalized_v = normalize(v)
        pd.testing.assert_series_equal(normalized_v, v)

if __name__ == '__main__':
    unittest.main()
