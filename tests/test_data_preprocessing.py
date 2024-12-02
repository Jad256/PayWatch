import pytest
import pandas as pd
from src.data_preprocessing import preprocess_data # Assuming this function exists and is testable
import os

# Define the path to the test dataset
TEST_DATA_PATH = os.path.join('tests', 'data', 'test_data.csv')

expected_number_of_rows = 10  # Define expected row count

def test_data_loading():
    """Test that the data loads correctly and is not empty."""
    data = pd.read_csv(TEST_DATA_PATH)
    assert not data.empty, "Loaded data should not be empty"

def test_preprocess_data():
    """Test specific aspects of the data preprocessing."""
    data = pd.read_csv(TEST_DATA_PATH)
    processed_data = preprocess_data(data)  # This needs preprocess_data to return DataFrame
    # Here we test a few conditions that should be true if preprocessing is done right
    assert 'ExpectedColumn' in processed_data.columns, "Processed data must include 'ExpectedColumn'"
    assert processed_data.isnull().sum().sum() == 0, "Processed data should have no missing values"
    assert len(processed_data) == expected_number_of_rows, "Processed data should have X number of rows"

@pytest.mark.parametrize("column", ['V1', 'V2', 'V3'])  # Example of testing multiple columns
def test_columns_exist(column):
    """Test that essential columns exist after preprocessing."""
    data = pd.read_csv(TEST_DATA_PATH)
    processed_data = preprocess_data(data)
    assert column in processed_data.columns, f"{column} should be in the processed data"

# Add more tests as needed
