import pytest
import numpy as np
import pandas as pd
import torch
from cap.utils.data_utils import load_time_series_data, preprocess_data

def test_load_time_series_data():
    # Create a sample time series dataset
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    values = np.random.randn(100)
    df = pd.DataFrame({'date': dates, 'value': values})
    
    # Test loading data
    data = load_time_series_data(df, 'date', 'value')
    assert isinstance(data, np.ndarray)
    assert data.shape[0] == 100

def test_preprocess_data():
    # Create sample data
    data = np.random.randn(100, 1)
    
    # Test preprocessing
    processed_data = preprocess_data(data, normalize=True)
    assert isinstance(processed_data, torch.Tensor)
    assert processed_data.shape[0] == 100
    assert processed_data.shape[1] == 1 