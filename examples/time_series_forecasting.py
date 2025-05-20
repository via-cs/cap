#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use CAP's time series forecasting models.
"""

import numpy as np
import pandas as pd
import torch
from cap.models import (
    Transformer,
    FEDFormer,
    Autoformer,
    TimesNet,
    Informer,
    LSTM
)
from cap.utils.data_utils import load_time_series_data, preprocess_data

def create_sample_data():
    """Create a sample time series dataset."""
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    # Create a time series with trend, seasonality, and noise
    trend = np.linspace(0, 10, 1000)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(1000) / 24)  # Daily seasonality
    noise = np.random.normal(0, 1, 1000)
    values = trend + seasonality + noise
    return pd.DataFrame({'date': dates, 'value': values})

def train_model(model, train_data, val_data, epochs=10):
    """Train a time series forecasting model."""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(train_data)
        loss = criterion(outputs, train_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_data)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

def main():
    # Create sample data
    df = create_sample_data()
    data = load_time_series_data(df, 'date', 'value')
    processed_data = preprocess_data(data, normalize=True)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(processed_data))
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:]
    
    # Initialize models
    models = {
        'Transformer': Transformer(),
        'FEDFormer': FEDFormer(),
        'Autoformer': Autoformer(),
        'TimesNet': TimesNet(),
        'Informer': Informer(),
        'LSTM': LSTM()
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f'\nTraining {name}...')
        train_model(model, train_data, val_data)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(val_data)
            print(f'{name} predictions shape:', predictions.shape)

if __name__ == '__main__':
    main() 