Data Processing
==============

This guide explains how to prepare and process your time series data for use with CAP models.

Data Format
----------

CAP expects time series data in the following format:

1. **CSV Files**:
   - Must have a timestamp column (can be named 'timestamp', 'date', 'time', etc.)
   - One or more value columns
   - Timestamps should be in a standard format (e.g., 'YYYY-MM-DD HH:MM:SS')

2. **Pandas DataFrame**:
   - Index should be a DatetimeIndex
   - One or more value columns

Example data format:

.. code-block:: python

    import pandas as pd

    # From CSV
    data = pd.read_csv('data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Or create directly
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    data = pd.DataFrame({
        'value': [...],  # Your time series values
    }, index=dates)

Data Loading
-----------

CAP provides utilities for loading data from various sources:

.. code-block:: python

    from cap.data import load_data

    # Load from CSV
    data = load_data('data.csv')

    # Load from DataFrame
    data = load_data(df)

    # Load with specific column names
    data = load_data('data.csv', timestamp_col='date', value_cols=['value1', 'value2'])

Data Preprocessing
----------------

Time Series Dataset
~~~~~~~~~~~~~~~~~

The `TimeSeriesDataset` class handles the creation of input-output pairs for training:

.. code-block:: python

    from cap.data import TimeSeriesDataset

    # Create dataset
    dataset = TimeSeriesDataset(
        data=data,
        input_size=24,    # 24 hours of input
        output_size=24,   # 24 hours of output
        stride=1          # Step size between samples
    )

    # Get a sample
    x, y = dataset[0]

Data Normalization
~~~~~~~~~~~~~~~~

CAP provides several normalization methods:

.. code-block:: python

    from cap.data.transforms import Normalizer

    # Z-score normalization
    normalizer = Normalizer(method='zscore')
    normalized_data = normalizer.fit_transform(data)

    # Min-max normalization
    normalizer = Normalizer(method='minmax')
    normalized_data = normalizer.fit_transform(data)

    # Robust normalization
    normalizer = Normalizer(method='robust')
    normalized_data = normalizer.fit_transform(data)

Data Splitting
-------------

Split your data into training, validation, and test sets:

.. code-block:: python

    from cap.data import split_data

    # Split data
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Split with specific dates
    train_data, val_data, test_data = split_data(
        data,
        train_end='2023-10-31',
        val_end='2023-11-30',
        test_end='2023-12-31'
    )

Data Augmentation
---------------

CAP provides several data augmentation techniques:

.. code-block:: python

    from cap.data.augmentation import (
        AddNoise,
        TimeWarping,
        WindowWarping,
        MagnitudeWarping
    )

    # Add Gaussian noise
    augmenter = AddNoise(noise_level=0.1)
    augmented_data = augmenter(data)

    # Time warping
    augmenter = TimeWarping(warp_ratio=0.1)
    augmented_data = augmenter(data)

    # Window warping
    augmenter = WindowWarping(window_size=24, warp_ratio=0.1)
    augmented_data = augmenter(data)

    # Magnitude warping
    augmenter = MagnitudeWarping(warp_ratio=0.1)
    augmented_data = augmenter(data)

Data Visualization
----------------

Visualize your time series data:

.. code-block:: python

    from cap.utils import plot_time_series

    # Basic plot
    plot_time_series(data, title='Time Series Data')

    # Plot with multiple columns
    plot_time_series(
        data,
        columns=['value1', 'value2'],
        title='Multiple Time Series'
    )

    # Plot with confidence intervals
    plot_time_series(
        data,
        confidence_intervals={
            'lower': lower_bound,
            'upper': upper_bound
        },
        title='Time Series with Confidence Intervals'
    )

Best Practices
-------------

1. **Data Quality**:
   - Handle missing values appropriately
   - Remove or handle outliers
   - Ensure consistent sampling frequency

2. **Preprocessing**:
   - Normalize data before training
   - Use appropriate input/output sizes
   - Consider seasonal decomposition for seasonal data

3. **Data Splitting**:
   - Maintain temporal order in splits
   - Use sufficient validation data
   - Consider multiple test sets for robustness

4. **Augmentation**:
   - Use augmentation sparingly
   - Validate augmented data quality
   - Consider domain-specific augmentations

For more detailed information about data processing functions, see the :ref:`api_reference` section. 