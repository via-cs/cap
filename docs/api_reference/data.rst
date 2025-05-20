Data Handling
=============

CAP provides utilities for loading, preprocessing, and managing time series data. This section documents the available data handling functions and classes.

Data Loading
-----------

.. autofunction:: cap.data.load_data
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.data import load_data

       # Load data from CSV
       data = load_data('data.csv')

       # Load data from pandas DataFrame
       import pandas as pd
       df = pd.read_csv('data.csv')
       data = load_data(df)

Data Preprocessing
----------------

.. autoclass:: cap.data.TimeSeriesDataset
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap.data import TimeSeriesDataset

       # Create dataset
       dataset = TimeSeriesDataset(
           data=data,
           input_size=24,  # 24 hours of input
           output_size=24,  # 24 hours of output
           stride=1  # Step size between samples
       )

       # Get a sample
       x, y = dataset[0]

Data Transforms
--------------

.. autoclass:: cap.data.transforms.Normalizer
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap.data.transforms import Normalizer

       # Create normalizer
       normalizer = Normalizer(method='zscore')

       # Fit and transform
       normalized_data = normalizer.fit_transform(data)

       # Inverse transform
       original_data = normalizer.inverse_transform(normalized_data)

Data Splitting
-------------

.. autofunction:: cap.data.split_data
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.data import split_data

       # Split data into train, validation, and test sets
       train_data, val_data, test_data = split_data(
           data,
           train_ratio=0.7,
           val_ratio=0.15,
           test_ratio=0.15
       ) 