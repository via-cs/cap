.. _quickstart:

Quickstart
==========

This guide will show you how to use CAP for time series forecasting with a simple example.

1. Load and Prepare Data
-----------------------

First, let's load some time series data. CAP supports various data formats, but for this example, we'll use a simple CSV file.

.. code-block:: python

    import pandas as pd
    from cap.data import load_data

    # Load your time series data
    data = load_data('your_data.csv')
    
    # The data should have a timestamp column and one or more value columns
    print(data.head())

2. Create and Configure a Model
------------------------------

CAP provides several state-of-the-art forecasting models. Let's use the Transformer model as an example:

.. code-block:: python

    from cap import Transformer

    # Initialize the model with default configuration
    model = Transformer(
        input_size=1,          # Number of input features
        output_size=1,         # Number of output features
        d_model=512,          # Dimension of the model
        nhead=8,              # Number of attention heads
        num_layers=3,         # Number of transformer layers
        dropout=0.1           # Dropout rate
    )

3. Train the Model
-----------------

Now we can train the model on our data:

.. code-block:: python

    # Split data into train and validation sets
    train_data = data[:'2023-12-31']
    val_data = data['2024-01-01':]

    # Train the model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=100,
        batch_size=32
    )

4. Make Predictions
------------------

Once the model is trained, you can use it to make predictions:

.. code-block:: python

    # Make predictions for the next 24 hours
    predictions = model.predict(
        data,
        forecast_horizon=24,
        return_confidence=True
    )

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(predictions.index, predictions['forecast'], label='Forecast')
    plt.fill_between(
        predictions.index,
        predictions['lower_bound'],
        predictions['upper_bound'],
        alpha=0.2,
        label='Confidence Interval'
    )
    plt.legend()
    plt.show()

5. Evaluate the Model
--------------------

You can evaluate the model's performance using various metrics:

.. code-block:: python

    from cap.metrics import evaluate_forecast

    metrics = evaluate_forecast(
        actual=val_data,
        predicted=predictions,
        metrics=['mae', 'mse', 'rmse', 'mape']
    )

    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

For more advanced usage and configuration options, please refer to the :ref:`user_guides` section.
