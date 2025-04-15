Models
======

CAP provides several state-of-the-art models for time series forecasting. This guide explains how to use each model and their specific features.

Overview
--------

CAP implements the following models:

1. **Transformer**: Standard transformer architecture adapted for time series forecasting
2. **FEDFormer**: Frequency Enhanced Decomposition Transformer
3. **Autoformer**: Auto-Correlation based Transformer
4. **TimesNet**: Time Series Network with temporal decomposition
5. **Informer**: Transformer with ProbSparse self-attention
6. **LSTM**: Long Short-Term Memory network

Model Selection
-------------

Choosing the right model depends on your specific use case:

- **Transformer**: Good general-purpose model, works well with most time series data
- **FEDFormer**: Best for data with strong seasonal patterns
- **Autoformer**: Excellent for long-term forecasting
- **TimesNet**: Good for data with multiple temporal patterns
- **Informer**: Efficient for very long sequences
- **LSTM**: Simple and effective for shorter sequences

Basic Usage
----------

All models follow a similar interface:

.. code-block:: python

    from cap import Transformer  # or any other model

    # Initialize model
    model = Transformer(
        input_size=1,          # Number of input features
        output_size=1,         # Number of output features
        d_model=512,          # Dimension of the model
        nhead=8,              # Number of attention heads
        num_layers=3,         # Number of transformer layers
        dropout=0.1           # Dropout rate
    )

    # Train model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=100,
        batch_size=32
    )

    # Make predictions
    predictions = model.predict(
        data,
        forecast_horizon=24,
        return_confidence=True
    )

Model-Specific Features
---------------------

Transformer
~~~~~~~~~~

- Standard transformer architecture
- Multi-head self-attention
- Position-wise feed-forward networks
- Positional encoding

FEDFormer
~~~~~~~~~

- Frequency Enhanced Decomposition
- Seasonal-Trend decomposition
- Frequency domain attention
- Adaptive frequency selection

Autoformer
~~~~~~~~~

- Auto-Correlation mechanism
- Series decomposition
- Seasonal-Trend decomposition
- Efficient self-attention

TimesNet
~~~~~~~~

- Temporal decomposition
- Multi-scale temporal patterns
- Adaptive temporal attention
- Trend and seasonal components

Informer
~~~~~~~~

- ProbSparse self-attention
- Distilling operation
- Generative style decoder
- Long sequence efficiency

LSTM
~~~~

- Simple LSTM architecture
- Configurable layers
- Bidirectional option
- Dropout for regularization

Advanced Configuration
--------------------

All models support advanced configuration through YAML files:

.. code-block:: yaml

    model:
      name: transformer
      input_size: 1
      output_size: 1
      d_model: 512
      nhead: 8
      num_layers: 3
      dropout: 0.1
      activation: gelu
      norm_first: true

    training:
      epochs: 100
      batch_size: 32
      learning_rate: 0.0001
      weight_decay: 0.0001
      scheduler: cosine
      warmup_steps: 1000

    data:
      input_size: 24
      output_size: 24
      stride: 1
      normalization: zscore

Model Comparison
--------------

Here's a comparison of model characteristics:

+------------+-------------+-------------+-------------+-------------+
| Model      | Long Seq    | Seasonal    | Trend       | Memory      |
+============+=============+=============+=============+=============+
| Transformer| Good        | Good        | Good        | High        |
+------------+-------------+-------------+-------------+-------------+
| FEDFormer  | Excellent   | Excellent   | Good        | Medium      |
+------------+-------------+-------------+-------------+-------------+
| Autoformer | Excellent   | Good        | Excellent   | Medium      |
+------------+-------------+-------------+-------------+-------------+
| TimesNet   | Good        | Excellent   | Excellent   | Medium      |
+------------+-------------+-------------+-------------+-------------+
| Informer   | Excellent   | Good        | Good        | Low         |
+------------+-------------+-------------+-------------+-------------+
| LSTM       | Fair        | Good        | Good        | Low         |
+------------+-------------+-------------+-------------+-------------+

For more detailed information about each model's implementation, see the :ref:`api_reference` section. 