Architecture
===========

This guide explains the architecture of CAP and how its components work together.

Overview
--------

CAP is built with a modular architecture that separates concerns into distinct components:

1. **Models**: Forecasting models (Transformer, FEDFormer, etc.)
2. **Data**: Data loading, preprocessing, and augmentation
3. **Training**: Training loops, optimizers, and schedulers
4. **Evaluation**: Metrics and visualization
5. **Utils**: Configuration, logging, and other utilities

Core Components
-------------

Models
~~~~~~

The models component is organized as follows:

.. code-block:: text

    cap/models/
    ├── base.py           # Base model class
    ├── transformer.py    # Transformer model
    ├── fedformer.py      # FEDFormer model
    ├── autoformer.py     # Autoformer model
    ├── timesnet.py       # TimesNet model
    ├── informer.py       # Informer model
    └── lstm.py           # LSTM model

Each model inherits from the base model class and implements:

- ``__init__``: Model initialization
- ``forward``: Forward pass
- ``fit``: Training
- ``predict``: Inference
- ``save``: Model saving
- ``load``: Model loading

Data
~~~~

The data component handles data processing:

.. code-block:: text

    cap/data/
    ├── __init__.py
    ├── dataset.py        # Dataset classes
    ├── transforms.py     # Data transforms
    └── augmentation.py   # Data augmentation

Key classes:

- ``TimeSeriesDataset``: Dataset for time series data
- ``Normalizer``: Data normalization
- ``AddNoise``: Noise augmentation
- ``TimeWarping``: Time warping augmentation

Training
~~~~~~~

The training component manages model training:

.. code-block:: text

    cap/training/
    ├── __init__.py
    ├── trainer.py        # Trainer class
    ├── optimizer.py      # Optimizer utilities
    └── scheduler.py      # Learning rate schedulers

Key classes:

- ``Trainer``: Training loop
- ``get_optimizer``: Optimizer factory
- ``get_scheduler``: Scheduler factory

Evaluation
~~~~~~~~~

The evaluation component handles metrics and visualization:

.. code-block:: text

    cap/metrics/
    ├── __init__.py
    ├── metrics.py        # Evaluation metrics
    ├── visualization.py  # Visualization tools
    └── statistical.py    # Statistical tests

Key functions:

- ``evaluate_forecast``: Forecast evaluation
- ``plot_forecast``: Forecast visualization
- ``statistical_tests``: Statistical analysis

Utils
~~~~~

The utils component provides common utilities:

.. code-block:: text

    cap/utils/
    ├── __init__.py
    ├── config.py         # Configuration
    ├── logging.py        # Logging
    ├── checkpoint.py     # Checkpointing
    └── device.py         # Device management

Key functions:

- ``load_config``: Configuration loading
- ``setup_logger``: Logger setup
- ``save_checkpoint``: Model checkpointing
- ``get_device``: Device management

Data Flow
--------

1. **Data Loading**:
   - Load raw data
   - Apply transforms
   - Create dataset

2. **Model Training**:
   - Initialize model
   - Create trainer
   - Train model
   - Save checkpoint

3. **Model Inference**:
   - Load model
   - Make predictions
   - Evaluate results
   - Visualize output

Example Flow
-----------

.. code-block:: python

    # 1. Data Loading
    from cap.data import load_data, TimeSeriesDataset
    from cap.data.transforms import Normalizer

    # Load data
    data = load_data('data.csv')

    # Normalize data
    normalizer = Normalizer(method='zscore')
    normalized_data = normalizer.fit_transform(data)

    # Create dataset
    dataset = TimeSeriesDataset(
        data=normalized_data,
        input_size=24,
        output_size=24
    )

    # 2. Model Training
    from cap import Transformer
    from cap.training import Trainer
    from cap.training import get_optimizer, get_scheduler

    # Initialize model
    model = Transformer(
        input_size=1,
        output_size=1,
        d_model=512
    )

    # Create optimizer and scheduler
    optimizer = get_optimizer(model.parameters())
    scheduler = get_scheduler(optimizer)

    # Create trainer
    trainer = Trainer(model, optimizer, scheduler)

    # Train model
    trainer.fit(dataset)

    # 3. Model Inference
    from cap.metrics import evaluate_forecast, plot_forecast

    # Make predictions
    predictions = model.predict(data)

    # Evaluate results
    metrics = evaluate_forecast(
        actual=data,
        predicted=predictions
    )

    # Visualize output
    plot_forecast(
        actual=data,
        predicted=predictions
    )

Extension Points
--------------

1. **Custom Models**:
   - Inherit from ``BaseModel``
   - Implement required methods
   - Add to model registry

2. **Custom Transforms**:
   - Inherit from ``BaseTransform``
   - Implement transform methods
   - Add to transform registry

3. **Custom Metrics**:
   - Add new metric functions
   - Register with metric registry
   - Update evaluation logic

4. **Custom Visualization**:
   - Inherit from ``BasePlotter``
   - Implement plotting methods
   - Add to visualization registry

Best Practices
------------

1. **Code Organization**:
   - Follow modular design
   - Use clear naming
   - Add proper documentation

2. **Error Handling**:
   - Use custom exceptions
   - Add proper validation
   - Provide helpful messages

3. **Testing**:
   - Write unit tests
   - Add integration tests
   - Test edge cases

4. **Documentation**:
   - Document public API
   - Add usage examples
   - Keep docs up to date

For more information about contributing to CAP, see the :ref:`contributing` guide. 