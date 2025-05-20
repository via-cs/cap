Advanced Usage
=============

This guide covers advanced features and usage patterns in CAP.

Model Configuration
-----------------

YAML Configuration
~~~~~~~~~~~~~~~~

CAP supports configuration through YAML files:

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

Load configuration:

.. code-block:: python

    from cap.utils import load_config

    # Load from file
    config = load_config('config.yaml')

    # Load from dictionary
    config = load_config({
        'model': {
            'name': 'transformer',
            'input_size': 1,
            'output_size': 1
        }
    })

Custom Models
-----------

Creating Custom Models
~~~~~~~~~~~~~~~~~~~

You can create custom models by inheriting from the base model class:

.. code-block:: python

    from cap.models.base import BaseModel

    class CustomModel(BaseModel):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__(input_size, output_size)
            # Initialize your custom model

        def forward(self, x):
            # Implement forward pass
            return x

        def predict(self, data, forecast_horizon, **kwargs):
            # Implement prediction logic
            return predictions

Model Ensembles
-------------

Create and use model ensembles:

.. code-block:: python

    from cap.ensemble import ModelEnsemble

    # Create ensemble
    ensemble = ModelEnsemble([
        Transformer(input_size=1, output_size=1),
        FEDFormer(input_size=1, output_size=1),
        Autoformer(input_size=1, output_size=1)
    ])

    # Train ensemble
    ensemble.fit(train_data, validation_data=val_data)

    # Make predictions
    predictions = ensemble.predict(data, forecast_horizon=24)

    # Get individual model predictions
    individual_predictions = ensemble.predict_individual(data, forecast_horizon=24)

Advanced Training
---------------

Custom Training Loop
~~~~~~~~~~~~~~~~~

Implement custom training loops:

.. code-block:: python

    from cap.training import Trainer

    class CustomTrainer(Trainer):
        def train_step(self, batch):
            # Implement custom training step
            return loss

        def validation_step(self, batch):
            # Implement custom validation step
            return metrics

    # Use custom trainer
    trainer = CustomTrainer(model, optimizer, scheduler)
    trainer.fit(train_data, validation_data=val_data)

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~

Use different learning rate schedulers:

.. code-block:: python

    from cap.training import get_scheduler

    # Cosine scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_warmup_steps=1000,
        num_training_steps=10000
    )

    # Linear scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='linear',
        num_warmup_steps=1000,
        num_training_steps=10000
    )

    # Custom scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='custom',
        scheduler_fn=lambda step: 1.0 / (1.0 + step / 1000)
    )

Advanced Data Processing
----------------------

Custom Data Transforms
~~~~~~~~~~~~~~~~~~~

Create custom data transforms:

.. code-block:: python

    from cap.data.transforms import BaseTransform

    class CustomTransform(BaseTransform):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def fit(self, data):
            # Implement fit logic
            return self

        def transform(self, data):
            # Implement transform logic
            return transformed_data

        def inverse_transform(self, data):
            # Implement inverse transform logic
            return original_data

Custom Datasets
~~~~~~~~~~~~~

Create custom datasets:

.. code-block:: python

    from cap.data import BaseDataset

    class CustomDataset(BaseDataset):
        def __init__(self, data, **kwargs):
            super().__init__(data, **kwargs)

        def __getitem__(self, idx):
            # Implement data loading logic
            return x, y

        def __len__(self):
            # Implement length calculation
            return length

Advanced Visualization
--------------------

Custom Plotting
~~~~~~~~~~~~~

Create custom visualizations:

.. code-block:: python

    from cap.visualization import BasePlotter

    class CustomPlotter(BasePlotter):
        def plot_forecast(self, actual, predicted, **kwargs):
            # Implement custom forecast plotting
            pass

        def plot_errors(self, actual, predicted, **kwargs):
            # Implement custom error plotting
            pass

Interactive Visualization
~~~~~~~~~~~~~~~~~~~~~~

Use interactive visualization tools:

.. code-block:: python

    from cap.visualization import InteractivePlotter

    # Create interactive plotter
    plotter = InteractivePlotter()

    # Plot with interactive features
    plotter.plot_forecast(
        actual=actual_data,
        predicted=predicted_data,
        confidence_intervals=confidence_intervals
    )

    # Add interactive controls
    plotter.add_controls(
        metrics=['mae', 'rmse'],
        time_range=['1d', '1w', '1m']
    )

Best Practices
------------

1. **Model Configuration**:
   - Use YAML files for complex configurations
   - Version control your configurations
   - Document configuration parameters

2. **Custom Models**:
   - Follow the base model interface
   - Implement all required methods
   - Add proper documentation

3. **Training**:
   - Use appropriate learning rate schedules
   - Monitor training progress
   - Save checkpoints regularly

4. **Data Processing**:
   - Create reusable transforms
   - Handle edge cases
   - Maintain data consistency

5. **Visualization**:
   - Create informative plots
   - Add interactive features when needed
   - Document visualization parameters

For more detailed information about advanced features, see the :ref:`api_reference` section. 