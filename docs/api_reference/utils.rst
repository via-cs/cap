Utilities
========

CAP provides various utility functions to help with common tasks. This section documents the available utilities.

Configuration
------------

.. autofunction:: cap.utils.load_config
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.utils import load_config

       # Load configuration from YAML file
       config = load_config('config.yaml')

       # Load configuration from dictionary
       config = load_config({
           'model': 'transformer',
           'input_size': 1,
           'output_size': 1,
           'd_model': 512
       })

Logging
-------

.. autofunction:: cap.utils.setup_logger
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.utils import setup_logger

       # Setup logger
       logger = setup_logger('cap', level='INFO')

       # Use logger
       logger.info('Training started')
       logger.warning('Validation loss increased')
       logger.error('Training failed')

Checkpointing
------------

.. autofunction:: cap.utils.save_checkpoint
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.utils import save_checkpoint, load_checkpoint

       # Save model checkpoint
       save_checkpoint(
           model=model,
           optimizer=optimizer,
           epoch=epoch,
           path='checkpoints/model.pt'
       )

       # Load model checkpoint
       model, optimizer, epoch = load_checkpoint(
           model=model,
           optimizer=optimizer,
           path='checkpoints/model.pt'
       )

Device Management
---------------

.. autofunction:: cap.utils.get_device
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.utils import get_device

       # Get device (CPU or GPU)
       device = get_device()

       # Move model to device
       model = model.to(device)

       # Move data to device
       data = data.to(device)

Data Visualization
----------------

.. autofunction:: cap.utils.plot_time_series
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.utils import plot_time_series

       # Plot time series data
       plot_time_series(
           data=data,
           title='Time Series Data',
           xlabel='Time',
           ylabel='Value'
       )

Progress Tracking
---------------

.. autofunction:: cap.utils.ProgressBar
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.utils import ProgressBar

       # Create progress bar
       progress = ProgressBar(total=100, desc='Training')

       # Update progress
       for i in range(100):
           # Training step
           progress.update(1) 