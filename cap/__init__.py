"""
CATF: A Time Series Forecasting Framework
"""

__version__ = '0.7.1.dev0'

from .models import (
    Transformer,
    FEDformer,
    Autoformer,
    TimesNet,
    Informer,
    TimeSeriesLSTM,
    catf,
)

from .data.data import (
    get_dataloaders,
)

from .training import (
    train_model,
    evaluate_model,
    load_model,
    catf_trainer,
)

# Import CATF-specific functions from their correct modules
from .models.catf import create_worker_pool, available_models

__all__ = [
    'Transformer',
    'FEDformer',
    'Autoformer',
    'TimesNet',
    'Informer',
    'TimeSeriesLSTM',
    'train_model',
    'evaluate_model',
    'load_model',
    'catf_trainer',
    'catf',
    'get_dataloaders',
    'create_worker_pool',
    'available_models',
]
