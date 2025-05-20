# CAP: Time Series Forecasting Framework

CAP is a Python package for time series forecasting that implements various state-of-the-art models including Transformer, FEDFormer, Autoformer, TimesNet, Informer, and LSTM.

## Installation

```bash
pip install cap
```

## Quick Start

### As a Python Package

```python
from cap import Transformer, FEDFormer, Autoformer, train_model, evaluate_model

# Initialize and train a model
model = train_model(
    train_loader=train_loader,
    valid_loader=valid_loader,
    input_dim=10,
    output_dim=1,
    seq_len=24,
    pred_len=12,
    model_type='transformer'
)

# Evaluate the model
test_loss = evaluate_model(model, test_loader, model_type='transformer')
```

### From Command Line

```bash
# Get help
cap --help

# Train a model
cap train --model transformer --data input.csv --output model.pt --epochs 10 --lr 0.001

# Make predictions with a trained model
cap predict --model transformer --data input.csv --model-path model.pt --output predictions.csv
```

## Available Models

- Transformer
- FEDFormer
- Autoformer
- TimesNet
- Informer
- LSTM

## Training

The framework provides a unified training interface for all models:

```python
from cap import train_model

# Train a model
model = train_model(
    train_loader=train_loader,
    valid_loader=valid_loader,
    input_dim=input_dim,
    output_dim=output_dim,
    seq_len=seq_len,
    pred_len=pred_len,
    hidden_dim=128,  # for LSTM
    num_layers=2,    # for LSTM
    epochs=10,
    lr=0.001,
    patience=5,      # early stopping patience
    device='cuda',   # or 'cpu'
    model_type='lstm'  # or 'transformer', 'autoformer', 'informer', 'fedformer'
)
```

## Evaluation

Models can be evaluated using the provided evaluation function:

```python
from cap import evaluate_model

# Evaluate a model
test_loss = evaluate_model(
    model=model,
    test_loader=test_loader,
    device='cuda',
    model_type='lstm'
)
```

## Configuration

Models can be configured using YAML configuration files:

```yaml
model:
  type: transformer
  hidden_size: 512
  num_layers: 6
  num_heads: 8
```

## Requirements

- Python >= 3.8
- PyTorch
- Other dependencies listed in requirements.txt

## License

MIT License

