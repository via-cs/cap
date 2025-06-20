# CATP (Collaborative Adaptive Time-series Prediction) - Usage Guide

This guide explains how to run CATP experiments with 6 different worker models for time series forecasting.

## 📁 Files Overview

- `run_catp_experiment.py` - Full experiment script with comprehensive logging and evaluation
- `quick_catp_test.py` - Quick test script for verifying everything works
- `catp_config.py` - Configuration file for customizing experiments
- `test_data_loading.py` - Test script to verify data loading works correctly
- `README_CATP.md` - This guide

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have:
- Python 3.7+
- PyTorch
- Required dependencies installed
- Your dataset ready

### 2. Test Data Loading

First, test that your data loading works correctly:

```bash
python test_data_loading.py
```

This will verify that:
- Your dataset path is correct
- Data is loaded in the expected format: `(X, Y)` pairs
- X has shape `[batch, seq_len, in_dim]`
- Y has shape `[batch, pred_len, out_dim]`

### 3. Quick Test

To quickly verify everything works:

```bash
python quick_catp_test.py
```

This will:
- Run a 5-epoch training session
- Use 6 different worker models
- Show basic results
- Take ~5-10 minutes

### 4. Full Experiment

For a complete experiment:

```bash
python run_catp_experiment.py
```

This will:
- Run 100 epochs with early stopping
- Save models and plots
- Generate detailed evaluation metrics
- Take several hours

## 🔧 Configuration

### Data Format

The CATP framework expects data in the following format:
- **Input (X)**: Shape `[batch, seq_len, in_dim]` - Historical time series data
- **Target (Y)**: Shape `[batch, pred_len, out_dim]` - Future values to predict

The data loader automatically handles:
- Data normalization
- Train/validation/test splits
- Batch creation
- Sequence and prediction length configuration

### Using the Configuration File

The `catp_config.py` file contains all experiment settings. You can:

1. **Modify data path**:
   ```python
   DATA_CONFIG = {
       'path': 'your/dataset/path.csv',  # Update this
       'batch_size': 32,
       # ...
   }
   ```

2. **Change experiment variants**:
   ```python
   # In your script
   from catp_config import get_experiment_config
   
   config = get_experiment_config('quick_test')  # or 'full_experiment'
   ```

3. **Customize worker models**:
   ```python
   WORKER_CONFIGS = [
       {
           'name': 'MyLSTM',
           'model_name': 'lstm',
           'hidden_dim': 256,  # Customize parameters
           # ...
       },
       # Add more workers...
   ]
   ```

### Available Experiment Variants

- `quick_test`: 5 epochs, small batch, no early stopping
- `full_experiment`: 100 epochs, standard batch, with early stopping
- `large_batch`: 100 epochs, large batch size

## 🏗️ Worker Models

The scripts include 6 different worker models:

1. **LSTM-High**: High-capacity LSTM (128 hidden units, 2 layers)
2. **LSTM-Low**: Low-capacity LSTM (32 hidden units, 1 layer)
3. **Transformer**: Standard transformer architecture
4. **Autoformer**: Autoformer with moving average
5. **FEDFormer**: FEDFormer with Fourier decomposition
6. **Informer**: Informer with sparse attention

## 📊 Output and Results

### Training Output

The scripts will show:
- Training progress with loss values
- Validation metrics
- Worker selection rates
- Final test performance

### Saved Files

- **Models**: `saved_models/catp_experiment/`
- **Logs**: `runs/catp_experiment/`
- **Plots**: Training curves and worker selection analysis
- **Predictions**: CSV file with test predictions

### Key Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Worker Selection**: How often each worker is chosen

## 🔍 Understanding CATP

### How It Works

1. **Manager Model**: Learns to select the best worker for each input
2. **Worker Models**: Different architectures for time series prediction
3. **Collaborative Training**: Manager and workers train together
4. **Adaptive Selection**: Manager adapts worker selection based on performance

### Key Components

- **ManagerModel**: Transformer-based model that learns worker selection
- **WorkerWrapper**: Wrapper that handles different model interfaces
- **CATPTrainer**: Training loop with collaborative learning
- **Sinkhorn Distance**: Used for optimal transport in worker selection

### Data Flow

```
Input Data (X) → Manager Model → Worker Selection
                                    ↓
Target Data (Y) ← Worker Models ← Selected Workers
```

The manager learns to select the most appropriate worker model for each input sequence, and the workers learn to make accurate predictions for their assigned inputs.

## 🛠️ Troubleshooting

### Common Issues

1. **Data not found**:
   - Update the data path in `catp_config.py`
   - Ensure your dataset is in the correct format (CSV or TXT)
   - Run `test_data_loading.py` to verify data loading

2. **CUDA out of memory**:
   - Reduce batch size in configuration
   - Use smaller model dimensions

3. **Import errors**:
   - Make sure you're in the correct directory
   - Check that all dependencies are installed

4. **Data format issues**:
   - Ensure your data provides (X, Y) pairs
   - Check that X and Y have the correct shapes
   - Verify that the data loader returns the expected format

### Debug Mode

For debugging, you can modify the quick test script:

```python
# Add this to see more details
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Customization Examples

### Adding a New Worker Model

1. Add your model to the models directory
2. Update `available_models()` in `catp.py`
3. Add configuration to `WORKER_CONFIGS`

### Changing the Manager Architecture

Modify `MANAGER_CONFIG` in `catp_config.py`:

```python
MANAGER_CONFIG = {
    'd_model': 256,      # Larger model
    'n_heads': 8,        # More attention heads
    'd_ff': 512,         # Larger feedforward
    'num_layers': 4,     # More layers
    'dropout': 0.2       # More dropout
}
```

### Using Different Loss Functions

```python
LOSS_CONFIG = {
    'criterion': nn.L1Loss(),  # MAE instead of MSE
    # or
    'criterion': nn.HuberLoss(),  # Huber loss
}
```

## 📚 Advanced Usage

### Multi-GPU Training

For multi-GPU setups, modify the trainer initialization:

```python
trainer = CATPTrainer(
    # ... other parameters ...
    device=torch.device('cuda'),
    # Add DataParallel if needed
)
```

### Custom Evaluation Metrics

Add custom metrics to the evaluation:

```python
def custom_metric(predictions, targets):
    # Your custom metric
    return metric_value

# Add to evaluation
test_metrics = trainer.evaluate(test_loader, custom_metrics=[custom_metric])
```

### Hyperparameter Tuning

Use the configuration system for hyperparameter search:

```python
# Create multiple configurations
configs = []
for lr in [0.001, 0.0005, 0.0001]:
    for batch_size in [16, 32, 64]:
        config = get_experiment_config('full_experiment')
        config['optimizer']['manager_lr'] = lr
        config['data']['batch_size'] = batch_size
        configs.append(config)

# Run experiments
for i, config in enumerate(configs):
    run_experiment(config, f"experiment_{i}")
```

## 🤝 Contributing

To add new features:

1. **New Models**: Add to models directory and update `available_models()`
2. **New Metrics**: Add to evaluation functions
3. **New Configurations**: Add to `EXPERIMENT_VARIANTS`

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Verify your data format using `test_data_loading.py`
3. Check PyTorch and dependency versions
4. Review the error messages for specific guidance

---

**Happy experimenting with CATP! 🎉** 