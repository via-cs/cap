# Tutorials

CAP is a framework for time series forecasting. The following tutorials help you navigate: training models on different datasets, comparing model architectures, and tuning hyperparameters.

## Available Tutorials

| Tutorial | Description |
|----------|-------------|
| [ET_Data_Training.ipynb](ET_Data_Training.ipynb) | Train and compare different models on the Electricity Transformer (ET) dataset. Includes data loading, model training, evaluation, visualization, and hyperparameter tuning. |
<!-- | [CAP_Tutorial.ipynb](CAP_Tutorial.ipynb) | General tutorial for using the CAP framework with various datasets and models. | -->

## Getting Started

To run the tutorials:

1. Make sure you have installed the CAP package:
   ```bash
   pip install -e .
   ```

2. Start Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   ```

3. Navigate to the tutorials directory and open the desired notebook.

## ET-Data Training Tutorial

The [ET_Data_Training.ipynb](ET_Data_Training.ipynb) notebook demonstrates:

- Loading and preprocessing the ET-data
- Training different models (LSTM, Transformer, FEDformer)
- Evaluating and visualizing results
- Comparing model architectures
- Tuning hyperparameters for optimal performance

This tutorial is ideal for understanding how to use the CAP framework for time series forecasting tasks.