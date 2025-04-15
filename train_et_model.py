#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a model for ET-data using the CAP framework.
"""

import os
import yaml
import torch
import argparse
from cap.data.data import get_dataloaders
from cap.training import train_model, evaluate_model, load_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for ET-data')
    parser.add_argument('--config', type=str, default='cap/configs/config_et.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Override model type from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device from config')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    if args.model_type:
        config['model']['type'] = args.model_type
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.device:
        config['training']['device'] = args.device

    # Print configuration
    print("Configuration:")
    print(f"Dataset: {config['dataset']['path']}")
    print(f"Model: {config['model']['type']}")
    print(f"Training: {config['training']['epochs']} epochs, {config['training']['device']}")

    # Get dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(
        path=config['dataset']['path'],
        batch_size=config['dataset']['batch_size'],
        model_type=config['model']['type'],
        normalization=True
    )

    # Get input and output dimensions from the first batch
    for batch in train_loader:
        inputs, targets = batch
        input_dim = inputs.shape[-1]
        output_dim = targets.shape[-1]
        seq_len = inputs.shape[1]
        pred_len = targets.shape[1]
        break

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Prediction length: {pred_len}")

    # Train model
    model = train_model(
        train_loader=train_loader,
        valid_loader=valid_loader,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'],
        patience=config['training']['patience'],
        device=config['training']['device'],
        model_type=config['model']['type']
    )

    # Save model
    model_path = f"saved_models/et_{config['model']['type']}_model.pth"
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate model
    mse = evaluate_model(model, test_loader, device=config['training']['device'], model_type=config['model']['type'])
    print(f"Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main() 