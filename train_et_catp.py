#!/usr/bin/env python3
"""
ET (Electricity Transformer) CATP Training Script
=================================================

A training script for CATP with 7 worker models on the Electricity Transformer dataset.
This script uses a YAML configuration file for easy parameter management.

Usage:
    python train_et_catp.py --config configs/et_training_config.yaml
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path

# Add the cap package to the path
sys.path.append(str(Path(__file__).parent / "cap"))

import cap
from cap import get_dataloaders
from cap.models.catp import ManagerModel, create_worker_pool, available_models
from cap.training.catp_trainer import CATPTrainer
from cap.utils.config_loader import load_config, validate_config, get_device


def create_worker_configs(config, input_dim, output_dim, seq_len, pred_len):
    """
    Create worker configurations from the config file, supporting multiple instances.
    
    Args:
        config: Configuration dictionary
        input_dim: Input dimension from data
        output_dim: Output dimension from data
        seq_len: Sequence length from data
        pred_len: Prediction length from data
        
    Returns:
        Tuple of (List of worker configurations, List of worker names)
    """
    worker_configs = []
    worker_names = []
    
    for worker_config in config['models']['workers']:
        # Base configuration
        base_config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'seq_len': seq_len,
            'pred_len': pred_len,
        }
        
        # Add worker-specific parameters
        if 'count' in worker_config:
            for i in range(worker_config['count']):
                worker_configs.append({
                    'type': worker_config['type'],
                    'hidden_dim': worker_config['hidden_dim'],
                    'num_layers': worker_config['num_layers'],
                    'dropout': worker_config['dropout'],
                    **base_config
                })
        else:
            worker_configs.append({
                'type': worker_config['type'],
                'hidden_dim': worker_config['hidden_dim'],
                'num_layers': worker_config['num_layers'],
                'dropout': worker_config['dropout'],
                **base_config
            })
    
    return worker_configs


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CATP on ET dataset")
    parser.add_argument("--config", type=str, default="cap/configs/et_training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--validate-config", action="store_true",
                       help="Validate configuration and exit")
    
    args = parser.parse_args()
    
    # Load and validate configuration
    print(" Loading configuration...")
    config = load_config(args.config)
    
    if args.validate_config:
        try:
            validate_config(config)
            print(" Configuration is valid!")
            return
        except ValueError as e:
            print(f" Configuration validation failed: {e}")
            return
    
    # Get device
    device = get_device(config)
    print(f" Using device: {device}")
    
    # Check if data exists
    data_path = config['data']['path']
    if not os.path.exists(data_path):
        print(f" Data not found: {data_path}")
        print("Please update the data path in the configuration file.")
        return
    
    try:
        # 1. Load data
        print(" Loading data...")

        train_loader, val_loader, test_loader = get_dataloaders(
            path=data_path,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            train_size=config['data']['train_size'],
            valid_size=config['data']['valid_size'],
            test_size=config['data']['test_size'],
            normalization=config['data']['normalization'],
            seq_len=config['data']['seq_len'],
            pred_len=config['data']['pred_len']
        )
        
        # Get dimensions from first batch
        for batch in train_loader:
            inputs, targets = batch
            input_dim = inputs.shape[-1]
            output_dim = targets.shape[-1]
            seq_len = inputs.shape[1]
            pred_len = targets.shape[1]
            break
        
        print(f" Data loaded: {input_dim}D input, {output_dim}D output")
        print(f"   Sequence: {seq_len}, Prediction: {pred_len}")
        
        # 2. Create worker configurations
        print("  Creating worker configurations...")
        worker_configs = create_worker_configs(config, input_dim, output_dim, seq_len, pred_len)
        
        # 3. Create models
        print("  Creating models...")
        worker_models = create_worker_pool(worker_configs, available_models())
        
        manager_config = config['models']['manager']
        manager_model = ManagerModel(
            input_dim=input_dim,
            worker_count=len(worker_models),
            d_model=manager_config['d_model'],
            n_heads=manager_config['n_heads'],
            d_ff=manager_config['d_ff'],
            num_layers=manager_config['num_layers'],
            dropout=manager_config['dropout']
        )
        
        print(f" Created {len(worker_models)} worker models:")
        for i, worker in enumerate(worker_models):
            model_name = type(worker.model).__name__
            print(f"   Worker {i+1}: {model_name}")
        
        # 4. Create trainer
        print(" Creating trainer...")
        trainer = CATPTrainer(
            manager_model=manager_model,
            worker_models=worker_models,
            criterion=nn.MSELoss(),
            device=device,
            manager_lr=config['training']['manager_lr'],
            worker_lr=config['training']['worker_lr'],
            log_dir=config['logging']['log_dir'],
            clip_value=config['training']['clip_value'],
            worker_update_steps=config['training']['worker_update_steps']
        )
        
        # 5. Training
        print(" Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['epochs'],
            checkpoint_dir=config['logging']['checkpoint_dir'],
            early_stopping_patience=None,  # No early stopping for quick test
            plot_metrics=config['logging']['plot_metrics']
        )
        
        # 6. Evaluation
        print(" Evaluating on test set...")
        test_loss = trainer.validate(test_loader)
        
        # 7. Results
        print("\n Training completed!")
        print(f"   Final training loss: {history['train_losses'][-1]:.4f}")
        print(f"   Final validation loss: {history['val_losses'][-1]:.4f}")
        print(f"   Test loss: {test_loss:.4f}")
        print(f"   Best validation loss: {min(history['val_losses']):.4f}")
        
        # 8. Worker selection analysis
        final_selections = history['worker_selections'][-1]
        worker_names = [worker['name'] for worker in config['models']['workers']]
        print("\n Final worker selection rates:")
        for name, rate in zip(worker_names, final_selections):
            print(f"   {name}: {rate:.3f}")
        
        print("\n CATP training completed successfully!")
        
    except Exception as e:
        print(f" Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 