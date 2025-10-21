#!/usr/bin/env python3
"""
ET (Electricity Transformer) CATF Training Script

A training script for CATF with 7 worker models on the Electricity Transformer dataset.

Usage:
    python train_et_catf.py --config configs/et_training_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add the cap package to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "cap"))

import cap
from cap import get_dataloaders
from cap.models.catf import ManagerModel, create_worker_pool, available_models
from cap.training.catf_trainer import CATFTrainer
from cap.utils.config_loader import load_config, validate_config, get_device


def save_training_results(
    history, 
    test_loss, 
    dataset_name, 
    model_config, 
    log_dir, 
    config_path,
    test_loss_file_path,
    config,
    training_time=None
):
    """
    Append training results to the existing test loss file for record keeping.
    
    Args:
        history: Training history dictionary
        test_loss: Final test loss
        dataset_name: Name of the dataset
        model_config: Model configuration description
        log_dir: Directory to save the log file
        config_path: Path to the configuration file used
        test_loss_file_path: Path to the existing test loss file
        config: Configuration dictionary
        training_time: Optional training duration
    """
    # Compute final worker selection rates
    worker_selections = history['worker_selections']
    if worker_selections:
        selections_array = np.array(worker_selections)
        overall_rates = np.mean(selections_array, axis=0)
        final_epoch_rates = selections_array[-1] if len(selections_array) > 0 else np.zeros(len(selections_array[0]))
        
        if len(selections_array) >= 10:
            last_10_epochs_rates = np.mean(selections_array[-10:], axis=0)
        else:
            last_10_epochs_rates = overall_rates
    else:
        overall_rates = np.array([])
        final_epoch_rates = np.array([])
        last_10_epochs_rates = np.array([])
    
    # Append to the existing test loss file
    with open(test_loss_file_path, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("COMPREHENSIVE TRAINING RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Configuration File: {config_path}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if training_time:
            f.write(f"Training Duration: {training_time}\n")
        f.write("\n")
        
        f.write("FINAL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Training Loss: {history['train_losses'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {history['val_losses'][-1]:.6f}\n")
        f.write(f"Best Validation Loss: {min(history['val_losses']):.6f}\n")
        f.write(f"Best Validation Epoch: {history['val_losses'].index(min(history['val_losses'])) + 1}\n")
        f.write("\n")
        
        f.write("TRAINING HISTORY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Epochs: {len(history['train_losses'])}\n")
        f.write(f"Training Losses: {[f'{loss:.6f}' for loss in history['train_losses']]}\n")
        f.write(f"Validation Losses: {[f'{loss:.6f}' for loss in history['val_losses']]}\n")
        f.write("\n")
        
        if len(overall_rates) > 0:
            f.write("FINAL WORKER SELECTION RATES:\n")
            f.write("-" * 40 + "\n")
            for i, rate in enumerate(overall_rates):
                f.write(f"Worker {i}: {rate:.6f} ({rate*100:.2f}%)\n")
            f.write("\n")
            
            # Add summary statistics for worker selections
            f.write("WORKER SELECTION SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Most Selected Worker: Worker {overall_rates.argmax()} ({overall_rates.max()*100:.2f}%)\n")
            f.write(f"Least Selected Worker: Worker {overall_rates.argmin()} ({overall_rates.min()*100:.2f}%)\n")
            f.write(f"Selection Rate Range: {overall_rates.max()*100:.2f}% - {overall_rates.min()*100:.2f}%\n")
            f.write(f"Selection Rate Std Dev: {overall_rates.std()*100:.2f}%\n")
            f.write(f"Selection Rate Variance: {overall_rates.var()*100:.2f}%\n")
            f.write("\n")
            
            # Add fairness metrics
            f.write("FAIRNESS METRICS:\n")
            f.write("-" * 40 + "\n")
            gini_coeff = compute_gini_coefficient(overall_rates)
            entropy_val = compute_entropy(overall_rates)
            f.write(f"Gini Coefficient: {gini_coeff:.4f}\n")
            f.write(f"Entropy: {entropy_val:.4f}\n")
            f.write(f"Fairness Score (1 - Gini): {1 - gini_coeff:.4f}\n")
            f.write("\n")
        
        # Add training configuration details
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        if len(overall_rates) > 0:
            f.write(f"Number of Workers: {len(overall_rates)}\n")
        f.write(f"Training Epochs: {len(history['train_losses'])}\n")
        f.write(f"Early Stopping: {'Yes' if len(history['train_losses']) < config.get('training', {}).get('epochs', 0) else 'No'}\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"Training results appended to: {test_loss_file_path}")
    
    # Also print a summary to console
    if len(overall_rates) > 0:
        print(f"\n{'='*60}")
        print("FINAL WORKER SELECTION RATES SUMMARY")
        print(f"{'='*60}")
        print(f"Most Selected: Worker {overall_rates.argmax()} ({overall_rates.max()*100:.2f}%)")
        print(f"Least Selected: Worker {overall_rates.argmin()} ({overall_rates.min()*100:.2f}%)")
        print(f"Fairness Score: {1 - compute_gini_coefficient(overall_rates):.4f}")
        print(f"{'='*60}")
    
    return test_loss_file_path


def compute_gini_coefficient(rates):
    """Compute Gini coefficient as a measure of inequality in selection rates."""
    if not len(rates) or len(rates) < 2:
        return 0.0
    
    rates = np.array(rates)
    n = len(rates)
    sorted_rates = np.sort(rates)
    
    # Gini coefficient formula
    cumsum = np.cumsum(sorted_rates)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_entropy(rates):
    """Compute entropy as a measure of diversity in selection rates."""
    if not len(rates):
        return 0.0
    
    rates = np.array(rates)
    # Normalize to probabilities
    probs = rates / np.sum(rates)
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    return -np.sum(probs * np.log(probs))


def create_worker_configs(config, input_dim, output_dim, seq_len, pred_len, num_worker_models=None):
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
    
    i = 0
    for worker_config in config['models']['workers']:
        # Base configuration
        base_config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'seq_len': seq_len,
            'pred_len': pred_len,
        }
        
        if num_worker_models:
            if i >= len(num_worker_models):
                raise ValueError("--num-worker length must match the number of different worker model types")
            cur_worker_count = num_worker_models[i]
            i += 1
        else:
            if 'count' in worker_config:
                cur_worker_count = worker_config['count']
            else:
                cur_worker_count = 1

        # Add worker-specific parameters
        # if 'count' in worker_config:
        #     for i in range(worker_config['count']):
        #         worker_configs.append({
        #             'type': worker_config['type'],
        #             'd_model': worker_config['d_model'],
        #             'num_layers': worker_config['num_layers'],
        #             'dropout': worker_config['dropout'],
        #             **base_config
        #         })
        # else:
        #     worker_configs.append({
        #         'type': worker_config['type'],
        #         'd_model': worker_config['d_model'],
        #         'num_layers': worker_config['num_layers'],
        #         'dropout': worker_config['dropout'],
        #         **base_config
        #     })
        
        for i in range(cur_worker_count):
            worker_configs.append({
                'type': worker_config['type'],
                'd_model': worker_config['d_model'],
                'num_layers': worker_config['num_layers'],
                'dropout': worker_config['dropout'],
                **base_config
            })
    
    return worker_configs


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CATF on ET dataset")
    parser.add_argument("--config", type=str, default="../../cap/configs/ETTh1/et_cap_times.yaml",
                       help="Path to configuration file")
    parser.add_argument("--validate-config", action="store_true",
                       help="Validate configuration and exit")
    parser.add_argument("--num-workers", type=int, nargs='+', default=None,
                        help="List of worker model counts")
    
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
        # Record start time
        start_time = datetime.now()
        
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
        worker_configs = create_worker_configs(config, input_dim, output_dim, seq_len, pred_len, args.num_workers)
        
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
        trainer = CATFTrainer(
            manager_model=manager_model,
            worker_models=worker_models,
            criterion=nn.MSELoss(),
            device=device,
            manager_lr=config['training']['manager_lr'],
            worker_lr=config['training']['worker_lr'],
            log_dir=config['logging']['log_dir'],
            clip_value=config['training']['clip_value'],
            worker_update_steps=config['training']['worker_update_steps'],
            use_multi_gpu=False,  # Disable multi-GPU training to avoid NCCL errors
            distributed=False  # Use DataParallel instead of DDP for now
        )
        
        # 5. Training
        print(" Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['epochs'],
            checkpoint_dir=config['logging']['checkpoint_dir'],
            early_stopping_patience=config['training']['patience'],  # No early stopping for quick test
            plot_metrics=config['logging']['plot_metrics'],
            pre_training_epochs=config['training']['pre_training_epochs']
        )
        
        # 6. Evaluation
        print(" Evaluating on test set...")
        test_loss = trainer.validate(test_loader)
        
        # Calculate training time
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # Get dataset name from config
        dataset_name = os.path.basename(config['data']['path']).replace('.csv', '').replace('.txt', '')
        
        # Create model configuration summary
        worker_configs = []
        i = 0
        for worker_config in config['models']['workers']:
            if args.num_workers:
                worker_configs.append(f"{worker_config['type']}x{args.num_workers[i]}")
                i += 1
            else:
                if 'count' in worker_config:
                    worker_configs.append(f"{worker_config['type']}x{worker_config['count']}")
                else:
                    worker_configs.append(worker_config['type'])
        model_config = f"Manager({config['models']['manager']['d_model']}d_{config['models']['manager']['num_layers']}l) + Workers({', '.join(worker_configs)})"
        
        # Save test loss to text file with dataset and model info
        test_loss_file_path = trainer.save_test_loss(test_loss, config['logging']['log_dir'], dataset_name, model_config, history)
        
        # Save comprehensive training results
        if test_loss_file_path:  # Only append if file was created successfully
            save_training_results(
                history=history,
                test_loss=test_loss,
                dataset_name=dataset_name,
                model_config=model_config,
                log_dir=config['logging']['log_dir'],
                config_path=args.config,
                test_loss_file_path=test_loss_file_path,
                config=config,
                training_time=training_time
            )
        
        # 7. Results
        print("\n Training completed!")
        print(f"   Final training loss: {history['train_losses'][-1]:.4f}")
        print(f"   Final validation loss: {history['val_losses'][-1]:.4f}")
        print(f"   Test loss: {test_loss:.4f}")
        print(f"   Best validation loss: {min(history['val_losses']):.4f}")
        print(f"   Training time: {training_time}")
        
        # 8. Worker selection analysis
        final_selections = history['worker_selections'][-1]
        print("\n Final worker selection rates:")
        for i, rate in enumerate(final_selections):
            print(f"   Worker {i}: {rate:.3f}")
        
        print("\n CATF training completed successfully!")
        
    except Exception as e:
        print(f" Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 