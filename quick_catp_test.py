#!/usr/bin/env python3
"""
Quick CATP Test Script
======================

A simple script to quickly test CATP with 6 worker models.
This is useful for verifying that everything works before running the full experiment.

Usage:
    python quick_catp_test.py
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add the cap package to the path
sys.path.append(str(Path(__file__).parent / "cap"))

import cap
from cap import get_dataloaders
from cap.models.catp import ManagerModel, create_worker_pool, available_models
from cap.training.catp_trainer import CATPTrainer

def main():
    """Quick test of CATP with 6 worker models."""
    print(" Quick CATP Test with 6 Worker Models")
    print("=" * 40)
    
    # Configuration
    data_path = 'dataset/ElectricityTransformer/ETTh1.csv'  # Fixed path
    batch_size = 16  # Smaller batch for quick test
    epochs = 5  # Few epochs for quick test
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f" Data not found: {data_path}")
        print("Please update the data_path variable.")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    try:
        # 1. Load data
        print(" Loading data...")
        

        train_loader, val_loader, test_loader = get_dataloaders(
            path=data_path,
            batch_size=batch_size,
            shuffle=True,
            train_size=0.8,
            valid_size=0.1,
            test_size=0.1,
            normalization=True,
            seq_len=96,
            pred_len=96
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
        worker_configs = [
            # Worker 1: LSTM (High capacity)
            {
                'type': 'lstm',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 128,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'num_layers': 3,
                'dropout': 0.1
            },
            # Worker 2: LSTM (Low capacity)
            {
                'type': 'lstm',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 64,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'num_layers': 2,
                'dropout': 0.1
            },
            # Worker 3: Transformer
            {
                'type': 'transformer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 128,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'num_layers': 2,
                'dropout': 0.1
            },
            # Worker 4: Autoformer
            {
                'type': 'autoformer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 128,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'num_layers': 2,
                'dropout': 0.1
            },
            # Worker 5: FEDFormer
            {
                'type': 'fedformer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 128,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'num_layers': 2,
                'dropout': 0.1
            },
            # Worker 6: Informer
            {
                'type': 'informer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 128,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'num_layers': 2,
                'dropout': 0.1
            },
            # Worker 7: TimesNet
            {
                'type': 'timesnet',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 128,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'num_layers': 2,
                'dropout': 0.1
            }
        ]
        
        # 3. Create models
        print("🏗️  Creating models...")
        worker_models = create_worker_pool(worker_configs, available_models())
        manager_model = ManagerModel(
            input_dim=input_dim,
            worker_count=len(worker_models),
            d_model=128,
            n_heads=4,
            d_ff=256,
            num_layers=2,
            dropout=0.1
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
            manager_lr=0.001,
            worker_lr=0.0005,
            log_dir="runs/quick_test",
            clip_value=1.0,
            worker_update_steps=2
        )
        
        # 5. Quick training
        print(" Starting quick training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            checkpoint_dir="saved_models/quick_test",
            early_stopping_patience=None,  # No early stopping for quick test
            plot_metrics=False  # No plotting for quick test
        )
        
        # 6. Quick evaluation
        print(" Quick evaluation...")
        test_loss = trainer.validate(test_loader)
        
        # 7. Results
        print("\n Quick test completed!")
        print(f"   Final training loss: {history['train_losses'][-1]:.4f}")
        print(f"   Final validation loss: {history['val_losses'][-1]:.4f}")
        print(f"   Test loss: {test_loss:.4f}")
        print(f"   Best validation loss: {min(history['val_losses']):.4f}")
        
        # 8. Worker selection analysis
        final_selections = history['worker_selections'][-1]
        worker_names = ['LSTM-High', 'LSTM-Low', 'Transformer', 'Autoformer', 'FEDFormer', 'Informer', 'TimesNet']
        print("\n Final worker selection rates:")
        for name, rate in zip(worker_names, final_selections):
            print(f"   {name}: {rate:.3f}")
        
        print("\n CATP is working correctly!")
        
    except Exception as e:
        print(f" Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 