import os
import torch
import torch.nn as nn
from torch.optim import Adam
from ..models.catp import ManagerModel, create_worker_pool
from ..training.catp_trainer import CATPTrainer
from ..data.data import get_dataloaders
from ..models.transformer import Transformer
from ..models.lstm import TimeSeriesLSTM
from ..models.Informer import Informer
from ..models.Autoformer import Autoformer
from ..models.FEDFormer import FEDformer
import math
import numpy as np


def main():
    # Configuration
    config = {
        'data_path': '/home/yeqchen/cap/cap/dataset/ElectricityTransformer/et_data.txt',
        'batch_size': 32,
        'epochs': 30,
        'manager_learning_rate': 0.005,  # Reduced learning rate for manager
        'worker_learning_rate': 0.001,    # Increased learning rate for workers
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'checkpoint_dir': 'checkpoints/catp',
        'log_dir': 'runs/catp',
        
        # Model parameters
        'input_dim': 7,
        'output_dim': 1,
        'seq_len': 32,
        'pred_len': 32,
        
        # Manager parameters
        'manager_d_model': 256,           # Reduced model size
        'manager_n_heads': 4,             # Reduced number of heads
        'manager_d_ff': 1024,             # Reduced feed-forward dimension
        'manager_layers': 2,              # Reduced number of layers
        'manager_dropout': 0.2,           # Increased dropout
        
        # Learning rate scheduler parameters
        'min_lr': 1e-5,
    }

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config['data_path'],
        batch_size=config['batch_size'],
        model_type='lstm',
        shuffle=True
    )

    # Define available models
    available_models = {
        'transformer': Transformer,
        'lstm': TimeSeriesLSTM,
        'informer': Informer,
        'autoformer': Autoformer,
        'fedformer': FEDformer
    }

    # Define worker configurations
    worker_configs = [
        # LSTM Workers (more emphasis on LSTM since it performed better)
        {
            'model_name': 'lstm',
            'input_dim': config['input_dim'],
            'hidden_dim': 256,
            'num_layers': 2,
            'output_dim': config['output_dim'],
            'dropout': 0.1
        },
        # {
        #     'model_name': 'lstm',
        #     'input_dim': config['input_dim'],
        #     'hidden_dim': 128,
        #     'num_layers': 1,
        #     'output_dim': config['output_dim'],
        #     'dropout': 0.1
        # },
        # {
        #     'model_name': 'lstm',
        #     'input_dim': config['input_dim'],
        #     'hidden_dim': 512,
        #     'num_layers': 3,
        #     'output_dim': config['output_dim'],
        #     'dropout': 0.2
        # },
        {
            'model_name': 'lstm',
            'input_dim': config['input_dim'],
            'hidden_dim': 64,
            'num_layers': 2,
            'output_dim': config['output_dim'],
            'dropout': 0.15
        },
        {
            'model_name': 'lstm',
            'input_dim': config['input_dim'],
            'hidden_dim': 192,
            'num_layers': 2,
            'output_dim': config['output_dim'],
            'dropout': 0.1
        },
        # Transformer Workers (simplified for shorter sequences)
        {
            'model_name': 'transformer',
            'input_dim': config['input_dim'],
            'output_dim': config['output_dim'],
            'seq_len': config['seq_len'],
            'pred_len': config['pred_len'],
            'd_model': 128,
            'n_heads': 4,
            'd_ff': 256,
            'num_layers': 2,
            'dropout': 0.1
        },
        {
            'model_name': 'transformer',
            'input_dim': config['input_dim'],
            'output_dim': config['output_dim'],
            'seq_len': config['seq_len'],
            'pred_len': config['pred_len'],
            'd_model': 64,
            'n_heads': 2,
            'd_ff': 128,
            'num_layers': 1,
            'dropout': 0.15
        }
    ]

    # Create models
    worker_models = create_worker_pool(worker_configs, available_models)
    manager_model = ManagerModel(
        input_dim=config['input_dim'],
        worker_count=len(worker_models),
        d_model=config['manager_d_model'],
        n_heads=config['manager_n_heads'],
        d_ff=config['manager_d_ff'],
        num_layers=config['manager_layers'],
        dropout=config['manager_dropout']
    )

    # Create optimizers with different learning rates
    manager_optimizer = Adam(manager_model.parameters(), lr=config['manager_learning_rate'], weight_decay=1e-5)
    worker_optimizers = [
        Adam(worker.parameters(), lr=config['worker_learning_rate'], weight_decay=1e-5)
        for worker in worker_models
    ]

    # Create learning rate schedulers
    def get_manager_lr(epoch):
        # Cosine decay: smoothly decrease from manager_learning_rate to min_lr
        progress = epoch / config['epochs']  # Progress from 0 to 1
        cosine_term = 1 + math.cos(math.pi * progress)
        return config['min_lr'] + 0.5 * (config['manager_learning_rate'] - config['min_lr']) * cosine_term

    def get_worker_lr(epoch):
        # Cosine decay: smoothly decrease from worker_learning_rate to min_lr
        progress = epoch / config['epochs']  # Progress from 0 to 1
        cosine_term = 1 + math.cos(math.pi * progress)
        return config['min_lr'] + 0.5 * (config['worker_learning_rate'] - config['min_lr']) * cosine_term

    # Create trainer
    trainer = CATPTrainer(
        manager_model=manager_model,
        worker_models=worker_models,
        criterion=nn.MSELoss(),
        manager_optimizer=manager_optimizer,
        worker_optimizers=worker_optimizers,
        device=config['device'],
        log_dir=config['log_dir'],
        clip_value=0.5,
        worker_update_steps=1
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        
        # Update learning rates
        current_manager_lr = get_manager_lr(epoch)
        current_worker_lr = get_worker_lr(epoch)
        
        for param_group in manager_optimizer.param_groups:
            param_group['lr'] = current_manager_lr
        for optimizer in worker_optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_worker_lr
        
        # Print learning rates with progress
        progress = epoch / config['epochs']
        print(f"Current learning rates - Manager: {current_manager_lr:.6f}, Worker: {current_worker_lr:.6f}")
        print(f"  (Training progress: {progress:.2%})")
        
        # Training
        for batch_idx, batch_data in enumerate(train_loader):
            metrics = trainer.train_step(
                batch_data,
                epoch,
                batch_idx,
                len(train_loader)
            )
            
            if batch_idx % 10 == 0:
                worker_selections = metrics['worker_selections']
                active_workers = metrics['active_workers']
                # Calculate selection percentages only for active workers
                selection_counts = np.bincount(worker_selections, minlength=len(worker_models))
                total_selections = len(worker_selections)
                selection_str = ", ".join([
                    f"Worker {i}: {count/total_selections:.1%}" 
                    for i, count in enumerate(selection_counts) 
                    if count > 0
                ])
                print(f"  Batch {batch_idx}: Worker Loss = {metrics['worker_loss']:.4f}, "
                      f"Manager Loss = {metrics['manager_loss']:.4f}")
                print(f"  Worker Selections: {selection_str}")
        
        # Validation
        val_loss = trainer.validate(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            trainer.save_checkpoint(checkpoint_path, epoch, best_val_loss)
            print(f"Saved checkpoint with validation loss: {best_val_loss:.4f}")

    # Final evaluation
    test_loss = trainer.validate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main() 