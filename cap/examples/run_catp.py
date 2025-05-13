import torch
import torch.nn as nn
import cap
from cap import *

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = cap.get_dataloaders(
        path='/home/yeqchen/cap/cap/dataset/ElectricityTransformer/et_data.txt',
        batch_size=32,
        model_type='lstm',
        shuffle=True
    )

    # Get input dimensions from first batch
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

    # Define worker configurations
    worker_configs = [
        # LSTM Workers
        {
            'model_name': 'lstm',
            'input_dim': input_dim,
            'hidden_dim': 256,
            'num_layers': 2,
            'output_dim': output_dim,
            'dropout': 0.1
        },
        {
            'model_name': 'lstm',
            'input_dim': input_dim,
            'hidden_dim': 64,
            'num_layers': 2,
            'output_dim': output_dim,
            'dropout': 0.15
        },
        # Transformer Worker
        {
            'model_name': 'transformer',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'seq_len': seq_len,
            'pred_len': pred_len,
            'd_model': 128,
            'n_heads': 4,
            'd_ff': 256,
            'num_layers': 2,
            'dropout': 0.1
        }
    ]

    # Create models
    worker_models = cap.catp.create_worker_pool(worker_configs, cap.catp.available_models())
    manager_model = cap.catp.ManagerModel(
        input_dim=input_dim,
        worker_count=len(worker_models),
        d_model=256,
        n_heads=4,
        d_ff=1024,
        num_layers=2,
        dropout=0.2
    )

    # Create trainer with default optimizers
    trainer = cap.catp_trainer.CATPTrainer(
        manager_model=manager_model,
        worker_models=worker_models,
        criterion=nn.MSELoss(),
        device=device,
        manager_lr=0.005,  # Optional: override default learning rate
        worker_lr=0.001    # Optional: override default learning rate
    )

    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        checkpoint_dir='checkpoints/catp',
        early_stopping_patience=5,  # Optional: enable early stopping
        plot_metrics=True           # Optional: show training plots
    )

    # Final evaluation
    test_loss = trainer.validate(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main() 