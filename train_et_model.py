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
from torch.utils.data import DataLoader, Subset
from cap.data.data import CSVSequenceDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for ET-data')
    parser.add_argument('--config', type=str, default='cap/configs/ett_config.yaml',
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

    # Check CUDA availability and fall back to CPU if needed
    cfg_dev = config['training']['device']
    if cfg_dev.lower() == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        config['training']['device'] = 'cpu'

    # Override config with command line arguments if provided
    if args.model_type:
        config['model']['type'] = args.model_type
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.device:
        config['training']['device'] = args.device
    
    # If Fedformer or TimesNet, force a larger window for stability
    mt = config['model']['type'].lower()
    if mt in ('fedformer', 'timesnet'):
        print(f"{mt.title()} detected — overriding sequence and prediction lengths for stability.")
        # assume your config.yaml has seq_len/pred_len under model
        config['dataset']['seq_len']  = config['model'].get('seq_len', 96)
        config['dataset']['pred_len'] = config['model'].get('pred_len', 24)

    # Print configuration
    print("Configuration:")
    print(f"Dataset: {config['dataset']['path']}")
    print(f"Model: {config['model']['type']}")
    print(f"Training: {config['training']['epochs']} epochs, {config['training']['device']}")

    # Get dataloaders
    # Default lengths
    seq_len  = config['model'].get('seq_len',  96)
    pred_len = config['model'].get('pred_len', 24)

    # Override for Autoformer (before passing to dataloader!)
    if config['model']['type'].lower() == 'autoformer':
        print("Autoformer detected — overriding sequence and prediction lengths for stability.")
        seq_len = 96
        pred_len = 24

    # Get dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(
        path=config['dataset']['path'],
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        train_size=config['dataset']['train_size'],
        valid_size=config['dataset']['valid_size'],
        test_size=config['dataset']['test_size'],
        model_type=config['model']['type'],
        normalization=config['dataset'].get('normalization', True),
        seq_len  = seq_len,
        pred_len = pred_len
    )


    model_type = config['model']['type'].lower()
    if model_type == 'lstm':
        # 1) load raw CSV with NO per-sample normalization
        raw_ds = CSVSequenceDataset(
            config['dataset']['path'],
            seq_len=seq_len,
            pred_len=pred_len,
            normalization=False
        )

        # 2) split into train/valid/test by index
        N = len(raw_ds)
        idxs = list(range(N))
        random.shuffle(idxs)
        n1 = int(N * config['dataset']['train_size'])
        n2 = int(N * config['dataset']['valid_size'])

        train_idx = idxs[:n1]
        valid_idx = idxs[n1:n1 + n2]
        test_idx  = idxs[n1 + n2:]

        train_base = Subset(raw_ds, train_idx)
        valid_base = Subset(raw_ds, valid_idx)
        test_base  = Subset(raw_ds, test_idx)

        # 3) fit a global StandardScaler on all train windows
        all_X = np.concatenate([train_base[i][0].numpy() for i in range(len(train_base))], axis=0)
        x_scaler = StandardScaler().fit(all_X)
        # 3b) fit a StandardScaler on all train targets
        all_Y = np.concatenate([train_base[i][1].numpy() for i in range(len(train_base))], axis=0)
        y_scaler = StandardScaler().fit(all_Y)

        # 4) wrap a dataset that applies this scaler
        class ScaledDataset(torch.utils.data.Dataset):
            def __init__(self, base_ds, x_scaler, y_scaler):
                self.base     = base_ds
                self.x_scaler = x_scaler
                self.y_scaler = y_scaler

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                X, Y = self.base[idx]                    # X: [seq_len, feat], Y: [pred_len, feat]
                # scale X exactly as before
                X_np = X.numpy().reshape(-1, X.shape[-1])
                Xs   = self.x_scaler.transform(X_np).reshape(X.shape)
                # now scale Y
                Y_np = Y.numpy().reshape(-1, Y.shape[-1])
                Ys   = self.y_scaler.transform(Y_np).reshape(Y.shape)
                return (
                    torch.tensor(Xs, dtype=torch.float32),
                    torch.tensor(Ys, dtype=torch.float32)
                )

        train_ds = ScaledDataset(train_base, x_scaler, y_scaler)
        valid_ds = ScaledDataset(valid_base, x_scaler, y_scaler)
        test_ds  = ScaledDataset(test_base,  x_scaler, y_scaler)

        # 5) build DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([x for x, y in batch]),
                torch.stack([y for x, y in batch])
            )
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=config['dataset']['batch_size'],
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([x for x, y in batch]),
                torch.stack([y for x, y in batch])
            )
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([x for x, y in batch]),
                torch.stack([y for x, y in batch])
            )
        )

    else:
        # everyone else—use your original per-sample normalization path
        train_loader, valid_loader, test_loader = get_dataloaders(
            path=config['dataset']['path'],
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            train_size=config['dataset']['train_size'],
            valid_size=config['dataset']['valid_size'],
            test_size=config['dataset']['test_size'],
            model_type=config['model']['type'],
            normalization=config['dataset'].get('normalization', True),
            seq_len=seq_len,
            pred_len=pred_len
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