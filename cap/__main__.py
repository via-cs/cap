#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import torch
from . import __version__

def main():
    parser = argparse.ArgumentParser(description='CAP: Time Series Forecasting Framework')
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    
    # Add subcommands for different operations
    subparsers = parser.add_subparsers(dest='operation', help='Available operations')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, required=True, 
                             choices=['transformer', 'fedformer', 'autoformer', 'timesnet', 'informer', 'lstm'],
                             help='Model type to train')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--data', type=str, required=True, help='Path to input data')
    train_parser.add_argument('--output', type=str, required=True, help='Path to save trained model')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                             help='Device to train on (cuda or cpu)')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--model', type=str, required=True,
                               choices=['transformer', 'fedformer', 'autoformer', 'timesnet', 'informer', 'lstm'],
                               help='Model type to use')
    predict_parser.add_argument('--config', type=str, help='Path to config file')
    predict_parser.add_argument('--data', type=str, required=True, help='Path to input data')
    predict_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    predict_parser.add_argument('--output', type=str, help='Path to save predictions')
    predict_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                               help='Device to use for prediction (cuda or cpu)')
    
    args = parser.parse_args()
    
    if not args.operation:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.operation == 'train':
            from .training import train_model
            from .data.data import get_dataloaders
            
            # Load data
            train_loader, valid_loader, _ = get_dataloaders(
                args.data, 
                batch_size=args.batch_size
            )
            
            # Get model dimensions from data
            sample_batch = next(iter(train_loader))
            if args.model in ['lstm', 'transformer']:
                input_dim = sample_batch[0].shape[-1]
                output_dim = sample_batch[1].shape[-1]
                seq_len = sample_batch[0].shape[1]
                pred_len = sample_batch[1].shape[1]
            else:  # autoformer, informer, fedformer
                input_dim = sample_batch[0].shape[-1]
                output_dim = sample_batch[2].shape[-1]
                seq_len = sample_batch[0].shape[1]
                pred_len = sample_batch[2].shape[1]
            
            # Train model
            model = train_model(
                train_loader=train_loader,
                valid_loader=valid_loader,
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                pred_len=pred_len,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                model_type=args.model
            )
            
            # Save model
            torch.save(model.state_dict(), args.output)
            print(f"Model saved to {args.output}")
            
        elif args.operation == 'predict':
            from .training import load_model, evaluate_model
            from .data.data import get_dataloaders
            
            # Load data
            _, _, test_loader = get_dataloaders(args.data)
            
            # Get model dimensions from data
            sample_batch = next(iter(test_loader))
            if args.model in ['lstm', 'transformer']:
                input_dim = sample_batch[0].shape[-1]
                output_dim = sample_batch[1].shape[-1]
                seq_len = sample_batch[0].shape[1]
                pred_len = sample_batch[1].shape[1]
            else:  # autoformer, informer, fedformer
                input_dim = sample_batch[0].shape[-1]
                output_dim = sample_batch[2].shape[-1]
                seq_len = sample_batch[0].shape[1]
                pred_len = sample_batch[2].shape[1]
            
            # Load model
            model = load_model(
                model_path=args.model_path,
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                pred_len=pred_len,
                device=args.device,
                model_type=args.model
            )
            
            # Evaluate model
            test_loss = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=args.device,
                model_type=args.model
            )
            
            print(f"Test Loss: {test_loss:.4f}")
            
            # Save predictions if output path provided
            if args.output:
                model.eval()
                predictions = []
                with torch.no_grad():
                    for batch in test_loader:
                        if args.model in ['lstm', 'transformer']:
                            inputs, _ = batch
                            outputs = model(inputs.to(args.device))
                        else:
                            x_enc, x_dec, _ = batch
                            outputs = model(x_enc.to(args.device), None, x_dec.to(args.device), None)
                        predictions.append(outputs.cpu())
                
                predictions = torch.cat(predictions, dim=0)
                torch.save(predictions, args.output)
                print(f"Predictions saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
