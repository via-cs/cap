#!/usr/bin/env python3
"""
Configuration Loader for CATP Training
======================================

Utility functions to load and validate YAML configuration files for CATP training.
"""

import yaml
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['data', 'models', 'training', 'logging', 'hardware']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate data configuration
    data_config = config['data']
    if not os.path.exists(data_config['path']):
        print(f"Warning: Data path does not exist: {data_config['path']}")
    
    # Validate model configuration
    models_config = config['models']
    if 'manager' not in models_config:
        raise ValueError("Missing manager configuration")
    if 'workers' not in models_config:
        raise ValueError("Missing workers configuration")
    
    # Validate training configuration
    training_config = config['training']
    if training_config['epochs'] <= 0:
        raise ValueError("Epochs must be positive")
    if training_config['manager_lr'] <= 0 or training_config['worker_lr'] <= 0:
        raise ValueError("Learning rates must be positive")
    
    return True


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get the device based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device object
    """
    device_config = config['hardware']['device']
    
    if device_config == "auto":
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_config == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device('cpu')
        return torch.device('cuda')
    elif device_config == "cpu":
        return torch.device('cpu')
    else:
        raise ValueError(f"Unknown device: {device_config}")


def create_experiment_dir(config: Dict[str, Any]) -> str:
    """
    Create experiment directory based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the experiment directory
    """
    experiment_name = config['experiment']['name']
    log_dir = config['logging']['log_dir']
    
    # Create experiment directory
    experiment_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    prediction_dir = os.path.join(experiment_dir, 'predictions')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    
    return experiment_dir


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def print_config_summary(config: Dict[str, Any]):
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 50)
    print("CATP Training Configuration Summary")
    print("=" * 50)
    
    # Data
    print(f"Data Path: {config['data']['path']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    
    # Models
    print(f"Manager Model: {config['models']['manager']['d_model']}D, {config['models']['manager']['num_layers']} layers")
    print(f"Worker Models: {len(config['models']['workers'])} workers")
    for worker in config['models']['workers']:
        print(f"  - {worker['name']}: {worker['type']}")
    
    # Training
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Manager LR: {config['training']['manager_lr']}")
    print(f"Worker LR: {config['training']['worker_lr']}")
    
    # Hardware
    device = get_device(config)
    print(f"Device: {device}")
    
    print("=" * 50)


def main():
    """Command line interface for configuration loading."""
    parser = argparse.ArgumentParser(description="Load and validate CATP configuration")
    parser.add_argument("config_path", help="Path to YAML configuration file")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--summary", action="store_true", help="Print configuration summary")
    parser.add_argument("--save", help="Save configuration to file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Validate if requested
    if args.validate:
        try:
            validate_config(config)
            print("✅ Configuration is valid!")
        except ValueError as e:
            print(f"❌ Configuration validation failed: {e}")
            return
    
    # Print summary if requested
    if args.summary:
        print_config_summary(config)
    
    # Save if requested
    if args.save:
        save_config(config, args.save)
        print(f"Configuration saved to: {args.save}")


if __name__ == "__main__":
    main() 