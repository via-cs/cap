#!/usr/bin/env python3
"""
Test Data Loading Script
========================

A simple script to test that the data loading works correctly
with the new format that provides (X, Y) pairs.
"""

import os
import sys
import torch
from pathlib import Path

# Add the cap package to the path
sys.path.append(str(Path(__file__).parent / "cap"))

from cap import get_dataloaders

def test_data_loading():
    """Test that data loading works correctly."""
    print("🧪 Testing Data Loading")
    print("=" * 30)
    
    # Configuration
    data_path = "dataset/ElectricityTransformer/ETTh1.csv"
    batch_size = 4  # Small batch for testing
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f" Data not found: {data_path}")
        print("Please make sure the dataset exists.")
        return False
    
    try:
        # Load data
        print(" Loading data...")
        train_loader, val_loader, test_loader = get_dataloaders(
            path=data_path,
            batch_size=batch_size,
            shuffle=True,
            seq_len=96,  # Set reasonable sequence length
            pred_len=24  # Set reasonable prediction length
        )
        
        print(f" Data loaders created successfully!")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test first batch
        print("\n Testing first batch...")
        for batch in train_loader:
            inputs, targets = batch
            print(f"   Input shape: {inputs.shape}")
            print(f"   Target shape: {targets.shape}")
            print(f"   Input dtype: {inputs.dtype}")
            print(f"   Target dtype: {targets.dtype}")
            print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"   Target range: [{targets.min():.3f}, {targets.max():.3f}]")
            
            # Check that we have the expected format
            assert len(batch) == 2, f"Expected 2 elements in batch, got {len(batch)}"
            assert inputs.dim() == 3, f"Expected 3D input tensor, got {inputs.dim()}D"
            assert targets.dim() == 3, f"Expected 3D target tensor, got {targets.dim()}D"
            assert inputs.shape[0] == batch_size, f"Expected batch size {batch_size}, got {inputs.shape[0]}"
            assert targets.shape[0] == batch_size, f"Expected batch size {batch_size}, got {targets.shape[0]}"
            
            print(" Batch format is correct!")
            break
        
        # Test multiple batches
        print("\n Testing multiple batches...")
        batch_count = 0
        for batch in train_loader:
            inputs, targets = batch
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        print(f" Successfully processed {batch_count} batches")
        
        # Test validation and test loaders
        print("\n Testing validation and test loaders...")
        for batch in val_loader:
            inputs, targets = batch
            print(f"   Validation batch: {inputs.shape}, {targets.shape}")
            break
            
        for batch in test_loader:
            inputs, targets = batch
            print(f"   Test batch: {inputs.shape}, {targets.shape}")
            break
        
        print("\n🎉 All data loading tests passed!")
        return True
        
    except Exception as e:
        print(f" Data loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n Data loading is working correctly!")
        print("You can now run the CATP experiments.")
    else:
        print("\n Data loading has issues. Please check the configuration.") 