#!/usr/bin/env python
"""
quick_start.py - Quick start script for Task V

This script runs the entire Task V pipeline:
1. Setup environment
2. Generate/load data
3. Train model
4. Evaluate results

Usage:
    python quick_start.py
"""

import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print('=' * 60)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error during: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def main():
    print("\n" + "=" * 60)
    print("  ML4SCI QMLHEP TASK V - QUICK START")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Verify environment")
    print("  2. Generate dataset")
    print("  3. Train model")
    print("  4. Evaluate results")
    print("  5. Report final AUC score")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Verify environment
    print("\n" + "-" * 60)
    print("Step 1: Verifying environment...")
    
    try:
        import torch
        import torch_geometric
        import numpy as np
        import sklearn
        print(f"  PyTorch: {torch.__version__}")
        print(f"  PyG: {torch_geometric.__version__}")
        print(f"  NumPy: {np.__version__}")
        print(f"  scikit-learn: {sklearn.__version__}")
        print("✅ Environment OK")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\nRun setup.sh first:")
        print("  bash setup.sh")
        sys.exit(1)
    
    # Step 2: Train model
    print("\n" + "-" * 60)
    print("Step 2: Training model...")
    
    train_cmd = "python train.py --epochs 50 --batch_size 32 --lr 0.001"
    success = run_command(train_cmd, "Training")
    
    if not success:
        print("Training failed. Check error messages above.")
        sys.exit(1)
    
    # Step 3: Find output directory
    print("\n" + "-" * 60)
    print("Step 3: Locating results...")
    
    output_dirs = [d for d in os.listdir('./outputs') 
                   if os.path.isdir(os.path.join('./outputs', d))]
    
    if not output_dirs:
        print("No output directory found!")
        sys.exit(1)
    
    latest_dir = max(output_dirs)
    model_path = os.path.join('./outputs', latest_dir, 'best_model.pt')
    
    print(f"Latest output: {latest_dir}")
    print(f"Model path: {model_path}")
    
    # Step 4: Final summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    
    # Load and print history
    import json
    history_path = os.path.join('./outputs', latest_dir, 'history.json')
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        best_auc = max(history['val_auc'])
        best_epoch = history['val_auc'].index(best_auc) + 1
        
        print(f"\n  Best AUC:     {best_auc:.4f}")
        print(f"  Best Epoch:   {best_epoch}")
        print(f"  Final AUC:    {history['val_auc'][-1]:.4f}")
        
        print("\n" + "-" * 60)
        
        if best_auc > 0.80:
            print("\n  ✅ TASK V COMPLETED SUCCESSFULLY!")
            print(f"     Target AUC > 0.80: ACHIEVED ({best_auc:.4f})")
            print("\n  You can now submit your results!")
        else:
            print("\n  ⚠️  Target not yet achieved.")
            print(f"     Current AUC: {best_auc:.4f}")
            print("\n  Try these improvements:")
            print("     python train.py --epochs 100")
            print("     python train.py --conv_channels 128 256 512")
    else:
        print("Could not find training history.")
    
    print("\n" + "=" * 60)
    print(f"  All outputs saved to: outputs/{latest_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
