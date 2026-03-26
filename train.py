#!/usr/bin/env python
"""
train.py - Training script for Quark-Gluon Jet Classification (Task V)

This script trains a Graph Neural Network (ParticleNet) for quark-gluon
jet classification as part of the ML4SCI QMLHEP Task V.

Target: AUC > 0.80

Usage:
    python train.py                           # Default settings
    python train.py --epochs 100 --lr 0.001  # Custom settings
    python train.py --config config.yaml     # From config file

Author: Dev Datya Pratap Bansal
For: ML4SCI QMLHEP Task V
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    ReduceLROnPlateau,
    OneCycleLR
)
from torch_geometric.loader import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Local imports
from utils.dataset import QuarkGluonDataset, get_dataloaders, analyze_dataset
from utils.metrics import (
    compute_metrics,
    plot_training_history, 
    plot_roc_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    print_classification_report
)
from models.particlenet import (
    ParticleNet, 
    ParticleNetLite, 
    SimpleGNN, 
    GATNet,
    count_parameters
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GNN for Quark-Gluon Jet Classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--k_neighbors', type=int, default=8,
                        help='Number of neighbors for k-NN graph')
    parser.add_argument('--max_particles', type=int, default=30,
                        help='Maximum particles per jet')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='Force regenerate synthetic data')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='particlenet',
                        choices=['particlenet', 'particlenet_lite', 'simple_gnn', 'gat'],
                        help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension (for lite models)')
    parser.add_argument('--conv_channels', type=int, nargs='+', 
                        default=[64, 128, 256],
                        help='Convolution channels for ParticleNet')
    parser.add_argument('--fc_channels', type=int, nargs='+',
                        default=[256, 128],
                        help='Fully connected channels')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers (for simple models)')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads (for GAT)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'onecycle', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Early stopping patience (0 to disable)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Save only the best model')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path (overrides other args)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
            
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    return args


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str):
    """Get torch device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def build_model(args, input_dim=4):
    """Build model based on arguments."""
    if args.model == 'particlenet':
        model = ParticleNet(
            input_dim=input_dim,
            num_classes=2,
            conv_channels=args.conv_channels,
            fc_channels=args.fc_channels,
            k_neighbors=args.k_neighbors,
            dropout=args.dropout
        )
    elif args.model == 'particlenet_lite':
        model = ParticleNetLite(
            input_dim=input_dim,
            num_classes=2,
            hidden_dim=args.hidden_dim,
            k_neighbors=args.k_neighbors,
            dropout=args.dropout
        )
    elif args.model == 'simple_gnn':
        model = SimpleGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model == 'gat':
        model = GATNet(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training", leave=False, ncols=100):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch)
        loss = criterion(logits, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, auc, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False, ncols=100):
        batch = batch.to(device)
        
        logits = model(batch)
        loss = criterion(logits, batch.y)
        
        total_loss += loss.item() * batch.num_graphs
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, auc, acc, all_preds, all_labels


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f'{args.model}_{timestamp}'
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Print header
    print("\n" + "=" * 70)
    print("  QUARK-GLUON JET CLASSIFICATION - ML4SCI QMLHEP TASK V")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Target: AUC > 0.80")
    print()
    
    # Load data
    print("Loading dataset...")
    train_loader, test_loader, dataset = get_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        k_neighbors=args.k_neighbors,
        max_particles=args.max_particles,
        num_workers=args.num_workers,
        force_regenerate=args.force_regenerate
    )
    
    # Analyze dataset
    analyze_dataset(dataset)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(args, input_dim=4)
    model = model.to(device)
    
    n_params = count_parameters(model)
    print(f"Model: {args.model}")
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', patience=args.lr_patience, 
            factor=0.5, verbose=True
        )
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
    else:
        scheduler = None
    
    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)
    
    history = {
        'train_loss': [], 'train_auc': [], 'train_acc': [],
        'val_loss': [], 'val_auc': [], 'val_acc': [],
        'lr': []
    }
    
    best_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_auc, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        val_loss, val_auc, val_acc, val_preds, val_labels = evaluate(
            model, test_loader, criterion, device
        )
        
        # Get learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_auc)
            elif args.scheduler == 'onecycle':
                scheduler.step()
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Check if target achieved
        if val_auc > 0.80:
            print(f"🎯 TARGET ACHIEVED! AUC = {val_auc:.4f} > 0.80")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'config': vars(args)
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"✓ New best model saved! AUC = {val_auc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Final results
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest Results:")
    print(f"  Best Val AUC: {best_auc:.4f} (Epoch {best_epoch})")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    _, _, _, final_preds, final_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Print classification report
    metrics = print_classification_report(final_labels, final_preds)
    
    # Save plots
    print("\nGenerating visualizations...")
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    plot_roc_curve(final_labels, final_preds, os.path.join(output_dir, 'roc_curve.png'))
    plot_confusion_matrix(final_labels, final_preds, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_score_distribution(final_labels, final_preds, os.path.join(output_dir, 'score_distribution.png'))
    
    # Save history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    
    # Save predictions
    np.savez(
        os.path.join(output_dir, 'predictions.npz'),
        labels=final_labels,
        predictions=final_preds
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    target_achieved = best_auc > 0.80
    print(f"\n  Task V Target: AUC > 0.80")
    print(f"  Achieved AUC:  {best_auc:.4f}")
    
    if target_achieved:
        print("\n  ✅ TASK V COMPLETED SUCCESSFULLY!")
        print("  Congratulations! You can now submit your results.")
    else:
        print("\n  ⚠️  Target not yet achieved.")
        print("  Suggestions:")
        print("    - Train for more epochs: --epochs 100")
        print("    - Increase model capacity: --conv_channels 64 128 256 512")
        print("    - Tune learning rate: --lr 0.0005")
        print("    - Use real JetNet dataset instead of synthetic")
    
    print(f"\n  All outputs saved to: {output_dir}")
    print("=" * 70 + "\n")
    
    return best_auc


if __name__ == "__main__":
    main()
