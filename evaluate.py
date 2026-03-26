#!/usr/bin/env python
"""
evaluate.py - Evaluate trained model on test set

Usage:
    python evaluate.py --model outputs/particlenet_20240101/best_model.pt
"""

import os
import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from utils.dataset import get_dataloaders
from utils.metrics import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    print_classification_report
)
from models.particlenet import ParticleNet, ParticleNetLite, SimpleGNN, GATNet


def load_model(checkpoint_path, device='auto'):
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config
    config = checkpoint.get('config', {})
    model_type = config.get('model', 'particlenet')
    
    # Build model
    if model_type == 'particlenet':
        model = ParticleNet(
            input_dim=4,
            num_classes=2,
            conv_channels=config.get('conv_channels', [64, 128, 256]),
            fc_channels=config.get('fc_channels', [256, 128]),
            k_neighbors=config.get('k_neighbors', 8),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'particlenet_lite':
        model = ParticleNetLite(
            input_dim=4,
            num_classes=2,
            hidden_dim=config.get('hidden_dim', 64),
            k_neighbors=config.get('k_neighbors', 8),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'simple_gnn':
        model = SimpleGNN(
            input_dim=4,
            hidden_dim=config.get('hidden_dim', 64),
            num_classes=2,
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'gat':
        model = GATNet(
            input_dim=4,
            hidden_dim=config.get('hidden_dim', 64),
            num_classes=2,
            num_layers=config.get('num_layers', 3),
            heads=config.get('heads', 4),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run evaluation."""
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    args = parser.parse_args()
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model)
    
    # Load model
    print(f"Loading model from: {args.model}")
    model, config = load_model(args.model, args.device)
    print(f"Model: {config.get('model', 'particlenet')}")
    
    # Load data
    print("Loading dataset...")
    _, test_loader, _ = get_dataloaders(
        root=args.data_dir,
        batch_size=64,
        train_ratio=0.8,
        k_neighbors=config.get('k_neighbors', 8),
        max_particles=config.get('max_particles', 30)
    )
    
    # Evaluate
    print("Running evaluation...")
    predictions, labels = evaluate_model(model, test_loader, model.device)
    
    # Print results
    metrics = print_classification_report(labels, predictions)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_roc_curve(labels, predictions, 
                  os.path.join(args.output_dir, 'eval_roc_curve.png'))
    plot_confusion_matrix(labels, predictions,
                         os.path.join(args.output_dir, 'eval_confusion_matrix.png'))
    plot_score_distribution(labels, predictions,
                           os.path.join(args.output_dir, 'eval_score_distribution.png'))
    
    print(f"\nResults saved to: {args.output_dir}")
    
    return metrics['auc']


if __name__ == "__main__":
    main()
