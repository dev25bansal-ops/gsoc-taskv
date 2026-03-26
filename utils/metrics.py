"""
utils/metrics.py - Evaluation metrics and visualization utilities

This module provides functions for computing evaluation metrics
and creating visualizations for the jet classification task.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    roc_curve, 
    precision_recall_curve,
    confusion_matrix,
    classification_report
)


def compute_metrics(labels, predictions, threshold=0.5):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        labels: True labels (0 or 1)
        predictions: Predicted probabilities for class 1
        threshold: Threshold for binary classification
    
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # Binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'auc': roc_auc_score(labels, predictions),
        'accuracy': accuracy_score(labels, binary_preds),
        'threshold': threshold,
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Derived metrics
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
        metrics['precision'] + metrics['recall']
    ) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # Specificity (true negative rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_training_history(history, output_path='training_history.png'):
    """
    Plot training history (loss and AUC over epochs).
    
    Args:
        history: Dictionary with training history
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # AUC plot
    ax = axes[0, 1]
    if 'train_auc' in history:
        ax.plot(history['train_auc'], 'b-', label='Train', linewidth=2)
    if 'val_auc' in history:
        ax.plot(history['val_auc'], 'r-', label='Validation', linewidth=2)
    ax.axhline(y=0.80, color='g', linestyle='--', label='Target (0.80)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('AUC-ROC Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Accuracy plot
    ax = axes[1, 0]
    if 'train_acc' in history:
        ax.plot(history['train_acc'], 'b-', label='Train', linewidth=2)
    if 'val_acc' in history:
        ax.plot(history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax.axhline(y=0.70, color='g', linestyle='--', label='Reference (0.70)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # Learning rate plot
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(history['lr'], 'g-', linewidth=2)
        ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {output_path}")


def plot_roc_curve(labels, predictions, output_path='roc_curve.png'):
    """
    Plot ROC curve.
    
    Args:
        labels: True labels
        predictions: Predicted probabilities
        output_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Fill area
    plt.fill_between(fpr, tpr, alpha=0.2)
    
    # Styling
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Quark/Gluon Jet Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add target indicator
    if auc > 0.80:
        plt.text(0.6, 0.2, '✓ TARGET ACHIEVED\n(AUC > 0.80)', 
                fontsize=12, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_path}")


def plot_confusion_matrix(labels, predictions, output_path='confusion_matrix.png', threshold=0.5):
    """
    Plot confusion matrix.
    
    Args:
        labels: True labels
        predictions: Predicted probabilities
        output_path: Path to save the plot
        threshold: Classification threshold
    """
    binary_preds = (np.array(predictions) >= threshold).astype(int)
    cm = confusion_matrix(labels, binary_preds)
    
    plt.figure(figsize=(8, 6))
    
    # Plot
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    # Labels
    classes = ['Gluon', 'Quark']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # Add numbers
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=16, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def plot_score_distribution(labels, predictions, output_path='score_distribution.png'):
    """
    Plot distribution of prediction scores for each class.
    
    Args:
        labels: True labels
        predictions: Predicted probabilities
        output_path: Path to save the plot
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    quark_scores = predictions[labels == 1]
    gluon_scores = predictions[labels == 0]
    
    plt.figure(figsize=(10, 6))
    
    # Histograms
    bins = np.linspace(0, 1, 50)
    plt.hist(gluon_scores, bins=bins, alpha=0.6, label='Gluon (label=0)', 
             color='blue', density=True)
    plt.hist(quark_scores, bins=bins, alpha=0.6, label='Quark (label=1)', 
             color='red', density=True)
    
    # Threshold line
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Prediction Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Score distribution saved to: {output_path}")


def print_classification_report(labels, predictions, threshold=0.5):
    """
    Print detailed classification report.
    
    Args:
        labels: True labels
        predictions: Predicted probabilities
        threshold: Classification threshold
    """
    binary_preds = (np.array(predictions) >= threshold).astype(int)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    
    print(classification_report(labels, binary_preds, target_names=['Gluon', 'Quark']))
    
    # Additional metrics
    metrics = compute_metrics(labels, predictions, threshold)
    
    print(f"\nSummary Metrics:")
    print(f"  AUC-ROC:     {metrics['auc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    if metrics['auc'] > 0.80:
        print("\n✅ TARGET ACHIEVED! AUC = {:.4f} > 0.80".format(metrics['auc']))
    
    return metrics


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    labels = np.random.randint(0, 2, 1000)
    predictions = np.random.rand(1000) * 0.3 + labels * 0.4 + 0.2  # Some correlation
    
    metrics = compute_metrics(labels, predictions)
    print("Metrics:", metrics)
    
    # Create test plots
    os.makedirs('./test_plots', exist_ok=True)
    plot_roc_curve(labels, predictions, './test_plots/roc.png')
    plot_confusion_matrix(labels, predictions, './test_plots/cm.png')
    plot_score_distribution(labels, predictions, './test_plots/dist.png')
    
    print("\n✅ All tests passed!")
