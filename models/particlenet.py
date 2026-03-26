"""
models/particlenet.py - ParticleNet-style GNN for Jet Classification

This module implements the ParticleNet architecture and variants for
quark-gluon jet classification.

Reference:
    "ParticleNet: Jet Tagging via Particle Clouds"
    Huilin Qu, Loukas Gouskos
    arXiv:1902.08570 (2020)

Author: Dev Datya Pratap Bansal
For: ML4SCI QMLHEP Task V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, 
    GATConv,
    global_mean_pool, 
    global_max_pool,
    knn_graph
)
import numpy as np


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scatter_mean(src, index, dim=0, dim_size=None):
    """
    Compute mean over groups specified by index.
    Pure PyTorch implementation (no torch_scatter dependency).
    
    Args:
        src: Source tensor
        index: Group indices
        dim: Dimension to scatter
        dim_size: Size of output dimension
    
    Returns:
        Mean values for each group
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output tensor
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    
    # Sum values
    index_expanded = index.view([-1] + [1] * (src.dim() - 1))
    index_expanded = index_expanded.expand_as(src)
    
    out.scatter_add_(dim, index_expanded, src)
    
    # Count occurrences
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    ones = torch.ones(src.size(dim), dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, ones)
    
    # Avoid division by zero
    count = count.clamp(min=1)
    
    # Compute mean
    count_expanded = count.view([-1] + [1] * (out.dim() - 1))
    out = out / count_expanded
    
    return out


class EdgeConvBlock(nn.Module):
    """
    EdgeConv block with dynamic graph construction.
    
    For each particle, computes edge features with k nearest neighbors,
    then aggregates using convolution. This captures local geometric
    structure in the particle cloud.
    
    The edge features are computed as:
        e_ij = MLP(concat(x_i, x_j - x_i))
    
    And aggregated as:
        x'_i = mean({e_ij : j in N(i)})
    """
    
    def __init__(self, in_channels, out_channels, k=8):
        super(EdgeConvBlock, self).__init__()
        
        self.k = k
        
        # MLP for edge features
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, batch=None):
        """
        Forward pass with dynamic graph construction.
        
        Args:
            x: Node features [N, in_channels]
            batch: Batch indices [N] (optional)
        
        Returns:
            Updated node features [N, out_channels]
        """
        # Build dynamic k-NN graph
        edge_index = self._build_knn_graph(x, batch)
        
        # Apply EdgeConv
        out = self._edge_conv(x, edge_index)
        
        return out
    
    def _build_knn_graph(self, x, batch=None):
        """
        Build k-NN graph based on spatial distance (eta, phi).
        """
        # Use (eta, phi) as spatial coordinates
        if x.shape[1] >= 2:
            pos = x[:, :2]
        else:
            pos = x
        
        # Build k-NN graph
        edge_index = knn_graph(
            pos, 
            k=self.k, 
            batch=batch, 
            loop=False,
            cosine=False
        )
        
        return edge_index
    
    def _edge_conv(self, x, edge_index):
        """
        Apply EdgeConv operation.
        """
        row, col = edge_index
        
        # Gather features
        x_i = x[row]  # Source nodes
        x_j = x[col]  # Target nodes
        
        # Compute edge features: [x_i, x_j - x_i]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        
        # Apply MLP
        edge_features = self.mlp(edge_features)
        
        # Aggregate (mean over neighbors) - using pure PyTorch
        out = scatter_mean(edge_features, row, dim=0, dim_size=x.size(0))
        
        return out


class ParticleNet(nn.Module):
    """
    ParticleNet: Jet Classification via Particle Clouds.
    
    Architecture:
        1. Input batch normalization
        2. Three EdgeConv blocks with increasing channels
        3. Global pooling (mean) for each block
        4. Concatenate pooled features
        5. Fully connected classifier
    """
    
    def __init__(self, 
                 input_dim=4,
                 num_classes=2,
                 conv_channels=[64, 128, 256],
                 fc_channels=[256, 128],
                 k_neighbors=16,
                 dropout=0.1):
        super(ParticleNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        
        # Input feature normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # EdgeConv blocks
        self.conv_blocks = nn.ModuleList()
        
        in_channels = input_dim
        for out_channels in conv_channels:
            self.conv_blocks.append(
                EdgeConvBlock(in_channels, out_channels, k=k_neighbors)
            )
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Fully connected classifier
        fc_layers = []
        in_features = sum(conv_channels)  # Concatenate all conv outputs
        
        for out_features in fc_channels:
            fc_layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = out_features
        
        # Output layer
        fc_layers.append(nn.Linear(in_features, num_classes))
        
        self.fc = nn.Sequential(*fc_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data object with x, edge_index, batch
        
        Returns:
            logits: [batch_size, num_classes]
        """
        x = data.x
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Input normalization
        x = self.input_bn(x)
        
        # EdgeConv blocks - collect outputs for skip connections
        conv_outputs = []
        for conv_block in self.conv_blocks:
            x = conv_block(x, batch)
            conv_outputs.append(x)
        
        # Global pooling for each conv output
        pooled_outputs = []
        for conv_out in conv_outputs:
            if batch is not None:
                pooled = self.global_pool(conv_out, batch)
            else:
                # Single graph case
                pooled = conv_out.mean(dim=0, keepdim=True)
            pooled_outputs.append(pooled)
        
        # Concatenate pooled features
        x = torch.cat(pooled_outputs, dim=1)
        
        # Fully connected classifier
        logits = self.fc(x)
        
        return logits
    
    def predict_proba(self, data):
        """Return probability predictions."""
        logits = self.forward(data)
        return torch.softmax(logits, dim=1)
    
    def predict(self, data):
        """Return class predictions."""
        proba = self.predict_proba(data)
        return torch.argmax(proba, dim=1)


class ParticleNetLite(nn.Module):
    """
    Lightweight ParticleNet for faster training.
    """
    
    def __init__(self, 
                 input_dim=4,
                 num_classes=2,
                 hidden_dim=64,
                 k_neighbors=8,
                 dropout=0.1):
        super(ParticleNetLite, self).__init__()
        
        # Feature embedding
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # EdgeConv layers
        self.conv1 = EdgeConvBlock(hidden_dim, hidden_dim, k=k_neighbors)
        self.conv2 = EdgeConvBlock(hidden_dim, hidden_dim, k=k_neighbors)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x = data.x
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Embed
        x = self.embed(x)
        
        # Convolve
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        
        # Pool
        if batch is not None:
            x1 = global_mean_pool(x1, batch)
            x2 = global_mean_pool(x2, batch)
        else:
            x1 = x1.mean(dim=0, keepdim=True)
            x2 = x2.mean(dim=0, keepdim=True)
        
        # Classify
        x = torch.cat([x1, x2], dim=1)
        logits = self.classifier(x)
        
        return logits


class SimpleGNN(nn.Module):
    """
    Simple baseline GNN using static pre-computed graph.
    """
    
    def __init__(self, 
                 input_dim=4,
                 hidden_dim=64,
                 num_classes=2,
                 num_layers=3,
                 dropout=0.1):
        super(SimpleGNN, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.input_bn(x)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        return self.classifier(x)


class GATNet(nn.Module):
    """
    Graph Attention Network for jet classification.
    """
    
    def __init__(self,
                 input_dim=4,
                 hidden_dim=64,
                 num_classes=2,
                 num_layers=3,
                 heads=4,
                 dropout=0.1):
        super(GATNet, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim * heads
            out_ch = hidden_dim
            
            self.convs.append(
                GATConv(in_ch, out_ch, heads=heads, dropout=dropout, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(out_ch * heads))
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim * heads, hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.input_bn(x)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        # Project
        x = self.final_proj(x)
        x = F.relu(x)
        
        # Pool
        x = global_mean_pool(x, batch)
        
        return self.classifier(x)


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("TESTING MODELS")
    print("=" * 60)
    
    from torch_geometric.data import Data, Batch
    
    # Create dummy data
    n_particles = 100
    x = torch.randn(n_particles, 4)
    edge_index = torch.randint(0, n_particles, (2, 500))
    batch = torch.zeros(n_particles, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test ParticleNet
    print("\n1. ParticleNet")
    model = ParticleNet(input_dim=4, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    out = model(data)
    print(f"   Output shape: {out.shape}")
    assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"
    print("   ✓ Test passed")
    
    # Test ParticleNetLite
    print("\n2. ParticleNetLite")
    model_lite = ParticleNetLite(input_dim=4, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
    out = model_lite(data)
    print(f"   Output shape: {out.shape}")
    assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"
    print("   ✓ Test passed")
    
    # Test SimpleGNN
    print("\n3. SimpleGNN")
    model_gnn = SimpleGNN(input_dim=4, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model_gnn.parameters()):,}")
    out = model_gnn(data)
    print(f"   Output shape: {out.shape}")
    assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"
    print("   ✓ Test passed")
    
    print("\n" + "=" * 60)
    print("ALL MODEL TESTS PASSED!")
    print("=" * 60)
