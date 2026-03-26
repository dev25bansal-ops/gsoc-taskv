"""
utils/dataset.py - Quark-Gluon Jet Dataset for PyTorch Geometric

This module provides a dataset class for quark-gluon jet classification
using Graph Neural Networks.

Author: Dev Datya Pratap Bansal
For: ML4SCI QMLHEP Task V
"""

import os
import urllib.request
import numpy as np
import h5py
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors


class QuarkGluonDataset(InMemoryDataset):
    """
    Quark-Gluon Jet Dataset for jet classification using Graph Neural Networks.
    
    Each jet is represented as a graph where:
    - Nodes = constituent particles
    - Node features = [pT, eta, phi, particle_id, ...]
    - Edges = k-NN connections based on (eta, phi) distance
    - Label = 0 (gluon) or 1 (quark)
    """
    
    def __init__(self, 
                 root, 
                 transform=None, 
                 pre_transform=None,
                 pre_filter=None,
                 k_neighbors=8,
                 max_particles=30,
                 force_regenerate=False):
        """
        Args:
            root: Root directory for data storage
            transform: Optional transform to apply to each data object
            pre_transform: Optional transform before saving to disk
            pre_filter: Optional filter for data objects
            k_neighbors: Number of neighbors for k-NN graph construction
            max_particles: Maximum number of particles per jet
            force_regenerate: Force regeneration of synthetic data
        """
        self.k_neighbors = k_neighbors
        self.max_particles = max_particles
        self.force_regenerate = force_regenerate
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if os.path.exists(self.processed_paths[0]) and not force_regenerate:
            self.load(self.processed_paths[0])
        else:
            self.process()
            self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        """List of raw file names"""
        return ['quark_gluon_dataset.h5']
    
    @property
    def processed_file_names(self):
        """List of processed file names"""
        return ['data.pt']
    
    def download(self):
        """Download the Quark-Gluon dataset from remote source"""
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        
        if os.path.exists(raw_path) and not self.force_regenerate:
            print(f"Dataset already exists at {raw_path}")
            return
        
        os.makedirs(self.raw_dir, exist_ok=True)
        
        print("=" * 60)
        print("DOWNLOADING/GENERATING DATASET")
        print("=" * 60)
        
        # Generate synthetic data
        print("\nGenerating synthetic dataset for testing...")
        self._generate_synthetic_data(raw_path)
    
    def _generate_synthetic_data(self, path, n_jets=10000):
        """
        Generate synthetic jet data for testing.
        
        Creates realistic-looking jet data with:
        - pT: Exponential distribution (typical for jets)
        - eta: Uniform in [-2.5, 2.5] (detector acceptance)
        - phi: Uniform in [-π, π] (azimuthal angle)
        - Particle ID: Integer representing particle type
        
        Quark jets tend to be narrower (fewer particles, more collimated)
        Gluon jets tend to be wider (more particles, broader distribution)
        
        Args:
            path: Path to save the HDF5 file
            n_jets: Number of jets to generate
        """
        print(f"Generating {n_jets} synthetic jets...")
        
        n_quarks = 0
        n_gluons = 0
        
        with h5py.File(path, 'w') as f:
            # Create datasets
            X = f.create_dataset('X', shape=(n_jets, self.max_particles, 4), dtype='float32')
            y = f.create_dataset('y', shape=(n_jets,), dtype='int32')
            
            # Add metadata
            f.attrs['description'] = 'Synthetic Quark-Gluon Jet Dataset'
            f.attrs['features'] = ['pT (GeV)', 'eta', 'phi', 'particle_id']
            f.attrs['labels'] = ['0=gluon', '1=quark']
            
            for i in tqdm(range(n_jets), desc="Generating jets"):
                # Determine if quark (1) or gluon (0)
                # Alternate for balanced dataset
                is_quark = i % 2
                y[i] = is_quark
                
                if is_quark:
                    n_quarks += 1
                else:
                    n_gluons += 1
                
                # Number of particles
                # Quark jets: fewer particles (10-20)
                # Gluon jets: more particles (15-30)
                if is_quark:
                    n_particles = np.random.randint(8, min(20, self.max_particles))
                else:
                    n_particles = np.random.randint(12, min(28, self.max_particles))
                
                # Jet axis (center of jet)
                jet_eta = np.random.uniform(-1.5, 1.5)
                jet_phi = np.random.uniform(-np.pi, np.pi)
                
                # Generate particles
                for j in range(n_particles):
                    # pT: Exponential distribution, higher for quarks on average
                    pt_scale = 8 if is_quark else 6
                    X[i, j, 0] = np.random.exponential(pt_scale) + 1
                    
                    # eta: Gaussian around jet axis
                    # Quark jets are narrower (smaller spread)
                    eta_spread = 0.15 if is_quark else 0.25
                    X[i, j, 1] = jet_eta + np.random.normal(0, eta_spread)
                    
                    # phi: Gaussian around jet axis
                    phi_spread = 0.12 if is_quark else 0.22
                    X[i, j, 2] = jet_phi + np.random.normal(0, phi_spread)
                    
                    # Wrap phi to [-π, π]
                    X[i, j, 2] = np.arctan2(np.sin(X[i, j, 2]), np.cos(X[i, j, 2]))
                    
                    # Particle ID: Random integer representing particle type
                    X[i, j, 3] = np.random.choice([1, 2, 3, 4, 5, 11, 13, 22], 
                                                   p=[0.3, 0.25, 0.2, 0.1, 0.05, 0.03, 0.02, 0.05])
                
                # Zero-pad remaining particles
                for j in range(n_particles, self.max_particles):
                    X[i, j, :] = 0
        
        print(f"Synthetic data saved to {path}")
        print(f"  - Quark jets: {n_quarks} (50%)")
        print(f"  - Gluon jets: {n_gluons} (50%)")
    
    def process(self):
        """Process raw data into PyG Data objects"""
        print("=" * 60)
        print("PROCESSING DATASET")
        print("=" * 60)
        
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        
        # Download or generate if not exists
        if not os.path.exists(raw_path):
            self.download()
        
        data_list = []
        
        # Load HDF5 file
        print(f"Loading from: {raw_path}")
        
        # Load all data into memory first (fixes HDF5 read issues)
        with h5py.File(raw_path, 'r') as f:
            print(f"Dataset keys: {list(f.keys())}")
            X = np.array(f['X'])
            y = np.array(f['y'])
        
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        print(f"Quark jets: {np.sum(y)} ({100*np.mean(y):.1f}%)")
        print(f"Gluon jets: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")
        
        # Process each jet
        skipped = 0
        for i in tqdm(range(len(X)), desc="Processing jets"):
            jet_features = X[i]
            jet_label = int(y[i])
            
            # Remove zero-padded particles (pT > 0)
            mask = jet_features[:, 0] > 0
            valid_features = jet_features[mask]
            
            # Skip jets with too few particles
            if len(valid_features) < 3:
                skipped += 1
                continue
            
            # Convert to tensor
            x = torch.tensor(valid_features, dtype=torch.float)
            
            # Create edge index using k-NN
            edge_index = self._create_knn_graph(valid_features)
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(jet_label, dtype=torch.long)
            )
            
            data_list.append(data)
        
        print(f"Processed {len(data_list)} jets (skipped {skipped} with <3 particles)")
        
        # Apply transforms if specified
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            print(f"After pre-filter: {len(data_list)} jets")
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Save processed data
        os.makedirs(self.processed_dir, exist_ok=True)
        self.save(data_list, self.processed_paths[0])
        print(f"Saved to: {self.processed_paths[0]}")
    
    def _create_knn_graph(self, features):
        """
        Create k-NN graph based on particle positions (eta, phi).
        
        Uses spatial distance in (eta, phi) space to connect particles.
        This preserves the geometric structure of the jet.
        
        Args:
            features: numpy array of shape (n_particles, n_features)
        
        Returns:
            edge_index: torch tensor of shape (2, n_edges)
        """
        n_particles = len(features)
        k = min(self.k_neighbors + 1, n_particles)  # +1 includes self
        
        # Get (eta, phi) positions (indices 1 and 2)
        if features.shape[1] >= 3:
            pos = features[:, 1:3]  # eta, phi
        else:
            pos = features[:, :2]
        
        # Build k-NN graph using sklearn
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(pos)
        distances, indices = nbrs.kneighbors(pos)
        
        # Create edge index (bidirectional)
        source = []
        target = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self-connection
                source.append(i)
                target.append(j)
                # Add reverse edge for undirected graph
                source.append(j)
                target.append(i)
        
        edge_index = torch.tensor([source, target], dtype=torch.long)
        return edge_index
    
    def get_num_classes(self):
        """Return number of classes"""
        return 2
    
    def get_num_features(self):
        """Return number of node features"""
        return 4


def get_dataloaders(root='./data', 
                    batch_size=32, 
                    train_ratio=0.8,
                    k_neighbors=8,
                    max_particles=30,
                    num_workers=0,
                    force_regenerate=False):
    """
    Create train and test dataloaders for jet classification.
    
    Args:
        root: Data directory
        batch_size: Batch size for training
        train_ratio: Fraction of data for training (rest for testing)
        k_neighbors: Number of neighbors for graph construction
        max_particles: Maximum particles per jet
        num_workers: Number of data loader workers
        force_regenerate: Force regeneration of data
    
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        dataset: Full dataset (for analysis)
    """
    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)
    
    # Load full dataset
    dataset = QuarkGluonDataset(
        root=root,
        k_neighbors=k_neighbors,
        max_particles=max_particles,
        force_regenerate=force_regenerate
    )
    
    # Split into train/test
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    # Use random split
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {n_total}")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, dataset


def analyze_dataset(dataset):
    """
    Analyze dataset statistics.
    
    Args:
        dataset: QuarkGluonDataset instance
    """
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    n_quarks = 0
    n_gluons = 0
    n_particles_list = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        if data.y.item() == 1:
            n_quarks += 1
        else:
            n_gluons += 1
        n_particles_list.append(data.x.shape[0])
    
    print(f"Total jets: {len(dataset)}")
    print(f"Quark jets: {n_quarks} ({100*n_quarks/len(dataset):.1f}%)")
    print(f"Gluon jets: {n_gluons} ({100*n_gluons/len(dataset):.1f}%)")
    print(f"Average particles per jet: {np.mean(n_particles_list):.1f}")
    print(f"Min particles: {min(n_particles_list)}")
    print(f"Max particles: {max(n_particles_list)}")


if __name__ == "__main__":
    # Test the dataset
    print("Testing QuarkGluonDataset...")
    
    dataset = QuarkGluonDataset(root='./data')
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample data: {dataset[0]}")
    print(f"Node features shape: {dataset[0].x.shape}")
    print(f"Edge index shape: {dataset[0].edge_index.shape}")
    print(f"Label: {dataset[0].y}")
    
    analyze_dataset(dataset)
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, test_loader, _ = get_dataloaders(root='./data')
    
    # Get one batch
    for batch in train_loader:
        print(f"\nBatch shape: {batch}")
        print(f"Number of graphs: {batch.num_graphs}")
        break
    
    print("\n✅ All tests passed!")
