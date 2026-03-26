#!/usr/bin/env bash
# setup.sh - Setup script for Task V
# 
# This script sets up the environment and installs all dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

echo ""
echo "============================================================"
echo "    ML4SCI QMLHEP Task V - Setup Script"
echo "============================================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo ""
    echo "[2/6] Creating conda environment..."
    
    # Check if environment already exists
    if conda env list | grep -q "^taskv "; then
        echo "Environment 'taskv' already exists. Activating..."
    else
        echo "Creating new environment 'taskv' with Python 3.10..."
        conda create -n taskv python=3.10 -y
    fi
    
    # Activate environment
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate taskv
    
else
    echo ""
    echo "[2/6] Conda not found. Using pip directly..."
    echo "Consider installing conda for better environment management."
fi

# Install PyTorch
echo ""
echo "[3/6] Installing PyTorch..."

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected. Installing CPU version of PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install PyTorch Geometric
echo ""
echo "[4/6] Installing PyTorch Geometric..."

# Get PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu')")
echo "PyTorch version: $TORCH_VERSION"
echo "CUDA version: $CUDA_VERSION"

pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Install other dependencies
echo ""
echo "[5/6] Installing other dependencies..."
pip install numpy pandas matplotlib scikit-learn tqdm h5py pyyaml tensorboard seaborn

# Verify installation
echo ""
echo "[6/6] Verifying installation..."

python << 'EOF'
import sys
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    import numpy as np
    import sklearn
    import matplotlib
    import h5py
    
    print("")
    print("=" * 60)
    print("INSTALLATION VERIFICATION")
    print("=" * 60)
    print(f"Python version:     {sys.version.split()[0]}")
    print(f"PyTorch version:    {torch.__version__}")
    print(f"PyG version:        {torch_geometric.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:       {torch.version.cuda}")
        print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"NumPy version:      {np.__version__}")
    print(f"scikit-learn:       {sklearn.__version__}")
    print(f"h5py version:       {h5py.__version__}")
    print("=" * 60)
    
    # Test basic operations
    x = torch.randn(10, 4)
    edge_index = torch.tensor([[0,1,2], [1,2,3]])
    data = Data(x=x, edge_index=edge_index)
    assert data.x.shape == (10, 4), "Data shape check failed"
    
    print("✅ All packages installed and working correctly!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error during verification: {e}")
    sys.exit(1)
EOF

echo ""
echo "============================================================"
echo "    Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment:    conda activate taskv"
echo "  2. Run training:            python train.py --epochs 50"
echo "  3. Check results:           ls outputs/"
echo ""
echo "For help:"
echo "  python train.py --help"
echo ""
