@echo off
REM setup.bat - Setup script for Task V (Windows)
REM
REM Usage: Double-click this file or run from command prompt

echo.
echo ============================================================
echo     ML4SCI QMLHEP Task V - Setup Script (Windows)
echo ============================================================
echo.

REM Check Python
echo [1/4] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)

REM Install PyTorch
echo.
echo [2/4] Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio

REM Install PyTorch Geometric
echo.
echo [3/4] Installing PyTorch Geometric...
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

REM Install other dependencies
echo.
echo [4/4] Installing other dependencies...
pip install numpy pandas matplotlib scikit-learn tqdm h5py pyyaml tensorboard seaborn

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; import torch_geometric; import numpy; import sklearn; print('All packages installed successfully!')"

echo.
echo ============================================================
echo     Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Run training:  python train.py --epochs 50
echo   2. Check results: dir outputs\
echo.
pause
