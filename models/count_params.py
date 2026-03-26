"""
models/count_params.py - Parameter counting utility
"""

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """
    Count all parameters (including non-trainable).
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_model_info(model):
    """
    Get detailed model information.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Model information dictionary
    """
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    
    return {
        'trainable_params': trainable,
        'total_params': total,
        'frozen_params': total - trainable,
        'size_mb': total * 4 / (1024 * 1024)  # Assuming float32
    }


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    
    # Test with a simple model
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    info = get_model_info(model)
    print(f"Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
