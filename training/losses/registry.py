import torch.nn as nn

_LOSS_REGISTRY = dict()


def register_loss(name: str):
    """Decorator to automatically add a loss function to the loss registry."""

    def decorator(cls):
        if name in _LOSS_REGISTRY:
            raise ValueError(f"Loss function '{name}' is already registered.")
        _LOSS_REGISTRY[name] = cls
        return cls

    return decorator


def get_loss_criterion(name: str):
    """Safely retrieves a standard PyTorch or custom loss class object."""
    # 1. Check custom decorated losses first
    if name in _LOSS_REGISTRY:
        return _LOSS_REGISTRY[name]

    # 2. Check native torch.nn classes next
    if hasattr(nn, name):
        return getattr(nn, name)

    raise ValueError(f"Loss '{name}' not found. Options: {list(_LOSS_REGISTRY.keys())}")
