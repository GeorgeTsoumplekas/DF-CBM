from losses.registry import get_loss_criterion, register_loss
from losses import cbm_loss  # noqa: F401 — register concept losses

__all__ = ["get_loss_criterion", "register_loss"]
