_MODEL_REGISTRY = {}
_MODELS_REGISTERED = False


def _ensure_models_registered() -> None:
    global _MODELS_REGISTERED
    if _MODELS_REGISTERED:
        return
    from models_ import classifier  # noqa: F401 — registers model classes

    _MODELS_REGISTERED = True


def register_model(name: str):
    """Decorator to automatically add a class to the model registry."""

    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model name '{name}' is already registered by {_MODEL_REGISTRY[name].__name__}"
            )
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str):
    """Safely retrieves a model class or throws a clean error."""
    _ensure_models_registered()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. Registered models are: {list(_MODEL_REGISTRY.keys())}. Check config and change model_config.name"
        )
    return _MODEL_REGISTRY[name]
