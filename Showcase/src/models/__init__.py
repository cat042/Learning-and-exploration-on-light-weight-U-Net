from .baseline_unet import UNet as BaselineUNet
from .basic_model import UNet as BasicModelUNet
from .premier_model import UNet as PremierModelUNet

MODEL_REGISTRY = {
    "baseline_unet": BaselineUNet,
    "basic_model": BasicModelUNet,
    "premier_model": PremierModelUNet,
}

MODEL_NAMES = tuple(MODEL_REGISTRY.keys())


def create_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError("Unknown model '{}'. Available models: {}".format(model_name, ", ".join(MODEL_NAMES)))
    return MODEL_REGISTRY[model_name](**kwargs)


__all__ = ["MODEL_NAMES", "MODEL_REGISTRY", "create_model"]
