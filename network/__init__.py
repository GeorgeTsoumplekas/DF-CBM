from .models import SegFaceCeleb


def get_model(backbone, input_resolution, model):
    if backbone == "segface_celeb":
        model = SegFaceCeleb(input_resolution, model)
    else:
        raise ValueError("Backbone not implemented")
    return model
