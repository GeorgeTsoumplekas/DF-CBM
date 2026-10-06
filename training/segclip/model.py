import os
import torch
from network import get_model
from transformers import CLIPModel
from typing import Dict, Any


def load_segclip_models(config: Dict[str, Any], device: torch.device):

    segface_backbone = config.get("segface_backbone", "segface_celeb")
    segface_encoder = config.get("segface_encoder", "swin_base")
    segface_resolution = int(
        config.get("segface_input_resolution", config["resolution"])
    )
    segface_model_path = config["segface_model_path"]
    if not os.path.isabs(segface_model_path):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        normalized = segface_model_path.replace("\\", "/").lstrip("./")
        while normalized.startswith("../"):
            normalized = normalized[3:]
        segface_model_path = os.path.normpath(os.path.join(repo_root, normalized))

    segface = get_model(segface_backbone, segface_resolution, segface_encoder).to(
        device
    )
    segface.eval()
    checkpoint = torch.load(segface_model_path, map_location=device)
    if "state_dict_backbone" not in checkpoint:
        raise ValueError(
            "Expected segface checkpoint to contain key 'state_dict_backbone' "
            f"(got keys: {list(checkpoint.keys())[:10]})"
        )
    segface.load_state_dict(checkpoint["state_dict_backbone"])

    clip_model_id = config.get("clip_model", "openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    return segface, clip_model
