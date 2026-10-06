import numpy as np
import cv2
import torch
import torch.nn.functional as F


CELEBAMASK_LABELS = [
    "background",
    "neck",
    "skin",
    "cloth",
    "l_ear",
    "r_ear",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "nose",
    "mouth",
    "l_lip",
    "u_lip",
    "hair",
    "eye_g",
    "hat",
    "ear_r",
    "neck_l",
]

MERGE_TO_BACKGROUND_NAMES = ("cloth", "hat", "ear_r", "neck_l")
MERGE_TO_BACKGROUND_INDICES = tuple(
    CELEBAMASK_LABELS.index(name) for name in MERGE_TO_BACKGROUND_NAMES
)

MERGE_TO_SKIN_NAMES = ("eye_g",)
MERGE_TO_SKIN_INDICES = tuple(
    CELEBAMASK_LABELS.index(name) for name in MERGE_TO_SKIN_NAMES
)
SKIN_CLASS_INDEX = CELEBAMASK_LABELS.index("skin")

_CLASS = {name: index for index, name in enumerate(CELEBAMASK_LABELS)}

BOUNDARY_DEFINITIONS = [
    ("Background - Neck", _CLASS["background"], _CLASS["neck"]),
    ("Background - Skin", _CLASS["background"], _CLASS["skin"]),
    ("Background - Left Ear", _CLASS["background"], _CLASS["l_ear"]),
    ("Background - Right Ear", _CLASS["background"], _CLASS["r_ear"]),
    ("Background - Hair", _CLASS["background"], _CLASS["hair"]),
    ("Neck - Skin", _CLASS["neck"], _CLASS["skin"]),
    ("Neck - Hair", _CLASS["neck"], _CLASS["hair"]),
    ("Skin - Left Ear", _CLASS["skin"], _CLASS["l_ear"]),
    ("Skin - Right Ear", _CLASS["skin"], _CLASS["r_ear"]),
    ("Skin - Left Eyebrow", _CLASS["skin"], _CLASS["l_brow"]),
    ("Skin - Right Eyebrow", _CLASS["skin"], _CLASS["r_brow"]),
    ("Skin - Left Eye", _CLASS["skin"], _CLASS["l_eye"]),
    ("Skin - Right Eye", _CLASS["skin"], _CLASS["r_eye"]),
    ("Skin - Nose", _CLASS["skin"], _CLASS["nose"]),
    ("Skin - Mouth", _CLASS["skin"], _CLASS["mouth"]),
    ("Skin - Lower Lip", _CLASS["skin"], _CLASS["l_lip"]),
    ("Skin - Upper Lip", _CLASS["skin"], _CLASS["u_lip"]),
    ("Skin - Hair", _CLASS["skin"], _CLASS["hair"]),
    ("Left Ear - Hair", _CLASS["l_ear"], _CLASS["hair"]),
    ("Right Ear - Hair", _CLASS["r_ear"], _CLASS["hair"]),
    ("Left Eyebrow - Hair", _CLASS["l_brow"], _CLASS["hair"]),
    ("Right Eyebrow - Hair", _CLASS["r_brow"], _CLASS["hair"]),
    ("Mouth - Lower Lip", _CLASS["mouth"], _CLASS["l_lip"]),
    ("Mouth - Upper Lip", _CLASS["mouth"], _CLASS["u_lip"]),
    ("Lower Lip - Upper Lip", _CLASS["l_lip"], _CLASS["u_lip"]),
]

_BOUNDARY_KERNEL = np.ones((3, 3), np.uint8)


def compute_directional_boundary(mask, class_a, class_b):
    neighbor_b = cv2.dilate(
        (mask == class_b).astype(np.uint8),
        _BOUNDARY_KERNEL,
    )
    return (mask == class_a) & (neighbor_b > 0)


def compute_pair_boundary(mask, class_a, class_b):
    return compute_directional_boundary(
        mask, class_a, class_b
    ) | compute_directional_boundary(mask, class_b, class_a)


def compute_boundary_masks(mask, definitions=BOUNDARY_DEFINITIONS):
    boundaries = {}
    for name, class_a, class_b in definitions:
        region = compute_pair_boundary(mask, class_a, class_b)
        if region.any():
            boundaries[name] = region
    return boundaries


@torch.no_grad()
def predict_segmentation(model, image_tensor, labels, dataset_tensor, input_resolution):
    device = image_tensor.device
    labels = {key: value.to(device) for key, value in labels.items()}
    dataset_tensor = dataset_tensor.to(device)
    seg_output = model(image_tensor, labels, dataset_tensor)
    seg_output = F.interpolate(
        seg_output,
        size=(input_resolution, input_resolution),
        mode="bilinear",
        align_corners=False,
    )
    return torch.argmax(seg_output.softmax(dim=1), dim=1)
