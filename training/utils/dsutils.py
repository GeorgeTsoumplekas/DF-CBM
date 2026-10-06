import os
import numpy as np
import cv2
from itertools import chain
from utils.ioutils import load_json


def read_rgb_uint8_square(image_path: str, size: int) -> np.ndarray:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return ensure_rgb_uint8_square(image_rgb, size)


def ensure_rgb_uint8_square(image_rgb: np.ndarray, size: int) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    if image_rgb.shape[0] != size or image_rgb.shape[1] != size:
        image_rgb = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    return image_rgb


def get_ff_filepath_video_id(filepath):
    return os.path.split(os.path.dirname(filepath))[-1][:3]


def video_id_from_frame_path(frame_path: str) -> str:
    """Stable video key: parent directory of the frame file."""
    normalized = os.path.normpath(str(frame_path).replace("\\", "/"))
    return os.path.dirname(normalized)


def get_ff_split_ids(split_json_file):
    data = load_json(split_json_file)
    unique_ids = set(chain.from_iterable(data))
    return unique_ids


def split_filter(data_rootpath, filepaths, splits):
    split_json_files = [
        os.path.join(data_rootpath, f"{split}.json") for split in splits
    ]
    valid_ids = set()
    for json_file in split_json_files:
        valid_ids.update(get_ff_split_ids(json_file))

    valid_filepaths = [
        fp for fp in filepaths if get_ff_filepath_video_id(fp) in valid_ids
    ]
    return valid_filepaths
