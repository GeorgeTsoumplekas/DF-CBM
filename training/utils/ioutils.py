import os
from typing import List, Union
import json


def get_filepaths(rootpath: str, exts: Union[List[str], str, None] = None) -> List[str]:

    if isinstance(exts, str):
        exts = [exts]

    if not os.path.exists(rootpath):
        raise ValueError(f"the given rootpath doesn't exist rootpath: {rootpath}")

    keepfiles = list()
    for root, _, files in os.walk(rootpath):
        for file in files:
            filepath = os.path.join(root, file)
            if not exts:
                keepfiles.append(filepath)
            else:
                if os.path.splitext(filepath)[1] in exts:
                    keepfiles.append(filepath)
    return keepfiles


def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
