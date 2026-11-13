"""Shared resource paths, gesture labels, and artifact format version."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURE_VERSION = "wrist_centered_scale_normalized_xyz_v1"


def gesture_names():
    data = json.loads((ROOT / "gesture_list.json").read_text())
    return [data[str(i)] for i in range(1, 6)]
