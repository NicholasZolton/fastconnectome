"""Safe, versioned artifact helpers without pickle payloads."""

import json
from pathlib import Path

import numpy as np


def read_npz_manifest(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if "manifest" not in archive:
            raise ValueError("Artifact has no manifest")
        raw: object = json.loads(str(archive["manifest"]))
    return object_dict(raw, "manifest")


def object_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be an object with string keys")
    return {str(key): item for key, item in value.items()}


def string_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def float_value(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
