"""Standalone utilities for Mantis-Alpha.

Constants, optional-package guard, observation queues and dtype helpers that the
SmolVLA implementation needs. Self-contained: nothing here imports LeRobot.
"""

import dataclasses
import importlib.util
from collections import deque
from typing import Any

import torch

# ── Feature-name constants (LeRobot dataset field conventions) ──────────────
OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"

ACTION = "action"
ACTION_PREFIX = ACTION + "."

# Additive mask value used to mask out attention entries (openpi convention).
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


# ── Optional package guard ──────────────────────────────────────────────────
def is_package_available(pkg_name: str, import_name: str | None = None) -> bool:
    return importlib.util.find_spec(import_name or pkg_name) is not None


_transformers_available = is_package_available("transformers")

_require_package_cache: dict[str, bool] = {}


def require_package(pkg_name: str, extra: str, import_name: str | None = None) -> None:
    """Raise an informative ImportError if a package required by an optional feature is missing."""
    cache_key = import_name or pkg_name
    if cache_key not in _require_package_cache:
        _require_package_cache[cache_key] = is_package_available(pkg_name, import_name)
    if not _require_package_cache[cache_key]:
        raise ImportError(
            f"'{pkg_name}' is required but not installed. Install it with: "
            f"pip install 'mantis-alpha[{extra}]'"
        )


# ── Inference observation queues ────────────────────────────────────────────
def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


# ── Dtype helpers ───────────────────────────────────────────────────────────
def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """mps is currently not compatible with float64."""
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def log_model_keys(module: torch.nn.Module, state_dict_keys: set[str]) -> tuple[list[str], list[str]]:
    """Return (missing, unexpected) key names for a relaxed checkpoint load."""
    model_keys = {k for k, _ in module.state_dict().items()}
    missing = sorted(model_keys - state_dict_keys)
    unexpected = sorted(state_dict_keys - model_keys)
    return missing, unexpected


def get_item_from_nested(obj: Any, key: str, default: Any = None) -> Any:
    """Fetch a key from a dict, returning `default` when absent or when obj is None."""
    if obj is None:
        return default
    return obj.get(key, default) if isinstance(obj, dict) else default


def json_ready(value: Any) -> Any:
    """Recursively convert dataclasses / tuples / nested structures into JSON-compatible values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: json_ready(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
