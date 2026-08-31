"""Slim stand-alone replacement for LeRobot's `PreTrainedPolicy`.

Provides just what the SmolVLA policy needs: config ownership, checkpoint
save/load (config.json + model.safetensors), and the PEFT validation hook kept
as a no-op so the copied model code runs unchanged.
"""

import json
import logging
import os
from typing import TypeVar

import torch
from safetensors.torch import load_file, save_model
from torch import nn

from .utils import json_ready

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="PreTrainedPolicy")


class PreTrainedPolicy(nn.Module):
    """Base class for Mantis-Alpha policies."""

    config_class: type | None = None
    name: str | None = None

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _validate_peft_config(self, peft_config) -> None:
        """PEFT is not part of the stand-alone build; kept as a no-op hook."""

    # ── Checkpoint I/O ──────────────────────────────────────────────────
    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        cfg = json_ready(self.config)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        # save_model handles shared/tied tensors by cloning them.
        save_model(self, os.path.join(path, "model.safetensors"))
        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def from_pretrained(cls: type[T], path: str, strict: bool = False, **overrides) -> T:
        """Load a policy from a checkpoint directory containing config.json + model.safetensors.

        Compatible with checkpoint directories produced either by this package or by
        LeRobot training (same safetensors layout for SmolVLA). Unknown config fields
        are ignored; missing weights stay at their initialization.
        """
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "model.safetensors")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found in {path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"model.safetensors not found in {path}")

        with open(config_path) as f:
            raw = json.load(f)
        if cls.config_class is None:
            raise ValueError(f"{cls.__name__} does not define config_class")

        # Also tolerate LeRobot-style nested {"policy": {...}} train configs.
        if "policy" in raw and isinstance(raw["policy"], dict):
            raw = raw["policy"]

        config = cls.config_class.from_dict(raw, **overrides)
        model = cls(config)

        state_dict = load_file(weights_path)
        missing, unexpected = [], []
        result = model.load_state_dict(state_dict, strict=strict)
        if not strict:
            missing, unexpected = result
            if missing:
                logger.warning("Missing weights (initialized fresh): %d, e.g. %s", len(missing), missing[:5])
            if unexpected:
                logger.warning("Unexpected weights (ignored): %d, e.g. %s", len(unexpected), unexpected[:5])
        return model
