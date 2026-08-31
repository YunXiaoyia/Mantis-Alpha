"""Standalone preprocessing for Mantis-Alpha training and inference.

Replaces LeRobot's processor pipeline with a small, explicit module:
- tokenizes task strings with the SmolVLM2 tokenizer,
- mean/std-normalizes state and action using dataset stats,
- assembles the batch dict consumed by `SmolVLAPolicy.forward`, and
- unnormalizes predicted action chunks back to dataset units.
"""

import json
import os

import numpy as np
import torch

from .utils import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def load_dataset_stats(dataset_root: str) -> dict:
    with open(os.path.join(dataset_root, "meta", "stats.json")) as f:
        return json.load(f)


def save_dataset_stats(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f)


class SmolVLABatchProcessor:
    """Builds training / inference batches for SmolVLAPolicy."""

    def __init__(self, config, tokenizer, stats: dict | None = None):
        self.config = config
        self.tokenizer = tokenizer
        self.stats = stats or {}
        self.state_key = "observation.state"
        self.state_dim = config.state_feature.shape[0] if config.state_feature else 8
        self.action_dim = config.action_feature.shape[0] if config.action_feature else 7
        self._state_mean = torch.tensor(
            self.stats.get("observation.state", {}).get("mean", [0.0] * self.state_dim), dtype=torch.float32
        )
        self._state_std = torch.tensor(
            self.stats.get("observation.state", {}).get("std", [1.0] * self.state_dim), dtype=torch.float32
        )
        self._action_mean = torch.tensor(
            self.stats.get("action", {}).get("mean", [0.0] * self.action_dim), dtype=torch.float32
        )
        self._action_std = torch.tensor(
            self.stats.get("action", {}).get("std", [1.0] * self.action_dim), dtype=torch.float32
        )

    # ── Normalization ────────────────────────────────────────────────────
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self._state_mean.to(state.device)) / self._state_std.to(state.device)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._action_mean.to(action.device)) / self._action_std.to(action.device)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """action: (..., action_dim) in normalized units -> dataset units."""
        return action * self._action_std.to(action.device) + self._action_mean.to(action.device)

    # ── Tokenization ─────────────────────────────────────────────────────
    @staticmethod
    def _with_newline(task: str) -> str:
        """SmolVLA / PaliGemma tokenizers expect a trailing newline on the prompt."""
        return task if task.endswith("\n") else f"{task}\n"

    def tokenize(self, tasks: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [self._with_newline(t) for t in tasks],
            max_length=self.config.tokenizer_max_length,
            padding=self.config.pad_language_to,
            truncation=True,
            return_tensors="pt",
        )
        return {
            OBS_LANGUAGE_TOKENS: encoded["input_ids"],
            OBS_LANGUAGE_ATTENTION_MASK: encoded["attention_mask"].bool(),
        }

    # ── Batch assembly ───────────────────────────────────────────────────
    def train_batch(self, samples: list[dict], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Collate a list of raw dataset samples into a policy-ready batch.

        Images are stacked as uint8 (cheap through DataLoader worker IPC); call
        `to_policy_batch` (or convert manually) before feeding the policy.
        """
        batch: dict[str, torch.Tensor] = {}
        for key in self.config.image_features:
            batch[key] = torch.stack([s[key] for s in samples])  # [B, 3, H, W] uint8

        state = torch.stack([s[self.state_key] for s in samples]).float()
        batch[OBS_STATE] = self.normalize_state(state)

        action = torch.stack([s[ACTION] for s in samples]).float()
        batch[ACTION] = self.normalize_action(action)

        batch["action_is_pad"] = torch.stack([s["action_is_pad"] for s in samples]).bool()

        batch.update(self.tokenize([s["task"] for s in samples]))

        if device is not None:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return batch

    def to_policy_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert a collated batch (uint8 images) into policy-ready float tensors."""
        out = dict(batch)
        for key in self.config.image_features:
            if key in out and out[key].dtype == torch.uint8:
                out[key] = out[key].float() / 255.0
        return out

    def infer_batch(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        task: str | list[str],
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assemble an inference batch from raw (unnormalized) observations.

        images: mapping camera key -> tensor [B, 3, H, W] (uint8 or float in [0, 1]).
        state:  tensor [B, state_dim] raw units.
        """
        if isinstance(task, str):
            task = [task] * len(state)
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(np.asarray(state, dtype=np.float32))
        batch: dict[str, torch.Tensor] = {}
        for key in self.config.image_features:
            img = images[key]
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            batch[key] = img.float()
        batch[OBS_STATE] = self.normalize_state(state.float())
        batch.update(self.tokenize(task))
        if device is not None:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return batch
