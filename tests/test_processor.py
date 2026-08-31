"""Processor normalization and SmolVLA prompt newline."""

from __future__ import annotations

import torch

from mantis_alpha.config import PolicyFeature, SmolVLAConfig
from mantis_alpha.processor import SmolVLABatchProcessor
from mantis_alpha.utils import OBS_LANGUAGE_TOKENS


class _FakeTokenizer:
    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        assert all(t.endswith("\n") for t in texts), texts
        ids = torch.ones(len(texts), min(4, max_length), dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


def test_tokenize_appends_newline() -> None:
    cfg = SmolVLAConfig()
    cfg.input_features["observation.images.image"] = PolicyFeature(type="VISUAL", shape=(3, 256, 256))
    cfg.input_features["observation.state"] = PolicyFeature(type="STATE", shape=(8,))
    cfg.output_features["action"] = PolicyFeature(type="ACTION", shape=(7,))
    proc = SmolVLABatchProcessor(cfg, _FakeTokenizer(), stats={})
    out = proc.tokenize(["pick up the cup", "pick up the cup\n"])
    assert out[OBS_LANGUAGE_TOKENS].shape[0] == 2


def test_mean_std_normalize_roundtrip() -> None:
    cfg = SmolVLAConfig()
    cfg.input_features["observation.state"] = PolicyFeature(type="STATE", shape=(2,))
    cfg.output_features["action"] = PolicyFeature(type="ACTION", shape=(2,))
    stats = {
        "observation.state": {"mean": [1.0, 3.0], "std": [2.0, 4.0]},
        "action": {"mean": [0.0, 1.0], "std": [0.5, 0.5]},
    }
    proc = SmolVLABatchProcessor(cfg, _FakeTokenizer(), stats=stats)
    state = torch.tensor([[1.0, 3.0], [3.0, 7.0]])
    norm = proc.normalize_state(state)
    assert torch.allclose(norm, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    action = torch.tensor([[[0.0, 1.0]]])
    assert torch.allclose(proc.unnormalize_action(proc.normalize_action(action)), action)
