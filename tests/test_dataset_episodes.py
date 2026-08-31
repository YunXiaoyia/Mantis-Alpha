"""LIBERO / LeRobot-v3 dataset loader: episode slice and action chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

LIBERO = Path("/home/adminroot/Desktop/vla/datasets/libero")

pytestmark = pytest.mark.skipif(not (LIBERO / "meta" / "info.json").exists(), reason="LIBERO dataset not on this machine")


def test_episode_zero_length_and_chunk_padding() -> None:
    from mantis_alpha.dataset import LeRobotDataset

    ds = LeRobotDataset(str(LIBERO), chunk_size=50, episodes=[0])
    assert ds.num_episodes == 1
    assert len(ds) == 214
    assert set(int(x) for x in np.unique(ds.episode_index)) == {0}

    first = ds[0]
    assert first["action"].shape == (50, 7)
    assert bool(first["action_is_pad"][0]) is False
    assert "observation.images.image" in first
    assert first["observation.images.image"].shape[0] == 3
    assert first["task"]

    last = ds[len(ds) - 1]
    assert bool(last["action_is_pad"][0]) is False
    assert bool(last["action_is_pad"][-1]) is True
    assert int(last["action_is_pad"].sum()) == 49
