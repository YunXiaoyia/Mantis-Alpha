"""Standalone LeRobot-format (v3.x) dataset reader for Mantis-Alpha.

Reads parquet data files (with inline JPEG/PNG image bytes) and meta files
produced by LeRobot's dataset converter. No LeRobot import is required.

Design notes:
- Tabular columns (state, action, episode/frame/task indices) are loaded once
  into RAM (~30 MB for 270k frames) for fast action-chunk assembly.
- Images are decoded lazily per sample from the parquet byte columns; a small
  per-worker LRU on row-group columns keeps repeated reads cheap.
"""

import io
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class _FileRef:
    path: str
    num_rows: int


class LeRobotDataset(Dataset):
    """Minimal LeRobot v3 dataset: images (inline bytes), state, action chunks, task strings."""

    def __init__(
        self,
        root: str,
        chunk_size: int = 50,
        episodes: list[int] | None = None,
        video_backend_check: bool = False,
        row_group_cache_bytes: int = 268_435_456,
    ):
        self.root = root
        self.chunk_size = chunk_size
        self.episodes = None if episodes is None else [int(e) for e in episodes]
        self._row_group_cache_bytes = row_group_cache_bytes

        with open(os.path.join(root, "meta", "info.json")) as f:
            self.info = json.load(f)

        features = self.info["features"]
        self.image_keys = sorted(
            k for k, v in features.items() if v.get("dtype") == "image"
        )
        if not self.image_keys:
            raise ValueError(f"No image features found in {self.info['features']}")
        self.state_key = next(
            (k for k, v in features.items() if k.endswith(".state")), "observation.state"
        )
        self.action_key = "action"
        self.state_dim = features[self.state_key]["shape"][0]
        self.action_dim = features[self.action_key]["shape"][0]
        self.fps = float(self.info.get("fps", 10.0))

        # Task strings: tasks.parquet with either a `task` column or the strings as index.
        import pandas as pd

        tasks_df = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
        if "task" in tasks_df.columns:
            ordered = tasks_df.sort_values("task_index")["task"].tolist()
        else:
            ordered = [str(t) for t in tasks_df.index.tolist()]
            # `task_index` column aligns with the index position.
            if "task_index" in tasks_df.columns:
                idx = tasks_df["task_index"].tolist()
                ordered = [t for _, t in sorted(zip(idx, ordered))]
        self.tasks: list[str] = ordered

        # ── Scan data files and load the tabular table into RAM ──────────
        data_pattern = self.info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        self._files: list[_FileRef] = []
        self._data_dir = os.path.join(root, "data")
        for dirpath, _, filenames in sorted(os.walk(self._data_dir)):
            for name in sorted(filenames):
                if name.endswith(".parquet"):
                    self._files.append(_FileRef(path=os.path.join(dirpath, name), num_rows=0))
        if not self._files:
            raise FileNotFoundError(f"No parquet files under {self._data_dir}")

        import pandas as pd

        frames = []
        for file_id, ref in enumerate(self._files):
            df = pd.read_parquet(
                ref.path,
                columns=[self.state_key, self.action_key, "episode_index", "frame_index", "task_index"],
            )
            ref.num_rows = len(df)
            df["_file_id"] = file_id
            frames.append(df)
        table = pd.concat(frames, ignore_index=True)

        if len(table) != sum(r.num_rows for r in self._files):
            raise RuntimeError("Row count mismatch between table and file index")

        self._file_offsets = np.cumsum([0] + [r.num_rows for r in self._files])
        self.episode_index = table["episode_index"].to_numpy()
        self.frame_index = table["frame_index"].to_numpy()
        self.task_index = table["task_index"].to_numpy()
        self.state = np.stack(table[self.state_key].to_numpy()).astype(np.float32)
        self.action = np.stack(table[self.action_key].to_numpy()).astype(np.float32)
        self._file_id = table["_file_id"].to_numpy()
        self._row_in_file = np.concatenate(
            [np.arange(r.num_rows) for r in self._files]
        )

        if self.episodes is not None:
            keep = np.isin(self.episode_index, np.asarray(self.episodes))
            if not bool(keep.any()):
                raise ValueError(f"No frames found for episodes={self.episodes} in {root}")
            self.episode_index = self.episode_index[keep]
            self.frame_index = self.frame_index[keep]
            self.task_index = self.task_index[keep]
            self.state = self.state[keep]
            self.action = self.action[keep]
            self._file_id = self._file_id[keep]
            self._row_in_file = self._row_in_file[keep]

        # Group rows into episodes (rows of one episode are contiguous after sorting by index).
        order = np.lexsort((self.frame_index, self.episode_index))
        self._order = order
        uniq, starts = np.unique(self.episode_index[order], return_index=True)
        self._episode_starts = dict(zip(uniq.tolist(), starts.tolist()))
        ends = np.concatenate([starts[1:], [len(order)]])
        self._episode_len = dict(zip(uniq.tolist(), (ends - starts).tolist()))
        self.num_episodes = len(uniq)
        logger.info(
            "Loaded %d frames / %d episodes / %d files from %s (episodes=%s)",
            len(self.episode_index), self.num_episodes, len(self._files), root, self.episodes,
        )

        # Worker-local caches (lazily created so DataLoader fork workers get private copies).
        self._pf_handles: dict[int, pq.ParquetFile] = {}
        self._rg_cache: "OrderedDict[tuple, list]" = OrderedDict()
        self._rg_cache_bytes = 0

    # ── Basic protocol ───────────────────────────────────────────────────
    def __len__(self) -> int:
        return int(self._order.shape[0])

    @property
    def num_frames(self) -> int:
        return len(self)

    # ── Parquet image access ─────────────────────────────────────────────
    def _parquet_file(self, file_id: int) -> pq.ParquetFile:
        pf = self._pf_handles.get(file_id)
        if pf is None:
            pf = pq.ParquetFile(self._files[file_id].path)
            self._pf_handles[file_id] = pf
        return pf

    def _image_bytes_column(self, file_id: int, row_group: int, key: str) -> list:
        cache_key = (file_id, row_group, key)
        cached = self._rg_cache.get(cache_key)
        if cached is not None:
            self._rg_cache.move_to_end(cache_key)
            return cached
        pf = self._parquet_file(file_id)
        column = pf.read_row_group(row_group, columns=[key]).column(key).to_pylist()
        nbytes = sum(len(v["bytes"]) for v in column if v is not None and v.get("bytes"))
        self._rg_cache[cache_key] = column
        self._rg_cache_bytes += nbytes
        while self._rg_cache_bytes > self._row_group_cache_bytes and len(self._rg_cache) > 1:
            _, evicted = self._rg_cache.popitem(last=False)
            self._rg_cache_bytes -= sum(
                len(v["bytes"]) for v in evicted if v is not None and v.get("bytes")
            )
        return column

    def _decode_image(self, file_id: int, row: int, key: str) -> torch.Tensor:
        pf = self._parquet_file(file_id)
        # Locate the row group holding this row (files here have 1 row group; general fallback).
        row_group = 0
        cum = 0
        for rg in range(pf.metadata.num_row_groups):
            rows = pf.metadata.row_group(rg).num_rows
            if cum <= row < cum + rows:
                row_group = rg
                row_in_group = row - cum
                break
            cum += rows
        else:
            raise IndexError(f"row {row} out of range for file {file_id}")
        column = self._image_bytes_column(file_id, row_group, key)
        record = column[row_in_group]
        img = Image.open(io.BytesIO(record["bytes"])).convert("RGB")
        arr = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)  # CHW uint8
        return arr
    # ── Items ────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        row = self._order[idx]
        episode = int(self.episode_index[row])
        start = self._episode_starts[episode]
        ep_len = self._episode_len[episode]
        pos = idx - start  # position within the episode (idx and start share the sorted order)

        # Action chunk with end-of-episode padding.
        horizon = min(self.chunk_size, ep_len - pos)
        chunk_rows = self._order[start + pos : start + pos + horizon]
        action_chunk = self.action[chunk_rows]
        if horizon < self.chunk_size:
            pad = np.repeat(action_chunk[-1:], self.chunk_size - horizon, axis=0)
            action_chunk = np.concatenate([action_chunk, pad], axis=0)
        action_is_pad = np.zeros(self.chunk_size, dtype=bool)
        action_is_pad[horizon:] = True

        file_id = int(self._file_id[row])
        row_in_file = int(self._row_in_file[row])
        # Images stay uint8 (cheap IPC through DataLoader workers); the trainer
        # converts to float in [0, 1] on the main process.
        images = {key: self._decode_image(file_id, row_in_file, key) for key in self.image_keys}

        return {
            **images,
            self.state_key: torch.from_numpy(self.state[row]),
            self.action_key: torch.from_numpy(action_chunk.astype(np.float32)),
            "action_is_pad": torch.from_numpy(action_is_pad),
            "task": self.tasks[int(self.task_index[row])] if self.task_index[row] < len(self.tasks) else "",
            "episode_index": episode,
            "frame_index": int(self.frame_index[row]),
        }
