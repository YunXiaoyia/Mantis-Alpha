"""Standalone LIBERO (robosuite hdf5) -> LeRobot v3 dataset converter.

Converts raw LIBERO demo hdf5 files into the LeRobot v3 layout consumed by
`mantis_alpha.dataset.LeRobotDataset` (parquet with inline JPEG images +
meta/info.json + meta/tasks.parquet + meta/stats.json). No LeRobot import.

Field mapping (verified against the original LIBERO distribution):
- observation.images.image  <- obs/agentview_rgb   (upscaled to --image_size)
- observation.images.image2 <- obs/eye_in_hand_rgb (upscaled to --image_size)
- observation.state (8)     <- obs/ee_states (6: xyz + euler) + obs/gripper_states (2)
- action (7)                <- actions (6 delta pose + 1 gripper)
- task string               <- hdf5 filename without the SCENE prefix and "_demo" suffix

Usage:
```bash
python scripts/convert_libero.py \
    --src /home/adminroot/Desktop/vla/data/libero/libero_10 \
    --dst /home/adminroot/Desktop/vla/datasets/libero_10
```
"""

import argparse
import io
import json
import os
import re
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

FPS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIBERO hdf5 demos to LeRobot v3 format")
    parser.add_argument("--src", type=str, required=True, help="Directory with LIBERO *_demo.hdf5 files")
    parser.add_argument("--dst", type=str, required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--image_size", type=int, default=256, help="Stored image resolution (square)")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--rows_per_file", type=int, default=1000)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--image_stats_samples", type=int, default=512,
                        help="Frames sampled for image channel stats (informational)")
    return parser.parse_args()


def task_name_from_filename(path: str) -> str:
    stem = os.path.basename(path)
    if stem.endswith("_demo.hdf5"):
        stem = stem[: -len("_demo.hdf5")]
    # Strip the leading scene prefix, e.g. "LIVING_ROOM_SCENE5_put_the_white_mug..."
    stem = re.sub(r"^(KITCHEN_SCENE\d+|LIVING_ROOM_SCENE\d+|STUDY_SCENE\d+)_", "", stem)
    return stem.replace("_", " ").strip()


def convert_demo(task: dict) -> dict:
    """Read one demo from its hdf5 file and return per-frame rows (JPEG bytes in RAM)."""
    image_size, quality = task["image_size"], task["jpeg_quality"]
    with h5py.File(task["path"], "r") as h:
        demo = h["data"][task["demo"]]
        ee = demo["obs/ee_states"][:]            # (T, 6) xyz + euler
        gripper = demo["obs/gripper_states"][:]  # (T, 2)
        states = np.concatenate([ee, gripper], axis=1).astype(np.float32)
        actions = demo["actions"][:].astype(np.float32)  # (T, 7)
        agentview = demo["obs/agentview_rgb"]    # (T, h, w, 3) uint8
        eye = demo["obs/eye_in_hand_rgb"]

        jpeg_main, jpeg_wrist = [], []
        for t in range(states.shape[0]):
            main = Image.fromarray(agentview[t]).resize((image_size, image_size), Image.BILINEAR)
            wrist = Image.fromarray(eye[t]).resize((image_size, image_size), Image.BILINEAR)
            buf_main = io.BytesIO()
            main.convert("RGB").save(buf_main, format="JPEG", quality=quality)
            buf_wrist = io.BytesIO()
            wrist.convert("RGB").save(buf_wrist, format="JPEG", quality=quality)
            jpeg_main.append(buf_main.getvalue())
            jpeg_wrist.append(buf_wrist.getvalue())

    return {
        "task_index": task["task_index"],
        "states": states,
        "actions": actions,
        "jpeg_main": jpeg_main,
        "jpeg_wrist": jpeg_wrist,
    }


def image_stats(frames: list[tuple[str, str]], image_size: int) -> dict:
    """Per-channel mean/std over a sample of JPEG frame pairs (informational only)."""
    sums = np.zeros(6, dtype=np.float64)  # [main rgb sums, wrist rgb sums]
    sqs = np.zeros(6, dtype=np.float64)
    count = 0
    for main_b, wrist_b in frames:
        for i, b in enumerate((main_b, wrist_b)):
            arr = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float64) / 255.0
            sums[i * 3 : i * 3 + 3] += arr.reshape(-1, 3).sum(axis=0)
            sqs[i * 3 : i * 3 + 3] += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        count += 1
    if count == 0:
        return {}
    n_px = count * image_size * image_size
    mean = (sums / n_px).tolist()
    std = np.sqrt(np.maximum(sqs / n_px - np.array(mean) ** 2, 0)).tolist()
    return {
        "observation.images.image": {"mean": mean[0:3], "std": std[0:3], "count": [count]},
        "observation.images.image2": {"mean": mean[3:6], "std": std[3:6], "count": [count]},
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.dst, exist_ok=True)
    os.makedirs(os.path.join(args.dst, "meta"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "data", "chunk-000"), exist_ok=True)

    hdf5_files = sorted(f for f in os.listdir(args.src) if f.endswith(".hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No hdf5 files under {args.src}")
    tasks = [{"path": os.path.join(args.src, f), "task_index": i, "task": task_name_from_filename(f)}
             for i, f in enumerate(hdf5_files)]
    task_strings = [t["task"] for t in tasks]
    print(f"Found {len(tasks)} tasks:")
    for t in tasks:
        print(f"  {t['task_index']}: {t['task']}")

    # Expand to (task, demo) units in deterministic global order.
    units = []
    for t in tasks:
        with h5py.File(t["path"], "r") as h:
            demos = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
        for d in demos:
            units.append({**t, "demo": d, "image_size": args.image_size, "jpeg_quality": args.jpeg_quality})
    print(f"Total episodes: {len(units)}")

    img_struct = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    schema = pa.schema([
        ("observation.images.image", img_struct),
        ("observation.images.image2", img_struct),
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float64()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("task_index", pa.int64()),
        ("index", pa.int64()),
    ])

    global_index = 0
    episode_index = 0
    total_frames = 0
    file_idx = 0
    buffer = []
    stats_state, stats_action = [], []
    stats_frames = []

    def flush() -> None:
        nonlocal file_idx, buffer
        if not buffer:
            return
        path = os.path.join(args.dst, "data", "chunk-000", f"file-{file_idx:03d}.parquet")
        table = pa.Table.from_pylist(buffer, schema=schema)
        pq.write_table(table, path, compression="snappy")
        print(f"  wrote {path} ({len(buffer)} rows)")
        file_idx += 1
        buffer = []

    with Pool(args.num_procs) as pool:
        for result in pool.imap(convert_demo, units, chunksize=1):
            n = result["states"].shape[0]
            stats_state.append(result["states"])
            stats_action.append(result["actions"])
            if len(stats_frames) < args.image_stats_samples and episode_index % 17 == 0:
                for t in range(0, n, max(1, n // 3)):
                    if len(stats_frames) < args.image_stats_samples:
                        stats_frames.append((result["jpeg_main"][t], result["jpeg_wrist"][t]))
            for t in range(n):
                buffer.append({
                    "observation.images.image": {"bytes": result["jpeg_main"][t], "path": None},
                    "observation.images.image2": {"bytes": result["jpeg_wrist"][t], "path": None},
                    "observation.state": result["states"][t].tolist(),
                    "action": result["actions"][t].tolist(),
                    "timestamp": t / FPS,
                    "frame_index": t,
                    "episode_index": episode_index,
                    "task_index": result["task_index"],
                    "index": global_index,
                })
                global_index += 1
            episode_index += 1
            total_frames += n
            if len(buffer) >= args.rows_per_file:
                flush()
    flush()

    # ── Meta files ────────────────────────────────────────────────────────
    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.image": {
                "dtype": "image", "shape": [args.image_size, args.image_size, 3],
                "names": ["height", "width", "channel"], "fps": FPS,
            },
            "observation.images.image2": {
                "dtype": "image", "shape": [args.image_size, args.image_size, 3],
                "names": ["height", "width", "channel"], "fps": FPS,
            },
            "observation.state": {"dtype": "float32", "shape": [8], "names": ["state"], "fps": FPS},
            "action": {"dtype": "float32", "shape": [7], "names": ["actions"], "fps": FPS},
        },
    }
    with open(os.path.join(args.dst, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    pd.DataFrame({"task": task_strings, "task_index": range(len(task_strings))}).to_parquet(
        os.path.join(args.dst, "meta", "tasks.parquet"), index=False
    )

    state_arr = np.concatenate(stats_state, axis=0)
    action_arr = np.concatenate(stats_action, axis=0)
    stats = {
        "observation.state": {
            "min": state_arr.min(axis=0).tolist(), "max": state_arr.max(axis=0).tolist(),
            "mean": state_arr.mean(axis=0).tolist(), "std": state_arr.std(axis=0).tolist(),
            "count": [int(state_arr.shape[0])],
        },
        "action": {
            "min": action_arr.min(axis=0).tolist(), "max": action_arr.max(axis=0).tolist(),
            "mean": action_arr.mean(axis=0).tolist(), "std": action_arr.std(axis=0).tolist(),
            "count": [int(action_arr.shape[0])],
        },
    }
    stats.update(image_stats(stats_frames, args.image_size))
    with open(os.path.join(args.dst, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Done: {episode_index} episodes, {total_frames} frames -> {args.dst}")


if __name__ == "__main__":
    main()
