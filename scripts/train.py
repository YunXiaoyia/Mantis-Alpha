"""Mantis-Alpha standalone training script.

Trains the vendored SmolVLA policy on a LeRobot-format dataset (e.g. LIBERO)
without importing LeRobot. Runs anywhere a conda env with torch/transformers
is available:

```bash
python scripts/train.py \
    --dataset_root /home/adminroot/Desktop/vla/datasets/libero \
    --vlm_model_name /home/adminroot/Desktop/vla/models/smolvlm2 \
    --batch_size 8 --steps 20
```

Every run appends to `<output_dir>/train.log`: the full command line, a
pretty-printed dump of all parameters (dataset / policy / optimizer /
scheduler) and per-step progress lines.
"""

import argparse
import json
import logging
import math
import os
import pprint
import random
import sys
import time

import numpy as np
import torch

logger = logging.getLogger("mantis_train")

LOG_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# Active progress bar (console rendering only; never written to the log file).
_BAR = None


class ConsoleLogHandler(logging.Handler):
    """Console handler that renders above the active progress bar.

    Log records carrying ``extra={"console": False}`` (per-step metrics) are
    skipped here; they go to the log file only, so the console stays a clean
    progress bar.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if not getattr(record, "console", True):
            return
        msg = self.format(record)
        if _BAR is not None:
            _BAR.write(msg)
        else:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()


class _FallbackBar:
    """Minimal tqdm-style progress bar used only when tqdm is unavailable."""

    _BLOCKS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]

    def __init__(self, total: int, desc: str = "Training", unit: str = "step"):
        self.total, self.desc, self.unit = total, desc, unit
        self.n = 0
        self.t0 = time.time()
        self._line = ""

    def update(self, n: int = 1) -> None:
        self.n += n
        self._draw()

    def write(self, msg: str) -> None:
        sys.stdout.write("\r" + " " * len(self._line) + "\r" + msg + "\n")
        sys.stdout.flush()
        self._draw()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _draw(self) -> None:
        elapsed = max(time.time() - self.t0, 1e-6)
        rate = self.n / elapsed
        frac = min(self.n / self.total, 1.0)
        width = 26
        filled = frac * width
        full = int(filled)
        bar = "█" * full
        if full < width:
            bar += self._BLOCKS[min(int((filled - full) * 8), 8)]
            bar = bar.ljust(width)
        if self.n > 0 and rate < 1:
            rate_str = f"{1 / rate:5.2f}s/{self.unit}"
        else:
            rate_str = f"{rate:5.2f}{self.unit}/s"
        line = (
            f"{self.desc}: {frac:3.0%}|{bar}| {self.n}/{self.total} "
            f"[{_fmt_duration(elapsed)}<{_fmt_duration((self.total - self.n) / rate if rate > 0 else 0)}, {rate_str}]"
        )
        sys.stdout.write("\r" + line.ljust(len(self._line)))
        sys.stdout.flush()
        self._line = line


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def make_progress_bar(total: int, desc: str = "Training", unit: str = "step"):
    """tqdm-style progress bar; falls back to a hand-rolled bar without tqdm."""
    global _BAR
    if tqdm is not None:
        _BAR = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
    else:
        _BAR = _FallbackBar(total, desc=desc, unit=unit)
    return _BAR


def setup_logging(output_dir: str, level: int = logging.INFO) -> str:
    """Console + file logging (LeRobot style). Appends across runs sharing an output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = ConsoleLogHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mantis-Alpha training (standalone SmolVLA)")
    parser.add_argument("--dataset_root", type=str, default="/home/adminroot/Desktop/vla/datasets/libero")
    parser.add_argument(
        "--vlm_model_name", type=str, default="/home/adminroot/Desktop/vla/models/smolvlm2",
        help="Local SmolVLM2 path or HuggingFace repo id",
    )
    parser.add_argument("--output_dir", type=str, default="/home/adminroot/Desktop/vla/outputs/mantis_alpha")
    parser.add_argument("--policy_path", type=str, default=None,
                        help="Optional checkpoint dir (config.json + model.safetensors) to warm-start from")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=4_000)
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--scheduler_warmup_steps", type=int, default=500)
    parser.add_argument("--scheduler_decay_steps", type=int, default=None,
                        help="Cosine decay horizon (default: --steps)")
    parser.add_argument("--scheduler_decay_lr", type=float, default=2.5e-6)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices to train on, e.g. 0 or 0,1,2 (default: all)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--add_image_special_tokens", type=lambda s: str(s).lower() in ("1", "true", "yes"),
                        default=False, help="Image start/end tokens (paper/lerobot default: false)")
    parser.add_argument("--pad_language_to", type=str, default="longest", choices=["max_length", "longest"],
                        help="Language padding (paper/lerobot default: longest)")
    parser.add_argument("--no_vlm_weights", action="store_true",
                        help="Do not load SmolVLM2 pretrained weights (train expert from scratch)")
    parser.add_argument("--train_vlm", action="store_true",
                        help="Unfreeze the VLM language layers during training")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(args) -> callable:
    """Linear warmup to peak lr, then cosine decay to scheduler_decay_lr (LeRobot preset)."""
    warmup = args.scheduler_warmup_steps
    decay_steps = args.scheduler_decay_steps or args.steps
    decay_lr = args.scheduler_decay_lr
    peak = 1.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        if step >= decay_steps:
            return decay_lr / peak if peak > 0 else 1.0
        progress = (step - warmup) / (decay_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return (decay_lr + (peak - decay_lr) * cosine) / peak

    return lr_lambda


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    log_path = setup_logging(args.output_dir)
    command = f"{sys.executable} {' '.join(sys.argv)}"
    logger.info(f"Command: {command}")

    from mantis_alpha.config import PolicyFeature, SmolVLAConfig
    from mantis_alpha.dataset import LeRobotDataset
    from mantis_alpha.modeling import SmolVLAPolicy
    from mantis_alpha.processor import SmolVLABatchProcessor, load_dataset_stats, save_dataset_stats
    from mantis_alpha.utils import json_ready

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    episodes = None
    if args.episodes:
        episodes = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = LeRobotDataset(args.dataset_root, chunk_size=args.chunk_size, episodes=episodes)
    stats = load_dataset_stats(args.dataset_root)
    logger.info(
        f"Dataset: {len(dataset)} frames, {dataset.num_episodes} episodes, "
        f"{len(dataset.image_keys)} cameras {dataset.image_keys}, "
        f"state_dim={dataset.state_dim}, action_dim={dataset.action_dim}, tasks={len(dataset.tasks)}"
    )

    # ── Config / features ────────────────────────────────────────────────
    img_h, img_w = dataset.info["features"][dataset.image_keys[0]]["shape"][:2]
    config = SmolVLAConfig(
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=not args.no_vlm_weights,
        freeze_vision_encoder=True,
        train_expert_only=not args.train_vlm,
        resize_imgs_with_padding=(args.image_size, args.image_size),
        add_image_special_tokens=args.add_image_special_tokens,
        pad_language_to=args.pad_language_to,
        optimizer_lr=args.lr,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_decay_steps=args.scheduler_decay_steps or args.steps,
        scheduler_decay_lr=args.scheduler_decay_lr,
        device=str(device),
    )
    for key in dataset.image_keys:
        config.input_features[key] = PolicyFeature(type="VISUAL", shape=(3, img_h, img_w))
    config.input_features["observation.state"] = PolicyFeature(type="STATE", shape=(dataset.state_dim,))
    config.output_features["action"] = PolicyFeature(type="ACTION", shape=(dataset.action_dim,))
    config.validate_features()

    # ── Full parameter dump (LeRobot style) ──────────────────────────────
    all_params = {
        "command": command,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "num_workers": args.num_workers,
        "prefetch_factor": 2 if args.num_workers > 0 else None,
        "persistent_workers": args.num_workers > 0,
        "dataset": {
            "root": args.dataset_root,
            "repo_id": f"local/{os.path.basename(args.dataset_root.rstrip('/'))}",
            "frames": len(dataset),
            "episodes": dataset.num_episodes,
            "fps": dataset.fps,
            "image_keys": dataset.image_keys,
            "image_shape": [img_h, img_w],
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "num_tasks": len(dataset.tasks),
            "chunk_size": args.chunk_size,
            "selected_episodes": episodes,
        },
        "policy": json_ready(config),
        "policy_path": args.policy_path,
        "optimizer": {
            "type": "adamw",
            "lr": args.lr,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-10,
            "grad_clip_norm": 10.0,
        },
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "peak_lr": args.lr,
            "decay_lr": args.scheduler_decay_lr,
            "num_warmup_steps": args.scheduler_warmup_steps,
            "num_decay_steps": args.scheduler_decay_steps or args.steps,
        },
        "output_dir": args.output_dir,
        "log_file": log_path,
        "log_freq": args.log_freq,
        "save_freq": args.save_freq,
        "save_checkpoint": True,
        "resume": False,
    }
    logger.info("Training configuration:\n" + pprint.pformat(all_params, width=100, sort_dicts=True))

    # ── Policy ───────────────────────────────────────────────────────────
    logger.info(f"Building SmolVLAPolicy (vlm={args.vlm_model_name}, load_vlm_weights={not args.no_vlm_weights})...")
    policy = SmolVLAPolicy(config)
    if args.policy_path:
        from safetensors.torch import load_file

        state = load_file(os.path.join(args.policy_path, "model.safetensors"))
        missing, unexpected = policy.load_state_dict(state, strict=False)
        logger.info(f"Warm-started from {args.policy_path} (missing={len(missing)}, unexpected={len(unexpected)})")
    policy.to(device)
    policy.train()

    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_trainable / 1e6:.1f}M in {len(trainable)} tensors")

    # ── Tokenizer / processor ────────────────────────────────────────────
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(args.vlm_model_name).tokenizer
    processor = SmolVLABatchProcessor(config, tokenizer, stats)

    # ── Data loading ─────────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=lambda samples: processor.train_batch(samples),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # ── Optimizer / schedule ─────────────────────────────────────────────
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(args))

    # ── Train loop ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_dataset_stats(stats, os.path.join(args.output_dir, "stats.json"))
    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump({"command": command, **vars(args)}, f, indent=2)
    with open(os.path.join(args.output_dir, "train_params.json"), "w") as f:
        json.dump(all_params, f, indent=2)

    step = 0
    t0 = time.time()
    running_loss = 0.0
    done = False
    bar = make_progress_bar(args.steps)
    while not done:
        for batch in loader:
            if step >= args.steps:
                done = True
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch = processor.to_policy_batch(batch)
            loss, loss_dict = policy(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            scheduler.step()
            loss_val = loss.detach().item()
            running_loss += loss_val
            step += 1
            bar.update(1)

            if step % args.log_freq == 0:
                fps = step / (time.time() - t0)
                # Per-step metrics go to train.log only; the console shows the progress bar.
                logger.info(
                    f"step {step}/{args.steps} | loss {loss_val:.4f} (avg {running_loss / step:.4f}) "
                    f"| after_forward {loss_dict['losses_after_forward']:.4f} "
                    f"| grad_norm {float(grad_norm):.2f} | lr {scheduler.get_last_lr()[0]:.2e} "
                    f"| {fps:.2f} steps/s",
                    extra={"console": False},
                )

            if step % args.save_freq == 0 or step >= args.steps:
                ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"{step:06d}")
                policy.save_pretrained(ckpt_dir)
                save_dataset_stats(stats, os.path.join(ckpt_dir, "stats.json"))
                logger.info(f"Saved checkpoint to {ckpt_dir}")

    bar.close()
    dt = time.time() - t0
    logger.info(
        f"Training loop finished: {step} steps in {dt:.1f}s "
        f"(avg loss {running_loss / max(step, 1):.4f})."
    )


if __name__ == "__main__":
    sys.exit(main())
