"""Mantis-Alpha standalone offline evaluator.

Computes action-MSE of a trained checkpoint over held-out samples of a
LeRobot-format dataset (e.g. LIBERO) without importing LeRobot. Closed-loop
simulation against the LIBERO simulator can be added on top of this script.

```bash
python scripts/eval_libero.py \
    --checkpoint /home/adminroot/Desktop/vla/outputs/mantis_alpha/checkpoints/000020 \
    --dataset_root /home/adminroot/Desktop/vla/datasets/libero \
    --num_samples 64
```
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

logger = logging.getLogger("mantis_eval")

LOG_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mantis-Alpha offline action-MSE evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="/home/adminroot/Desktop/vla/datasets/libero")
    parser.add_argument(
        "--vlm_model_name", type=str, default="/home/adminroot/Desktop/vla/models/smolvlm2",
        help="Only used if the checkpoint config lacks a resolvable vlm path",
    )
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output_dir", type=str, default="/home/adminroot/Desktop/vla/outputs/mantis_eval",
        help="Where to write eval.log",
    )
    return parser.parse_args()


def setup_logging(output_dir: str, level: int = logging.INFO) -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval.log")
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)
    return log_path


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.output_dir)
    logger.info(f"Command: {sys.executable} {' '.join(sys.argv)}")
    logger.info(f"Params: {vars(args)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from mantis_alpha.dataset import LeRobotDataset
    from mantis_alpha.modeling import SmolVLAPolicy
    from mantis_alpha.processor import SmolVLABatchProcessor, load_dataset_stats

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info(f"Loading policy from {args.checkpoint}...")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, vlm_model_name=args.vlm_model_name)
    policy.to(device)
    policy.eval()

    dataset = LeRobotDataset(args.dataset_root, chunk_size=policy.config.chunk_size)
    stats = load_dataset_stats(args.dataset_root)

    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(args.vlm_model_name).tokenizer
    processor = SmolVLABatchProcessor(policy.config, tokenizer, stats)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)

    sq_sum, n_elem, n_batch = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start : start + args.batch_size]
            samples = [dataset[int(i)] for i in idx]
            batch = processor.train_batch(samples)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch = processor.to_policy_batch(batch)
            actions_gt = batch["action"]
            pred = policy.predict_action_chunk(batch)
            pred = pred[:, :, : actions_gt.shape[-1]]
            horizon = min(pred.shape[1], actions_gt.shape[1])
            se = (pred[:, :horizon] - actions_gt[:, :horizon]) ** 2
            valid = (~batch["action_is_pad"][:, :horizon]).unsqueeze(-1)
            sq_sum += float((se * valid).sum())
            n_elem += int(valid.sum()) * se.shape[-1]  # count action dims too
            n_batch += 1
            logger.info(f"batch {n_batch}: running MSE {sq_sum / max(n_elem, 1):.4f}")

    mse = sq_sum / max(n_elem, 1)
    logger.info(f"Final action MSE (normalized units) over {n_elem} elements: {mse:.4f}")


if __name__ == "__main__":
    sys.exit(main())
