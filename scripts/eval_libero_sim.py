"""Standalone closed-loop LIBERO simulation evaluation for Mantis-Alpha.

Rolls out a trained checkpoint in the LIBERO simulator (via the `libero`
package + robosuite) and reports per-task success rates — the real benchmark
metric, as opposed to offline action-MSE. No LeRobot import.

```bash
MUJOCO_GL=egl python scripts/eval_libero_sim.py \
    --checkpoint /home/adminroot/Desktop/vla/outputs/mantis_libero10_b128_chunk50/checkpoints/002000 \
    --suite libero_10 --n_episodes 10
```
"""

import os
import argparse
import json
import logging
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before mujoco/robosuite import

import numpy as np
import torch

logger = logging.getLogger("mantis_sim_eval")

LOG_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

SUITE_LABELS = {
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_10": "LIBERO_10 (长时序复合)",
    "libero_90": "LIBERO_90",
    "libero_100": "LIBERO_100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mantis-Alpha closed-loop LIBERO evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_10",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_100"])
    parser.add_argument("--n_episodes", type=int, default=10, help="Rollouts per task")
    parser.add_argument("--n_action_steps", type=int, default=10,
                        help="Actions executed per predicted chunk before replanning "
                             "(SmolVLA paper ablation optimum: 10; 50 = full chunk, worst)")
    parser.add_argument("--max_steps", type=int, default=600, help="Env steps per episode")
    parser.add_argument("--image_size", type=int, default=128, help="Camera resolution (matches demo data)")
    parser.add_argument("--flip_images", action="store_true",
                        help="Rotate rendered images 180 deg to match training data orientation. "
                             "The official HuggingFaceVLA/libero conversion stores images rotated "
                             "180 deg (upright but horizontally mirrored vs the live env); raw hdf5 "
                             "conversions store the raw render and need no transform.")
    parser.add_argument("--stats", type=str, default=None, help="stats.json (default: <checkpoint>/stats.json)")
    parser.add_argument("--output_dir", type=str, default="/home/adminroot/Desktop/vla/outputs/mantis_sim_eval")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "sim_eval.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)
    return log_path


def obs_to_inputs(obs, task_text: str, flip: bool = False) -> tuple[dict, np.ndarray]:
    """Map a robosuite observation to (images dict, 8-dim state), matching training format."""
    from robosuite.utils.transform_utils import quat2axisangle

    images = {
        "observation.images.image": obs["agentview_image"],
        "observation.images.image2": obs["robot0_eye_in_hand_image"],
    }
    images = {k: torch.from_numpy(np.array(v)).permute(2, 0, 1) for k, v in images.items()}
    if flip:
        # Official conversion stores images rotated 180 deg vs the live render
        # (verified pixel-level: rot180 MSE ~1.4k vs flipud ~2k vs raw ~14k).
        images = {k: torch.flip(v, dims=(1, 2)) for k, v in images.items()}  # CHW: rot180
    state = np.hstack(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)
    return images, state


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)
    logger.info(f"Command: {sys.executable} {' '.join(sys.argv)}")
    logger.info(f"Execution horizon: n_action_steps={args.n_action_steps} (replan every {args.n_action_steps} steps)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    # ── Policy / processor ───────────────────────────────────────────────
    from mantis_alpha import SmolVLAPolicy, SmolVLABatchProcessor, load_dataset_stats
    from transformers import AutoProcessor

    logger.info(f"Loading policy from {args.checkpoint} ...")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    stats_path = args.stats or os.path.join(args.checkpoint, "stats.json")
    if os.path.isdir(stats_path):
        stats = load_dataset_stats(stats_path)  # dataset root -> meta/stats.json
    elif os.path.isfile(stats_path):
        stats = json.load(open(stats_path))  # direct stats.json file
    else:
        raise FileNotFoundError(
            f"stats.json not found at {stats_path}; pass --stats pointing to the training dataset stats"
        )
    tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer
    processor = SmolVLABatchProcessor(policy.config, tokenizer, stats)

    # ── Benchmark ────────────────────────────────────────────────────────
    bench = benchmark.get_benchmark(args.suite)()
    n_tasks = bench.get_num_tasks()
    logger.info(f"Suite {args.suite}: {n_tasks} tasks, {args.n_episodes} episodes each")

    results = {}
    t_start = time.time()
    for i in range(n_tasks):
        bddl_path = bench.get_task_bddl_file_path(i)
        init_states = bench.get_task_init_states(i)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=args.image_size,
            camera_widths=args.image_size,
        )
        task_text = env.language_instruction.strip()
        env.seed(0)

        n_success = 0
        episode_lines = []
        for ep in range(args.n_episodes):
            obs = env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])

            success = False
            action_queue: list = []
            for _ in range(args.max_steps):
                if not action_queue:
                    images, state = obs_to_inputs(obs, task_text, flip=args.flip_images)
                    batch = processor.infer_batch(
                        {k: v[None] for k, v in images.items()},
                        state[None],
                        [task_text],
                        device=device,
                    )
                    with torch.inference_mode():
                        chunk = policy.predict_action_chunk(batch)
                    # Receding-horizon execution: only the first n_action_steps of each
                    # 50-action chunk are executed before re-observing and replanning.
                    action_queue = processor.unnormalize_action(chunk)[0, : args.n_action_steps].cpu().numpy().tolist()

                action = action_queue.pop(0)
                obs, reward, done, info = env.step(action)
                if env.check_success():
                    success = True
                    break
                if done:
                    break
            n_success += int(success)
            episode_lines.append(f"ep{ep}: {'✅' if success else '❌'}")
            logger.info(
                f"[{i:02d}] {task_text[:56]} ep{ep + 1}/{args.n_episodes}: "
                f"{'SUCCESS' if success else 'FAIL'}"
            )
        env.close()

        rate = n_success / args.n_episodes
        results[task_text] = {"successes": n_success, "episodes": args.n_episodes, "rate": rate}
        logger.info(f"[{i:02d}] {task_text[:56]} -> {n_success}/{args.n_episodes} = {rate:.0%}")

    total_success = sum(r["successes"] for r in results.values())
    total_eps = sum(r["episodes"] for r in results.values())

    # ── Scorecard ────────────────────────────────────────────────────────
    label = SUITE_LABELS.get(args.suite, args.suite)
    bar = "=" * 82
    lines = [bar, f"Mantis-Alpha closed-loop evaluation | checkpoint: {args.checkpoint}", bar,
             f"📂 Suite: {label}   [总分: {total_success}/{total_eps} = {total_success / total_eps:.1%}]", "-" * 82,
             " ID  | Success  | Rate   | 任务", "-" * 82]
    for i, (task_text, r) in enumerate(results.items()):
        star = "⭐" if r["rate"] == 1.0 else ("✅" if r["rate"] >= 0.8 else ("⚠️" if r["rate"] < 0.5 else ""))
        lines.append(f"[{i:02d}] | {r['successes']:3d}/{r['episodes']:<3d} | {r['rate']:5.0%}  | {star} {task_text}")
    lines.append(bar)
    lines.append(f" 🏆 综合成功率: {total_success}/{total_eps} ({total_success / total_eps:.1%})  "
                 f"| 用时 {time.time() - t_start:.0f}s")
    lines.append(bar)
    scorecard = "\n".join(lines)
    print(scorecard)
    logger.info("\n" + scorecard)

    with open(os.path.join(args.output_dir, "sim_eval_results.json"), "w") as f:
        json.dump(
            {"suite": args.suite, "checkpoint": args.checkpoint,
             "total_success": total_success, "total_episodes": total_eps, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"Results saved to {os.path.join(args.output_dir, 'sim_eval_results.json')}")


if __name__ == "__main__":
    main()
