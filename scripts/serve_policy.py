"""Mantis-Alpha standalone HTTP policy server.

Serves a trained checkpoint over REST without importing LeRobot. Request
images as base64 JPEG/PNG; returns the predicted (unnormalized) action chunk
from flow-matching sampling.
"""

import argparse
import base64
import io
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:  # pragma: no cover
    print("FastAPI / Uvicorn not found. Install with: pip install fastapi uvicorn")
    raise

app = FastAPI(title="Mantis-Alpha Policy Server", version="0.2.0")
policy = None
processor = None
device = None


class ActionRequest(BaseModel):
    task: str
    image_main_b64: str            # Base64-encoded RGB image (top/front view)
    image_wrist_b64: Optional[str] = None  # Base64-encoded RGB image (wrist view)
    state: List[float]             # Low-dim proprioceptive state (e.g. 7-DoF / 8-DoF)


class ActionResponse(BaseModel):
    action: List[float]            # First action of the chunk (dataset units)
    action_chunk: Optional[List[List[float]]] = None


def _decode_image(b64: str) -> torch.Tensor:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)
    return arr.float() / 255.0


@app.post("/act", response_model=ActionResponse)
def get_action(req: ActionRequest):
    if policy is None or processor is None:
        raise HTTPException(status_code=500, detail="Policy model not loaded")

    image_keys = policy.config.image_features
    if not image_keys:
        raise HTTPException(status_code=500, detail="Policy has no image features")
    images = {image_keys[0]: _decode_image(req.image_main_b64)[None]}
    if len(image_keys) > 1:
        wrist = _decode_image(req.image_wrist_b64)[None] if req.image_wrist_b64 else torch.zeros_like(
            images[image_keys[0]]
        )
        images[image_keys[1]] = wrist

    state = torch.tensor([req.state], dtype=torch.float32)
    batch = processor.infer_batch(images, state, req.task, device=device)
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    chunk = processor.unnormalize_action(chunk)[0]  # [chunk_size, action_dim] dataset units
    first = chunk[0, : len(req.state)] if len(req.state) <= chunk.shape[1] else chunk[0]
    return ActionResponse(action=first.tolist(), action_chunk=chunk.tolist())


def main():
    global policy, processor, device
    parser = argparse.ArgumentParser(description="Serve Mantis-Alpha Policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--vlm_model_name", type=str, default=None, help="Fallback VLM path if checkpoint config lacks one"
    )
    parser.add_argument("--stats", type=str, default=None, help="stats.json path (default: alongside checkpoint)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from mantis_alpha.modeling import SmolVLAPolicy
    from mantis_alpha.processor import SmolVLABatchProcessor, load_dataset_stats
    from transformers import AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {"vlm_model_name": args.vlm_model_name} if args.vlm_model_name else {}
    print(f"Loading policy from {args.checkpoint}...")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, **overrides)
    policy.to(device)
    policy.eval()

    stats_path = args.stats or os.path.join(args.checkpoint, "stats.json")
    stats = load_dataset_stats(stats_path) if os.path.isfile(stats_path) else {}
    tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer
    processor = SmolVLABatchProcessor(policy.config, tokenizer, stats)

    print(f"Starting Mantis-Alpha Policy Server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
