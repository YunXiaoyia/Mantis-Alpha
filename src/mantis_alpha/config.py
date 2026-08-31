"""Mantis-Alpha policy configuration (stand-alone SmolVLA).

Field set and defaults mirror LeRobot's SmolVLAConfig so that the copied model
code runs unchanged and checkpoints stay interoperable, plus a few Mantis-Alpha
specific extras. Depends only on the standard library.
"""

from dataclasses import dataclass, field, fields

from .rtc.configuration_rtc import RTCConfig

# Dataset field-name constants (kept in sync with mantis_alpha.utils to avoid
# a circular import at config-import time; the policy imports utils directly).
OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"
ACTION = "action"


class FeatureType:
    VISUAL = "VISUAL"
    STATE = "STATE"
    ACTION = "ACTION"


@dataclass
class PolicyFeature:
    type: str
    shape: tuple[int, ...]


@dataclass
class SmolVLAConfig:
    """Configuration class for the Mantis-Alpha (SmolVLA-based) policy."""

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )

    # Dataset features (filled in by the trainer / dataset loader).
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (256, 256)

    # Add empty images. Used when a checkpoint expects more cameras than the env provides.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding (number of flow-matching Euler steps)
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 40_000
    scheduler_decay_lr: float = 2.5e-6

    # VLM backbone. Use a local path or a HuggingFace repo id.
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = True  # False: train the expert from scratch. True: init from SmolVLM2/SmolVLA weights.

    add_image_special_tokens: bool = False  # LeRobot SmolVLA default: no extra image start/end tokens.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1  # < 0 disables fixed prefix padding.

    pad_language_to: str = "longest"  # "max_length" | "longest"

    num_expert_layers: int = -1  # <= 0: expert depth equals the VLM depth.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers).
    self_attn_every_n_layers: int = 2  # Interleave self-attention layers every N layers.
    expert_width_multiplier: float = 0.75  # Action expert hidden size relative to the VLM.

    min_period: float = 4e-3  # Sensitivity range for the timestep sine-cosine positional encoding.
    max_period: float = 4.0

    # Real-Time Chunking (RTC) inference configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # Device ("cuda", "cpu", ... or None for auto)
    device: str | None = None

    # ── Mantis-Alpha extras (informational / trainer hints) ──────────────
    # Task string of the training dataset, stored only for provenance.
    dataset_repo_id: str | None = None

    def __post_init__(self):
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if isinstance(self.resize_imgs_with_padding, list):
            self.resize_imgs_with_padding = tuple(self.resize_imgs_with_padding)
        if isinstance(self.optimizer_betas, list):
            self.optimizer_betas = tuple(self.optimizer_betas)

    def validate_features(self) -> None:
        """Register synthetic camera features for `empty_cameras` (mirrors LeRobot behavior)."""
        for i in range(self.empty_cameras):
            self.input_features[f"{OBS_IMAGES}.empty_camera_{i}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )

    # ── Feature accessors ────────────────────────────────────────────────
    @property
    def image_features(self) -> list[str]:
        return [k for k, v in self.input_features.items() if v.type == FeatureType.VISUAL]

    @property
    def state_feature(self) -> PolicyFeature | None:
        return self.input_features.get(OBS_STATE)

    @property
    def action_feature(self) -> PolicyFeature | None:
        return self.output_features.get(ACTION)

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    # ── (De)serialization ────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, raw: dict, **overrides) -> "SmolVLAConfig":
        """Build a config from a JSON-ish dict, ignoring unknown fields (LeRobot checkpoints tolerated)."""
        known = {f.name: f for f in fields(cls)}
        kwargs: dict = {}
        for key, value in raw.items():
            if key not in known or value is None:
                continue
            if key in ("input_features", "output_features") and isinstance(value, dict):
                value = {
                    fk: PolicyFeature(type=feat.get("type", "VISUAL"), shape=tuple(feat.get("shape", ())))
                    for fk, feat in value.items()
                    if isinstance(feat, dict)
                }
            elif key == "rtc_config" and isinstance(value, dict):
                value = RTCConfig(**{k: v for k, v in value.items() if k in {f.name for f in fields(RTCConfig)}})
            kwargs[key] = value
        kwargs.update(overrides)
        return cls(**kwargs)


# Backwards-friendly alias for the original Mantis-Alpha config name.
MantisAlphaConfig = SmolVLAConfig
