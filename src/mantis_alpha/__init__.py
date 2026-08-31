"""
Mantis-Alpha: High-Frequency Edge Vision-Language-Action Policy Package
Built on the SmolVLA / SmolVLM2 Architectural Foundation.

Stand-alone implementation: the SmolVLA model code is vendored in this
repository and no LeRobot import is required at runtime.
"""

__version__ = "0.2.0"

from .config import MantisAlphaConfig, PolicyFeature, SmolVLAConfig
from .dataset import LeRobotDataset
from .ensemble import TemporalEnsemble
from .modeling import SmolVLAPolicy, VLAFlowMatching
from .processor import SmolVLABatchProcessor, load_dataset_stats
from .smolvlm_with_expert import SmolVLMWithExpertModel

__all__ = [
    "SmolVLAPolicy",
    "SmolVLAConfig",
    "MantisAlphaConfig",
    "PolicyFeature",
    "VLAFlowMatching",
    "SmolVLMWithExpertModel",
    "LeRobotDataset",
    "SmolVLABatchProcessor",
    "TemporalEnsemble",
    "load_dataset_stats",
]
