"""Flow Matching with Classifier-Free Guidance on 2D Gaussians.

A numpy-based implementation demonstrating:
- Flow matching training with label dropout
- Classifier-free guidance for improved class separation
- Deterministic Euler ODE sampling
- Visualization of CFG effects and flow evolution
"""

from flow_gaussians.data import (
    DATASET_CONFIGS,
    generate_multimodal_non_overlapping,
    generate_non_overlapping_gaussians,
    generate_overlapping_gaussians,
)
from flow_gaussians.model import AdamOptimizer, SimpleFlowNetwork
from flow_gaussians.sampling import sample_euler, sample_euler_full_trajectory, sample_euler_with_trajectory
from flow_gaussians.train import train_flow_matching

__all__ = [
    # Data
    "DATASET_CONFIGS",
    "generate_overlapping_gaussians",
    "generate_non_overlapping_gaussians",
    "generate_multimodal_non_overlapping",
    # Model
    "SimpleFlowNetwork",
    "AdamOptimizer",
    # Training
    "train_flow_matching",
    # Sampling
    "sample_euler",
    "sample_euler_full_trajectory",
    "sample_euler_with_trajectory",
]
