"""Sampling functions for distilled flow models."""

from typing import List

import numpy as np

from flow_gaussians.distill_model import DistilledFlowNetwork


def sample_distilled_euler(
    model: DistilledFlowNetwork,
    n_samples: int,
    label: int,
    w: float,
    num_steps: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample using deterministic Euler ODE with distilled model (single forward pass).

    The distilled model directly predicts the CFG-combined velocity,
    so we only need one forward pass per step instead of two.

    Args:
        model: Trained DistilledFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        w: Guidance scale (the model was trained to handle any scale in [w_min, w_max])
        num_steps: Number of Euler steps
        seed: Random seed for reproducibility

    Returns:
        x: (n_samples, 2) generated samples
    """
    if seed is not None:
        np.random.seed(seed)

    # Start from noise (t=0)
    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_arr = np.full(n_samples, label, dtype=float)
    w_arr = np.full((n_samples, 1), w)

    for step in range(num_steps):
        t = np.full((n_samples, 1), step * dt)

        # Single forward pass to get CFG-combined velocity
        v = model.predict(x, t, labels_arr, w_arr)

        # Euler step
        x = x + v * dt

    return x


def sample_distilled_euler_full_trajectory(
    model: DistilledFlowNetwork,
    n_samples: int,
    label: int,
    w: float,
    num_steps: int = 100,
    seed: int | None = None,
) -> List[np.ndarray]:
    """
    Sample with full trajectory tracking at every step using distilled model.

    Args:
        model: Trained DistilledFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        w: Guidance scale
        num_steps: Number of Euler steps
        seed: Random seed for reproducibility

    Returns:
        trajectory: List of (n_samples, 2) arrays, one per step [x_0, x_1, ..., x_T]
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_arr = np.full(n_samples, label, dtype=float)
    w_arr = np.full((n_samples, 1), w)

    trajectory = [x.copy()]

    for step in range(num_steps):
        t = np.full((n_samples, 1), step * dt)

        # Single forward pass
        v = model.predict(x, t, labels_arr, w_arr)

        x = x + v * dt
        trajectory.append(x.copy())

    return trajectory
