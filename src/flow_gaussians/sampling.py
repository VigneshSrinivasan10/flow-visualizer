"""Sampling functions for flow matching with CFG."""

from typing import Dict, List, Tuple

import numpy as np

from flow_gaussians.model import SimpleFlowNetwork


def sample_euler(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    cfg_scale: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample using deterministic Euler ODE with classifier-free guidance.

    CFG Formula: v = v_uncond + cfg_scale * (v_cond - v_uncond)

    Args:
        model: Trained SimpleFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        num_steps: Number of Euler steps
        cfg_scale: CFG strength (0 = unconditional, 1 = conditional, >1 = extrapolation)
        seed: Random seed for reproducibility

    Returns:
        x: (n_samples, 2) generated samples
    """
    if seed is not None:
        np.random.seed(seed)

    # Start from noise (t=0)
    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)

    for step in range(num_steps):
        t = np.full((n_samples, 1), step * dt)

        # Compute conditional velocity
        v_cond = model.predict(x, t, labels_cond)

        # Compute unconditional velocity
        v_uncond = model.predict(x, t, labels_uncond)

        # Apply CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step (deterministic)
        x = x + v * dt

    return x


def sample_euler_with_trajectory(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    cfg_scale: float = 0.0,
    save_times: List[float] | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """
    Sample with trajectory tracking at specified time points.

    Args:
        model: Trained SimpleFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        num_steps: Number of Euler steps
        cfg_scale: CFG strength
        save_times: List of time points to save (default: [0, 0.25, 0.5, 0.75, 1.0])
        seed: Random seed for reproducibility

    Returns:
        x: Final samples
        trajectories: Dictionary mapping time to samples at that time
    """
    if save_times is None:
        save_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    if seed is not None:
        np.random.seed(seed)

    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)

    trajectories = {0.0: x.copy()}

    for step in range(num_steps):
        t_val = step * dt
        t = np.full((n_samples, 1), t_val)

        v_cond = model.predict(x, t, labels_cond)
        v_uncond = model.predict(x, t, labels_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        x = x + v * dt

        # Save trajectory at specified times
        current_time = (step + 1) * dt
        for save_time in save_times:
            if abs(current_time - save_time) < dt / 2 and save_time not in trajectories:
                trajectories[save_time] = x.copy()

    # Ensure final time is saved
    trajectories[1.0] = x.copy()

    return x, trajectories


def sample_euler_full_trajectory(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    cfg_scale: float = 0.0,
    seed: int | None = None,
) -> List[np.ndarray]:
    """
    Sample with full trajectory tracking at every step.

    Args:
        model: Trained SimpleFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        num_steps: Number of Euler steps
        cfg_scale: CFG strength
        seed: Random seed for reproducibility

    Returns:
        trajectory: List of (n_samples, 2) arrays, one per step [x_0, x_1, ..., x_T]
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)

    trajectory = [x.copy()]

    for step in range(num_steps):
        t = np.full((n_samples, 1), step * dt)

        v_cond = model.predict(x, t, labels_cond)
        v_uncond = model.predict(x, t, labels_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        x = x + v * dt
        trajectory.append(x.copy())

    return trajectory


def sample_euler_rectified_cfg_plusplus(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    lambda_max: float = 1.0,
    gamma: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample using Rectified CFG++ (predictor-corrector guidance).

    Rectified CFG++ uses a predictor-corrector scheme that:
    1. Predictor: Take half-step using conditional velocity
    2. Corrector: Compute guidance at intermediate point
    3. Final velocity: v = v_cond + alpha(t) * (v_cond_mid - v_uncond_mid)

    Guidance schedule: alpha(t) = lambda_max * (1-t)^gamma

    Reference: "Rectified-CFG++ for Flow Based Models" (NeurIPS 2025)
    https://arxiv.org/abs/2510.07631

    Args:
        model: Trained SimpleFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        num_steps: Number of Euler steps
        lambda_max: Maximum guidance strength (default: 1.0)
        gamma: Schedule decay power (default: 1.0, linear decay)
        seed: Random seed for reproducibility

    Returns:
        x: (n_samples, 2) generated samples
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)

    for step in range(num_steps):
        t_val = step * dt
        t = np.full((n_samples, 1), t_val)

        # Compute conditional velocity at current point
        v_cond = model.predict(x, t, labels_cond)

        # Predictor: half-step using conditional velocity
        x_mid = x + (dt / 2) * v_cond
        t_mid = np.full((n_samples, 1), t_val + dt / 2)

        # Compute velocities at intermediate point
        v_cond_mid = model.predict(x_mid, t_mid, labels_cond)
        v_uncond_mid = model.predict(x_mid, t_mid, labels_uncond)

        # Guidance schedule: alpha(t) = lambda_max * (1-t)^gamma
        alpha_t = lambda_max * ((1 - t_val) ** gamma)

        # Rectified CFG++: v = v_cond + alpha(t) * (v_cond_mid - v_uncond_mid)
        v = v_cond + alpha_t * (v_cond_mid - v_uncond_mid)

        x = x + v * dt

    return x


def sample_euler_rectified_cfg_plusplus_full_trajectory(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    lambda_max: float = 1.0,
    gamma: float = 1.0,
    seed: int | None = None,
) -> List[np.ndarray]:
    """
    Sample with Rectified CFG++ and full trajectory tracking.

    Args:
        model: Trained SimpleFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        num_steps: Number of Euler steps
        lambda_max: Maximum guidance strength
        gamma: Schedule decay power
        seed: Random seed for reproducibility

    Returns:
        trajectory: List of (n_samples, 2) arrays, one per step [x_0, x_1, ..., x_T]
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)

    trajectory = [x.copy()]

    for step in range(num_steps):
        t_val = step * dt
        t = np.full((n_samples, 1), t_val)

        # Compute conditional velocity at current point
        v_cond = model.predict(x, t, labels_cond)

        # Predictor: half-step using conditional velocity
        x_mid = x + (dt / 2) * v_cond
        t_mid = np.full((n_samples, 1), t_val + dt / 2)

        # Compute velocities at intermediate point
        v_cond_mid = model.predict(x_mid, t_mid, labels_cond)
        v_uncond_mid = model.predict(x_mid, t_mid, labels_uncond)

        # Guidance schedule: alpha(t) = lambda_max * (1-t)^gamma
        alpha_t = lambda_max * ((1 - t_val) ** gamma)

        # Rectified CFG++: v = v_cond + alpha(t) * (v_cond_mid - v_uncond_mid)
        v = v_cond + alpha_t * (v_cond_mid - v_uncond_mid)

        x = x + v * dt
        trajectory.append(x.copy())

    return trajectory


def classify_samples(
    samples: np.ndarray,
    class0_centers: List[List[float]] | None = None,
    class1_centers: List[List[float]] | None = None,
) -> np.ndarray:
    """
    Classify samples based on distance to nearest class center.

    Args:
        samples: (n_samples, 2) array of 2D points
        class0_centers: List of [x, y] centers for class 0
        class1_centers: List of [x, y] centers for class 1

    Returns:
        labels: (n_samples,) array of predicted class labels (0 or 1)
    """
    if class0_centers is None:
        class0_centers = [[-0.3, -0.3]]
    if class1_centers is None:
        class1_centers = [[0.3, 0.3]]

    class0_centers_arr = np.array(class0_centers)
    class1_centers_arr = np.array(class1_centers)

    # Compute minimum distance to any class 0 center
    dist_to_class0 = np.min(
        [np.sqrt(np.sum((samples - center) ** 2, axis=1)) for center in class0_centers_arr],
        axis=0,
    )

    # Compute minimum distance to any class 1 center
    dist_to_class1 = np.min(
        [np.sqrt(np.sum((samples - center) ** 2, axis=1)) for center in class1_centers_arr],
        axis=0,
    )

    # Classify based on nearest center
    return (dist_to_class1 < dist_to_class0).astype(int)
