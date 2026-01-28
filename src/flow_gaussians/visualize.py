"""Visualization functions for flow matching experiments."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig, OmegaConf
from scipy.stats import gaussian_kde

from flow_gaussians.data import DATASET_CONFIGS
from flow_gaussians.model import SimpleFlowNetwork
from flow_gaussians.sampling import (
    classify_samples,
    sample_euler,
    sample_euler_full_trajectory,
    sample_euler_rectified_cfg_plusplus,
    sample_euler_rectified_cfg_plusplus_full_trajectory,
    sample_euler_with_trajectory,
)

logger = logging.getLogger(__name__)


def plot_training_data(
    data: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Training Data",
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
):
    """Visualize the training data."""
    if class0_centers is None:
        class0_centers = [[-0.3, -0.3]]
    if class1_centers is None:
        class1_centers = [[0.3, 0.3]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot each class
    mask_0 = labels == 0
    mask_1 = labels == 1

    ax.scatter(data[mask_0, 0], data[mask_0, 1], c="gray", alpha=0.5, s=10, label="Class 0")
    ax.scatter(data[mask_1, 0], data[mask_1, 1], c="lightcoral", alpha=0.5, s=10, label="Class 1")

    # Mark the centers
    for i, center in enumerate(class0_centers):
        label = "Mean 0" if i == 0 else None
        ax.scatter([center[0]], [center[1]], c="black", marker="x", s=200, linewidths=3, label=label)
    for i, center in enumerate(class1_centers):
        label = "Mean 1" if i == 0 else None
        ax.scatter(
            [center[0]], [center[1]], c="darkred", marker="x", s=200, linewidths=3, label=label
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(f"Training Data: {title}", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_training_loss(losses: List[float], save_path: Optional[str] = None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(losses, color="#2E86AB", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Flow Matching Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_cfg_comparison(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    target_label: int = 0,
    cfg_scales: Optional[List[float]] = None,
    n_samples: int = 500,
    save_path: Optional[str] = None,
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    seed: int = 123,
):
    """Create a grid visualization comparing different CFG scales."""
    if cfg_scales is None:
        cfg_scales = [0, 1, 3, 5, 7, 9]

    n_cfg = len(cfg_scales)
    fig, axes = plt.subplots(2, (n_cfg + 1) // 2, figsize=(4 * ((n_cfg + 1) // 2), 8))
    axes = axes.flatten()

    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1

    for idx, cfg_scale in enumerate(cfg_scales):
        ax = axes[idx]

        # Generate samples
        samples = sample_euler(model, n_samples, target_label, num_steps=100, cfg_scale=cfg_scale, seed=seed)

        # Plot training data (faded)
        ax.scatter(data[mask_0, 0], data[mask_0, 1], c="gray", alpha=0.15, s=5)
        ax.scatter(data[mask_1, 0], data[mask_1, 1], c="lightcoral", alpha=0.15, s=5)

        # Plot generated samples
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            c="#2E86AB",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.7,
            s=30,
        )

        # Compute statistics
        predicted_classes = classify_samples(samples, class0_centers, class1_centers)
        n_class0 = np.sum(predicted_classes == 0)
        n_class1 = np.sum(predicted_classes == 1)
        target_count = n_class0 if target_label == 0 else n_class1
        ratio = target_count / n_samples
        std_x = np.std(samples[:, 0])
        std_y = np.std(samples[:, 1])

        # Statistics box
        stats_text = f"Target: {target_count}/{n_samples}\nRatio: {ratio:.2%}\nStd: ({std_x:.2f}, {std_y:.2f})"

        # Color code by effectiveness
        if ratio >= 0.9:
            box_color = "#90EE90"  # Light green
        elif ratio >= 0.7:
            box_color = "#FFFF99"  # Light yellow
        else:
            box_color = "#FFB6C1"  # Light pink

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"CFG Scale = {cfg_scale}", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Hide extra axes if odd number
    for idx in range(n_cfg, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"CFG Scale Comparison (Target: Class {target_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_flow_evolution(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    cfg_scales: Optional[List[float]] = None,
    target_labels: Optional[List[int]] = None,
    time_steps: Optional[List[float]] = None,
    n_samples: int = 300,
    save_path: Optional[str] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    seed: int = 456,
):
    """Create a grid showing flow evolution over time for different CFG scales."""
    if cfg_scales is None:
        cfg_scales = [0, 1, 2, 4]
    if target_labels is None:
        target_labels = [0]
    if time_steps is None:
        time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_rows = len(cfg_scales)
    n_cols = len(time_steps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1

    for row_idx, cfg_scale in enumerate(cfg_scales):
        # Generate samples with trajectory tracking
        target_label = (
            target_labels[0] if len(target_labels) == 1 else target_labels[row_idx % len(target_labels)]
        )
        _, trajectories = sample_euler_with_trajectory(
            model,
            n_samples,
            target_label,
            num_steps=100,
            cfg_scale=cfg_scale,
            save_times=time_steps,
            seed=seed,
        )

        for col_idx, t in enumerate(time_steps):
            ax = axes[row_idx, col_idx]

            # Get samples at this time
            samples = trajectories.get(
                t, trajectories[min(trajectories.keys(), key=lambda x: abs(x - t))]
            )

            # Plot training data (faded)
            ax.scatter(data[mask_0, 0], data[mask_0, 1], c="gray", alpha=0.1, s=3)
            ax.scatter(data[mask_1, 0], data[mask_1, 1], c="lightcoral", alpha=0.1, s=3)

            # Plot samples
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                c="#2E86AB",
                edgecolors="black",
                linewidths=0.3,
                alpha=0.6,
                s=20,
            )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Labels
            if row_idx == 0:
                ax.set_title(f"t = {t:.2f}", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"CFG = {cfg_scale}", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Flow Evolution (Target: Class {target_labels[0]})", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_both_classes_cfg(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    cfg_scales: Optional[List[float]] = None,
    n_samples: int = 400,
    save_path: Optional[str] = None,
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    seed: int = 789,
):
    """Show CFG comparison for both classes side by side."""
    if cfg_scales is None:
        cfg_scales = [0, 1, 3, 5]

    n_cfg = len(cfg_scales)
    fig, axes = plt.subplots(2, n_cfg, figsize=(4 * n_cfg, 8))

    mask_0 = labels == 0
    mask_1 = labels == 1

    for row_idx, target_label in enumerate([0, 1]):
        for col_idx, cfg_scale in enumerate(cfg_scales):
            ax = axes[row_idx, col_idx]

            # Generate samples
            samples = sample_euler(
                model, n_samples, target_label, num_steps=100, cfg_scale=cfg_scale, seed=seed + row_idx
            )

            # Plot training data (faded)
            ax.scatter(data[mask_0, 0], data[mask_0, 1], c="gray", alpha=0.15, s=5)
            ax.scatter(data[mask_1, 0], data[mask_1, 1], c="lightcoral", alpha=0.15, s=5)

            # Color samples by target class
            color = "#3498db" if target_label == 0 else "#e74c3c"
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                c=color,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.7,
                s=30,
            )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"CFG = {cfg_scale}", fontsize=25)
            if col_idx == 0:
                ax.set_ylabel(f"Class {target_label}", fontsize=25)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def create_probability_path_animation(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    cfg_scales: List[float] = None,
    n_samples: int = 500,
    num_steps: int = 50,
    save_path: Optional[str] = None,
    fps: int = 8,
    dpi: int = 120,
    grid_size: int = 100,
    seed: int = 42,
    hold_end_seconds: float = 2.0,
):
    """
    Create animated probability path visualization showing density flow from source to target.

    Minimal, clean style inspired by Diffusion Explorer:
    - Plain white background
    - Blue source distribution (left)
    - Red target distribution (right)
    - Yellow/gold flowing density
    - Slow, clear animation

    Args:
        model: Trained SimpleFlowNetwork
        data: Training data (n_samples, 2)
        labels: Training labels (n_samples,)
        cfg_scales: List of CFG scales to visualize (default: [1, 5, 9])
        n_samples: Number of samples for trajectory
        num_steps: Number of Euler steps (frames in animation)
        save_path: Path to save the GIF
        fps: Frames per second (default: 8 for slow animation)
        dpi: Output resolution
        grid_size: Resolution for KDE grid
        seed: Random seed for reproducibility
        hold_end_seconds: How long to hold the final frame
    """
    if cfg_scales is None:
        cfg_scales = [1, 5, 9]

    n_cfg = len(cfg_scales)
    x_offset = 4.0  # Clear separation
    target_labels = [0, 1]

    # Generate trajectories for each class and CFG scale
    logger.info(f"Generating trajectories for CFG scales {cfg_scales}...")
    trajectories = {}
    for target_label in target_labels:
        trajectories[target_label] = {}
        for cfg_scale in cfg_scales:
            traj = sample_euler_full_trajectory(
                model, n_samples, target_label, num_steps=num_steps, cfg_scale=cfg_scale, seed=seed
            )
            trajectories[target_label][cfg_scale] = traj

    # Setup figure: 2 rows (classes), n columns (CFG scales) - 4x size for quality
    fig, axes = plt.subplots(2, n_cfg, figsize=(16 * n_cfg, 24), facecolor='white')
    fig.patch.set_facecolor('white')

    # Setup KDE grid
    x_grid = np.linspace(-8, 8, grid_size * 2)
    y_grid = np.linspace(-4, 4, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Get source and target for static display
    sources = {}
    targets = {}
    for target_label in target_labels:
        sources[target_label] = {cfg: trajectories[target_label][cfg][0] for cfg in cfg_scales}
        targets[target_label] = {cfg: trajectories[target_label][cfg][-1] for cfg in cfg_scales}

    # Calculate frames
    n_animation_frames = num_steps + 1
    n_hold_frames = int(hold_end_seconds * fps)
    n_frames = n_animation_frames + n_hold_frames

    # Custom light orange colormap for density
    from matplotlib.colors import LinearSegmentedColormap
    orange_cmap = LinearSegmentedColormap.from_list(
        'orange_density',
        ['white', '#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726'],
        N=256
    )

    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1

    def update(frame_idx):
        actual_frame = min(frame_idx, num_steps)
        t = actual_frame / num_steps

        for row_idx, target_label in enumerate(target_labels):
            for col_idx, cfg_scale in enumerate(cfg_scales):
                ax = axes[row_idx, col_idx]
                ax.clear()
                ax.set_facecolor('white')

                # Get current samples
                current_samples = trajectories[target_label][cfg_scale][actual_frame].copy()

                # Training data in gray (on target/right side)
                train_data_right = data.copy()
                train_data_right[:, 0] += x_offset
                ax.scatter(
                    train_data_right[:, 0],
                    train_data_right[:, 1],
                    s=5,
                    color='#9E9E9E',  # Gray
                    alpha=0.3,
                    edgecolors='none',
                )

                # Source distribution (shifted left) - BLUE
                source_shifted = sources[target_label][cfg_scale].copy()
                source_shifted[:, 0] -= x_offset
                ax.scatter(
                    source_shifted[:, 0],
                    source_shifted[:, 1],
                    s=8,
                    color='#2196F3',  # Material blue
                    alpha=0.6,
                    edgecolors='none',
                )

                # Target distribution (shifted right) - RED
                target_shifted = targets[target_label][cfg_scale].copy()
                target_shifted[:, 0] += x_offset
                ax.scatter(
                    target_shifted[:, 0],
                    target_shifted[:, 1],
                    s=8,
                    color='#F44336',  # Material red
                    alpha=0.6,
                    edgecolors='none',
                )

                # Current samples shifted based on time (flow from left to right)
                data_shifted = current_samples.copy()
                data_shifted[:, 0] += x_offset * (2 * t - 1)

                # KDE density visualization - LIGHT ORANGE
                try:
                    kde = gaussian_kde(data_shifted.T, bw_method=0.2)
                    Z = kde(positions).reshape(grid_size, grid_size * 2)

                    # Normalize and threshold
                    Z_max = Z.max()
                    if Z_max > 0:
                        Z = Z / Z_max
                        levels = np.linspace(0.05, 1.0, 20)
                        ax.contourf(X, Y, Z, levels=levels, cmap=orange_cmap, alpha=0.85)
                except (np.linalg.LinAlgError, ValueError):
                    ax.scatter(
                        data_shifted[:, 0],
                        data_shifted[:, 1],
                        s=8,
                        color='#FFB74D',
                        alpha=0.6,
                        edgecolors='none',
                    )

                # Minimal labels
                if row_idx == 0:
                    ax.set_title(f"CFG = {cfg_scale}", fontsize=14, fontweight='normal', pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"Class {target_label}", fontsize=12, fontweight='normal')

                # Simple time indicator (bottom center, only bottom row)
                if row_idx == 1:
                    ax.text(0, -3.5, f"t = {t:.2f}", ha='center', fontsize=10, color='#666666')

                # Clean axis
                ax.set_xlim(-8, 8)
                ax.set_ylim(-4, 4)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        return axes.flatten()

    logger.info(f"Creating probability path animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    plt.tight_layout()

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        logger.info(f"Saved: {save_path}")

    plt.close(fig)


def plot_cfg_vs_rectified_comparison(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    cfg_scales: Optional[List[float]] = None,
    lambda_maxs: Optional[List[float]] = None,
    n_samples: int = 500,
    save_path: Optional[str] = None,
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    seed: int = 123,
    gamma: float = 1.0,
):
    """
    Compare Standard CFG vs Rectified CFG++ side by side.

    Creates a grid with:
    - 2 rows: Class 0, Class 1
    - n columns: different guidance scales
    - Top half: Standard CFG
    - Bottom half: Rectified CFG++
    """
    if cfg_scales is None:
        cfg_scales = [1, 3, 5, 7]
    if lambda_maxs is None:
        lambda_maxs = cfg_scales  # Use same values for comparison

    n_cols = len(cfg_scales)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))

    mask_0 = labels == 0
    mask_1 = labels == 1

    methods = [
        ("Standard CFG", sample_euler, "cfg_scale"),
        ("Rectified CFG++", sample_euler_rectified_cfg_plusplus, "lambda_max"),
    ]

    for method_idx, (method_name, sample_fn, param_name) in enumerate(methods):
        scales = cfg_scales if method_name == "Standard CFG" else lambda_maxs

        for target_label in [0, 1]:
            row_idx = method_idx * 2 + target_label

            for col_idx, scale in enumerate(scales):
                ax = axes[row_idx, col_idx]

                # Generate samples
                kwargs = {
                    "model": model,
                    "n_samples": n_samples,
                    "label": target_label,
                    "num_steps": 100,
                    "seed": seed + target_label,
                }
                kwargs[param_name] = scale
                if method_name == "Rectified CFG++":
                    kwargs["gamma"] = gamma

                samples = sample_fn(**kwargs)

                # Plot training data (faded)
                ax.scatter(data[mask_0, 0], data[mask_0, 1], c="gray", alpha=0.15, s=5)
                ax.scatter(data[mask_1, 0], data[mask_1, 1], c="lightcoral", alpha=0.15, s=5)

                # Color samples by target class
                color = "#3498db" if target_label == 0 else "#e74c3c"
                ax.scatter(
                    samples[:, 0],
                    samples[:, 1],
                    c=color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.7,
                    s=30,
                )

                # Compute accuracy
                predicted = classify_samples(samples, class0_centers, class1_centers)
                target_count = np.sum(predicted == target_label)
                accuracy = target_count / n_samples

                # Stats box
                box_color = "#90EE90" if accuracy >= 0.9 else ("#FFFF99" if accuracy >= 0.7 else "#FFB6C1")
                ax.text(
                    0.02, 0.98, f"Acc: {accuracy:.1%}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
                )

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

                # Labels
                if row_idx == 0:
                    ax.set_title(f"Scale = {scale}", fontsize=12, fontweight="bold")
                if col_idx == 0:
                    label_text = f"{method_name}\nClass {target_label}"
                    ax.set_ylabel(label_text, fontsize=10, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def create_rectified_cfg_probability_path_animation(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    lambda_maxs: List[float] = None,
    n_samples: int = 500,
    num_steps: int = 50,
    save_path: Optional[str] = None,
    fps: int = 8,
    dpi: int = 120,
    grid_size: int = 100,
    seed: int = 42,
    gamma: float = 1.0,
    hold_end_seconds: float = 2.0,
):
    """
    Create animated probability path visualization for Rectified CFG++.

    Minimal, clean style:
    - Plain white background
    - Blue source distribution (left)
    - Red target distribution (right)
    - Yellow/gold flowing density
    - Slow, clear animation
    """
    if lambda_maxs is None:
        lambda_maxs = [1, 5, 9]

    n_cfg = len(lambda_maxs)
    x_offset = 4.0
    target_labels = [0, 1]

    # Generate trajectories
    logger.info(f"Generating Rectified CFG++ trajectories for lambda_max={lambda_maxs}...")
    trajectories = {}
    for target_label in target_labels:
        trajectories[target_label] = {}
        for lambda_max in lambda_maxs:
            traj = sample_euler_rectified_cfg_plusplus_full_trajectory(
                model, n_samples, target_label,
                num_steps=num_steps, lambda_max=lambda_max, gamma=gamma, seed=seed
            )
            trajectories[target_label][lambda_max] = traj

    # Setup figure - 4x size for quality
    fig, axes = plt.subplots(2, n_cfg, figsize=(16 * n_cfg, 24), facecolor='white')
    fig.patch.set_facecolor('white')

    # Setup KDE grid
    x_grid = np.linspace(-8, 8, grid_size * 2)
    y_grid = np.linspace(-4, 4, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Get source and target
    sources = {}
    targets = {}
    for target_label in target_labels:
        sources[target_label] = {lm: trajectories[target_label][lm][0] for lm in lambda_maxs}
        targets[target_label] = {lm: trajectories[target_label][lm][-1] for lm in lambda_maxs}

    n_animation_frames = num_steps + 1
    n_hold_frames = int(hold_end_seconds * fps)
    n_frames = n_animation_frames + n_hold_frames

    # Custom light orange colormap for density
    from matplotlib.colors import LinearSegmentedColormap
    orange_cmap = LinearSegmentedColormap.from_list(
        'orange_density',
        ['white', '#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726'],
        N=256
    )

    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1

    def update(frame_idx):
        actual_frame = min(frame_idx, num_steps)
        t = actual_frame / num_steps

        for row_idx, target_label in enumerate(target_labels):
            for col_idx, lambda_max in enumerate(lambda_maxs):
                ax = axes[row_idx, col_idx]
                ax.clear()
                ax.set_facecolor('white')

                current_samples = trajectories[target_label][lambda_max][actual_frame].copy()

                # Training data in gray (on target/right side)
                train_data_right = data.copy()
                train_data_right[:, 0] += x_offset
                ax.scatter(
                    train_data_right[:, 0],
                    train_data_right[:, 1],
                    s=5,
                    color='#9E9E9E',
                    alpha=0.3,
                    edgecolors='none',
                )

                # Source (left) - BLUE
                source_shifted = sources[target_label][lambda_max].copy()
                source_shifted[:, 0] -= x_offset
                ax.scatter(
                    source_shifted[:, 0],
                    source_shifted[:, 1],
                    s=8,
                    color='#2196F3',
                    alpha=0.6,
                    edgecolors='none',
                )

                # Target (right) - RED
                target_shifted = targets[target_label][lambda_max].copy()
                target_shifted[:, 0] += x_offset
                ax.scatter(
                    target_shifted[:, 0],
                    target_shifted[:, 1],
                    s=8,
                    color='#F44336',
                    alpha=0.6,
                    edgecolors='none',
                )

                # Current flow - LIGHT ORANGE
                data_shifted = current_samples.copy()
                data_shifted[:, 0] += x_offset * (2 * t - 1)

                try:
                    kde = gaussian_kde(data_shifted.T, bw_method=0.2)
                    Z = kde(positions).reshape(grid_size, grid_size * 2)
                    Z_max = Z.max()
                    if Z_max > 0:
                        Z = Z / Z_max
                        levels = np.linspace(0.05, 1.0, 20)
                        ax.contourf(X, Y, Z, levels=levels, cmap=orange_cmap, alpha=0.85)
                except (np.linalg.LinAlgError, ValueError):
                    ax.scatter(
                        data_shifted[:, 0],
                        data_shifted[:, 1],
                        s=8,
                        color='#FFB74D',
                        alpha=0.6,
                        edgecolors='none',
                    )

                # Minimal labels
                if row_idx == 0:
                    ax.set_title(f"λ = {lambda_max}", fontsize=14, fontweight='normal', pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"Class {target_label}", fontsize=12, fontweight='normal')

                # Simple time indicator
                if row_idx == 1:
                    ax.text(0, -3.5, f"t = {t:.2f}", ha='center', fontsize=10, color='#666666')

                # Clean axis
                ax.set_xlim(-8, 8)
                ax.set_ylim(-4, 4)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        return axes.flatten()

    logger.info(f"Creating Rectified CFG++ animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    plt.tight_layout()

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        logger.info(f"Saved: {save_path}")

    plt.close(fig)


def create_cfg_vs_rectified_side_by_side_animation(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    guidance_scale: float = 5.0,
    n_samples: int = 500,
    num_steps: int = 50,
    save_path: Optional[str] = None,
    fps: int = 8,
    dpi: int = 120,
    grid_size: int = 100,
    seed: int = 42,
    gamma: float = 1.0,
    hold_end_seconds: float = 2.0,
):
    """
    Create side-by-side animation comparing Standard CFG vs Rectified CFG++.

    Minimal, clean style:
    - Plain white background
    - Blue source, red target, yellow density
    - Slow, clear animation
    """
    x_offset = 4.0
    target_labels = [0, 1]

    # Generate trajectories for both methods
    logger.info(f"Generating CFG vs Rectified CFG++ trajectories (scale={guidance_scale})...")

    cfg_trajectories = {}
    rectified_trajectories = {}

    for target_label in target_labels:
        cfg_trajectories[target_label] = sample_euler_full_trajectory(
            model, n_samples, target_label,
            num_steps=num_steps, cfg_scale=guidance_scale, seed=seed
        )
        rectified_trajectories[target_label] = sample_euler_rectified_cfg_plusplus_full_trajectory(
            model, n_samples, target_label,
            num_steps=num_steps, lambda_max=guidance_scale, gamma=gamma, seed=seed
        )

    # Setup figure: 2 rows (methods) x 2 columns (classes) - 4x size for quality
    fig, axes = plt.subplots(2, 2, figsize=(40, 32), facecolor='white')
    fig.patch.set_facecolor('white')

    # KDE grid
    x_grid = np.linspace(-8, 8, grid_size * 2)
    y_grid = np.linspace(-4, 4, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    n_animation_frames = num_steps + 1
    n_hold_frames = int(hold_end_seconds * fps)
    n_frames = n_animation_frames + n_hold_frames

    methods = [
        ("Standard CFG", cfg_trajectories),
        ("Rectified CFG++", rectified_trajectories),
    ]

    # Custom light orange colormap
    from matplotlib.colors import LinearSegmentedColormap
    orange_cmap = LinearSegmentedColormap.from_list(
        'orange_density',
        ['white', '#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726'],
        N=256
    )

    def update(frame_idx):
        actual_frame = min(frame_idx, num_steps)
        t = actual_frame / num_steps

        for row_idx, (method_name, trajectories) in enumerate(methods):
            for col_idx, target_label in enumerate(target_labels):
                ax = axes[row_idx, col_idx]
                ax.clear()
                ax.set_facecolor('white')

                current_samples = trajectories[target_label][actual_frame].copy()

                # Training data in gray (on target/right side)
                train_data_right = data.copy()
                train_data_right[:, 0] += x_offset
                ax.scatter(
                    train_data_right[:, 0],
                    train_data_right[:, 1],
                    s=5,
                    color='#9E9E9E',
                    alpha=0.3,
                    edgecolors='none',
                )

                # Source (left) - BLUE
                source = trajectories[target_label][0].copy()
                source[:, 0] -= x_offset
                ax.scatter(
                    source[:, 0],
                    source[:, 1],
                    s=8,
                    color='#2196F3',
                    alpha=0.6,
                    edgecolors='none',
                )

                # Target (right) - RED
                target = trajectories[target_label][-1].copy()
                target[:, 0] += x_offset
                ax.scatter(
                    target[:, 0],
                    target[:, 1],
                    s=8,
                    color='#F44336',
                    alpha=0.6,
                    edgecolors='none',
                )

                # Current flow - LIGHT ORANGE
                data_shifted = current_samples.copy()
                data_shifted[:, 0] += x_offset * (2 * t - 1)

                try:
                    kde = gaussian_kde(data_shifted.T, bw_method=0.2)
                    Z = kde(positions).reshape(grid_size, grid_size * 2)
                    Z_max = Z.max()
                    if Z_max > 0:
                        Z = Z / Z_max
                        levels = np.linspace(0.05, 1.0, 20)
                        ax.contourf(X, Y, Z, levels=levels, cmap=orange_cmap, alpha=0.85)
                except (np.linalg.LinAlgError, ValueError):
                    ax.scatter(
                        data_shifted[:, 0],
                        data_shifted[:, 1],
                        s=8,
                        color='#FFB74D',
                        alpha=0.6,
                        edgecolors='none',
                    )

                # Minimal labels
                if row_idx == 0:
                    ax.set_title(f"Class {target_label}", fontsize=14, fontweight='normal', pad=10)
                if col_idx == 0:
                    ax.set_ylabel(method_name, fontsize=12, fontweight='normal')

                # Simple time indicator (bottom row)
                if row_idx == 1:
                    ax.text(0, -3.5, f"t = {t:.2f}", ha='center', fontsize=10, color='#666666')

                # Clean axis
                ax.set_xlim(-8, 8)
                ax.set_ylim(-4, 4)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        return axes.flatten()

    logger.info(f"Creating CFG vs Rectified CFG++ animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    fig.suptitle(f"Scale = {guidance_scale}", fontsize=14, fontweight='normal', y=0.98)
    plt.tight_layout()

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        logger.info(f"Saved: {save_path}")

    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main visualization function."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.training.seed)

    # Get dataset configuration
    dataset_name = cfg.data.dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    class0_centers = dataset_config["class0_centers"]
    class1_centers = dataset_config["class1_centers"]

    # Determine axis limits based on data centers
    all_centers = class0_centers + class1_centers
    max_coord = max(max(abs(c[0]), abs(c[1])) for c in all_centers) + 1.5
    xlim = (-max_coord, max_coord)
    ylim = (-max_coord, max_coord)

    # Generate data for visualization
    logger.info("Generating data...")
    data, labels = dataset_config["generator"](n_samples=cfg.data.n_samples)

    # Load model
    output_dir = Path(cfg.training.output_dir)
    model_path = output_dir / "flow_model.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    logger.info(f"Loading model from {model_path}")
    model = SimpleFlowNetwork(hidden_dim=cfg.model.hidden_dim)
    model.load(str(model_path))

    # Create visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Training data visualization
    logger.info("Creating training data visualization...")
    plot_training_data(
        data,
        labels,
        save_path=str(viz_dir / "training_data.png"),
        title=dataset_config["title"],
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim,
        ylim=ylim,
    )

    # Load and plot losses if available
    losses_path = output_dir / "losses.npy"
    if losses_path.exists():
        logger.info("Creating loss curve visualization...")
        losses = np.load(losses_path)
        plot_training_loss(losses.tolist(), save_path=str(viz_dir / "training_loss.png"))

    # CFG comparison
    cfg_scales = list(cfg.visualization.cfg_scales)
    n_samples = cfg.visualization.n_samples

    logger.info("Creating CFG comparison visualizations...")
    for target_label in [0, 1]:
        plot_cfg_comparison(
            model,
            data,
            labels,
            target_label=target_label,
            cfg_scales=cfg_scales,
            n_samples=n_samples,
            save_path=str(viz_dir / f"cfg_comparison_class{target_label}.png"),
            class0_centers=class0_centers,
            class1_centers=class1_centers,
            xlim=xlim,
            ylim=ylim,
        )

    # Both classes comparison
    logger.info("Creating both classes CFG comparison...")
    plot_both_classes_cfg(
        model,
        data,
        labels,
        cfg_scales=cfg_scales,
        n_samples=n_samples,
        save_path=str(viz_dir / "both_classes_cfg.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=(-3.5, 3.5),
        ylim=(-3.5, 3.5),
    )

    # Flow evolution
    logger.info("Creating flow evolution visualizations...")
    evolution_cfg_scales = list(cfg.visualization.evolution_cfg_scales)
    for target_label in [0, 1]:
        plot_flow_evolution(
            model,
            data,
            labels,
            cfg_scales=evolution_cfg_scales,
            target_labels=[target_label],
            n_samples=n_samples,
            save_path=str(viz_dir / f"flow_evolution_class{target_label}.png"),
            xlim=(-3.5, 3.5),
            ylim=(-3.5, 3.5),
        )

    # Probability path animation
    logger.info("Creating probability path animation...")
    anim_cfg_scales = list(cfg.visualization.get("animation_cfg_scales", [1, 5, 9]))
    anim_n_samples = cfg.visualization.get("animation_n_samples", 500)
    anim_num_steps = cfg.visualization.get("animation_num_steps", 50)
    anim_fps = cfg.visualization.get("animation_fps", 15)

    create_probability_path_animation(
        model,
        data,
        labels,
        cfg_scales=anim_cfg_scales,
        n_samples=anim_n_samples,
        num_steps=anim_num_steps,
        save_path=str(viz_dir / "probability_path.gif"),
        fps=anim_fps,
    )

    logger.info(f"All visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()
