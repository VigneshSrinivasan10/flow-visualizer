"""Visualization functions for distilled flow matching models."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig, OmegaConf
from scipy.stats import gaussian_kde

from flow_gaussians.data import DATASET_CONFIGS
from flow_gaussians.distill_model import DistilledFlowNetwork
from flow_gaussians.model import SimpleFlowNetwork
from flow_gaussians.sampling import classify_samples, sample_euler
from flow_gaussians.sampling_distill import (
    sample_distilled_euler,
    sample_distilled_euler_full_trajectory,
)

logger = logging.getLogger(__name__)


def plot_distillation_loss(losses: List[float], save_path: Optional[str] = None):
    """Plot distillation training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(losses, color="#9C27B0", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Distillation Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_distilled_guidance_comparison(
    model: DistilledFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    guidance_scales: Optional[List[float]] = None,
    n_samples: int = 500,
    save_path: Optional[str] = None,
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    seed: int = 123,
):
    """
    Compare different guidance scales using the distilled model.

    Grid layout: 2 rows (classes) x N columns (guidance scales)
    """
    if guidance_scales is None:
        guidance_scales = [0, 1, 3, 5, 7, 9]

    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    mask_0 = labels == 0
    mask_1 = labels == 1

    for row_idx, target_label in enumerate([0, 1]):
        for col_idx, w in enumerate(guidance_scales):
            ax = axes[row_idx, col_idx]

            # Generate samples using distilled model (single forward pass per step)
            samples = sample_distilled_euler(
                model, n_samples, target_label, w=w, num_steps=100, seed=seed + row_idx
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

            # Compute accuracy
            predicted = classify_samples(samples, class0_centers, class1_centers)
            target_count = np.sum(predicted == target_label)
            accuracy = target_count / n_samples

            # Stats box
            box_color = "#90EE90" if accuracy >= 0.9 else ("#FFFF99" if accuracy >= 0.7 else "#FFB6C1")
            ax.text(
                0.02,
                0.98,
                f"Acc: {accuracy:.1%}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
            )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"w = {w}", fontsize=25)
            if col_idx == 0:
                ax.set_ylabel(f"Class {target_label}", fontsize=25)

    fig.suptitle("Distilled Model: Guidance Scale Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_distilled_vs_teacher_comparison(
    teacher_model: SimpleFlowNetwork,
    student_model: DistilledFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    guidance_scales: Optional[List[float]] = None,
    n_samples: int = 500,
    save_path: Optional[str] = None,
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    seed: int = 123,
):
    """
    Compare teacher (CFG) vs student (distilled) model outputs.

    Grid layout:
    - 4 rows: Teacher Class 0, Teacher Class 1, Student Class 0, Student Class 1
    - N columns: guidance scales
    """
    if guidance_scales is None:
        guidance_scales = [1, 3, 5, 7]

    n_cols = len(guidance_scales)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))

    mask_0 = labels == 0
    mask_1 = labels == 1

    configs = [
        ("Teacher (CFG)", teacher_model, True),
        ("Student (Distilled)", student_model, False),
    ]

    for model_idx, (model_name, model, is_teacher) in enumerate(configs):
        for target_label in [0, 1]:
            row_idx = model_idx * 2 + target_label

            for col_idx, w in enumerate(guidance_scales):
                ax = axes[row_idx, col_idx]

                # Generate samples
                if is_teacher:
                    samples = sample_euler(
                        model, n_samples, target_label, num_steps=100, cfg_scale=w, seed=seed + target_label
                    )
                else:
                    samples = sample_distilled_euler(
                        model, n_samples, target_label, w=w, num_steps=100, seed=seed + target_label
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

                # Compute accuracy
                predicted = classify_samples(samples, class0_centers, class1_centers)
                target_count = np.sum(predicted == target_label)
                accuracy = target_count / n_samples

                # Stats box
                box_color = "#90EE90" if accuracy >= 0.9 else ("#FFFF99" if accuracy >= 0.7 else "#FFB6C1")
                ax.text(
                    0.02,
                    0.98,
                    f"Acc: {accuracy:.1%}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
                )

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

                # Labels
                if row_idx == 0:
                    ax.set_title(f"w = {w}", fontsize=12, fontweight="bold")
                if col_idx == 0:
                    label_text = f"{model_name}\nClass {target_label}"
                    ax.set_ylabel(label_text, fontsize=10, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def create_distilled_trajectory_animation(
    model: DistilledFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    guidance_scales: List[float] = None,
    n_particles: int = 50,
    n_samples: int = 500,
    num_steps: int = 50,
    save_path: Optional[str] = None,
    fps: int = 15,
    dpi: int = 120,
    seed: int = 42,
    hold_end_seconds: float = 2.0,
) -> None:
    """
    Create trajectory animation showing particles flowing for different guidance scales.

    Uses the distilled model (single forward pass per step).
    Grid layout: 2 rows (classes) x N columns (guidance scales w).
    """
    if guidance_scales is None:
        guidance_scales = [1, 5, 9]

    n_cfg = len(guidance_scales)
    x_offset = 4.0
    target_labels = [0, 1]

    np.random.seed(seed)

    # Generate trajectories for each class and guidance scale
    logger.info(f"Generating trajectories for guidance scales {guidance_scales}...")
    trajectories = {}
    for target_label in target_labels:
        trajectories[target_label] = {}
        for w in guidance_scales:
            traj = sample_distilled_euler_full_trajectory(
                model, n_samples, target_label, w=w, num_steps=num_steps, seed=seed
            )
            trajectories[target_label][w] = traj

    # Select particle indices to track
    particle_indices = np.random.choice(n_samples, size=n_particles, replace=False)

    # Get source distributions (t=0)
    sources = {}
    for target_label in target_labels:
        sources[target_label] = {}
        for w in guidance_scales:
            sources[target_label][w] = trajectories[target_label][w][0]

    # Extract particle paths with left-right transformation
    particle_paths = {}
    for target_label in target_labels:
        particle_paths[target_label] = {}
        for w in guidance_scales:
            traj = trajectories[target_label][w]
            n_frames = len(traj)
            paths = []
            for idx in particle_indices:
                path = []
                for frame_idx in range(n_frames):
                    t = frame_idx / (n_frames - 1)
                    pt = traj[frame_idx][idx]
                    x_pos = pt[0] + x_offset * (2 * t - 1)
                    path.append([x_pos, pt[1]])
                paths.append(np.array(path))
            particle_paths[target_label][w] = np.array(paths)

    # Setup figure
    fig, axes = plt.subplots(2, n_cfg, figsize=(5 * n_cfg, 5), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.02, wspace=0.02, top=0.92, bottom=0.05)

    # Orange color for particles
    particle_color = "#FFA726"

    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1

    # Calculate frames
    n_animation_frames = num_steps + 1
    n_hold_frames = int(hold_end_seconds * fps)
    n_frames = n_animation_frames + n_hold_frames

    def update(frame_idx):
        actual_frame = min(frame_idx, num_steps)
        t = actual_frame / num_steps

        for row_idx, target_label in enumerate(target_labels):
            for col_idx, w in enumerate(guidance_scales):
                ax = axes[row_idx, col_idx]
                ax.clear()
                ax.set_facecolor("white")

                paths = particle_paths[target_label][w]

                # Training data on right side
                train_data_right = data.copy()
                train_data_right[:, 0] += x_offset
                ax.scatter(
                    train_data_right[mask_1, 0],
                    train_data_right[mask_1, 1],
                    s=30,
                    color="lightcoral",
                    alpha=0.2,
                    edgecolors="none",
                )
                ax.scatter(
                    train_data_right[mask_0, 0],
                    train_data_right[mask_0, 1],
                    s=30,
                    color="gray",
                    alpha=0.2,
                    edgecolors="none",
                )

                # Source distribution (shifted left) - BLUE
                source_shifted = sources[target_label][w].copy()
                source_shifted[:, 0] -= x_offset
                ax.scatter(
                    source_shifted[:, 0],
                    source_shifted[:, 1],
                    s=40,
                    color="#3498db",
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                )

                # Draw full trajectory lines (faded)
                for i, path in enumerate(paths):
                    ax.plot(
                        path[:, 0],
                        path[:, 1],
                        alpha=0.15,
                        linewidth=1,
                        color="#FFCC80",
                    )

                # Draw orange trails up to current frame
                for i, path in enumerate(paths):
                    if actual_frame > 0:
                        ax.plot(
                            path[: actual_frame + 1, 0],
                            path[: actual_frame + 1, 1],
                            alpha=0.6,
                            linewidth=1.5,
                            color=particle_color,
                        )

                    # Draw current position
                    ax.scatter(
                        path[actual_frame, 0],
                        path[actual_frame, 1],
                        s=30,
                        color=particle_color,
                        edgecolors="black",
                        linewidth=0.5,
                        zorder=10,
                    )

                # Labels
                if row_idx == 0:
                    ax.set_title(f"w = {w}", fontsize=25, fontweight="normal", pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"Class {target_label}", fontsize=25, fontweight="normal")

                # Time indicator
                if row_idx == 1:
                    ax.text(0, -3.5, f"t = {t:.2f}", ha="center", fontsize=20, color="#666666")

                # Clean axis
                ax.set_xlim(-8, 8)
                ax.set_ylim(-4, 4)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        return axes.flatten()

    logger.info(f"Creating distilled trajectory animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        logger.info(f"Saved: {save_path}")

    plt.close(fig)


def create_distilled_probability_path_animation(
    model: DistilledFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    guidance_scales: List[float] = None,
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
    Create animated probability path visualization for distilled model.

    Shows density flow from source (left) to target (right) for different guidance scales.
    """
    if guidance_scales is None:
        guidance_scales = [1, 5, 9]

    n_cfg = len(guidance_scales)
    x_offset = 4.0
    target_labels = [0, 1]

    # Generate trajectories
    logger.info(f"Generating distilled trajectories for guidance scales {guidance_scales}...")
    trajectories = {}
    for target_label in target_labels:
        trajectories[target_label] = {}
        for w in guidance_scales:
            traj = sample_distilled_euler_full_trajectory(
                model, n_samples, target_label, w=w, num_steps=num_steps, seed=seed
            )
            trajectories[target_label][w] = traj

    # Setup figure
    fig, axes = plt.subplots(2, n_cfg, figsize=(5 * n_cfg, 5), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.02, wspace=0.02, top=0.92, bottom=0.05)

    # KDE grid
    x_grid = np.linspace(-8, 8, grid_size * 2)
    y_grid = np.linspace(-4, 4, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Get source and target
    sources = {}
    targets = {}
    for target_label in target_labels:
        sources[target_label] = {w: trajectories[target_label][w][0] for w in guidance_scales}
        targets[target_label] = {w: trajectories[target_label][w][-1] for w in guidance_scales}

    n_animation_frames = num_steps + 1
    n_hold_frames = int(hold_end_seconds * fps)
    n_frames = n_animation_frames + n_hold_frames

    # Orange colormap
    orange_cmap = LinearSegmentedColormap.from_list(
        "orange_density",
        ["white", "#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726"],
        N=256,
    )

    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1

    def update(frame_idx):
        actual_frame = min(frame_idx, num_steps)
        t = actual_frame / num_steps

        for row_idx, target_label in enumerate(target_labels):
            for col_idx, w in enumerate(guidance_scales):
                ax = axes[row_idx, col_idx]
                ax.clear()
                ax.set_facecolor("white")

                current_samples = trajectories[target_label][w][actual_frame].copy()

                # Training data (on target/right side)
                train_data_right = data.copy()
                train_data_right[:, 0] += x_offset
                ax.scatter(
                    train_data_right[mask_1, 0],
                    train_data_right[mask_1, 1],
                    s=30,
                    color="lightcoral",
                    alpha=0.2,
                    edgecolors="none",
                )
                ax.scatter(
                    train_data_right[mask_0, 0],
                    train_data_right[mask_0, 1],
                    s=30,
                    color="gray",
                    alpha=0.2,
                    edgecolors="none",
                )

                # Source (left) - BLUE
                source_shifted = sources[target_label][w].copy()
                source_shifted[:, 0] -= x_offset
                ax.scatter(
                    source_shifted[:, 0],
                    source_shifted[:, 1],
                    s=40,
                    color="#3498db",
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                )

                # Target (right) - BLUE
                target_shifted = targets[target_label][w].copy()
                target_shifted[:, 0] += x_offset
                ax.scatter(
                    target_shifted[:, 0],
                    target_shifted[:, 1],
                    s=40,
                    color="#2196F3",
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                )

                # Current flow - ORANGE
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
                        s=40,
                        color="#FFA726",
                        alpha=0.7,
                        edgecolors="black",
                        linewidths=0.5,
                    )

                # Labels
                if row_idx == 0:
                    ax.set_title(f"w = {w}", fontsize=25, fontweight="normal", pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"Class {target_label}", fontsize=25, fontweight="normal")

                # Time indicator
                if row_idx == 1:
                    ax.text(0, -3.5, f"t = {t:.2f}", ha="center", fontsize=20, color="#666666")

                # Clean axis
                ax.set_xlim(-8, 8)
                ax.set_ylim(-4, 4)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        return axes.flatten()

    logger.info(f"Creating distilled probability path animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        logger.info(f"Saved: {save_path}")

    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="config_distill")
def main(cfg: DictConfig) -> None:
    """Main visualization function for distilled models."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.training.seed)

    # Get dataset configuration
    dataset_name = cfg.data.dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    class0_centers = dataset_config["class0_centers"]
    class1_centers = dataset_config["class1_centers"]

    # Determine axis limits
    all_centers = class0_centers + class1_centers
    max_coord = max(max(abs(c[0]), abs(c[1])) for c in all_centers) + 1.5
    xlim = (-max_coord, max_coord)
    ylim = (-max_coord, max_coord)

    # Generate data for visualization
    logger.info("Generating data...")
    data, labels = dataset_config["generator"](n_samples=cfg.data.n_samples)

    # Load distilled model
    output_dir = Path(cfg.distillation.output_dir)
    model_path = output_dir / "distilled_model.npz"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Distilled model not found at {model_path}. "
            f"Run `uv run fg-distill data.dataset={dataset_name}` first."
        )

    logger.info(f"Loading distilled model from {model_path}")
    student_model = DistilledFlowNetwork(hidden_dim=cfg.model.hidden_dim)
    student_model.load(str(model_path))

    # Load teacher model for comparison
    teacher_dir = Path(cfg.distillation.teacher_dir)
    teacher_path = teacher_dir / "flow_model.npz"

    teacher_model = None
    if teacher_path.exists():
        logger.info(f"Loading teacher model from {teacher_path}")
        teacher_model = SimpleFlowNetwork(hidden_dim=cfg.model.hidden_dim)
        teacher_model.load(str(teacher_path))

    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load and plot distillation losses if available
    losses_path = output_dir / "losses.npy"
    if losses_path.exists():
        logger.info("Creating distillation loss curve...")
        losses = np.load(losses_path)
        plot_distillation_loss(losses.tolist(), save_path=str(viz_dir / "distillation_loss.png"))

    # Guidance scale comparison
    guidance_scales = list(cfg.visualization.guidance_scales)
    n_samples = cfg.visualization.n_samples

    logger.info("Creating guidance scale comparison...")
    plot_distilled_guidance_comparison(
        student_model,
        data,
        labels,
        guidance_scales=guidance_scales,
        n_samples=n_samples,
        save_path=str(viz_dir / "guidance_scale_comparison.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim,
        ylim=ylim,
    )

    # Teacher vs Student comparison
    if teacher_model is not None:
        logger.info("Creating teacher vs student comparison...")
        plot_distilled_vs_teacher_comparison(
            teacher_model,
            student_model,
            data,
            labels,
            guidance_scales=[1, 3, 5, 7],
            n_samples=n_samples,
            save_path=str(viz_dir / "teacher_vs_student.png"),
            class0_centers=class0_centers,
            class1_centers=class1_centers,
            xlim=xlim,
            ylim=ylim,
        )

    # Trajectory animation
    anim_guidance_scales = list(cfg.visualization.animation_guidance_scales)
    anim_n_samples = cfg.visualization.animation_n_samples
    anim_num_steps = cfg.visualization.animation_num_steps
    anim_fps = cfg.visualization.animation_fps

    logger.info("Creating trajectory animation...")
    create_distilled_trajectory_animation(
        student_model,
        data,
        labels,
        guidance_scales=anim_guidance_scales,
        n_samples=anim_n_samples,
        num_steps=anim_num_steps,
        save_path=str(viz_dir / "trajectory_guidance_scales.gif"),
        fps=anim_fps,
    )

    # Probability path animation
    logger.info("Creating probability path animation...")
    create_distilled_probability_path_animation(
        student_model,
        data,
        labels,
        guidance_scales=anim_guidance_scales,
        n_samples=anim_n_samples,
        num_steps=anim_num_steps,
        save_path=str(viz_dir / "probability_path_guidance_scales.gif"),
        fps=anim_fps,
    )

    logger.info(f"All visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()
