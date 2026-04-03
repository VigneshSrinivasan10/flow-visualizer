"""Temperature-scaled sampling for flow Gaussians.

Temperature reduction: v_scaled = v * temperature, where temperature < 1
produces more concentrated samples (sharper distribution).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from flow_gaussians.data import DATASET_CONFIGS
from flow_gaussians.model import SimpleFlowNetwork
from flow_gaussians.sampling import classify_samples

logger = logging.getLogger(__name__)


def sample_euler_temperature(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    cfg_scale: float = 1.0,
    temperature: float = 1.0,
    seed: int | None = None,
    clamp_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Sample with Euler ODE + CFG + temperature scaling.

    Args:
        temperature: Multiply velocity by this. <1 = more concentrated, >1 = more spread.
        clamp_range: If provided, clip x to (lo, hi) after each Euler step.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps

    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)

    for step in range(num_steps):
        t = np.full((n_samples, 1), step * dt)
        v_cond = model.predict(x, t, labels_cond)
        v_uncond = model.predict(x, t, labels_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        v = v * temperature
        x = x + v * dt
        if clamp_range is not None:
            x = np.clip(x, clamp_range[0], clamp_range[1])

    return x


def plot_temperature_comparison(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    temperature_values: List[float],
    cfg_scale: float = 1.0,
    n_samples: int = 500,
    save_path: Optional[str] = None,
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    xlim: Tuple[float, float] = (-3.5, 3.5),
    ylim: Tuple[float, float] = (-3.5, 3.5),
    seed: int = 42,
    dataset_title: str = "",
):
    """Create comparison grid: rows=classes, cols=temperature values."""
    n_cols = len(temperature_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    mask_0 = labels == 0
    mask_1 = labels == 1

    # Pre-generate all samples to compute per-column axis limits
    all_samples = {}
    for row_idx, target_label in enumerate([0, 1]):
        for col_idx, temp in enumerate(temperature_values):
            samples = sample_euler_temperature(
                model, n_samples, target_label,
                num_steps=100, cfg_scale=cfg_scale,
                temperature=temp, seed=seed + row_idx,
            )
            all_samples[(row_idx, col_idx)] = samples

    # Compute per-column limits (shared across both classes)
    col_limits = {}
    for col_idx, temp in enumerate(temperature_values):
        all_col_pts = np.vstack([all_samples[(r, col_idx)] for r in range(2)])
        margin = 0.5
        max_abs = max(
            np.abs(all_col_pts).max(),
            max(max(abs(c[0]), abs(c[1])) for c in (class0_centers or []) + (class1_centers or [])) + 1.0,
        )
        col_limits[col_idx] = (-max_abs - margin, max_abs + margin)

    for row_idx, target_label in enumerate([0, 1]):
        for col_idx, temp in enumerate(temperature_values):
            ax = axes[row_idx, col_idx]
            samples = all_samples[(row_idx, col_idx)]

            # Training data (faded)
            ax.scatter(data[mask_0, 0], data[mask_0, 1], c="gray", alpha=0.15, s=5)
            ax.scatter(data[mask_1, 0], data[mask_1, 1], c="lightcoral", alpha=0.15, s=5)

            # Generated samples
            color = "#3498db" if target_label == 0 else "#e74c3c"
            ax.scatter(
                samples[:, 0], samples[:, 1],
                c=color, edgecolors="black", linewidths=0.5,
                alpha=0.7, s=30,
            )

            # Accuracy
            predicted = classify_samples(samples, class0_centers, class1_centers)
            accuracy = np.sum(predicted == target_label) / n_samples
            box_color = "#90EE90" if accuracy >= 0.9 else ("#FFFF99" if accuracy >= 0.7 else "#FFB6C1")
            ax.text(
                0.02, 0.98, f"Acc: {accuracy:.1%}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
            )

            cl = col_limits[col_idx]
            ax.set_xlim(cl)
            ax.set_ylim(cl)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"T = {temp}", fontsize=25)
            if col_idx == 0:
                ax.set_ylabel(f"Class {target_label}", fontsize=25)

    fig.suptitle(
        f"Temperature Scaling — {dataset_title} (CFG = {cfg_scale})",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_temperature_clamping_comparison(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    temperature_values: List[float],
    clamp_range: tuple[float, float],
    cfg_scale: float = 1.0,
    n_samples: int = 500,
    save_path: Optional[str] = None,
    class0_centers: Optional[List[List[float]]] = None,
    class1_centers: Optional[List[List[float]]] = None,
    seed: int = 42,
    dataset_title: str = "",
):
    """Comparison grid: 4 rows (class0 unclamped, class0 clamped, class1 unclamped, class1 clamped) × N temperature cols."""
    n_cols = len(temperature_values)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 16))

    mask_0 = labels == 0
    mask_1 = labels == 1

    # Row config: (target_label, use_clamping, row_label)
    row_configs = [
        (0, False, "Class 0 — unclamped"),
        (0, True, "Class 0 — clamped"),
        (1, False, "Class 1 — unclamped"),
        (1, True, "Class 1 — clamped"),
    ]

    # Pre-generate all samples
    all_samples = {}
    for row_idx, (target_label, use_clamp, _) in enumerate(row_configs):
        for col_idx, temp in enumerate(temperature_values):
            samples = sample_euler_temperature(
                model, n_samples, target_label,
                num_steps=100, cfg_scale=cfg_scale,
                temperature=temp,
                seed=seed + target_label,
                clamp_range=clamp_range if use_clamp else None,
            )
            all_samples[(row_idx, col_idx)] = samples

    # Compute per-column axis limits (shared across all 4 rows)
    col_limits = {}
    for col_idx, temp in enumerate(temperature_values):
        all_col_pts = np.vstack([all_samples[(r, col_idx)] for r in range(4)])
        margin = 0.5
        max_abs = max(
            np.abs(all_col_pts).max(),
            max(max(abs(c[0]), abs(c[1])) for c in (class0_centers or []) + (class1_centers or [])) + 1.0,
        )
        col_limits[col_idx] = (-max_abs - margin, max_abs + margin)

    for row_idx, (target_label, use_clamp, row_label) in enumerate(row_configs):
        for col_idx, temp in enumerate(temperature_values):
            ax = axes[row_idx, col_idx]
            samples = all_samples[(row_idx, col_idx)]

            # Training data (faded)
            ax.scatter(data[mask_0, 0], data[mask_0, 1], c="gray", alpha=0.15, s=5)
            ax.scatter(data[mask_1, 0], data[mask_1, 1], c="lightcoral", alpha=0.15, s=5)

            # Generated samples
            color = "#3498db" if target_label == 0 else "#e74c3c"
            ax.scatter(
                samples[:, 0], samples[:, 1],
                c=color, edgecolors="black", linewidths=0.5,
                alpha=0.7, s=30,
            )

            # Accuracy
            predicted = classify_samples(samples, class0_centers, class1_centers)
            accuracy = np.sum(predicted == target_label) / n_samples
            box_color = "#90EE90" if accuracy >= 0.9 else ("#FFFF99" if accuracy >= 0.7 else "#FFB6C1")
            ax.text(
                0.02, 0.98, f"Acc: {accuracy:.1%}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8),
            )

            cl = col_limits[col_idx]
            ax.set_xlim(cl)
            ax.set_ylim(cl)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"T = {temp}", fontsize=25)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=14)

    fig.suptitle(
        f"Temperature Clamping Comparison — {dataset_title} (CFG = {cfg_scale})",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate temperature-scaled comparison PNGs for flow Gaussians."""
    logger.info("Starting temperature-scaled visualization...")

    np.random.seed(cfg.training.seed)

    dataset_name = cfg.data.dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    class0_centers = dataset_config["class0_centers"]
    class1_centers = dataset_config["class1_centers"]

    all_centers = class0_centers + class1_centers
    max_coord = max(max(abs(c[0]), abs(c[1])) for c in all_centers) + 1.5
    xlim = (-max_coord, max_coord)
    ylim = (-max_coord, max_coord)

    # Generate data
    data, labels = dataset_config["generator"](n_samples=cfg.data.n_samples)

    # Load model
    output_dir = Path(cfg.training.output_dir)
    model_path = output_dir / "flow_model.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    logger.info(f"Loading model from {model_path}")
    model = SimpleFlowNetwork(hidden_dim=cfg.model.hidden_dim)
    model.load(str(model_path))

    # Output directory
    viz_dir = output_dir / "temperature_scaling"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Temperature values: <1 = more concentrated, 1 = baseline, >1 = more spread
    temperature_values = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0]
    cfg_scale = 0.0
    n_samples = cfg.visualization.get("n_samples", 500)

    # Data-driven clamp bounds (99.5th percentile + margin)
    clamp_bound = np.percentile(np.abs(data), 99.5) + 0.5
    clamp_range = (-clamp_bound, clamp_bound)
    logger.info(f"Clamp range: [{-clamp_bound:.2f}, {clamp_bound:.2f}]")

    logger.info(f"Generating temperature comparison for {dataset_name}...")
    plot_temperature_comparison(
        model, data, labels,
        temperature_values=temperature_values,
        cfg_scale=cfg_scale,
        n_samples=n_samples,
        save_path=str(viz_dir / "temperature_comparison.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim, ylim=ylim,
        dataset_title=dataset_config["title"],
    )

    logger.info(f"Generating clamping comparison for {dataset_name}...")
    plot_temperature_clamping_comparison(
        model, data, labels,
        temperature_values=temperature_values,
        clamp_range=clamp_range,
        cfg_scale=cfg_scale,
        n_samples=n_samples,
        save_path=str(viz_dir / "temperature_clamping_comparison.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        dataset_title=dataset_config["title"],
    )

    logger.info(f"Temperature visualization complete! Output in {viz_dir}")


if __name__ == "__main__":
    main()
