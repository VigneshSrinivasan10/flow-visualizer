"""Visualization script for Rectified Flow model."""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import gaussian_kde

from flow_visualizer.data import CustomDataset
from flow_visualizer.model import FlowMLP

logger = logging.getLogger(__name__)


class FlowMatchingModel:
    """Simple wrapper for sampling from FlowMLP."""

    def __init__(self, velocity_net, device="cpu"):
        self.velocity_net = velocity_net.to(device)
        self.device = device

    @torch.no_grad()
    def sample_trajectory(self, n_samples, n_steps=100, data_dim=2):
        self.velocity_net.eval()
        x = torch.randn(n_samples, data_dim, device=self.device)
        trajectory = [x.cpu().clone()]
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, time=t)
            x = x + v * dt
            trajectory.append(x.cpu().clone())
        return trajectory


def create_trajectory_curvature_animation(
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    n_particles: int = 50,
    fps: int = 20,
    dpi: int = 100,
):
    """Create trajectory curvature animation with left-right flow layout."""
    n_frames = len(trajectory)
    n_samples = trajectory[0].shape[0]

    x_offset = 2.5

    particle_indices = np.random.choice(n_samples, size=n_particles, replace=False)

    particle_paths = []
    for idx in particle_indices:
        path = []
        for frame_idx, traj in enumerate(trajectory):
            t = frame_idx / (n_frames - 1)
            pt = traj[idx].numpy()
            x_pos = pt[0] + x_offset * (2 * t - 1)
            path.append([x_pos, pt[1]])
        particle_paths.append(np.array(path))
    particle_paths = np.array(particle_paths)

    all_source_data = trajectory[0].numpy()
    all_source_shifted = all_source_data.copy()
    all_source_shifted[:, 0] -= x_offset

    all_target_data = target_data[:n_samples]
    all_target_shifted = all_target_data.copy()
    all_target_shifted[:, 0] += x_offset

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_particles))

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)

        ax.scatter(
            all_source_shifted[:, 0],
            all_source_shifted[:, 1],
            alpha=0.4,
            s=15,
            color="dodgerblue",
            edgecolors="none",
        )

        ax.scatter(
            all_target_shifted[:, 0],
            all_target_shifted[:, 1],
            alpha=0.4,
            s=15,
            color="crimson",
            edgecolors="none",
        )

        for i, path in enumerate(particle_paths):
            ax.plot(
                path[:, 0],
                path[:, 1],
                alpha=0.15,
                linewidth=1,
                color="gray",
            )

        for i, path in enumerate(particle_paths):
            if frame > 0:
                ax.plot(
                    path[: frame + 1, 0],
                    path[: frame + 1, 1],
                    alpha=0.6,
                    linewidth=1.5,
                    color=colors[i],
                )

            ax.scatter(
                path[frame, 0],
                path[frame, 1],
                s=30,
                color=colors[i],
                edgecolors="black",
                linewidth=0.5,
                zorder=10,
            )

        ax.set_title("Rectified Flow: Trajectory Curvature", fontsize=14, fontweight="bold")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-2.8, 2)
        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        slider_y = -2.4
        ax.plot([-3.5, 3.5], [slider_y, slider_y], color="gray", linewidth=2, alpha=0.5)
        slider_x = -3.5 + 7.0 * t
        ax.scatter([slider_x], [slider_y], s=100, color="black", zorder=20)
        ax.text(-3.5, slider_y - 0.35, "t=0", ha="center", fontsize=10)
        ax.text(3.5, slider_y - 0.35, "t=1", ha="center", fontsize=10)
        ax.text(slider_x, slider_y + 0.25, f"t={t:.2f}", ha="center", fontsize=9, fontweight="bold")

    logger.info(f"Creating trajectory curvature animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Trajectory curvature animation saved to {save_path}")
    plt.close()


def create_probability_path_animation(
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
    grid_size: int = 80,
):
    """Create probability path animation with left-right flow layout."""
    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)
    n_samples = trajectory_subset[0].shape[0]

    x_offset = 2.5

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    x = np.linspace(-4.5, 4.5, grid_size * 2)
    y = np.linspace(-2.5, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    all_source_data = trajectory_subset[0].numpy()
    all_source_shifted = all_source_data.copy()
    all_source_shifted[:, 0] -= x_offset

    all_target_data = target_data[:n_samples]
    all_target_shifted = all_target_data.copy()
    all_target_shifted[:, 0] += x_offset

    def update(frame):
        ax.clear()
        ax.set_facecolor('white')
        t = (frame * subsample) / (len(trajectory) - 1)

        data = trajectory_subset[frame].numpy()
        data_shifted = data.copy()
        data_shifted[:, 0] += x_offset * (2 * t - 1)

        ax.scatter(
            all_source_shifted[:, 0],
            all_source_shifted[:, 1],
            alpha=0.4,
            s=15,
            color="dodgerblue",
            edgecolors="none",
        )

        ax.scatter(
            all_target_shifted[:, 0],
            all_target_shifted[:, 1],
            alpha=0.4,
            s=15,
            color="crimson",
            edgecolors="none",
        )

        try:
            kde = gaussian_kde(data_shifted.T, bw_method=0.15)
            Z = kde(positions).reshape(grid_size, grid_size * 2)

            levels = np.linspace(0, Z.max() * 0.95, 15)
            ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.9)
            ax.contour(X, Y, Z, levels=levels[::2], colors="darkblue", alpha=0.3, linewidths=0.5)

        except np.linalg.LinAlgError:
            ax.scatter(
                data_shifted[:, 0],
                data_shifted[:, 1],
                alpha=0.5,
                s=10,
                color="blue",
            )

        ax.set_title("Rectified Flow: Probability Path", fontsize=14, fontweight="bold")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-2.8, 2.2)
        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(-x_offset, 1.7, "Source", ha="center", fontsize=11, color="gray", fontweight="bold")
        ax.text(x_offset, 1.7, "Target", ha="center", fontsize=11, color="gray", fontweight="bold")

        slider_y = -2.3
        ax.plot([-3.5, 3.5], [slider_y, slider_y], color="gray", linewidth=2, alpha=0.5)
        slider_x = -3.5 + 7.0 * t
        ax.scatter([slider_x], [slider_y], s=100, color="black", zorder=20)
        ax.text(-3.5, slider_y - 0.35, "t=0", ha="center", fontsize=10)
        ax.text(3.5, slider_y - 0.35, "t=1", ha="center", fontsize=10)
        ax.text(slider_x, slider_y + 0.25, f"t={t:.2f}", ha="center", fontsize=9, fontweight="bold")

    logger.info(f"Creating probability path animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Probability path animation saved to {save_path}")
    plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main visualization function for rectified flow."""
    logger.info("Starting rectified flow visualization...")

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info("Loading rectified flow model...")
    velocity_net = FlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
    )

    model_path = Path(cfg.training.output_dir) / "velocity_net.pt"
    velocity_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = FlowMatchingModel(velocity_net=velocity_net, device=device)

    logger.info("Model loaded successfully")

    logger.info("Generating target data...")
    target_dataset = CustomDataset(size=cfg.data.n_samples)
    target_data = target_dataset.data.numpy()

    logger.info("Generating trajectory...")
    n_vis_samples = min(2000, cfg.data.n_samples)
    trajectory = model.sample_trajectory(
        n_samples=n_vis_samples,
        n_steps=cfg.visualization.n_sampling_steps,
    )

    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating trajectory curvature animation...")
    create_trajectory_curvature_animation(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "rectified_flow_trajectory_curvature.gif",
        n_particles=cfg.visualization.get("n_particles", 50),
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
    )

    logger.info("Creating probability path animation...")
    create_probability_path_animation(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "rectified_flow_probability_path.gif",
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
        grid_size=cfg.visualization.get("density_grid_size", 80),
    )

    logger.info("Rectified flow visualization complete!")


if __name__ == "__main__":
    main()
