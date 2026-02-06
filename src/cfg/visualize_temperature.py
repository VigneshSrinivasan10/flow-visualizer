"""Temperature-scaled sampling for CFG Flow Matching model.

Temperature reduction is achieved by dividing the velocity by a temperature factor > 1.
Lower temperatures (higher divisor) produce more concentrated samples.
"""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.stats import gaussian_kde
import torch
from omegaconf import DictConfig

from cfg.data import FaceDataset
from cfg.model import CFGFlowMLP

logger = logging.getLogger(__name__)

# Class colors: left eye (blue), right eye (red)
CLASS_COLORS = ['#1f77b4', '#d62728']
CLASS_NAMES = ['Left Eye', 'Right Eye']


class TemperatureScaledCFGModel:
    """Wrapper for temperature-scaled CFG sampling."""

    def __init__(self, velocity_net, device="cpu"):
        self.velocity_net = velocity_net.to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples, class_labels, n_steps=100, data_dim=2,
               guidance_scale=1.0, temperature=1.0):
        """Sample with CFG and temperature scaling.

        Args:
            temperature: Divide velocity by this factor. >1 = lower temp = more concentrated.
        """
        self.velocity_net.eval()
        x = torch.randn(n_samples, data_dim, device=self.device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net.forward_cfg(x, time=t, class_labels=class_labels,
                                              guidance_scale=guidance_scale)
            # Temperature scaling: divide velocity by temperature factor
            v = v / temperature
            x = x + v * dt

        return x.cpu()

    @torch.no_grad()
    def sample_trajectory(self, n_samples, class_labels, n_steps=100, data_dim=2,
                          guidance_scale=1.0, temperature=1.0):
        """Sample trajectory with CFG and temperature scaling."""
        self.velocity_net.eval()
        x = torch.randn(n_samples, data_dim, device=self.device)
        trajectory = [x.cpu().clone()]
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net.forward_cfg(x, time=t, class_labels=class_labels,
                                              guidance_scale=guidance_scale)
            # Temperature scaling: divide velocity by temperature factor
            v = v / temperature
            x = x + v * dt
            trajectory.append(x.cpu().clone())

        return trajectory


def create_temperature_trajectory_animation(
    trajectory: list[torch.Tensor],
    class_labels: torch.Tensor,
    save_path: Path,
    temperature: float = 1.0,
    guidance_scale: float = 2.0,
    n_particles: int = 50,
    fps: int = 20,
    dpi: int = 100,
):
    """Create trajectory curvature animation with temperature label."""
    n_frames = len(trajectory)
    n_samples = trajectory[0].shape[0]
    labels_np = class_labels.numpy()

    x_offset = 2.5

    particle_indices = np.random.choice(n_samples, size=min(n_particles, n_samples), replace=False)

    particle_paths = []
    particle_classes = []
    for idx in particle_indices:
        path = []
        for frame_idx, traj in enumerate(trajectory):
            t = frame_idx / (n_frames - 1)
            pt = traj[idx].numpy()
            x_pos = pt[0] + x_offset * (2 * t - 1)
            path.append([x_pos, pt[1]])
        particle_paths.append(np.array(path))
        particle_classes.append(labels_np[idx])
    particle_paths = np.array(particle_paths)
    particle_classes = np.array(particle_classes)

    all_source_data = trajectory[0].numpy()
    all_source_shifted = all_source_data.copy()
    all_source_shifted[:, 0] -= x_offset
    source_labels = labels_np

    all_generated_data = trajectory[-1].numpy()
    all_generated_shifted = all_generated_data.copy()
    all_generated_shifted[:, 0] += x_offset

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)

        # Plot static Gaussian source on left
        for c in range(2):
            mask = source_labels == c
            ax.scatter(
                all_source_shifted[mask, 0],
                all_source_shifted[mask, 1],
                alpha=0.4,
                s=15,
                color=CLASS_COLORS[c],
                edgecolors="none",
            )

        # Plot static generated on right
        for c in range(2):
            mask = source_labels == c
            ax.scatter(
                all_generated_shifted[mask, 0],
                all_generated_shifted[mask, 1],
                alpha=0.4,
                s=15,
                color=CLASS_COLORS[c],
                edgecolors="none",
            )

        ax.text(-x_offset, 1.8, "Source Distribution", ha="center", fontsize=11, fontweight="bold")
        ax.text(x_offset, 1.8, "Target Distribution", ha="center", fontsize=11, fontweight="bold")

        # Draw full trajectory lines (faded)
        for i, path in enumerate(particle_paths):
            ax.plot(path[:, 0], path[:, 1], alpha=0.15, linewidth=1, color="gray")

        # Draw trajectory lines up to current frame
        for i, path in enumerate(particle_paths):
            color = CLASS_COLORS[particle_classes[i]]

            if frame > 0:
                ax.plot(path[: frame + 1, 0], path[: frame + 1, 1],
                        alpha=0.6, linewidth=1.5, color=color)

            ax.scatter(path[frame, 0], path[frame, 1], s=30, color=color,
                       edgecolors="black", linewidth=0.5, zorder=10)

        title = f"Temperature = 1/{temperature:.1f} (CFG = {guidance_scale:.0f})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-2.8, 2)
        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Time slider
        slider_y = -2.4
        ax.plot([-3.5, 3.5], [slider_y, slider_y], color="gray", linewidth=2, alpha=0.5)
        slider_x = -3.5 + 7.0 * t
        ax.scatter([slider_x], [slider_y], s=100, color="black", zorder=20)
        ax.text(-3.5, slider_y - 0.35, "t=0", ha="center", fontsize=10)
        ax.text(3.5, slider_y - 0.35, "t=1", ha="center", fontsize=10)
        ax.text(slider_x, slider_y + 0.25, f"t={t:.2f}", ha="center", fontsize=9, fontweight="bold")

        # Legend
        for c in range(2):
            ax.scatter([], [], color=CLASS_COLORS[c], label=CLASS_NAMES[c], s=50)
        ax.legend(loc='upper right', fontsize=9)

    logger.info(f"Creating temperature trajectory animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Temperature trajectory animation saved to {save_path}")
    plt.close()


def create_temperature_probability_path(
    trajectory: list[torch.Tensor],
    class_labels: torch.Tensor,
    save_path: Path,
    temperature: float = 1.0,
    guidance_scale: float = 2.0,
    fps: int = 20,
    dpi: int = 100,
    grid_size: int = 100,
):
    """Create probability path animation showing density evolution."""
    n_frames = len(trajectory)
    n_samples = trajectory[0].shape[0]
    labels_np = class_labels.numpy()

    x_offset = 2.5

    # Static endpoints
    source_data = trajectory[0].numpy()
    source_shifted = source_data.copy()
    source_shifted[:, 0] -= x_offset

    target_data = trajectory[-1].numpy()
    target_shifted = target_data.copy()
    target_shifted[:, 0] += x_offset

    fig, ax = plt.subplots(figsize=(12, 6))

    # Grid for KDE
    x_grid = np.linspace(-5, 5, grid_size)
    y_grid = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)

        # Current positions with left-right offset
        current_data = trajectory[frame].numpy()
        current_shifted = current_data.copy()
        current_shifted[:, 0] += x_offset * (2 * t - 1)

        # Compute KDE for current distribution
        try:
            kde = gaussian_kde(current_shifted.T)
            Z = kde(positions).reshape(X.shape)
        except Exception:
            Z = np.zeros_like(X)

        # Plot density contours
        ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.7)
        ax.contour(X, Y, Z, levels=10, colors='blue', alpha=0.3, linewidths=0.5)

        # Plot static source (faded) on left
        for c in range(2):
            mask = labels_np == c
            ax.scatter(source_shifted[mask, 0], source_shifted[mask, 1],
                       alpha=0.2, s=8, color=CLASS_COLORS[c], edgecolors="none")

        # Plot static target (faded) on right
        for c in range(2):
            mask = labels_np == c
            ax.scatter(target_shifted[mask, 0], target_shifted[mask, 1],
                       alpha=0.2, s=8, color=CLASS_COLORS[c], edgecolors="none")

        # Plot current samples
        for c in range(2):
            mask = labels_np == c
            ax.scatter(current_shifted[mask, 0], current_shifted[mask, 1],
                       alpha=0.5, s=15, color=CLASS_COLORS[c], edgecolors="none")

        ax.text(-x_offset, 2.4, "Source", ha="center", fontsize=11, fontweight="bold")
        ax.text(x_offset, 2.4, "Target", ha="center", fontsize=11, fontweight="bold")

        title = f"Probability Path: Temperature = 1/{temperature:.1f}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Time slider
        slider_y = -2.6
        ax.plot([-3.5, 3.5], [slider_y, slider_y], color="gray", linewidth=2, alpha=0.5)
        slider_x = -3.5 + 7.0 * t
        ax.scatter([slider_x], [slider_y], s=100, color="black", zorder=20)
        ax.text(-3.5, slider_y - 0.3, "t=0", ha="center", fontsize=10)
        ax.text(3.5, slider_y - 0.3, "t=1", ha="center", fontsize=10)
        ax.text(slider_x, slider_y + 0.2, f"t={t:.2f}", ha="center", fontsize=9, fontweight="bold")

    logger.info(f"Creating probability path animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Probability path animation saved to {save_path}")
    plt.close()


def plot_temperature_comparison(
    model: TemperatureScaledCFGModel,
    dataset: FaceDataset,
    temperature_values: list = [1.0, 1.5, 2.0, 3.0],
    guidance_scale: float = 2.0,
    n_samples: int = 2000,
    figsize: tuple = (16, 4),
    save_path: Path = None,
    dpi: int = 150,
):
    """Create comparison grid showing effect of different temperatures."""
    n_cols = len(temperature_values)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    all_data = dataset.data.numpy()
    all_labels = dataset.labels.numpy()

    device = model.device
    n_per_class = n_samples // 2
    class_labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones(n_samples - n_per_class, dtype=torch.long),
    ]).to(device)
    labels_np = class_labels.cpu().numpy()

    for col, temp in enumerate(temperature_values):
        ax = axes[col]

        with torch.no_grad():
            generated = model.sample(
                n_samples=n_samples,
                class_labels=class_labels,
                n_steps=100,
                guidance_scale=guidance_scale,
                temperature=temp,
            ).numpy()

        # Plot training data as gray background
        ax.scatter(all_data[:, 0], all_data[:, 1], alpha=0.15, s=5,
                   color='lightgray', edgecolors='none')

        # Plot generated samples colored by class
        for c in range(2):
            mask = labels_np == c
            ax.scatter(generated[mask, 0], generated[mask, 1], alpha=0.5,
                       s=10, color=CLASS_COLORS[c], edgecolors='none')

        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 2)
        ax.set_aspect('equal')
        ax.set_title(f"T = 1/{temp:.1f}", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Temperature Scaling Comparison (CFG = {guidance_scale})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Temperature comparison plot saved to {save_path}")

    plt.close()
    return fig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main visualization function for temperature-scaled CFG sampling."""
    logger.info("Starting temperature-scaled CFG visualization...")

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    logger.info("Loading model...")
    dataset = FaceDataset(
        n_samples=cfg.data.n_samples,
        left_eye_center=tuple(cfg.data.get('left_eye_center', [-0.5, 0.5])),
        right_eye_center=tuple(cfg.data.get('right_eye_center', [0.5, 0.5])),
        eye_sigma=cfg.data.get('eye_sigma', 0.15),
    )

    # Auto-detect num_classes from checkpoint
    model_path = Path(cfg.training.output_dir) / "cfg_velocity_net.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # class_embedding.weight shape is [num_classes + 1, class_emb_dim]
    num_classes_from_ckpt = checkpoint['class_embedding.weight'].shape[0] - 1
    logger.info(f"Detected {num_classes_from_ckpt} classes from checkpoint")

    velocity_net = CFGFlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
        num_classes=num_classes_from_ckpt,
        class_emb_dim=cfg.model.class_emb_dim,
    )

    velocity_net.load_state_dict(checkpoint)
    model = TemperatureScaledCFGModel(velocity_net=velocity_net, device=device)

    logger.info("Model loaded successfully")

    # Create output directory for temperature experiments
    output_dir = Path(cfg.visualization.output_dir).parent / "temperature_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    n_vis_samples = min(2000, cfg.data.n_samples)
    n_per_class = n_vis_samples // 2
    guidance_scale = cfg.visualization.get("guidance_scale", 2.0)

    # Temperature values to test (dividing velocity by these values)
    temperature_values = [1.0, 1.5, 2.0, 3.0, 5.0]

    # Create class labels
    class_labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones(n_vis_samples - n_per_class, dtype=torch.long),
    ]).to(device)
    class_labels_cpu = class_labels.cpu()

    # 1. Static comparison plot
    logger.info("Creating temperature comparison plot...")
    plot_temperature_comparison(
        model=model,
        dataset=dataset,
        temperature_values=temperature_values,
        guidance_scale=guidance_scale,
        n_samples=n_vis_samples,
        save_path=output_dir / "temperature_comparison.png",
        dpi=150,
    )

    # 2. Generate trajectory and probability path animations for each temperature
    for temp in temperature_values:
        logger.info(f"Generating visualizations for temperature = 1/{temp}...")

        trajectory = model.sample_trajectory(
            n_samples=n_vis_samples,
            class_labels=class_labels,
            n_steps=cfg.visualization.n_sampling_steps,
            guidance_scale=guidance_scale,
            temperature=temp,
        )

        # Trajectory animation
        create_temperature_trajectory_animation(
            trajectory,
            class_labels_cpu,
            output_dir / f"trajectory_temp_{temp:.1f}.gif",
            temperature=temp,
            guidance_scale=guidance_scale,
            n_particles=cfg.visualization.get("n_particles", 100),
            fps=cfg.visualization.get("animation_fps", 20),
            dpi=cfg.visualization.get("animation_dpi", 100),
        )

        # Probability path animation
        create_temperature_probability_path(
            trajectory,
            class_labels_cpu,
            output_dir / f"probability_path_temp_{temp:.1f}.gif",
            temperature=temp,
            guidance_scale=guidance_scale,
            fps=cfg.visualization.get("animation_fps", 20),
            dpi=cfg.visualization.get("animation_dpi", 100),
            grid_size=cfg.visualization.get("density_grid_size", 100),
        )

    logger.info(f"Temperature-scaled visualization complete! Outputs in {output_dir}")


if __name__ == "__main__":
    main()
