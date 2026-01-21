"""Visualization script for CFG Flow Matching model with class-colored output."""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from omegaconf import DictConfig

from cfg.data import FaceDataset
from cfg.model import CFGFlowMLP

logger = logging.getLogger(__name__)

# Class colors: left eye (blue), right eye (green), mouth (red)
CLASS_COLORS = ['#1f77b4', '#2ca02c', '#d62728']
CLASS_NAMES = ['Left Eye', 'Right Eye', 'Mouth']


class CFGFlowMatchingModel:
    """Wrapper for sampling from CFG FlowMLP."""

    def __init__(self, velocity_net, device="cpu"):
        self.velocity_net = velocity_net.to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples, class_labels, n_steps=100, data_dim=2, guidance_scale=1.0):
        """Sample with CFG."""
        self.velocity_net.eval()
        x = torch.randn(n_samples, data_dim, device=self.device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net.forward_cfg(x, time=t, class_labels=class_labels, guidance_scale=guidance_scale)
            x = x + v * dt

        return x.cpu()

    @torch.no_grad()
    def sample_trajectory(self, n_samples, class_labels, n_steps=100, data_dim=2, guidance_scale=1.0):
        """Sample trajectory with CFG."""
        self.velocity_net.eval()
        x = torch.randn(n_samples, data_dim, device=self.device)
        trajectory = [x.cpu().clone()]
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net.forward_cfg(x, time=t, class_labels=class_labels, guidance_scale=guidance_scale)
            x = x + v * dt
            trajectory.append(x.cpu().clone())

        return trajectory


def create_cfg_trajectory_curvature_animation(
    trajectory: list[torch.Tensor],
    class_labels: torch.Tensor,
    save_path: Path,
    n_particles: int = 50,
    fps: int = 20,
    dpi: int = 100,
):
    """
    Create trajectory curvature animation with left-right flow layout.

    Layout: Gaussian (source) on left, generated on right, with curved particle paths
    flowing between them. Time slider at bottom shows progress.
    Particles colored by class.
    """
    n_frames = len(trajectory)
    n_samples = trajectory[0].shape[0]
    labels_np = class_labels.numpy()

    # Offsets for left-right layout
    x_offset = 2.5

    # Select random particles to track (ensure we get some from each class)
    particle_indices = np.random.choice(n_samples, size=min(n_particles, n_samples), replace=False)

    # Extract full particle paths and transform to left-right layout
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

    # Static source (Gaussian) - ALL samples
    all_source_data = trajectory[0].numpy()
    all_source_shifted = all_source_data.copy()
    all_source_shifted[:, 0] -= x_offset
    source_labels = labels_np

    # Generated endpoints - ALL samples
    all_generated_data = trajectory[-1].numpy()
    all_generated_shifted = all_generated_data.copy()
    all_generated_shifted[:, 0] += x_offset

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)

        # Plot static Gaussian source on left - colored by class
        for c in range(3):
            mask = source_labels == c
            ax.scatter(
                all_source_shifted[mask, 0],
                all_source_shifted[mask, 1],
                alpha=0.4,
                s=15,
                color=CLASS_COLORS[c],
                edgecolors="none",
            )

        # Plot static generated on right - colored by class
        for c in range(3):
            mask = source_labels == c
            ax.scatter(
                all_generated_shifted[mask, 0],
                all_generated_shifted[mask, 1],
                alpha=0.4,
                s=15,
                color=CLASS_COLORS[c],
                edgecolors="none",
            )

        # Draw full trajectory lines (faded)
        for i, path in enumerate(particle_paths):
            ax.plot(
                path[:, 0],
                path[:, 1],
                alpha=0.15,
                linewidth=1,
                color="gray",
            )

        # Draw trajectory lines up to current frame
        for i, path in enumerate(particle_paths):
            cls = particle_classes[i]
            color = CLASS_COLORS[cls]

            if frame > 0:
                ax.plot(
                    path[: frame + 1, 0],
                    path[: frame + 1, 1],
                    alpha=0.6,
                    linewidth=1.5,
                    color=color,
                )

            ax.scatter(
                path[frame, 0],
                path[frame, 1],
                s=30,
                color=color,
                edgecolors="black",
                linewidth=0.5,
                zorder=10,
            )

        ax.set_title("CFG Trajectory Curvature", fontsize=14, fontweight="bold")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-2.8, 2)
        ax.set_aspect("equal")

        # Remove ticks and box
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add time slider at bottom
        slider_y = -2.4
        ax.plot([-3.5, 3.5], [slider_y, slider_y], color="gray", linewidth=2, alpha=0.5)
        slider_x = -3.5 + 7.0 * t
        ax.scatter([slider_x], [slider_y], s=100, color="black", zorder=20)
        ax.text(-3.5, slider_y - 0.35, "t=0", ha="center", fontsize=10)
        ax.text(3.5, slider_y - 0.35, "t=1", ha="center", fontsize=10)
        ax.text(slider_x, slider_y + 0.25, f"t={t:.2f}", ha="center", fontsize=9, fontweight="bold")

        # Add legend
        for c in range(3):
            ax.scatter([], [], color=CLASS_COLORS[c], label=CLASS_NAMES[c], s=50)
        ax.legend(loc='upper right', fontsize=9)

    logger.info(f"Creating CFG trajectory curvature animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"CFG trajectory curvature animation saved to {save_path}")
    plt.close()


def create_cfg_vector_field_animation(
    model: CFGFlowMatchingModel,
    trajectory: list[torch.Tensor],
    class_labels: torch.Tensor,
    save_path: Path,
    guidance_scale: float = 2.0,
    fps: int = 20,
    dpi: int = 100,
    grid_size: int = 8,
):
    """
    Create vector field animation with left-right layout showing uncond/cond/CFG arrows.

    Layout: Gaussian on left, generated on right, with 3 arrow types for one point per class.
    - Gray arrows: Unconditional velocity
    - Orange arrows: Conditional velocity
    - Purple arrows: Final CFG velocity
    Time slider at bottom.
    """
    n_frames = len(trajectory)
    n_samples = trajectory[0].shape[0]
    labels_np = class_labels.numpy()

    x_offset = 2.5

    # Static source and generated
    all_source_data = trajectory[0].numpy()
    all_source_shifted = all_source_data.copy()
    all_source_shifted[:, 0] -= x_offset

    all_generated_data = trajectory[-1].numpy()
    all_generated_shifted = all_generated_data.copy()
    all_generated_shifted[:, 0] += x_offset

    # Select one representative point per class (fixed throughout animation)
    representative_indices = []
    for c in range(3):
        class_indices = np.where(labels_np == c)[0]
        representative_indices.append(class_indices[0])
    representative_indices = np.array(representative_indices)

    fig, ax = plt.subplots(figsize=(12, 6))

    model.velocity_net.eval()
    device = model.device

    null_class_idx = model.velocity_net.null_class_idx

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)

        # Current data position (shifted for left-right layout)
        data = trajectory[frame].numpy()
        data_shifted = data.copy()
        data_shifted[:, 0] += x_offset * (2 * t - 1)

        # Get positions and classes for representative points
        rep_positions = data_shifted[representative_indices]
        rep_classes = labels_np[representative_indices]

        # Compute velocities for representative points
        with torch.no_grad():
            pos_tensor = torch.from_numpy(data[representative_indices]).float().to(device)
            t_tensor = torch.ones(3, device=device) * t
            class_tensor = torch.from_numpy(rep_classes).long().to(device)

            # Conditional velocity
            v_cond = model.velocity_net(pos_tensor, time=t_tensor, class_labels=class_tensor).cpu().numpy()

            # Unconditional velocity
            null_labels = torch.full_like(class_tensor, null_class_idx)
            v_uncond = model.velocity_net(pos_tensor, time=t_tensor, class_labels=null_labels).cpu().numpy()

            # CFG velocity
            v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Plot static source on left
        for c in range(3):
            mask = labels_np == c
            ax.scatter(
                all_source_shifted[mask, 0],
                all_source_shifted[mask, 1],
                alpha=0.3,
                s=12,
                color=CLASS_COLORS[c],
                edgecolors="none",
            )

        # Plot static generated on right
        for c in range(3):
            mask = labels_np == c
            ax.scatter(
                all_generated_shifted[mask, 0],
                all_generated_shifted[mask, 1],
                alpha=0.3,
                s=12,
                color=CLASS_COLORS[c],
                edgecolors="none",
            )

        # Plot current distribution (moving)
        for c in range(3):
            mask = labels_np == c
            ax.scatter(
                data_shifted[mask, 0],
                data_shifted[mask, 1],
                alpha=0.5,
                s=15,
                color=CLASS_COLORS[c],
                edgecolors='white',
                linewidth=0.3,
            )

        # Highlight representative points
        ax.scatter(
            rep_positions[:, 0],
            rep_positions[:, 1],
            s=100,
            color=[CLASS_COLORS[c] for c in rep_classes],
            edgecolors='black',
            linewidth=2,
            zorder=15,
        )

        # Plot vector fields at representative positions (one per class)
        scale = 6
        # Unconditional (gray)
        ax.quiver(rep_positions[:, 0], rep_positions[:, 1],
                  v_uncond[:, 0], v_uncond[:, 1],
                  alpha=0.7, scale=scale, color='gray', width=0.008,
                  zorder=20)
        # Conditional (orange)
        ax.quiver(rep_positions[:, 0], rep_positions[:, 1],
                  v_cond[:, 0], v_cond[:, 1],
                  alpha=0.8, scale=scale, color='orange', width=0.008,
                  zorder=21)
        # CFG (purple) - thicker
        ax.quiver(rep_positions[:, 0], rep_positions[:, 1],
                  v_cfg[:, 0], v_cfg[:, 1],
                  alpha=0.9, scale=scale, color='purple', width=0.010,
                  zorder=22)

        ax.set_title("CFG Vector Field\nGray=Uncond | Orange=Cond | Purple=CFG", fontsize=12, fontweight="bold")
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

    logger.info(f"Creating CFG vector field animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"CFG vector field animation saved to {save_path}")
    plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main visualization function for CFG."""
    logger.info("Starting CFG visualization...")

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    logger.info("Loading model...")
    dataset = FaceDataset(n_samples=cfg.data.n_samples)

    velocity_net = CFGFlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
        num_classes=dataset.num_classes,
        class_emb_dim=cfg.model.class_emb_dim,
    )

    model_path = Path(cfg.training.output_dir) / "cfg_velocity_net.pt"
    velocity_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = CFGFlowMatchingModel(velocity_net=velocity_net, device=device)

    logger.info("Model loaded successfully")

    # Generate samples for each class
    n_vis_samples = min(2000, cfg.data.n_samples)
    n_per_class = n_vis_samples // 3

    # Create class labels for visualization
    class_labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones(n_per_class, dtype=torch.long),
        torch.full((n_vis_samples - 2 * n_per_class,), 2, dtype=torch.long),
    ]).to(device)

    # Generate trajectory with CFG
    logger.info("Generating trajectory with CFG...")
    guidance_scale = cfg.visualization.get("guidance_scale", 2.0)
    trajectory = model.sample_trajectory(
        n_samples=n_vis_samples,
        class_labels=class_labels,
        n_steps=cfg.visualization.n_sampling_steps,
        guidance_scale=guidance_scale,
    )

    # Move class_labels to CPU for visualization
    class_labels_cpu = class_labels.cpu()

    # Create output directory
    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create animations
    logger.info("Creating CFG trajectory curvature animation...")
    create_cfg_trajectory_curvature_animation(
        trajectory,
        class_labels_cpu,
        output_dir / "cfg_trajectory_curvature.gif",
        n_particles=cfg.visualization.get("n_particles", 100),
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
    )

    logger.info("Creating CFG vector field animation...")
    create_cfg_vector_field_animation(
        model,
        trajectory,
        class_labels_cpu,
        output_dir / "cfg_vector_field.gif",
        guidance_scale=guidance_scale,
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        grid_size=cfg.visualization.get("grid_size", 10),
    )

    logger.info("CFG Visualization complete!")


if __name__ == "__main__":
    main()
