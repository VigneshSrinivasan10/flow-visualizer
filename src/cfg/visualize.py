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

    Shows gaussian on left, generated on right, with 3 representative points showing arrows.
    - Gray arrows: Unconditional velocity
    - Orange arrows: Conditional velocity
    - Purple arrows: Final CFG velocity
    Time slider at bottom.
    """
    n_frames = len(trajectory)
    labels_np = class_labels.numpy()

    x_offset = 2.5

    # Static source (Gaussian) - ALL samples
    all_source_data = trajectory[0].numpy()
    all_source_shifted = all_source_data.copy()
    all_source_shifted[:, 0] -= x_offset

    # Generated endpoints - ALL samples
    all_generated_data = trajectory[-1].numpy()
    all_generated_shifted = all_generated_data.copy()
    all_generated_shifted[:, 0] += x_offset

    # Select one representative point per class (fixed throughout animation)
    representative_indices = []
    for c in range(3):
        class_indices = np.where(labels_np == c)[0]
        if c == 0:  # Blue (left eye) - pick a different point (1/4 through)
            representative_indices.append(class_indices[len(class_indices) // 4])
        else:  # Green and red - keep middle point
            representative_indices.append(class_indices[len(class_indices) // 2])
    representative_indices = np.array(representative_indices)
    rep_classes = labels_np[representative_indices]

    # Build full paths for representative points
    rep_paths = []
    for idx in representative_indices:
        path = []
        for frame_idx, traj in enumerate(trajectory):
            t = frame_idx / (n_frames - 1)
            pt = traj[idx].numpy()
            x_pos = pt[0] + x_offset * (2 * t - 1)
            path.append([x_pos, pt[1]])
        rep_paths.append(np.array(path))
    rep_paths = np.array(rep_paths)  # Shape: (3, n_frames, 2)

    fig, ax = plt.subplots(figsize=(12, 6))

    model.velocity_net.eval()
    device = model.device
    null_class_idx = model.velocity_net.null_class_idx

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)

        # Current positions for representative points
        current_positions = rep_paths[:, frame, :]

        # Get original (non-shifted) positions for velocity computation
        data = trajectory[frame].numpy()
        original_positions = data[representative_indices]

        # Compute velocities for representative points
        with torch.no_grad():
            pos_tensor = torch.from_numpy(original_positions).float().to(device)
            t_tensor = torch.ones(3, device=device) * t
            class_tensor = torch.from_numpy(rep_classes).long().to(device)

            # Conditional velocity
            v_cond = model.velocity_net(pos_tensor, time=t_tensor, class_labels=class_tensor).cpu().numpy()

            # Unconditional velocity
            null_labels = torch.full_like(class_tensor, null_class_idx)
            v_uncond = model.velocity_net(pos_tensor, time=t_tensor, class_labels=null_labels).cpu().numpy()

            # CFG velocity
            v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Add x-offset velocity to account for left-right layout shift
        # The shift is x_offset * (2*t - 1), so dx/dt = 2 * x_offset per unit time
        # Scale to match the visual motion
        shift_velocity = 2 * x_offset
        v_uncond_display = v_uncond.copy()
        v_uncond_display[:, 0] += shift_velocity
        v_cond_display = v_cond.copy()
        v_cond_display[:, 0] += shift_velocity
        v_cfg_display = v_cfg.copy()
        v_cfg_display[:, 0] += shift_velocity

        # Plot static Gaussian source on left - colored by class
        for c in range(3):
            mask = labels_np == c
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
            mask = labels_np == c
            ax.scatter(
                all_generated_shifted[mask, 0],
                all_generated_shifted[mask, 1],
                alpha=0.4,
                s=15,
                color=CLASS_COLORS[c],
                edgecolors="none",
            )

        # Draw full trajectory paths (faded)
        for i, c in enumerate(rep_classes):
            ax.plot(
                rep_paths[i, :, 0],
                rep_paths[i, :, 1],
                alpha=0.2,
                linewidth=2,
                color=CLASS_COLORS[c],
            )

        # Draw trajectory up to current frame
        for i, c in enumerate(rep_classes):
            if frame > 0:
                ax.plot(
                    rep_paths[i, :frame+1, 0],
                    rep_paths[i, :frame+1, 1],
                    alpha=0.6,
                    linewidth=3,
                    color=CLASS_COLORS[c],
                )

        # Current points (large, highlighted)
        for i, c in enumerate(rep_classes):
            ax.scatter(
                current_positions[i, 0],
                current_positions[i, 1],
                s=200,
                color=CLASS_COLORS[c],
                edgecolors='black',
                linewidth=2,
                zorder=15,
            )

        # Plot velocity arrows at current positions (using display velocities with shift)
        scale = 15
        arrow_width = 0.010
        # Unconditional (gray)
        ax.quiver(current_positions[:, 0], current_positions[:, 1],
                  v_uncond_display[:, 0], v_uncond_display[:, 1],
                  alpha=0.8, scale=scale, color='gray', width=arrow_width,
                  zorder=20)
        # Conditional (orange)
        ax.quiver(current_positions[:, 0], current_positions[:, 1],
                  v_cond_display[:, 0], v_cond_display[:, 1],
                  alpha=0.9, scale=scale, color='orange', width=arrow_width,
                  zorder=21)
        # CFG (purple)
        ax.quiver(current_positions[:, 0], current_positions[:, 1],
                  v_cfg_display[:, 0], v_cfg_display[:, 1],
                  alpha=1.0, scale=scale, color='purple', width=arrow_width * 1.3,
                  zorder=22)

        ax.set_title("CFG Vector Field: Gray=Uncond | Orange=Cond | Purple=CFG", fontsize=12, fontweight="bold")
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
