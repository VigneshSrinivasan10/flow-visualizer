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


def create_cfg_flow_animation(
    trajectory: list[torch.Tensor],
    class_labels: torch.Tensor,
    target_data: np.ndarray,
    target_labels: np.ndarray,
    save_path: Path,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
):
    """Create flow animation with class-specific colors."""
    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)

    fig, ax = plt.subplots(figsize=(8, 8))
    labels_np = class_labels.numpy()

    def update(frame):
        ax.clear()
        data = trajectory_subset[frame].numpy()
        t = (frame * subsample) / (len(trajectory) - 1)

        # Plot generated samples by class
        for c in range(3):
            mask = labels_np == c
            ax.scatter(
                data[mask, 0],
                data[mask, 1],
                alpha=0.6,
                s=20,
                color=CLASS_COLORS[c],
                edgecolors='white',
                linewidth=0.3,
                label=CLASS_NAMES[c] if frame == 0 else None,
            )

        # Plot target distribution (faded)
        for c in range(3):
            mask = target_labels == c
            ax.scatter(
                target_data[mask, 0],
                target_data[mask, 1],
                alpha=0.1,
                s=5,
                color=CLASS_COLORS[c],
            )

        ax.set_title(f"CFG Flow Matching: t = {t:.3f}", fontsize=16, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if frame == 0:
            ax.legend(loc='upper right')

    logger.info(f"Creating CFG flow animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"CFG flow animation saved to {save_path}")
    plt.close()


def create_cfg_particle_trajectories_animation(
    trajectory: list[torch.Tensor],
    class_labels: torch.Tensor,
    target_data: np.ndarray,
    target_labels: np.ndarray,
    save_path: Path,
    n_particles: int = 100,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
    trail_length: int = 10,
):
    """Create particle trajectories animation with class-specific colors."""
    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)
    labels_np = class_labels.numpy()

    n_samples = trajectory_subset[0].shape[0]
    particle_indices = np.random.choice(n_samples, size=min(n_particles, n_samples), replace=False)

    particle_paths = []
    particle_classes = []
    for idx in particle_indices:
        path = np.array([traj[idx].numpy() for traj in trajectory_subset])
        particle_paths.append(path)
        particle_classes.append(labels_np[idx])
    particle_paths = np.array(particle_paths)
    particle_classes = np.array(particle_classes)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        t = (frame * subsample) / (len(trajectory) - 1)

        # Plot target distribution (very faded)
        for c in range(3):
            mask = target_labels == c
            ax.scatter(
                target_data[mask, 0],
                target_data[mask, 1],
                alpha=0.05,
                s=3,
                color=CLASS_COLORS[c],
            )

        # Plot particle trails and current positions
        for i, (path, cls) in enumerate(zip(particle_paths, particle_classes)):
            color = CLASS_COLORS[cls]

            start_idx = max(0, frame - trail_length)
            trail = path[start_idx : frame + 1]

            if len(trail) > 1:
                ax.plot(
                    trail[:, 0],
                    trail[:, 1],
                    alpha=0.4,
                    linewidth=1.5,
                    color=color,
                )

            ax.scatter(
                path[frame, 0],
                path[frame, 1],
                s=50,
                color=color,
                edgecolors="black",
                linewidth=1,
                zorder=10,
            )

        ax.set_title(
            f"CFG Particle Trajectories: t = {t:.3f}\n{len(particle_indices)} particles",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Add legend
        for c in range(3):
            ax.scatter([], [], color=CLASS_COLORS[c], label=CLASS_NAMES[c], s=50)
        ax.legend(loc='upper right')

    logger.info(f"Creating CFG particle trajectories animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"CFG particle trajectories animation saved to {save_path}")
    plt.close()


def create_cfg_vector_field_animation(
    model: CFGFlowMatchingModel,
    trajectory: list[torch.Tensor],
    class_labels: torch.Tensor,
    target_data: np.ndarray,
    target_labels: np.ndarray,
    save_path: Path,
    guidance_scale: float = 2.0,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
    grid_size: int = 12,
):
    """
    Create vector field animation showing conditional, unconditional, and CFG arrows.

    - Gray arrows: Unconditional velocity
    - Orange arrows: Conditional velocity
    - Purple arrows: Final CFG velocity (combination)
    """
    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)
    labels_np = class_labels.numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create grid for vector field
    x = np.linspace(-1.3, 1.3, grid_size)
    y = np.linspace(-1.3, 1.3, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)

    model.velocity_net.eval()
    device = model.device

    # Use class 0 (left eye) for grid visualization
    grid_class_labels = torch.zeros(len(positions), dtype=torch.long, device=device)
    null_class_idx = model.velocity_net.null_class_idx

    def update(frame):
        ax.clear()
        data = trajectory_subset[frame].numpy()
        t = (frame * subsample) / (len(trajectory) - 1)

        # Compute velocities at current time
        with torch.no_grad():
            pos_tensor = torch.from_numpy(positions).float().to(device)
            t_tensor = torch.ones(len(positions), device=device) * t

            # Conditional velocity
            v_cond = model.velocity_net(pos_tensor, time=t_tensor, class_labels=grid_class_labels).cpu().numpy()

            # Unconditional velocity
            null_labels = torch.full_like(grid_class_labels, null_class_idx)
            v_uncond = model.velocity_net(pos_tensor, time=t_tensor, class_labels=null_labels).cpu().numpy()

            # CFG velocity
            v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        U_uncond = v_uncond[:, 0].reshape(grid_size, grid_size)
        V_uncond = v_uncond[:, 1].reshape(grid_size, grid_size)
        U_cond = v_cond[:, 0].reshape(grid_size, grid_size)
        V_cond = v_cond[:, 1].reshape(grid_size, grid_size)
        U_cfg = v_cfg[:, 0].reshape(grid_size, grid_size)
        V_cfg = v_cfg[:, 1].reshape(grid_size, grid_size)

        # Plot vector fields with different colors
        # Unconditional (gray)
        ax.quiver(X, Y, U_uncond, V_uncond, alpha=0.4, scale=15, color='gray', width=0.004,
                  label='Unconditional')
        # Conditional (orange)
        ax.quiver(X, Y, U_cond, V_cond, alpha=0.5, scale=15, color='orange', width=0.004,
                  label='Conditional')
        # CFG (purple) - thicker
        ax.quiver(X, Y, U_cfg, V_cfg, alpha=0.7, scale=15, color='purple', width=0.006,
                  label=f'CFG (w={guidance_scale})')

        # Plot current distribution
        for c in range(3):
            mask = labels_np == c
            ax.scatter(
                data[mask, 0],
                data[mask, 1],
                alpha=0.5,
                s=12,
                color=CLASS_COLORS[c],
                edgecolors='white',
                linewidth=0.3,
            )

        # Plot target distribution (faded)
        for c in range(3):
            mask = target_labels == c
            ax.scatter(
                target_data[mask, 0],
                target_data[mask, 1],
                alpha=0.08,
                s=3,
                color=CLASS_COLORS[c],
            )

        ax.set_title(
            f"CFG Vector Field: t = {t:.3f}\nGray=Uncond | Orange=Cond | Purple=CFG",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Add legend for arrows
        ax.quiver([], [], [], [], color='gray', label='Unconditional')
        ax.quiver([], [], [], [], color='orange', label='Conditional')
        ax.quiver([], [], [], [], color='purple', label=f'CFG (w={guidance_scale})')
        ax.legend(loc='upper right', fontsize=9)

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

    # Get target data with labels
    target_data = dataset.data.numpy()
    target_labels = dataset.labels.numpy()

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
    logger.info("Creating CFG flow animation...")
    create_cfg_flow_animation(
        trajectory,
        class_labels_cpu,
        target_data,
        target_labels,
        output_dir / "cfg_flow_animation.gif",
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
    )

    logger.info("Creating CFG particle trajectories animation...")
    create_cfg_particle_trajectories_animation(
        trajectory,
        class_labels_cpu,
        target_data,
        target_labels,
        output_dir / "cfg_particle_trajectories.gif",
        n_particles=cfg.visualization.get("n_particles", 100),
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
        trail_length=cfg.visualization.get("trail_length", 10),
    )

    logger.info("Creating CFG vector field animation...")
    create_cfg_vector_field_animation(
        model,
        trajectory,
        class_labels_cpu,
        target_data,
        target_labels,
        output_dir / "cfg_vector_field.gif",
        guidance_scale=guidance_scale,
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
        grid_size=cfg.visualization.get("grid_size", 12),
    )

    logger.info("CFG Visualization complete!")


if __name__ == "__main__":
    main()
