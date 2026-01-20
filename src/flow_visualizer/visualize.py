"""Visualization script for Flow Matching model."""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from flow_visualizer.data import CustomDataset
from flow_visualizer.model import FlowMLP


class FlowMatchingModel:
    """Simple wrapper for sampling from FlowMLP."""

    def __init__(self, velocity_net, device="cpu"):
        self.velocity_net = velocity_net.to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples, n_steps=100, data_dim=2):
        self.velocity_net.eval()
        x = torch.randn(n_samples, data_dim, device=self.device)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, time=t)
            x = x + v * dt
        return x.cpu()

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

logger = logging.getLogger(__name__)


def plot_trajectory(
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    steps_to_plot: list[int] = None,
):
    """
    Plot the sampling trajectory showing evolution from noise to data.

    Args:
        trajectory: List of tensors representing the sampling trajectory
        target_data: Target data distribution
        save_path: Path to save the figure
        steps_to_plot: List of step indices to plot (default: evenly spaced)
    """
    if steps_to_plot is None:
        n_plots = min(6, len(trajectory))
        steps_to_plot = np.linspace(0, len(trajectory) - 1, n_plots, dtype=int).tolist()

    n_plots = len(steps_to_plot)
    fig, axes = plt.subplots(2, n_plots // 2, figsize=(3 * n_plots // 2, 6))
    axes = axes.flatten()

    for i, step in enumerate(steps_to_plot):
        ax = axes[i]
        data = trajectory[step].numpy()

        # Plot generated samples
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, label="Generated", color="blue")

        # Plot target distribution (faded)
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            alpha=0.1,
            s=5,
            label="Target",
            color="red",
        )

        t = step / (len(trajectory) - 1)
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Trajectory plot saved to {save_path}")
    plt.close()


def plot_comparison(
    generated_samples: torch.Tensor,
    target_data: np.ndarray,
    save_path: Path,
):
    """
    Plot comparison between generated and target data.

    Args:
        generated_samples: Generated samples from the model
        target_data: Target data distribution
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot target data
    axes[0].scatter(target_data[:, 0], target_data[:, 1], alpha=0.5, s=10, color="red")
    axes[0].set_title("Target Distribution")
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # Plot generated samples
    generated_np = generated_samples.numpy()
    axes[1].scatter(generated_np[:, 0], generated_np[:, 1], alpha=0.5, s=10, color="blue")
    axes[1].set_title("Generated Distribution")
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    # Plot overlay
    axes[2].scatter(
        target_data[:, 0],
        target_data[:, 1],
        alpha=0.3,
        s=10,
        color="red",
        label="Target",
    )
    axes[2].scatter(
        generated_np[:, 0],
        generated_np[:, 1],
        alpha=0.3,
        s=10,
        color="blue",
        label="Generated",
    )
    axes[2].set_title("Overlay")
    axes[2].set_xlim(-1.5, 1.5)
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Comparison plot saved to {save_path}")
    plt.close()


def plot_vector_field(
    model: FlowMatchingModel,
    save_path: Path,
    grid_size: int = 20,
    t_values: list[float] = None,
):
    """
    Plot the learned velocity vector field at different times.

    Args:
        model: Trained FlowMatching model
        save_path: Path to save the figure
        grid_size: Number of grid points in each dimension
        t_values: List of time values to plot (default: [0.0, 0.33, 0.67, 1.0])
    """
    if t_values is None:
        t_values = [0.0, 0.33, 0.67, 1.0]

    n_plots = len(t_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Create grid
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)

    model.velocity_net.eval()
    device = next(model.velocity_net.parameters()).device

    for i, t in enumerate(t_values):
        ax = axes[i]

        # Compute velocity field
        with torch.no_grad():
            pos_tensor = torch.from_numpy(positions).float().to(device)
            t_tensor = torch.ones(len(positions), device=device) * t
            velocities = model.velocity_net(pos_tensor, time=t_tensor).cpu().numpy()

        U = velocities[:, 0].reshape(grid_size, grid_size)
        V = velocities[:, 1].reshape(grid_size, grid_size)

        # Plot vector field
        ax.quiver(X, Y, U, V, alpha=0.6)
        ax.set_title(f"Velocity Field at t = {t:.2f}")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Vector field plot saved to {save_path}")
    plt.close()


def create_flow_animation(
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
):
    """
    Create an animated GIF showing the flow from Gaussian noise to target distribution.

    Args:
        trajectory: List of tensors representing the sampling trajectory
        target_data: Target data distribution
        save_path: Path to save the GIF
        fps: Frames per second for the animation
        dpi: DPI for the output GIF
        subsample: Use every Nth frame to reduce file size (default: 1 = all frames)
    """
    # Subsample trajectory if requested
    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        data = trajectory_subset[frame].numpy()
        t = (frame * subsample) / (len(trajectory) - 1)

        # Plot current distribution
        ax.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.6,
            s=20,
            color="blue",
            edgecolors="darkblue",
            linewidth=0.5,
        )

        # Plot target distribution (faded)
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            alpha=0.1,
            s=5,
            color="red",
        )

        # Add title with time
        ax.set_title(f"Flow Matching: t = {t:.3f}", fontsize=16, fontweight="bold")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    logger.info(f"Creating flow animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Flow animation saved to {save_path}")
    plt.close()


def create_particle_trajectories_animation(
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    n_particles: int = 100,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
    trail_length: int = 10,
):
    """
    Create an animated GIF showing individual particle trajectories through the flow.

    Args:
        trajectory: List of tensors representing the sampling trajectory
        target_data: Target data distribution
        save_path: Path to save the GIF
        n_particles: Number of particle trajectories to visualize
        fps: Frames per second for the animation
        dpi: DPI for the output GIF
        subsample: Use every Nth frame to reduce file size
        trail_length: Number of previous positions to show as trails
    """
    # Subsample trajectory and select particles
    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)

    # Select random particles to track
    n_samples = trajectory_subset[0].shape[0]
    particle_indices = np.random.choice(n_samples, size=n_particles, replace=False)

    # Extract particle paths
    particle_paths = []
    for idx in particle_indices:
        path = np.array([traj[idx].numpy() for traj in trajectory_subset])
        particle_paths.append(path)
    particle_paths = np.array(particle_paths)  # Shape: (n_particles, n_frames, 2)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color map for particles
    colors = plt.cm.viridis(np.linspace(0, 1, n_particles))

    def update(frame):
        ax.clear()
        t = (frame * subsample) / (len(trajectory) - 1)

        # Plot target distribution (very faded)
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            alpha=0.05,
            s=3,
            color="gray",
        )

        # Plot particle trails and current positions
        for i, path in enumerate(particle_paths):
            # Trail
            start_idx = max(0, frame - trail_length)
            trail = path[start_idx : frame + 1]

            if len(trail) > 1:
                ax.plot(
                    trail[:, 0],
                    trail[:, 1],
                    alpha=0.4,
                    linewidth=1.5,
                    color=colors[i],
                )

            # Current position
            ax.scatter(
                path[frame, 0],
                path[frame, 1],
                s=50,
                color=colors[i],
                edgecolors="black",
                linewidth=1,
                zorder=10,
            )

        ax.set_title(
            f"Particle Trajectories: t = {t:.3f}\n{n_particles} particles flowing from noise to data",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    logger.info(f"Creating particle trajectories animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Particle trajectories animation saved to {save_path}")
    plt.close()


def create_density_animation(
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
    grid_size: int = 100,
):
    """
    Create an animated GIF showing the density evolution using heatmaps.

    Args:
        trajectory: List of tensors representing the sampling trajectory
        target_data: Target data distribution
        save_path: Path to save the GIF
        fps: Frames per second for the animation
        dpi: DPI for the output GIF
        subsample: Use every Nth frame to reduce file size
        grid_size: Resolution of the density grid
    """
    from scipy.stats import gaussian_kde

    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Create grid for density estimation
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    def update(frame):
        ax.clear()
        data = trajectory_subset[frame].numpy()
        t = (frame * subsample) / (len(trajectory) - 1)

        # Compute density using KDE
        try:
            kde = gaussian_kde(data.T)
            Z = kde(positions).reshape(grid_size, grid_size)

            # Plot density heatmap
            im = ax.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.8)

            # Overlay scatter plot
            ax.scatter(
                data[:, 0],
                data[:, 1],
                alpha=0.3,
                s=5,
                color="darkblue",
            )

        except np.linalg.LinAlgError:
            # Fallback if KDE fails (e.g., all points identical)
            ax.scatter(
                data[:, 0],
                data[:, 1],
                alpha=0.5,
                s=10,
                color="blue",
            )

        # Plot target distribution outline
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            alpha=0.1,
            s=2,
            color="red",
        )

        ax.set_title(
            f"Density Evolution: t = {t:.3f}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    logger.info(f"Creating density animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Density animation saved to {save_path}")
    plt.close()


def create_vector_field_animation(
    model: FlowMatchingModel,
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    fps: int = 20,
    dpi: int = 100,
    subsample: int = 1,
    grid_size: int = 20,
):
    """
    Create an animated GIF showing the flow with vector field overlay.

    Args:
        model: Trained FlowMatching model
        trajectory: List of tensors representing the sampling trajectory
        target_data: Target data distribution
        save_path: Path to save the GIF
        fps: Frames per second for the animation
        dpi: DPI for the output GIF
        subsample: Use every Nth frame to reduce file size
        grid_size: Resolution of the vector field grid
    """
    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create grid for vector field
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)

    model.velocity_net.eval()
    device = next(model.velocity_net.parameters()).device

    def update(frame):
        ax.clear()
        data = trajectory_subset[frame].numpy()
        t = (frame * subsample) / (len(trajectory) - 1)

        # Compute velocity field at current time
        with torch.no_grad():
            pos_tensor = torch.from_numpy(positions).float().to(device)
            t_tensor = torch.ones(len(positions), device=device) * t
            velocities = model.velocity_net(pos_tensor, time=t_tensor).cpu().numpy()

        U = velocities[:, 0].reshape(grid_size, grid_size)
        V = velocities[:, 1].reshape(grid_size, grid_size)

        # Plot vector field
        ax.quiver(X, Y, U, V, alpha=0.3, scale=20, color="gray")

        # Plot current distribution
        ax.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.6,
            s=15,
            color="blue",
            edgecolors="darkblue",
            linewidth=0.5,
            zorder=10,
        )

        # Plot target distribution (faded)
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            alpha=0.1,
            s=3,
            color="red",
        )

        ax.set_title(
            f"Flow with Velocity Field: t = {t:.3f}\nBlue = Generated | Red = Target | Arrows = Learned Velocity",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    logger.info(f"Creating vector field animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)

    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    logger.info(f"Vector field animation saved to {save_path}")
    plt.close()


def create_trajectory_curvature_animation(
    trajectory: list[torch.Tensor],
    target_data: np.ndarray,
    save_path: Path,
    n_particles: int = 50,
    fps: int = 20,
    dpi: int = 100,
):
    """
    Create an animated GIF showing trajectory curvature with left-right flow layout.

    Layout: Gaussian (source) on left, target on right, with curved particle paths
    flowing between them. Time slider at bottom shows progress.
    Similar to Figure 2 in https://alechelbling.com/blog/rectified-flow/

    Args:
        trajectory: List of tensors representing the sampling trajectory
        target_data: Target data distribution
        save_path: Path to save the GIF
        n_particles: Number of particle trajectories to visualize
        fps: Frames per second for the animation
        dpi: DPI for the output GIF
    """
    n_frames = len(trajectory)
    n_samples = trajectory[0].shape[0]

    # Offsets for left-right layout
    x_offset = 2.5  # Horizontal separation

    # Select random particles to track
    particle_indices = np.random.choice(n_samples, size=n_particles, replace=False)

    # Extract full particle paths and transform to left-right layout
    # Path goes from (x - x_offset, y) at t=0 to (x + x_offset, y) at t=1
    particle_paths = []
    for idx in particle_indices:
        path = []
        for frame_idx, traj in enumerate(trajectory):
            t = frame_idx / (n_frames - 1)
            pt = traj[idx].numpy()
            # Interpolate x position from left (-x_offset) to right (+x_offset)
            x_pos = pt[0] + x_offset * (2 * t - 1)
            path.append([x_pos, pt[1]])
        particle_paths.append(np.array(path))
    particle_paths = np.array(particle_paths)  # Shape: (n_particles, n_frames, 2)

    # Static source (Gaussian) - use ALL samples for denser visualization
    all_source_data = trajectory[0].numpy()
    all_source_shifted = all_source_data.copy()
    all_source_shifted[:, 0] -= x_offset

    # Static target - use ALL samples for denser visualization
    all_target_data = target_data[:n_samples]
    all_target_shifted = all_target_data.copy()
    all_target_shifted[:, 0] += x_offset

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color particles using a colormap
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_particles))

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)

        # Plot static Gaussian source on left (blue) - ALL points
        ax.scatter(
            all_source_shifted[:, 0],
            all_source_shifted[:, 1],
            alpha=0.4,
            s=15,
            color="dodgerblue",
            edgecolors="none",
        )

        # Plot static target on right (red) - ALL points
        ax.scatter(
            all_target_shifted[:, 0],
            all_target_shifted[:, 1],
            alpha=0.4,
            s=15,
            color="crimson",
            edgecolors="none",
        )

        # Draw full trajectory lines (faded) showing curvature
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
            if frame > 0:
                ax.plot(
                    path[: frame + 1, 0],
                    path[: frame + 1, 1],
                    alpha=0.6,
                    linewidth=1.5,
                    color=colors[i],
                )

            # Draw current position
            ax.scatter(
                path[frame, 0],
                path[frame, 1],
                s=30,
                color=colors[i],
                edgecolors="black",
                linewidth=0.5,
                zorder=10,
            )

        ax.set_title("Trajectory Curvature", fontsize=14, fontweight="bold")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-2.8, 2)
        ax.set_aspect("equal")

        # Remove ticks and box
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add time slider at bottom (further down)
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
    """
    Create an animated GIF showing the probability path with left-right flow layout.

    Layout: Gaussian density on left, target density on right, with probability mass
    flowing between them. Time slider at bottom shows progress.
    Similar to Figure 3 in https://alechelbling.com/blog/rectified-flow/

    Args:
        trajectory: List of tensors representing the sampling trajectory
        target_data: Target data distribution
        save_path: Path to save the GIF
        fps: Frames per second for the animation
        dpi: DPI for the output GIF
        subsample: Use every Nth frame to reduce file size
        grid_size: Resolution of the density grid
    """
    from scipy.stats import gaussian_kde

    trajectory_subset = trajectory[::subsample]
    n_frames = len(trajectory_subset)
    n_samples = trajectory_subset[0].shape[0]

    # Offsets for left-right layout
    x_offset = 2.5

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # Create grid for density estimation (wider for left-right layout)
    x = np.linspace(-4.5, 4.5, grid_size * 2)
    y = np.linspace(-2.5, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Precompute static source and target densities
    source_data = trajectory_subset[0].numpy()
    source_shifted = source_data.copy()
    source_shifted[:, 0] -= x_offset

    target_subset = target_data[:n_samples]
    target_shifted = target_subset.copy()
    target_shifted[:, 0] += x_offset

    try:
        source_kde = gaussian_kde(source_shifted.T, bw_method=0.15)
        Z_source = source_kde(positions).reshape(grid_size, grid_size * 2)
    except np.linalg.LinAlgError:
        Z_source = None

    try:
        target_kde = gaussian_kde(target_shifted.T, bw_method=0.15)
        Z_target = target_kde(positions).reshape(grid_size, grid_size * 2)
    except np.linalg.LinAlgError:
        Z_target = None

    def update(frame):
        ax.clear()
        ax.set_facecolor('white')
        t = (frame * subsample) / (len(trajectory) - 1)

        # Transform current data to left-right layout
        data = trajectory_subset[frame].numpy()
        data_shifted = data.copy()
        data_shifted[:, 0] += x_offset * (2 * t - 1)

        # Plot static source density on left (faded blue)
        if Z_source is not None:
            levels_source = np.linspace(0, Z_source.max() * 0.95, 10)
            ax.contourf(X, Y, Z_source, levels=levels_source, cmap="Blues", alpha=0.3)
            ax.contour(X, Y, Z_source, levels=levels_source[::2], colors="blue", alpha=0.2, linewidths=0.5)

        # Plot static target density on right (faded red)
        if Z_target is not None:
            levels_target = np.linspace(0, Z_target.max() * 0.95, 10)
            ax.contourf(X, Y, Z_target, levels=levels_target, cmap="Reds", alpha=0.3)
            ax.contour(X, Y, Z_target, levels=levels_target[::2], colors="red", alpha=0.2, linewidths=0.5)

        # Compute and plot current density (moving from left to right)
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

        ax.set_title("Probability Path", fontsize=14, fontweight="bold")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-2.8, 2.2)
        ax.set_aspect("equal")

        # Remove ticks and box
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add labels for source and target
        ax.text(-x_offset, 1.7, "Source", ha="center", fontsize=11, color="gray", fontweight="bold")
        ax.text(x_offset, 1.7, "Target", ha="center", fontsize=11, color="gray", fontweight="bold")

        # Add time slider at bottom (further down)
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
    """Main visualization function."""
    logger.info("Starting visualization...")

    # Set device
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    logger.info("Loading model...")
    velocity_net = FlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
    )

    model_path = Path(cfg.training.output_dir) / "velocity_net.pt"
    velocity_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = FlowMatchingModel(velocity_net=velocity_net, device=device)

    logger.info("Model loaded successfully")

    # Generate target data
    logger.info("Generating target data...")
    target_dataset = CustomDataset(size=cfg.data.n_samples)
    target_data = target_dataset.data.numpy()

    # Generate samples
    logger.info("Generating samples...")
    n_vis_samples = min(2000, cfg.data.n_samples)
    generated_samples = model.sample(
        n_samples=n_vis_samples,
        n_steps=cfg.visualization.n_sampling_steps,
    )

    # Generate trajectory
    logger.info("Generating trajectory...")
    trajectory = model.sample_trajectory(
        n_samples=n_vis_samples,
        n_steps=cfg.visualization.n_sampling_steps,
    )

    # Create visualizations
    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating comparison plot...")
    plot_comparison(
        generated_samples,
        target_data[:n_vis_samples],
        output_dir / "comparison.png",
    )

    logger.info("Creating trajectory plot...")
    plot_trajectory(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "trajectory.png",
    )

    logger.info("Creating vector field plot...")
    plot_vector_field(
        model,
        output_dir / "vector_field.png",
        grid_size=cfg.visualization.grid_size,
    )

    # Create animations
    logger.info("Creating flow animation...")
    create_flow_animation(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "flow_animation.gif",
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
    )

    logger.info("Creating particle trajectories animation...")
    create_particle_trajectories_animation(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "particle_trajectories.gif",
        n_particles=cfg.visualization.get("n_particles", 100),
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
        trail_length=cfg.visualization.get("trail_length", 10),
    )

    logger.info("Creating density animation...")
    create_density_animation(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "density_animation.gif",
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
        grid_size=cfg.visualization.get("density_grid_size", 100),
    )

    logger.info("Creating vector field animation...")
    create_vector_field_animation(
        model,
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "vector_field_animation.gif",
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
        grid_size=cfg.visualization.get("grid_size", 20),
    )

    logger.info("Creating trajectory curvature animation...")
    create_trajectory_curvature_animation(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "trajectory_curvature.gif",
        n_particles=cfg.visualization.get("n_particles", 50),
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
    )

    logger.info("Creating probability path animation...")
    create_probability_path_animation(
        trajectory,
        target_data[:n_vis_samples],
        output_dir / "probability_path.gif",
        fps=cfg.visualization.get("animation_fps", 20),
        dpi=cfg.visualization.get("animation_dpi", 100),
        subsample=cfg.visualization.get("animation_subsample", 1),
        grid_size=cfg.visualization.get("density_grid_size", 80),
    )

    logger.info("Visualization complete! Generated static plots and animated GIFs.")


if __name__ == "__main__":
    main()
