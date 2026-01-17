"""Visualization script for Flow Matching model."""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from flow_visualizer.data import generate_trex_data
from flow_visualizer.model import FlowMatchingModel, MLPVelocityNet

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
            t_tensor = torch.ones(len(positions), 1, device=device) * t
            velocities = model.velocity_net(pos_tensor, t_tensor).cpu().numpy()

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


@hydra.main(version_base=None, config_path="/home/user/flow-visualizer/conf", config_name="config")
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
    velocity_net = MLPVelocityNet(
        data_dim=cfg.model.data_dim,
        time_embed_dim=cfg.model.time_embed_dim,
        hidden_dims=cfg.model.hidden_dims,
    )

    model_path = Path(cfg.training.output_dir) / "velocity_net.pt"
    velocity_net.load_state_dict(torch.load(model_path, map_location=device))
    model = FlowMatchingModel(velocity_net=velocity_net, device=device)

    logger.info("Model loaded successfully")

    # Generate target data
    logger.info("Generating target data...")
    target_data = generate_trex_data(
        n_samples=cfg.data.n_samples,
        noise=cfg.data.noise,
    )

    # Generate samples
    logger.info("Generating samples...")
    n_vis_samples = min(2000, cfg.data.n_samples)
    generated_samples = model.sample(
        n_samples=n_vis_samples,
        n_steps=cfg.visualization.n_sampling_steps,
        data_dim=cfg.model.data_dim,
    )

    # Generate trajectory
    logger.info("Generating trajectory...")
    trajectory = model.sample_trajectory(
        n_samples=n_vis_samples,
        n_steps=cfg.visualization.n_sampling_steps,
        data_dim=cfg.model.data_dim,
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

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
