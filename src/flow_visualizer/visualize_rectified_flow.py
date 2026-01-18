"""Visualization script for comparing standard and rectified flow."""

import logging
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig, OmegaConf

from flow_visualizer.data import MoonsDataset, CirclesDataset
from flow_visualizer.model import RectifiedFlowModel, MLPVelocityNet

logger = logging.getLogger(__name__)


def plot_trajectory_comparison(
    trajectories_dict: dict[str, List[torch.Tensor]],
    target_data: np.ndarray,
    output_path: Path,
    title: str = "Flow Trajectory Comparison",
) -> None:
    """
    Plot trajectory comparison between different models.

    Args:
        trajectories_dict: Dictionary mapping model names to trajectories
        target_data: Target dataset samples
        output_path: Path to save the plot
        title: Plot title
    """
    n_models = len(trajectories_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, trajectory) in enumerate(trajectories_dict.items()):
        ax = axes[idx]

        # Plot target data
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            c='gray',
            alpha=0.3,
            s=1,
            label='Target Data'
        )

        # Plot trajectory samples at different timesteps
        n_steps = len(trajectory)
        timesteps_to_plot = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]

        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps_to_plot)))

        for i, step in enumerate(timesteps_to_plot):
            samples = trajectory[step].numpy()
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                c=[colors[i]],
                s=10,
                alpha=0.6,
                label=f't={step / (n_steps - 1):.2f}'
            )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(model_name)
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Trajectory comparison saved to {output_path}")


def plot_particle_paths_comparison(
    trajectories_dict: dict[str, List[torch.Tensor]],
    target_data: np.ndarray,
    output_path: Path,
    n_particles: int = 20,
) -> None:
    """
    Plot individual particle paths to show trajectory straightness.

    Args:
        trajectories_dict: Dictionary mapping model names to trajectories
        target_data: Target dataset samples
        output_path: Path to save the plot
        n_particles: Number of particle paths to visualize
    """
    n_models = len(trajectories_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, trajectory) in enumerate(trajectories_dict.items()):
        ax = axes[idx]

        # Plot target data
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            c='gray',
            alpha=0.2,
            s=1,
            label='Target Data'
        )

        # Plot particle paths
        trajectory_array = torch.stack(trajectory).numpy()  # (n_steps, n_samples, 2)

        for particle_idx in range(min(n_particles, trajectory_array.shape[1])):
            path = trajectory_array[:, particle_idx, :]  # (n_steps, 2)

            # Plot path
            ax.plot(
                path[:, 0],
                path[:, 1],
                alpha=0.5,
                linewidth=1,
                color='blue'
            )

            # Mark start and end
            ax.scatter(path[0, 0], path[0, 1], c='red', s=30, marker='o', zorder=10)
            ax.scatter(path[-1, 0], path[-1, 1], c='green', s=30, marker='s', zorder=10)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(model_name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Start (noise)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8, label='End (data)'),
            Line2D([0], [0], color='blue', linewidth=2, label='Trajectory'),
        ]
        ax.legend(handles=legend_elements)

    plt.suptitle("Particle Path Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Particle paths comparison saved to {output_path}")


def create_side_by_side_animation(
    trajectories_dict: dict[str, List[torch.Tensor]],
    target_data: np.ndarray,
    output_path: Path,
    fps: int = 20,
) -> None:
    """
    Create side-by-side animation comparing standard and rectified flow.

    Args:
        trajectories_dict: Dictionary mapping model names to trajectories
        target_data: Target dataset samples
        output_path: Path to save animation
        fps: Frames per second
    """
    n_models = len(trajectories_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    model_names = list(trajectories_dict.keys())
    trajectories = list(trajectories_dict.values())
    n_steps = len(trajectories[0])

    # Initialize scatter plots
    scatters = []
    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        # Plot target data
        ax.scatter(
            target_data[:, 0],
            target_data[:, 1],
            c='gray',
            alpha=0.2,
            s=1,
        )

        # Initialize scatter for generated samples
        scatter = ax.scatter([], [], c='blue', s=10, alpha=0.6)
        scatters.append(scatter)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(model_name)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def update(frame):
        for idx, (scatter, trajectory) in enumerate(zip(scatters, trajectories)):
            samples = trajectory[frame].numpy()
            scatter.set_offsets(samples)

            # Update title with timestep
            axes[idx].set_title(f"{model_names[idx]}\nt = {frame / (n_steps - 1):.2f}")

        return scatters

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000/fps, blit=True)

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()

    logger.info(f"Side-by-side animation saved to {output_path}")


@hydra.main(version_base=None, config_path="/home/user/flow-visualizer/conf", config_name="rectified_flow_config")
def main(cfg: DictConfig) -> None:
    """Main visualization function."""
    logger.info("Visualization Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Create dataset for target samples
    logger.info(f"Loading {cfg.data.dataset_type} dataset...")
    if cfg.data.dataset_type == "moons":
        dataset = MoonsDataset(
            n_samples=cfg.data.n_samples,
            noise=cfg.data.noise,
        )
    elif cfg.data.dataset_type == "circles":
        dataset = CirclesDataset(
            n_samples=cfg.data.n_samples,
            noise=cfg.data.noise,
            factor=cfg.data.factor,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.data.dataset_type}")

    target_data = dataset.data.numpy()

    # Load trained models
    model_dir = Path(cfg.training.output_dir)

    trajectories_dict = {}

    # Load initial (standard) model
    initial_model_path = model_dir / "velocity_net_initial.pt"
    if initial_model_path.exists():
        logger.info("Loading initial flow matching model...")
        velocity_net = MLPVelocityNet(
            data_dim=cfg.model.data_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            hidden_dims=cfg.model.hidden_dims,
        )
        velocity_net.load_state_dict(torch.load(initial_model_path, map_location=device))

        model = RectifiedFlowModel(velocity_net=velocity_net, device=device)

        logger.info("Generating trajectory for initial model...")
        trajectory = model.sample_trajectory(
            n_samples=1000,
            n_steps=cfg.visualization.n_sampling_steps,
            data_dim=cfg.model.data_dim,
        )
        trajectories_dict["Standard Flow"] = trajectory
    else:
        logger.warning(f"Initial model not found at {initial_model_path}")

    # Load reflow models
    for reflow_iter in range(cfg.training.n_reflow_iterations):
        reflow_model_path = model_dir / f"velocity_net_reflow_{reflow_iter + 1}.pt"
        if reflow_model_path.exists():
            logger.info(f"Loading reflow model {reflow_iter + 1}...")
            velocity_net = MLPVelocityNet(
                data_dim=cfg.model.data_dim,
                time_embed_dim=cfg.model.time_embed_dim,
                hidden_dims=cfg.model.hidden_dims,
            )
            velocity_net.load_state_dict(torch.load(reflow_model_path, map_location=device))

            model = RectifiedFlowModel(velocity_net=velocity_net, device=device)

            logger.info(f"Generating trajectory for reflow model {reflow_iter + 1}...")
            trajectory = model.sample_trajectory(
                n_samples=1000,
                n_steps=cfg.visualization.n_sampling_steps,
                data_dim=cfg.model.data_dim,
            )
            trajectories_dict[f"Rectified Flow {reflow_iter + 1}"] = trajectory
        else:
            logger.warning(f"Reflow model {reflow_iter + 1} not found at {reflow_model_path}")

    if not trajectories_dict:
        logger.error("No models found! Please run training first.")
        return

    # Create output directory
    vis_dir = Path(cfg.visualization.output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    logger.info("Creating trajectory comparison plot...")
    plot_trajectory_comparison(
        trajectories_dict,
        target_data,
        vis_dir / "trajectory_comparison.png",
    )

    logger.info("Creating particle paths comparison plot...")
    plot_particle_paths_comparison(
        trajectories_dict,
        target_data,
        vis_dir / "particle_paths_comparison.png",
        n_particles=30,
    )

    logger.info("Creating side-by-side animation...")
    create_side_by_side_animation(
        trajectories_dict,
        target_data,
        vis_dir / "flow_comparison.gif",
        fps=cfg.visualization.animation_fps,
    )

    logger.info("\n" + "="*60)
    logger.info("Visualization complete!")
    logger.info(f"Results saved to {vis_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
