"""
Visualization script for Rectified CFG++ comparison.

Creates side-by-side comparisons of Standard CFG, CFG++, and Rectified CFG++
showing how the different methods handle high guidance scales.
"""

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
from cfg.rectified_cfgpp import RectifiedCFGPlusPlusSampler, GuidanceMethod

logger = logging.getLogger(__name__)

# Colors
CLASS_COLORS = ['#1f77b4', '#d62728']
CLASS_NAMES = ['Left Eye', 'Right Eye']
METHOD_COLORS = {
    'cfg': '#E74C3C',
    'cfg++': '#9B59B6', 
    'rectified_cfg++': '#3498DB'
}
METHOD_NAMES = {
    'cfg': 'Standard CFG',
    'cfg++': 'CFG++',
    'rectified_cfg++': 'Rectified CFG++'
}


def create_method_comparison_animation(
    sampler: RectifiedCFGPlusPlusSampler,
    dataset: FaceDataset,
    class_labels: torch.Tensor,
    guidance_scale: float = 5.0,
    n_particles: int = 80,
    n_steps: int = 50,
    save_path: Path = None,
    fps: int = 15,
    dpi: int = 100,
):
    """
    Create side-by-side animation comparing CFG, CFG++, and Rectified CFG++.
    """
    methods = ['cfg', 'cfg++', 'rectified_cfg++']
    n_samples = len(class_labels)
    labels_np = class_labels.numpy()
    
    # Use same initial noise for fair comparison
    torch.manual_seed(42)
    initial_noise = torch.randn(n_samples, 2)
    
    # Get trajectories for each method
    trajectories = {}
    for method in methods:
        torch.manual_seed(42)  # Reset for same noise
        x = initial_noise.clone().to(sampler.device)
        class_labels_device = class_labels.to(sampler.device)
        
        traj = [x.cpu().numpy()]
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.ones(n_samples, device=sampler.device) * (step / n_steps)
            v_guided, _, _, _ = sampler.get_velocity(x, t, class_labels_device, guidance_scale, method)
            x = x + v_guided * dt
            traj.append(x.cpu().numpy())
            
        trajectories[method] = np.stack(traj)
    
    # Get target data for reference
    target_data = dataset.data.numpy()
    target_labels = dataset.labels.numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Select particles to visualize
    particle_indices = np.random.choice(n_samples, size=min(n_particles, n_samples), replace=False)
    
    def animate(frame):
        for ax, method in zip(axes, methods):
            ax.clear()
            
            # Plot target distribution (faded)
            for c in range(2):
                mask = target_labels == c
                ax.scatter(
                    target_data[mask, 0], target_data[mask, 1],
                    c=CLASS_COLORS[c], s=5, alpha=0.15
                )
            
            # Current positions
            traj = trajectories[method]
            pos = traj[frame]
            
            # Plot particles colored by class
            for c in range(2):
                mask = labels_np == c
                ax.scatter(
                    pos[mask, 0], pos[mask, 1],
                    c=METHOD_COLORS[method], s=20, alpha=0.7
                )
            
            # Trails for selected particles
            if frame > 0:
                for idx in particle_indices:
                    trail = traj[:frame+1, idx, :]
                    ax.plot(trail[:, 0], trail[:, 1], c=METHOD_COLORS[method], alpha=0.2, linewidth=0.5)
            
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
            ax.set_title(f"{METHOD_NAMES[method]}\n(scale={guidance_scale})", fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.suptitle(f"Method Comparison | t = {frame/n_steps:.2f}", fontsize=12)
        
        return []
    
    logger.info(f"Creating method comparison animation with {n_steps + 1} frames...")
    anim = FuncAnimation(fig, animate, frames=n_steps + 1, interval=1000 / fps)
    
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        logger.info(f"Method comparison animation saved to {save_path}")
    
    plt.close()
    return anim


def create_velocity_magnitude_plot(
    sampler: RectifiedCFGPlusPlusSampler,
    class_labels: torch.Tensor,
    guidance_scales: list = [1.0, 3.0, 7.0, 15.0],
    n_steps: int = 50,
    save_path: Path = None,
    dpi: int = 150,
):
    """
    Create plot showing velocity magnitudes over time for different methods and scales.
    Shows how Rectified CFG++ prevents magnitude explosion.
    """
    methods = ['cfg', 'cfg++', 'rectified_cfg++']
    n_samples = len(class_labels)
    
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(4*len(guidance_scales), 4))
    if len(guidance_scales) == 1:
        axes = [axes]
    
    for ax, scale in zip(axes, guidance_scales):
        torch.manual_seed(42)
        initial_noise = torch.randn(n_samples, 2)
        
        for method in methods:
            x = initial_noise.clone().to(sampler.device)
            class_labels_device = class_labels.to(sampler.device)
            
            magnitudes = []
            dt = 1.0 / n_steps
            
            for step in range(n_steps):
                t = torch.ones(n_samples, device=sampler.device) * (step / n_steps)
                v_guided, _, _, _ = sampler.get_velocity(x, t, class_labels_device, scale, method)
                
                mag = torch.norm(v_guided, dim=-1).mean().item()
                magnitudes.append(mag)
                
                x = x + v_guided * dt
            
            ax.plot(np.linspace(0, 1, n_steps), magnitudes, 
                   label=METHOD_NAMES[method], color=METHOD_COLORS[method], linewidth=2)
        
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Mean Velocity Magnitude")
        ax.set_title(f"CFG Scale = {scale}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Velocity Magnitude Over Time", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Velocity magnitude plot saved to {save_path}")
    
    plt.close()


def create_final_distribution_comparison(
    sampler: RectifiedCFGPlusPlusSampler,
    dataset: FaceDataset,
    class_labels: torch.Tensor,
    guidance_scales: list = [1.0, 5.0, 15.0],
    n_steps: int = 100,
    save_path: Path = None,
    dpi: int = 150,
):
    """
    Create plot showing final sample distributions for different methods and scales.
    """
    methods = ['cfg', 'cfg++', 'rectified_cfg++']
    n_samples = len(class_labels)
    labels_np = class_labels.numpy()
    
    # Get target data
    target_data = dataset.data.numpy()
    target_labels = dataset.labels.numpy()
    
    fig, axes = plt.subplots(len(methods), len(guidance_scales), 
                             figsize=(4*len(guidance_scales), 4*len(methods)))
    
    for i, method in enumerate(methods):
        for j, scale in enumerate(guidance_scales):
            ax = axes[i, j]
            
            # Plot target distribution (faded)
            for c in range(2):
                mask = target_labels == c
                ax.scatter(
                    target_data[mask, 0], target_data[mask, 1],
                    c=CLASS_COLORS[c], s=3, alpha=0.15
                )
            
            # Sample with this method
            torch.manual_seed(42)
            samples = sampler.sample(
                n_samples, class_labels, n_steps=n_steps,
                guidance_scale=scale, method=method
            ).numpy()
            
            # Plot samples colored by class
            for c in range(2):
                mask = labels_np == c
                ax.scatter(
                    samples[mask, 0], samples[mask, 1],
                    c='black', s=8, alpha=0.6
                )
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(f"Scale = {scale}", fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(METHOD_NAMES[method], fontsize=11, fontweight='bold')
    
    fig.suptitle("Final Sample Distributions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Final distribution comparison saved to {save_path}")
    
    plt.close()


def create_trajectory_straightness_plot(
    sampler: RectifiedCFGPlusPlusSampler,
    dataset: FaceDataset,
    class_labels: torch.Tensor,
    guidance_scale: float = 7.0,
    n_particles: int = 5,
    n_steps: int = 50,
    save_path: Path = None,
    dpi: int = 150,
):
    """
    Create plot showing trajectory paths for each method.
    CFG++ should have straighter/smoother trajectories.
    """
    methods = ['cfg', 'cfg++', 'rectified_cfg++']
    n_samples = len(class_labels)
    labels_np = class_labels.numpy()
    
    # Get target data
    target_data = dataset.data.numpy()
    target_labels = dataset.labels.numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Select particles
    particle_indices = np.random.choice(n_samples, size=min(n_particles, n_samples), replace=False)
    
    # Use same initial noise
    torch.manual_seed(123)
    initial_noise = torch.randn(n_samples, 2)
    
    for ax, method in zip(axes, methods):
        # Plot target distribution (faded)
        for c in range(2):
            mask = target_labels == c
            ax.scatter(
                target_data[mask, 0], target_data[mask, 1],
                c=CLASS_COLORS[c], s=5, alpha=0.15
            )
        
        # Get trajectory
        x = initial_noise.clone().to(sampler.device)
        class_labels_device = class_labels.to(sampler.device)
        
        trajectory = [x.cpu().numpy()]
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.ones(n_samples, device=sampler.device) * (step / n_steps)
            v_guided, _, _, _ = sampler.get_velocity(x, t, class_labels_device, guidance_scale, method)
            x = x + v_guided * dt
            trajectory.append(x.cpu().numpy())
        
        trajectory = np.stack(trajectory)
        
        # Plot trajectories for selected particles
        for idx in particle_indices:
            traj = trajectory[:, idx, :]
            c = labels_np[idx]
            ax.plot(traj[:, 0], traj[:, 1], c=METHOD_COLORS[method], linewidth=2, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], c='green', s=50, marker='o', zorder=10)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=50, marker='x', zorder=10)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(f"{METHOD_NAMES[method]}\n(scale={guidance_scale})", fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        ax.scatter([], [], c='green', s=50, marker='o', label='Start')
        ax.scatter([], [], c='red', s=50, marker='x', label='End')
        ax.legend(loc='upper right', fontsize=8)
    
    fig.suptitle("Trajectory Paths Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Trajectory straightness plot saved to {save_path}")
    
    plt.close()


def create_rectified_cfgpp_trajectory_animation(
    sampler: RectifiedCFGPlusPlusSampler,
    dataset: FaceDataset,
    class_labels: torch.Tensor,
    guidance_scale: float = 3.0,
    n_particles: int = 100,
    n_steps: int = 50,
    save_path: Path = None,
    fps: int = 15,
    dpi: int = 100,
):
    """
    Create animation for Rectified CFG++ showing velocity decomposition.
    
    Arrows:
    - Blue: Unconditional velocity
    - Green: Conditional velocity  
    - Purple: Rectified CFG++ velocity
    """
    method = 'rectified_cfg++'
    n_samples = len(class_labels)
    labels_np = class_labels.numpy()
    
    torch.manual_seed(42)
    x = torch.randn(n_samples, 2, device=sampler.device)
    class_labels_device = class_labels.to(sampler.device)
    
    trajectory = [x.cpu().numpy()]
    v_guided_list = []
    v_cond_list = []
    v_uncond_list = []
    
    dt = 1.0 / n_steps
    
    for step in range(n_steps):
        t = torch.ones(n_samples, device=sampler.device) * (step / n_steps)
        v_guided, v_cond, v_uncond, _ = sampler.get_velocity(x, t, class_labels_device, guidance_scale, method)
        
        v_guided_list.append(v_guided.cpu().numpy())
        v_cond_list.append(v_cond.cpu().numpy())
        v_uncond_list.append(v_uncond.cpu().numpy())
        
        x = x + v_guided * dt
        trajectory.append(x.cpu().numpy())
    
    trajectory = np.stack(trajectory)
    v_guided_arr = np.stack(v_guided_list)
    v_cond_arr = np.stack(v_cond_list)
    v_uncond_arr = np.stack(v_uncond_list)
    
    # Get target data
    target_data = dataset.data.numpy()
    target_labels = dataset.labels.numpy()
    
    # Select particles for arrows
    arrow_indices = np.random.choice(n_samples, size=min(15, n_samples), replace=False)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def animate(frame):
        ax.clear()
        
        # Plot target distribution (faded)
        for c in range(2):
            mask = target_labels == c
            ax.scatter(
                target_data[mask, 0], target_data[mask, 1],
                c=CLASS_COLORS[c], s=5, alpha=0.15
            )
        
        pos = trajectory[frame]
        
        # Plot particles
        for c in range(2):
            mask = labels_np == c
            ax.scatter(pos[mask, 0], pos[mask, 1], c='black', s=30, alpha=0.8)
        
        # Trails
        if frame > 0:
            for i in range(n_samples):
                trail = trajectory[:frame+1, i, :]
                ax.plot(trail[:, 0], trail[:, 1], c='gray', alpha=0.2, linewidth=0.5)
        
        # Velocity arrows
        if frame < n_steps:
            arrow_scale = 0.25
            
            for idx in arrow_indices:
                x_pos, y_pos = pos[idx]
                
                # Unconditional (blue)
                vx, vy = v_uncond_arr[frame, idx] * arrow_scale
                ax.annotate(
                    "", xy=(x_pos + vx, y_pos + vy), xytext=(x_pos, y_pos),
                    arrowprops=dict(arrowstyle="->", color="#3498DB", lw=2),
                    zorder=10
                )
                
                # Conditional (green)
                vx, vy = v_cond_arr[frame, idx] * arrow_scale
                ax.annotate(
                    "", xy=(x_pos + vx, y_pos + vy), xytext=(x_pos, y_pos),
                    arrowprops=dict(arrowstyle="->", color="#27AE60", lw=2),
                    zorder=10
                )
                
                # Rectified CFG++ (purple)
                vx, vy = v_guided_arr[frame, idx] * arrow_scale
                ax.annotate(
                    "", xy=(x_pos + vx, y_pos + vy), xytext=(x_pos, y_pos),
                    arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=2.5),
                    zorder=11
                )
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#3498DB", lw=2, label="Unconditional"),
            Line2D([0], [0], color="#27AE60", lw=2, label="Conditional"),
            Line2D([0], [0], color="#9B59B6", lw=2.5, label=f"Rectified CFG++ (scale={guidance_scale})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        
        ax.set_title(f"Rectified CFG++ Flow Matching | t = {frame/n_steps:.2f}", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        return []
    
    logger.info(f"Creating Rectified CFG++ trajectory animation with {n_steps + 1} frames...")
    anim = FuncAnimation(fig, animate, frames=n_steps + 1, interval=1000 / fps)
    
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        logger.info(f"Rectified CFG++ trajectory animation saved to {save_path}")
    
    plt.close()
    return anim


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main visualization function for Rectified CFG++."""
    logger.info("Starting Rectified CFG++ visualization...")

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

    # Get num_classes from checkpoint (infer from class_embedding shape)
    model_path = Path(cfg.training.output_dir) / "cfg_velocity_net.pt"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    num_classes_from_ckpt = state_dict['class_embedding.weight'].shape[0] - 1  # -1 for null class
    
    velocity_net = CFGFlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
        num_classes=num_classes_from_ckpt,
        class_emb_dim=cfg.model.class_emb_dim,
    )

    velocity_net.load_state_dict(state_dict)
    
    # Create sampler
    sampler = RectifiedCFGPlusPlusSampler(velocity_net=velocity_net, device=device)

    logger.info("Model loaded successfully")

    # Generate class labels
    n_vis_samples = min(2000, cfg.data.n_samples)
    n_per_class = n_vis_samples // 2
    class_labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones(n_vis_samples - n_per_class, dtype=torch.long),
    ])

    # Create output directory (same as main CFG visualizations)
    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Method comparison animation
    logger.info("Creating method comparison animation...")
    create_method_comparison_animation(
        sampler, dataset, class_labels,
        guidance_scale=5.0,
        n_particles=80,
        n_steps=50,
        save_path=output_dir / "method_comparison_rect.gif",
    )

    # 2. Rectified CFG++ trajectory with velocity arrows
    logger.info("Creating Rectified CFG++ trajectory animation...")
    create_rectified_cfgpp_trajectory_animation(
        sampler, dataset, class_labels,
        guidance_scale=3.0,
        n_particles=100,
        n_steps=50,
        save_path=output_dir / "cfgpp_trajectory_rect.gif",
    )

    # 3. Velocity magnitude comparison
    logger.info("Creating velocity magnitude plot...")
    create_velocity_magnitude_plot(
        sampler, class_labels,
        guidance_scales=[1.0, 3.0, 7.0, 15.0],
        n_steps=50,
        save_path=output_dir / "velocity_magnitude_rect.png",
    )

    # 4. Final distribution comparison
    logger.info("Creating final distribution comparison...")
    create_final_distribution_comparison(
        sampler, dataset, class_labels,
        guidance_scales=[1.0, 5.0, 15.0],
        n_steps=100,
        save_path=output_dir / "final_distributions_rect.png",
    )

    # 5. Trajectory straightness comparison
    logger.info("Creating trajectory straightness plot...")
    create_trajectory_straightness_plot(
        sampler, dataset, class_labels,
        guidance_scale=7.0,
        n_particles=5,
        n_steps=50,
        save_path=output_dir / "trajectory_paths_rect.png",
    )

    logger.info(f"All Rectified CFG++ visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
