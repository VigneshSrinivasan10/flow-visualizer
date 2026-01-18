"""Training script for Rectified Flow model."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from flow_visualizer.data import MoonsDataset, CirclesDataset
from flow_visualizer.model import RectifiedFlowModel, MLPVelocityNet

logger = logging.getLogger(__name__)


def train_epoch_standard(
    model: RectifiedFlowModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch using standard flow matching loss."""
    model.velocity_net.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epoch_reflow(
    model: RectifiedFlowModel,
    x0: torch.Tensor,
    x1: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch using reflow loss with paired samples."""
    model.velocity_net.train()

    # Create dataloader from paired samples
    dataset = TensorDataset(x0, x1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0

    for batch_x0, batch_x1 in dataloader:
        batch_x0 = batch_x0.to(device)
        batch_x1 = batch_x1.to(device)

        optimizer.zero_grad()
        loss = model.compute_reflow_loss(batch_x0, batch_x1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@hydra.main(version_base=None, config_path="/home/user/flow-visualizer/conf", config_name="rectified_flow_config")
def main(cfg: DictConfig) -> None:
    """Main training function for Rectified Flow."""
    logger.info("Rectified Flow Training Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Set device
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(cfg.training.seed)

    # Create dataset
    logger.info(f"Creating {cfg.data.dataset_type} dataset...")
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

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create output directory
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # Phase 1: Initial Flow Matching Training
    # ============================================================================
    logger.info("\n" + "="*60)
    logger.info("Phase 1: Initial Flow Matching Training")
    logger.info("="*60)

    velocity_net = MLPVelocityNet(
        data_dim=cfg.model.data_dim,
        time_embed_dim=cfg.model.time_embed_dim,
        hidden_dims=cfg.model.hidden_dims,
    )

    model = RectifiedFlowModel(velocity_net=velocity_net, device=device)
    optimizer = torch.optim.Adam(
        velocity_net.parameters(),
        lr=cfg.training.learning_rate,
    )

    logger.info("Training initial flow matching model...")
    for epoch in range(cfg.training.n_epochs):
        avg_loss = train_epoch_standard(model, dataloader, optimizer, device)

        if (epoch + 1) % cfg.training.log_interval == 0:
            logger.info(f"Epoch {epoch + 1}/{cfg.training.n_epochs}, Loss: {avg_loss:.6f}")

    # Compute and log straightness
    straightness = model.compute_trajectory_straightness(
        n_samples=1000,
        n_steps=cfg.visualization.n_sampling_steps
    )
    logger.info(f"Initial trajectory straightness: {straightness:.4f}")

    # Save initial model
    model_path = output_dir / "velocity_net_initial.pt"
    torch.save(velocity_net.state_dict(), model_path)
    logger.info(f"Initial model saved to {model_path}")

    # ============================================================================
    # Phase 2: Reflow Iterations
    # ============================================================================
    for reflow_iter in range(cfg.training.n_reflow_iterations):
        logger.info("\n" + "="*60)
        logger.info(f"Phase 2: Reflow Iteration {reflow_iter + 1}/{cfg.training.n_reflow_iterations}")
        logger.info("="*60)

        # Generate paired samples from current model
        logger.info("Generating paired samples for reflow...")
        x0, x1 = model.generate_reflow_pairs(
            n_samples=cfg.training.reflow_n_samples,
            n_steps=cfg.visualization.n_sampling_steps,
            data_dim=cfg.model.data_dim,
        )

        logger.info(f"Generated {len(x0)} paired samples")

        # Create new model for reflow training
        velocity_net_reflow = MLPVelocityNet(
            data_dim=cfg.model.data_dim,
            time_embed_dim=cfg.model.time_embed_dim,
            hidden_dims=cfg.model.hidden_dims,
        )

        model = RectifiedFlowModel(velocity_net=velocity_net_reflow, device=device)
        optimizer = torch.optim.Adam(
            velocity_net_reflow.parameters(),
            lr=cfg.training.learning_rate,
        )

        # Train on paired samples
        logger.info(f"Training reflow model (iteration {reflow_iter + 1})...")
        for epoch in range(cfg.training.reflow_n_epochs):
            avg_loss = train_epoch_reflow(
                model, x0, x1,
                cfg.training.batch_size,
                optimizer,
                device
            )

            if (epoch + 1) % cfg.training.log_interval == 0:
                logger.info(
                    f"Reflow {reflow_iter + 1} - Epoch {epoch + 1}/{cfg.training.reflow_n_epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        # Compute and log straightness
        straightness = model.compute_trajectory_straightness(
            n_samples=1000,
            n_steps=cfg.visualization.n_sampling_steps
        )
        logger.info(f"Reflow {reflow_iter + 1} trajectory straightness: {straightness:.4f}")

        # Save reflow model
        model_path = output_dir / f"velocity_net_reflow_{reflow_iter + 1}.pt"
        torch.save(velocity_net_reflow.state_dict(), model_path)
        logger.info(f"Reflow model {reflow_iter + 1} saved to {model_path}")

    # ============================================================================
    # Save Final Configuration
    # ============================================================================
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("\n" + "="*60)
    logger.info("Rectified Flow Training Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
