"""Training script for Flow Matching model."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_visualizer.data import TRexDataset
from flow_visualizer.model import FlowMatchingModel, MLPVelocityNet

logger = logging.getLogger(__name__)


def train_epoch(
    model: FlowMatchingModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Set device
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(cfg.training.seed)

    # Create dataset
    logger.info("Creating dataset...")
    dataset = TRexDataset(
        n_samples=cfg.data.n_samples,
        noise=cfg.data.noise,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create model
    logger.info("Creating model...")
    velocity_net = MLPVelocityNet(
        data_dim=cfg.model.data_dim,
        time_embed_dim=cfg.model.time_embed_dim,
        hidden_dims=cfg.model.hidden_dims,
    )

    model = FlowMatchingModel(velocity_net=velocity_net, device=device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        velocity_net.parameters(),
        lr=cfg.training.learning_rate,
    )

    # Training loop
    logger.info("Starting training...")
    for epoch in range(cfg.training.n_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        if (epoch + 1) % cfg.training.log_interval == 0:
            logger.info(f"Epoch {epoch + 1}/{cfg.training.n_epochs}, Loss: {avg_loss:.6f}")

    # Save model
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "velocity_net.pt"
    torch.save(velocity_net.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
