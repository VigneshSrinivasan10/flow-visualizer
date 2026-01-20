"""Training script for Rectified Flow model.

Rectified Flow straightens the trajectories by retraining on coupled pairs
(noise, generated_sample) from a base flow model.

Reference: https://arxiv.org/abs/2209.03003
"""

import logging
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from flow_visualizer.data import CustomDataset
from flow_visualizer.model import FlowMLP

logger = logging.getLogger(__name__)


class RectifiedFlowLoss:
    """Rectified Flow loss - trains on coupled pairs for straight trajectories."""

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def __call__(self, flow_model, x0, x1):
        """
        Compute rectified flow loss.

        Args:
            flow_model: The velocity network
            x0: Source samples (noise)
            x1: Target samples (generated from base model or data)
        """
        t = torch.rand(x0.shape[0], device=x0.device)

        # Linear interpolation between x0 and x1
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1

        # Target velocity is constant: x1 - x0 (straight line)
        target_velocity = x1 - x0

        # Predicted velocity
        predicted_velocity = flow_model(x_t, time=t)

        return (predicted_velocity - target_velocity).square().mean()


def generate_coupled_pairs(base_model, n_samples, n_steps=100, device="cpu"):
    """
    Generate coupled pairs (x0, x1) using the base flow model.

    x0: noise samples
    x1: generated samples from running the flow
    """
    base_model.eval()
    with torch.no_grad():
        # Sample noise
        x0 = torch.randn(n_samples, 2, device=device)

        # Run the flow to generate x1
        x = x0.clone()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.ones(n_samples, device=device) * (step / n_steps)
            v = base_model(x, time=t)
            x = x + v * dt

        x1 = x

    return x0, x1


def linear_decay_lr(step, num_iterations, learning_rate):
    """Linear learning rate decay."""
    return learning_rate * (1 - step / num_iterations)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for Rectified Flow."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg.training.seed)

    # Load base flow model
    logger.info("Loading base flow model...")
    base_model = FlowMLP(width=cfg.model.width, n_blocks=cfg.model.n_blocks)
    base_model_path = Path(cfg.training.base_model_path)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
    base_model.to(device)
    base_model.eval()
    logger.info(f"Base model loaded from {base_model_path}")

    # Create rectified flow model
    logger.info("Creating rectified flow model...")
    model = FlowMLP(width=cfg.model.width, n_blocks=cfg.model.n_blocks)
    model.to(device)

    # Configure optimizer
    optimizer, optimizer_settings = model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        betas=(
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.9),
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.95)
        ),
    )

    # Learning rate scheduler
    get_lr = lambda step: linear_decay_lr(step, cfg.training.n_epochs, cfg.training.learning_rate)

    # Loss function
    loss_fn = RectifiedFlowLoss(sigma_min=cfg.training.sigma_min)

    # Generate coupled pairs from base model
    logger.info("Generating coupled pairs from base model...")
    x0_data, x1_data = generate_coupled_pairs(
        base_model,
        n_samples=cfg.data.n_samples,
        n_steps=cfg.training.coupling_steps,
        device=device,
    )
    logger.info(f"Generated {cfg.data.n_samples} coupled pairs")

    logger.info("Starting rectified flow training...")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_size = cfg.training.batch_size

    for epoch in tqdm(range(cfg.training.n_epochs), desc="Training Rectified Flow"):
        model.zero_grad()

        # Sample a batch of coupled pairs
        indices = torch.randint(0, cfg.data.n_samples, (batch_size,))
        x0_batch = x0_data[indices]
        x1_batch = x1_data[indices]

        # Compute loss
        loss = loss_fn(model, x0_batch, x1_batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)

        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if epoch % cfg.training.log_interval == 0:
            logger.info(f"Epoch: {epoch} | LR: {lr:.6f} | Loss: {loss.item():.6f}")

        if math.isnan(loss.item()):
            logger.error("Loss became NaN, stopping training")
            break

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "velocity_net.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Rectified flow model saved to {model_path}")

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("Rectified flow training complete!")


if __name__ == "__main__":
    main()
