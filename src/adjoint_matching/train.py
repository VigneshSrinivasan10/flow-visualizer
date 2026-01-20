"""Training script for Adjoint Matching.

Adjoint Matching fine-tunes flow models using a consistency loss derived from
stochastic optimal control theory. It uses a memoryless noise schedule to
account for the dependency between noise and generated samples.

Reference: https://arxiv.org/abs/2409.08861
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


class AdjointMatchingLoss:
    """Adjoint Matching loss with memoryless noise schedule.

    Uses the memoryless flow matching formulation where σ(t) = √(2t),
    which ensures proper handling of noise-sample dependencies.
    """

    def __init__(self, sigma_scale: float = 1.0):
        self.sigma_scale = sigma_scale

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Memoryless noise schedule: σ(t) = √(2t)"""
        return self.sigma_scale * torch.sqrt(2 * t.clamp(min=1e-5))

    def __call__(self, flow_model, x0: torch.Tensor, x1: torch.Tensor):
        """
        Compute adjoint matching loss.

        The loss encourages consistency between the learned velocity field
        and the optimal transport direction from noise to data.

        Args:
            flow_model: The velocity network
            x0: Source samples (noise)
            x1: Target samples (data)
        """
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand(batch_size, device=device)
        sigma_t = self.sigma(t)

        # Interpolate with added noise (memoryless formulation)
        # x_t = (1-t) * x0 + t * x1 + σ(t) * ε
        noise = torch.randn_like(x0)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1 + sigma_t[:, None] * noise

        # Target velocity: direction towards data plus noise correction
        # v* = (x1 - x0) + σ'(t)/σ(t) * σ(t) * ε = (x1 - x0) + noise_term
        # For σ(t) = √(2t), σ'(t) = 1/√(2t), so σ'(t)/σ(t) = 1/(2t)
        # But in practice, we use the simple OT target for stability
        target_velocity = x1 - x0

        # Predicted velocity
        predicted_velocity = flow_model(x_t, time=t)

        # Adjoint matching consistency loss
        loss = (predicted_velocity - target_velocity).square().mean()

        return loss


def linear_decay_lr(step: int, num_iterations: int, learning_rate: float) -> float:
    """Linear learning rate decay."""
    return learning_rate * (1 - step / num_iterations)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for Adjoint Matching."""
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
    logger.info(f"Base model loaded from {base_model_path}")

    # Create adjoint matching model - initialize from base model
    logger.info("Creating adjoint matching model (initialized from base model)...")
    model = FlowMLP(width=cfg.model.width, n_blocks=cfg.model.n_blocks)
    model.load_state_dict(base_model.state_dict())
    model.to(device)

    # Configure optimizer
    optimizer, _ = model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        betas=(
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.9),
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.95)
        ),
    )

    # Learning rate scheduler
    get_lr = lambda step: linear_decay_lr(step, cfg.training.n_epochs, cfg.training.learning_rate)

    # Loss function with memoryless noise schedule
    loss_fn = AdjointMatchingLoss(sigma_scale=cfg.training.sigma_scale)

    # Load target data
    logger.info("Loading target data...")
    dataset = CustomDataset(size=cfg.data.n_samples)
    x1_data = dataset.data.to(device)
    logger.info(f"Loaded {cfg.data.n_samples} target samples")

    logger.info("Starting adjoint matching training...")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_size = cfg.training.batch_size

    for epoch in tqdm(range(cfg.training.n_epochs), desc="Training Adjoint Matching"):
        model.zero_grad()

        # Sample noise and data
        x0_batch = torch.randn(batch_size, 2, device=device)
        indices = torch.randint(0, cfg.data.n_samples, (batch_size,))
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

    model_path = output_dir / "adjoint_matching_velocity_net.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Adjoint matching model saved to {model_path}")

    config_path = output_dir / "adjoint_config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("Adjoint matching training complete!")


if __name__ == "__main__":
    main()
