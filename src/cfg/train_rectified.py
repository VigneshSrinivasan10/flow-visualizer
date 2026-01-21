"""Training script for Rectified Flow on CFG model.

Straightens the trajectories by retraining on coupled pairs
(noise, generated_sample) from a base CFG flow model.
"""

import logging
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cfg.data import FaceDataset
from cfg.model import CFGFlowMLP

logger = logging.getLogger(__name__)


class CFGRectifiedFlowLoss:
    """Rectified Flow loss for CFG - trains on coupled pairs for straight trajectories."""

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def __call__(self, flow_model, x0, x1, class_labels):
        """
        Compute rectified flow loss with class conditioning.

        Args:
            flow_model: The velocity network
            x0: Source samples (noise)
            x1: Target samples (generated from base model)
            class_labels: Class labels for conditioning
        """
        t = torch.rand(x0.shape[0], device=x0.device)

        # Linear interpolation between x0 and x1
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1

        # Target velocity is constant: x1 - x0 (straight line)
        target_velocity = x1 - x0

        # Predicted velocity (with class conditioning)
        predicted_velocity = flow_model(x_t, time=t, class_labels=class_labels)

        return (predicted_velocity - target_velocity).square().mean()


def generate_coupled_pairs(base_model, n_samples, class_labels, n_steps=100, device="cpu", guidance_scale=2.0):
    """
    Generate coupled pairs (x0, x1) using the base CFG flow model.

    x0: noise samples
    x1: generated samples from running the flow with CFG
    """
    base_model.eval()
    with torch.no_grad():
        # Sample noise
        x0 = torch.randn(n_samples, 2, device=device)

        # Run the flow to generate x1 using CFG
        x = x0.clone()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.ones(n_samples, device=device) * (step / n_steps)
            v = base_model.forward_cfg(x, time=t, class_labels=class_labels, guidance_scale=guidance_scale)
            x = x + v * dt

        x1 = x

    return x0, x1


def linear_decay_lr(step, num_iterations, learning_rate):
    """Linear learning rate decay."""
    return learning_rate * (1 - step / num_iterations)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for CFG Rectified Flow."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg.training.seed)

    # Create dataset for class labels
    dataset = FaceDataset(n_samples=cfg.data.n_samples)

    # Load base CFG model
    logger.info("Loading base CFG model...")
    base_model = CFGFlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
        num_classes=dataset.num_classes,
        class_emb_dim=cfg.model.class_emb_dim,
    )
    base_model_path = Path(cfg.training.output_dir) / "cfg_velocity_net.pt"
    base_model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
    base_model.to(device)
    base_model.eval()
    logger.info(f"Base model loaded from {base_model_path}")

    # Create rectified flow model - initialize from base model
    logger.info("Creating rectified flow model (initialized from base model)...")
    model = CFGFlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
        num_classes=dataset.num_classes,
        class_emb_dim=cfg.model.class_emb_dim,
    )
    model.load_state_dict(base_model.state_dict())
    model.to(device)

    # Configure optimizer with lower learning rate for fine-tuning
    rectified_lr = cfg.training.learning_rate * 0.5
    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=rectified_lr,
        betas=(
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.9),
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.95)
        ),
    )

    # Learning rate scheduler
    n_rectified_epochs = cfg.training.n_epochs // 2  # Fewer epochs for fine-tuning
    get_lr = lambda step: linear_decay_lr(step, n_rectified_epochs, rectified_lr)

    # Loss function
    loss_fn = CFGRectifiedFlowLoss(sigma_min=cfg.training.sigma_min)

    # Generate class labels (same distribution as training)
    n_per_class = cfg.data.n_samples // dataset.num_classes
    class_labels = torch.cat([
        torch.full((n_per_class,), i, dtype=torch.long)
        for i in range(dataset.num_classes)
    ])
    # Handle remainder
    remainder = cfg.data.n_samples - len(class_labels)
    if remainder > 0:
        class_labels = torch.cat([
            class_labels,
            torch.full((remainder,), dataset.num_classes - 1, dtype=torch.long)
        ])
    class_labels = class_labels.to(device)

    # Generate coupled pairs from base model
    guidance_scale = cfg.visualization.get("guidance_scale", 2.0)
    logger.info(f"Generating coupled pairs from base model (guidance_scale={guidance_scale})...")
    coupling_steps = 100
    x0_data, x1_data = generate_coupled_pairs(
        base_model,
        n_samples=cfg.data.n_samples,
        class_labels=class_labels,
        n_steps=coupling_steps,
        device=device,
        guidance_scale=guidance_scale,
    )
    logger.info(f"Generated {cfg.data.n_samples} coupled pairs")

    logger.info(f"Starting rectified flow training for {n_rectified_epochs} epochs...")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_size = cfg.training.get("batch_size", 256)

    for epoch in tqdm(range(n_rectified_epochs), desc="Training Rectified Flow"):
        model.zero_grad()

        # Sample a batch of coupled pairs
        indices = torch.randint(0, cfg.data.n_samples, (batch_size,))
        x0_batch = x0_data[indices]
        x1_batch = x1_data[indices]
        class_batch = class_labels[indices]

        # Compute loss
        loss = loss_fn(model, x0_batch, x1_batch, class_batch)
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

    # Save as the main model (overwrite)
    model_path = output_dir / "cfg_velocity_net.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Rectified flow model saved to {model_path}")

    logger.info("Rectified flow training complete!")


if __name__ == "__main__":
    main()
