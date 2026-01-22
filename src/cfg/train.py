"""Training script for CFG Flow Matching model."""

import logging
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cfg.data import FaceDataset
from cfg.loss import CFGFlowMatchingLoss
from cfg.model import CFGFlowMLP

logger = logging.getLogger(__name__)


def linear_decay_lr(step, num_iterations, learning_rate):
    """Linear learning rate decay."""
    return learning_rate * (1 - step / num_iterations)


def warmup_cooldown_lr(step, num_iterations, learning_rate, warmup_iters, warmdown_iters):
    """Learning rate schedule with warmup and cooldown."""
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    elif step < num_iterations - warmdown_iters:
        return learning_rate
    else:
        decay_ratio = (num_iterations - step) / warmdown_iters
        return learning_rate * decay_ratio


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for CFG."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg.training.seed)

    # Create dataset
    logger.info("Creating dataset...")
    dataset = FaceDataset(
        n_samples=cfg.data.n_samples,
        left_eye_center=tuple(cfg.data.get('left_eye_center', [-0.5, 0.5])),
        right_eye_center=tuple(cfg.data.get('right_eye_center', [0.5, 0.5])),
        eye_sigma=cfg.data.get('eye_sigma', 0.15),
    )

    # Create model with class conditioning
    logger.info("Creating CFG model...")
    model = CFGFlowMLP(
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
        num_classes=dataset.num_classes,
        class_emb_dim=cfg.model.class_emb_dim,
    )

    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        betas=(
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.9),
            1 - (cfg.data.n_samples / 5e5) * (1 - 0.95)
        ),
    )

    # Learning rate scheduler
    if cfg.training.lr_scheduler == "linear_decay":
        get_lr = lambda step: linear_decay_lr(step, cfg.training.n_epochs, cfg.training.learning_rate)
    else:
        warmup_iters = cfg.training.n_epochs // 10
        warmdown_iters = cfg.training.n_epochs - warmup_iters
        get_lr = lambda step: warmup_cooldown_lr(
            step, cfg.training.n_epochs, cfg.training.learning_rate, warmup_iters, warmdown_iters
        )

    # Loss function with 20% class dropout
    loss_fn = CFGFlowMatchingLoss(
        sigma_min=cfg.training.sigma_min,
        class_dropout_prob=cfg.training.class_dropout_prob,
        null_class_idx=dataset.num_classes,
    )

    logger.info("Starting training...")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in tqdm(range(cfg.training.n_epochs), desc="Training"):
        model.zero_grad()

        loss = loss_fn(model, dataset.create())
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)

        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if epoch % cfg.training.log_interval == 0:
            logger.info(f"Width: {cfg.model.width} | Epoch: {epoch} | LR: {lr:.6f} | Loss: {loss.item():.6f}")

        if math.isnan(loss.item()):
            logger.error("Loss became NaN, stopping training")
            break

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "cfg_velocity_net.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
