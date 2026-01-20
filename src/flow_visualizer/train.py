"""Training script for Flow Matching model."""

import logging
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from flow_visualizer.data import CustomDataset
from flow_visualizer.loss import ConditionalFlowMatchingLoss
from flow_visualizer.model import FlowMLP, FlowMatchingModel

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
    """Main training function."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

    torch.manual_seed(cfg.training.seed)

    logger.info("Creating dataset...")
    dataset = CustomDataset(size=cfg.data.n_samples)

    logger.info("Creating model...")
    model = FlowMLP(
        n_features=cfg.model.n_features,
        width=cfg.model.width,
        n_blocks=cfg.model.n_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )

    if cfg.training.lr_scheduler == "linear_decay":
        get_lr = lambda step: linear_decay_lr(step, cfg.training.n_epochs, cfg.training.learning_rate)
    else:
        warmup_iters = cfg.training.n_epochs // 10
        warmdown_iters = cfg.training.n_epochs - warmup_iters
        get_lr = lambda step: warmup_cooldown_lr(
            step, cfg.training.n_epochs, cfg.training.learning_rate, warmup_iters, warmdown_iters
        )

    loss_fn = ConditionalFlowMatchingLoss(sigma_min=cfg.training.sigma_min)

    logger.info("Starting training...")
    model.train()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True
    )

    for epoch in tqdm(range(cfg.training.n_epochs), desc="Training"):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Compute flow matching loss
            t = torch.rand(batch.shape[0], device=device)
            noise = torch.randn_like(batch)
            sigma_min = cfg.training.sigma_min

            x_t = (1 - (1 - sigma_min) * t[:, None]) * noise + t[:, None] * batch
            optimal_flow = batch - (1 - sigma_min) * noise
            predicted_flow = model(x_t, time=t)

            loss = (predicted_flow - optimal_flow).square().mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)

            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % cfg.training.log_interval == 0:
            logger.info(f"Width: {cfg.model.width} | Epoch: {epoch + 1} | LR: {lr:.6f} | Loss: {avg_loss:.6f}")

        if math.isnan(avg_loss):
            logger.error("Loss became NaN, stopping training")
            break

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "velocity_net.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
