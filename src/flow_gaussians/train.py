"""Training script for flow matching on 2D Gaussians."""

import logging
import os
from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from flow_gaussians.data import DATASET_CONFIGS
from flow_gaussians.model import SimpleFlowNetwork

logger = logging.getLogger(__name__)


def train_flow_matching(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    epochs: int = 500,
    batch_size: int = 128,
    label_dropout: float = 0.3,
    verbose: bool = True,
    log_interval: int = 50,
) -> List[float]:
    """
    Train the flow matching model with label dropout for CFG.

    Args:
        model: SimpleFlowNetwork instance
        data: (n_samples, 2) training data
        labels: (n_samples,) class labels
        epochs: Number of training epochs
        batch_size: Batch size
        label_dropout: Probability of dropping label (set to -1 for unconditional)
        verbose: Whether to print training progress
        log_interval: How often to log progress

    Returns:
        losses: List of average losses per epoch
    """
    n_samples = data.shape[0]
    losses = []

    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(n_samples)
        data_shuffled = data[perm]
        labels_shuffled = labels[perm]

        epoch_losses = []

        for i in range(0, n_samples, batch_size):
            # Get batch
            x1 = data_shuffled[i : i + batch_size]
            batch_labels = labels_shuffled[i : i + batch_size].copy()
            actual_batch_size = x1.shape[0]

            # Label dropout: randomly set labels to -1 (unconditional)
            dropout_mask = np.random.rand(actual_batch_size) < label_dropout
            batch_labels[dropout_mask] = -1

            # Sample random time t ∈ [0, 1]
            t = np.random.rand(actual_batch_size, 1)

            # Sample noise (starting point)
            x0 = np.random.randn(actual_batch_size, 2)

            # Straight-line interpolation: x_t = (1 - t) * x0 + t * x1
            x_t = (1 - t) * x0 + t * x1

            # Target velocity (constant along straight line): v = x1 - x0
            v_target = x1 - x0

            # Forward pass
            v_pred, cache = model.forward(x_t, t, batch_labels)

            # Compute MSE loss
            loss = np.mean((v_pred - v_target) ** 2)
            epoch_losses.append(loss)

            # Backward pass
            grads = model.backward(v_pred, v_target, cache)

            # Update parameters
            params = model.get_params()
            params = model.optimizer.step(params, grads)
            model.set_params(params)

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % log_interval == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return losses


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for flow matching on Gaussians."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.training.seed)

    # Get dataset configuration
    dataset_name = cfg.data.dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    logger.info(f"Using dataset: {dataset_config['title']}")

    # Generate data
    logger.info("Generating data...")
    data, labels = dataset_config["generator"](n_samples=cfg.data.n_samples)
    logger.info(f"Generated {len(data)} samples")
    logger.info(f"Class 0: {np.sum(labels == 0)} samples, Class 1: {np.sum(labels == 1)} samples")

    # Create model
    logger.info("Creating model...")
    model = SimpleFlowNetwork(
        hidden_dim=cfg.model.hidden_dim,
        lr=cfg.training.learning_rate,
    )

    # Check for existing model
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "flow_model.npz"

    if model_path.exists() and not cfg.training.force_retrain:
        logger.info(f"Loading existing model from {model_path}")
        model.load(str(model_path))
    else:
        logger.info("Training model...")
        logger.info(
            f"Epochs: {cfg.training.epochs}, Batch size: {cfg.training.batch_size}, "
            f"Label dropout: {cfg.training.label_dropout}"
        )

        np.random.seed(cfg.training.seed)  # Reset for training reproducibility
        losses = train_flow_matching(
            model,
            data,
            labels,
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            label_dropout=cfg.training.label_dropout,
            verbose=True,
            log_interval=cfg.training.log_interval,
        )

        # Save model
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save losses
        losses_path = output_dir / "losses.npy"
        np.save(losses_path, np.array(losses))
        logger.info(f"Losses saved to {losses_path}")

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
