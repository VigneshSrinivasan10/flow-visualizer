"""Distillation training script for flow matching models."""

import logging
from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from flow_gaussians.data import DATASET_CONFIGS
from flow_gaussians.distill_model import DistilledFlowNetwork
from flow_gaussians.model import SimpleFlowNetwork

logger = logging.getLogger(__name__)


def train_distillation(
    teacher_model: SimpleFlowNetwork,
    student_model: DistilledFlowNetwork,
    epochs: int = 200,
    batch_size: int = 256,
    n_samples_per_epoch: int = 10000,
    w_min: float = 0.0,
    w_max: float = 10.0,
    verbose: bool = True,
    log_interval: int = 20,
) -> List[float]:
    """
    Train the student model to match the teacher's CFG-combined velocity.

    The key insight is that we randomly sample the guidance scale `w` during
    training. This teaches the student to handle ANY guidance scale, not just
    a fixed one. At inference, you can use any w in [w_min, w_max].

    Teacher CFG formula:
        v_teacher = v_uncond + w * (v_cond - v_uncond)

    Student learns to match:
        v_student(x, t, label, w) ≈ v_teacher(x, t, label, w)

    Args:
        teacher_model: Pre-trained SimpleFlowNetwork (frozen)
        student_model: DistilledFlowNetwork to train
        epochs: Number of training epochs
        batch_size: Batch size
        n_samples_per_epoch: Number of random samples per epoch
        w_min: Minimum guidance scale
        w_max: Maximum guidance scale
        verbose: Whether to print training progress
        log_interval: How often to log progress

    Returns:
        losses: List of average losses per epoch
    """
    losses = []

    for epoch in range(epochs):
        epoch_losses = []

        for i in range(0, n_samples_per_epoch, batch_size):
            actual_batch_size = min(batch_size, n_samples_per_epoch - i)

            # Sample random positions from noise distribution (where flow starts)
            x = np.random.randn(actual_batch_size, 2)

            # Sample random times in [0, 1]
            t = np.random.rand(actual_batch_size, 1)

            # Sample random class labels (0 or 1)
            labels = np.random.choice([0, 1], actual_batch_size).astype(float)

            # *** CRITICAL: Sample guidance scale UNIFORMLY for each sample ***
            # This exposes the student to the full range of guidance behaviors
            w = np.random.uniform(w_min, w_max, (actual_batch_size, 1))

            # Teacher computes CFG velocity (2 forward passes)
            labels_uncond = np.full(actual_batch_size, -1, dtype=float)
            v_cond = teacher_model.predict(x, t, labels)
            v_uncond = teacher_model.predict(x, t, labels_uncond)
            v_teacher = v_uncond + w * (v_cond - v_uncond)

            # Student predicts the same output directly (1 forward pass)
            v_student, cache = student_model.forward(x, t, labels, w)

            # MSE loss
            loss = np.mean((v_student - v_teacher) ** 2)
            epoch_losses.append(loss)

            # Backward pass (student only, teacher is frozen)
            grads = student_model.backward(v_student, v_teacher, cache)

            # Update student parameters
            params = student_model.get_params()
            params = student_model.optimizer.step(params, grads)
            student_model.set_params(params)

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % log_interval == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return losses


@hydra.main(version_base=None, config_path="conf", config_name="config_distill")
def main(cfg: DictConfig) -> None:
    """Main distillation training function."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.training.seed)

    # Get dataset configuration
    dataset_name = cfg.data.dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    logger.info(f"Using dataset: {dataset_config['title']}")

    # Load teacher model
    teacher_dir = Path(cfg.distillation.teacher_dir)
    teacher_path = teacher_dir / "flow_model.npz"

    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Teacher model not found at {teacher_path}. "
            f"Run `uv run fg-train data.dataset={dataset_name}` first."
        )

    logger.info(f"Loading teacher model from {teacher_path}")
    teacher_model = SimpleFlowNetwork(hidden_dim=cfg.model.hidden_dim)
    teacher_model.load(str(teacher_path))

    # Create student model
    logger.info("Creating student (distilled) model...")
    student_model = DistilledFlowNetwork(
        hidden_dim=cfg.model.hidden_dim,
        lr=cfg.distillation.learning_rate,
    )

    # Create output directory
    output_dir = Path(cfg.distillation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "distilled_model.npz"

    if model_path.exists() and not cfg.distillation.force_retrain:
        logger.info(f"Loading existing distilled model from {model_path}")
        student_model.load(str(model_path))
    else:
        logger.info("Training distilled model...")
        logger.info(
            f"Epochs: {cfg.distillation.epochs}, Batch size: {cfg.distillation.batch_size}, "
            f"w_range: [{cfg.distillation.w_min}, {cfg.distillation.w_max}]"
        )

        np.random.seed(cfg.training.seed)  # Reset for training reproducibility
        losses = train_distillation(
            teacher_model,
            student_model,
            epochs=cfg.distillation.epochs,
            batch_size=cfg.distillation.batch_size,
            n_samples_per_epoch=cfg.distillation.n_samples_per_epoch,
            w_min=cfg.distillation.w_min,
            w_max=cfg.distillation.w_max,
            verbose=True,
            log_interval=cfg.distillation.log_interval,
        )

        # Save model
        student_model.save(str(model_path))
        logger.info(f"Distilled model saved to {model_path}")

        # Save losses
        losses_path = output_dir / "losses.npy"
        np.save(losses_path, np.array(losses))
        logger.info(f"Losses saved to {losses_path}")

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_path}")

    logger.info("Distillation complete!")


if __name__ == "__main__":
    main()
