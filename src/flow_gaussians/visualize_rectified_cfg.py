"""Visualization script specifically for Rectified CFG++ comparisons."""

import argparse
import logging
from pathlib import Path

import numpy as np

from flow_gaussians.data import DATASET_CONFIGS
from flow_gaussians.model import SimpleFlowNetwork
from flow_gaussians.visualize import (
    create_cfg_vs_rectified_side_by_side_animation,
    create_probability_path_animation,
    create_rectified_cfg_probability_path_animation,
    plot_both_classes_cfg,
    plot_cfg_vs_rectified_comparison,
    plot_training_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_visualizations(dataset_name: str, seed: int = 42):
    """Generate Rectified CFG++ visualizations for a given dataset.

    Args:
        dataset_name: One of 'overlapping', 'non_overlapping', 'multimodal'
        seed: Random seed for reproducibility
    """
    model_dir = Path(f"outputs/flow_gaussians/{dataset_name}")
    output_dir = model_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # Get dataset config
    dataset_config = DATASET_CONFIGS[dataset_name]
    class0_centers = dataset_config["class0_centers"]
    class1_centers = dataset_config["class1_centers"]

    # Axis limits based on dataset
    if dataset_name == "overlapping":
        xlim = (-3, 3)
        ylim = (-3, 3)
    else:
        xlim = (-4, 4)
        ylim = (-4, 4)

    # Generate data
    logger.info(f"Generating data for {dataset_name}...")
    data, labels = dataset_config["generator"](n_samples=5000)

    # Load model
    model_path = model_dir / "flow_model.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    logger.info(f"Loading model from {model_path}")
    model = SimpleFlowNetwork(hidden_dim=128)
    model.load(str(model_path))

    # Visualization parameters
    n_samples = 500
    cfg_scales = [1, 3, 5, 7, 9]
    animation_scales = [1, 5, 9]

    # 1. Training data visualization
    logger.info("Creating training data visualization...")
    plot_training_data(
        data, labels,
        save_path=str(output_dir / "training_data_rect.png"),
        title=dataset_config["title"],
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim, ylim=ylim,
    )

    # 2. CFG vs Rectified CFG++ comparison grid
    logger.info("Creating CFG vs Rectified CFG++ comparison...")
    plot_cfg_vs_rectified_comparison(
        model, data, labels,
        cfg_scales=cfg_scales,
        lambda_maxs=cfg_scales,
        n_samples=n_samples,
        save_path=str(output_dir / "cfg_vs_rectified_comparison_rect.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim, ylim=ylim,
        seed=seed,
        gamma=1.0,
    )

    # 3. Standard CFG probability path animation
    logger.info("Creating Standard CFG probability path animation...")
    create_probability_path_animation(
        model, data, labels,
        cfg_scales=animation_scales,
        n_samples=n_samples,
        num_steps=50,
        save_path=str(output_dir / "cfg_probability_path_rect.gif"),
        fps=15,
        seed=seed,
    )

    # 4. Rectified CFG++ probability path animation
    logger.info("Creating Rectified CFG++ probability path animation...")
    create_rectified_cfg_probability_path_animation(
        model, data, labels,
        lambda_maxs=animation_scales,
        n_samples=n_samples,
        num_steps=50,
        save_path=str(output_dir / "rectified_cfg_probability_path_rect.gif"),
        fps=15,
        seed=seed,
        gamma=1.0,
    )

    # 5. Side-by-side CFG vs Rectified CFG++ animation (at scale=5)
    logger.info("Creating CFG vs Rectified CFG++ side-by-side animation...")
    create_cfg_vs_rectified_side_by_side_animation(
        model, data, labels,
        guidance_scale=5.0,
        n_samples=n_samples,
        num_steps=50,
        save_path=str(output_dir / "cfg_vs_rectified_side_by_side_rect.gif"),
        fps=15,
        seed=seed,
        gamma=1.0,
    )

    # 6. Side-by-side at high guidance scale (9)
    logger.info("Creating CFG vs Rectified CFG++ side-by-side animation (high scale)...")
    create_cfg_vs_rectified_side_by_side_animation(
        model, data, labels,
        guidance_scale=9.0,
        n_samples=n_samples,
        num_steps=50,
        save_path=str(output_dir / "cfg_vs_rectified_side_by_side_high_rect.gif"),
        fps=15,
        seed=seed,
        gamma=1.0,
    )

    # 7. Both classes CFG comparison (standard CFG baseline)
    logger.info("Creating both classes CFG comparison...")
    plot_both_classes_cfg(
        model, data, labels,
        cfg_scales=cfg_scales,
        n_samples=n_samples,
        save_path=str(output_dir / "both_classes_cfg_rect.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim, ylim=ylim,
        seed=seed,
    )

    logger.info(f"All visualizations saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Rectified CFG++ visualizations")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["overlapping", "non_overlapping", "multimodal", "all"],
        help="Dataset to visualize (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["overlapping", "non_overlapping"]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'='*60}")
        try:
            generate_visualizations(dataset, seed=args.seed)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset}: {e}")


if __name__ == "__main__":
    main()
