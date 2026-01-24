"""Dataset generators for 2D Gaussian experiments."""

from typing import Callable, Dict, List, Tuple

import numpy as np


def generate_overlapping_gaussians(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate overlapping 2D Gaussian data for two classes.

    Class 0: mean = [-0.3, -0.3], std = 0.5
    Class 1: mean = [0.3, 0.3], std = 0.5

    The distributions overlap significantly, making CFG > 1 necessary for separation.

    Args:
        n_samples: Total number of samples (split evenly between classes)

    Returns:
        data: (n_samples, 2) array of 2D points
        labels: (n_samples,) array of class labels (0 or 1)
    """
    n_half = n_samples // 2
    mean1 = [-0.3, -0.3]
    mean2 = [0.3, 0.3]
    std = 0.5

    data1 = np.random.randn(n_half, 2) * std + mean1
    data2 = np.random.randn(n_samples - n_half, 2) * std + mean2

    data = np.vstack([data1, data2])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)])

    return data, labels


def generate_non_overlapping_gaussians(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate non-overlapping 2D Gaussian data for two classes - far apart.

    Class 0: mean = [-2, -2], std = 0.3
    Class 1: mean = [2, 2], std = 0.3

    The distributions are far apart with essentially no overlap.

    Args:
        n_samples: Total number of samples (split evenly between classes)

    Returns:
        data: (n_samples, 2) array of 2D points
        labels: (n_samples,) array of class labels (0 or 1)
    """
    n_half = n_samples // 2
    mean1 = [-2, -2]
    mean2 = [2, 2]
    std = 0.3

    data1 = np.random.randn(n_half, 2) * std + mean1
    data2 = np.random.randn(n_samples - n_half, 2) * std + mean2

    data = np.vstack([data1, data2])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)])

    return data, labels


def generate_multimodal_non_overlapping(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multimodal non-overlapping 2D data for two classes.

    Class 0: 3 modes at [-2, 0], [-1, -1.5], [-1.5, 1.5]
    Class 1: 3 modes at [2, 0], [1, 1.5], [1.5, -1.5]

    Each class has multiple clusters, positioned so they don't overlap with the other class.

    Args:
        n_samples: Total number of samples (split evenly between classes)

    Returns:
        data: (n_samples, 2) array of 2D points
        labels: (n_samples,) array of class labels (0 or 1)
    """
    n_half = n_samples // 2
    n_per_mode = n_half // 3
    std = 0.25

    # Class 0 modes (left side of space)
    modes_0 = [[-2, 0], [-1, -1.5], [-1.5, 1.5]]
    data0_list = []
    for i, mean in enumerate(modes_0):
        n = n_per_mode if i < 2 else n_half - 2 * n_per_mode
        data0_list.append(np.random.randn(n, 2) * std + mean)
    data0 = np.vstack(data0_list)

    # Class 1 modes (right side of space)
    n_class1 = n_samples - n_half
    n_per_mode_1 = n_class1 // 3
    modes_1 = [[2, 0], [1, 1.5], [1.5, -1.5]]
    data1_list = []
    for i, mean in enumerate(modes_1):
        n = n_per_mode_1 if i < 2 else n_class1 - 2 * n_per_mode_1
        data1_list.append(np.random.randn(n, 2) * std + mean)
    data1 = np.vstack(data1_list)

    data = np.vstack([data0, data1])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_class1)])

    return data, labels


# Dataset configurations registry
DATASET_CONFIGS: Dict[str, Dict] = {
    "overlapping": {
        "generator": generate_overlapping_gaussians,
        "title": "Overlapping Gaussians",
        "class0_centers": [[-0.3, -0.3]],
        "class1_centers": [[0.3, 0.3]],
    },
    "non_overlapping": {
        "generator": generate_non_overlapping_gaussians,
        "title": "Non-Overlapping Gaussians (Far Apart)",
        "class0_centers": [[-2, -2]],
        "class1_centers": [[2, 2]],
    },
    "multimodal": {
        "generator": generate_multimodal_non_overlapping,
        "title": "Multimodal Non-Overlapping",
        "class0_centers": [[-2, 0], [-1, -1.5], [-1.5, 1.5]],
        "class1_centers": [[2, 0], [1, 1.5], [1.5, -1.5]],
    },
}


def get_dataset_config(name: str) -> Dict:
    """Get a dataset configuration by name.

    Args:
        name: One of 'overlapping', 'non_overlapping', 'multimodal'

    Returns:
        Configuration dict with generator function, title, and class centers
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name]
