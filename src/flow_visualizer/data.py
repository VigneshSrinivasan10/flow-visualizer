"""Dataset generation for flow matching models."""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons, make_circles


def generate_trex_data(n_samples: int, noise: float = 0.02) -> np.ndarray:
    """
    Generate 2D T-Rex shaped dataset.

    Args:
        n_samples: Total number of samples to generate
        noise: Amount of Gaussian noise to add

    Returns:
        Array of shape (n_samples, 2) containing 2D points
    """
    points = []

    # Define T-Rex outline with different body parts
    # Each part gets a proportion of the total samples

    # Head (large, with jaw) - 20%
    n_head = int(0.20 * n_samples)
    t_head = np.linspace(0, np.pi, n_head)
    head_x = 0.8 + 0.3 * np.cos(t_head)
    head_y = 0.4 + 0.35 * np.sin(t_head)

    # Jaw (bottom part of head) - 10%
    n_jaw = int(0.10 * n_samples)
    jaw_x = np.linspace(0.5, 1.1, n_jaw)
    jaw_y = 0.4 - 0.15 * np.sin(np.linspace(0, np.pi, n_jaw))

    # Neck - 8%
    n_neck = int(0.08 * n_samples)
    neck_x = np.linspace(0.5, 0.3, n_neck)
    neck_y = np.linspace(0.4, 0.2, n_neck)

    # Body (rounded) - 15%
    n_body = int(0.15 * n_samples)
    t_body = np.linspace(0, 2 * np.pi, n_body)
    body_x = 0.0 + 0.35 * np.cos(t_body)
    body_y = 0.0 + 0.3 * np.sin(t_body)

    # Tail (long and tapering) - 15%
    n_tail = int(0.15 * n_samples)
    t_tail = np.linspace(0, 1, n_tail)
    tail_x = -0.35 - 0.8 * t_tail
    tail_y = 0.0 + 0.3 * np.sin(t_tail * np.pi) * (1 - t_tail * 0.7)

    # Left leg (back leg) - 12%
    n_leg1 = int(0.12 * n_samples)
    leg1_x = np.concatenate([
        np.linspace(-0.2, -0.15, n_leg1 // 2),  # thigh
        np.linspace(-0.15, -0.1, n_leg1 // 2)   # shin
    ])
    leg1_y = np.concatenate([
        np.linspace(-0.3, -0.6, n_leg1 // 2),
        np.linspace(-0.6, -0.8, n_leg1 // 2)
    ])

    # Right leg (front leg) - 12%
    n_leg2 = int(0.12 * n_samples)
    leg2_x = np.concatenate([
        np.linspace(0.15, 0.2, n_leg2 // 2),
        np.linspace(0.2, 0.25, n_leg2 // 2)
    ])
    leg2_y = np.concatenate([
        np.linspace(-0.3, -0.6, n_leg2 // 2),
        np.linspace(-0.6, -0.8, n_leg2 // 2)
    ])

    # Left arm (tiny) - 4%
    n_arm1 = int(0.04 * n_samples)
    arm1_x = np.linspace(0.2, 0.25, n_arm1)
    arm1_y = np.linspace(0.15, 0.0, n_arm1)

    # Right arm (tiny) - 4%
    n_arm2 = int(0.04 * n_samples)
    arm2_x = np.linspace(0.25, 0.3, n_arm2)
    arm2_y = np.linspace(0.1, -0.05, n_arm2)

    # Combine all parts
    all_x = np.concatenate([head_x, jaw_x, neck_x, body_x, tail_x,
                            leg1_x, leg2_x, arm1_x, arm2_x])
    all_y = np.concatenate([head_y, jaw_y, neck_y, body_y, tail_y,
                            leg1_y, leg2_y, arm1_y, arm2_y])

    # Stack into points
    data = np.stack([all_x, all_y], axis=1)

    # Add noise
    data += noise * np.random.randn(*data.shape)

    # Normalize to roughly [-1, 1]
    data = data / (np.abs(data).max() + 1e-8)

    return data.astype(np.float32)


class TRexDataset(Dataset):
    """PyTorch dataset for T-Rex shaped data."""

    def __init__(self, n_samples: int = 10000, noise: float = 0.02):
        """
        Initialize T-Rex dataset.

        Args:
            n_samples: Number of samples to generate
            noise: Amount of Gaussian noise to add
        """
        self.data = torch.from_numpy(
            generate_trex_data(n_samples, noise)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_moons_data(n_samples: int, noise: float = 0.05) -> np.ndarray:
    """
    Generate 2D two moons dataset.

    Args:
        n_samples: Total number of samples to generate
        noise: Amount of Gaussian noise to add

    Returns:
        Array of shape (n_samples, 2) containing 2D points
    """
    data, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    # Normalize to roughly [-1, 1]
    data = data / (np.abs(data).max() + 1e-8)
    return data.astype(np.float32)


def generate_circles_data(n_samples: int, noise: float = 0.05, factor: float = 0.5) -> np.ndarray:
    """
    Generate 2D concentric circles dataset.

    Args:
        n_samples: Total number of samples to generate
        noise: Amount of Gaussian noise to add
        factor: Scale factor between inner and outer circle (0 < factor < 1)

    Returns:
        Array of shape (n_samples, 2) containing 2D points
    """
    data, _ = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    # Normalize to roughly [-1, 1]
    data = data / (np.abs(data).max() + 1e-8)
    return data.astype(np.float32)


class MoonsDataset(Dataset):
    """PyTorch dataset for two moons shaped data."""

    def __init__(self, n_samples: int = 10000, noise: float = 0.05):
        """
        Initialize Moons dataset.

        Args:
            n_samples: Number of samples to generate
            noise: Amount of Gaussian noise to add
        """
        self.data = torch.from_numpy(
            generate_moons_data(n_samples, noise)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CirclesDataset(Dataset):
    """PyTorch dataset for concentric circles shaped data."""

    def __init__(self, n_samples: int = 10000, noise: float = 0.05, factor: float = 0.5):
        """
        Initialize Circles dataset.

        Args:
            n_samples: Number of samples to generate
            noise: Amount of Gaussian noise to add
            factor: Scale factor between inner and outer circle
        """
        self.data = torch.from_numpy(
            generate_circles_data(n_samples, noise, factor)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
