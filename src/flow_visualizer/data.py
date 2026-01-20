"""Dataset generation for flow matching models."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetCreator:
    """Creates the T-Rex shaped dataset from scaling-recipes."""

    def __init__(self, size: int):
        self.size = size

    def create(self) -> torch.Tensor:
        complex_points = torch.polar(torch.tensor(1.0), torch.rand(self.size) * 2 * torch.pi)
        X = torch.stack((complex_points.real, complex_points.imag)).T
        upper = complex_points.imag > 0
        left = complex_points.real < 0
        X[upper, 1] = 0.5
        X[upper & left, 0] = -0.5
        X[upper & ~left, 0] = 0.5
        noise = torch.zeros_like(X)
        noise[upper] = torch.randn_like(noise[upper]) * 0.10
        noise[~upper] = torch.randn_like(noise[~upper]) * 0.05
        X += noise
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        return X + noise


class CustomDataset(Dataset):
    """PyTorch dataset using scaling-recipes data generation."""

    def __init__(self, size: int = 10000):
        self.data = DatasetCreator(size).create()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def create(self, batch_size: int = 10000, shuffle: bool = True):
        return DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)


def generate_spiral_data(n_samples: int, noise: float = 0.1) -> np.ndarray:
    """
    Generate 2D two-spiral dataset.

    Args:
        n_samples: Total number of samples to generate
        noise: Amount of Gaussian noise to add

    Returns:
        Array of shape (n_samples, 2) containing 2D points
    """
    n_per_spiral = n_samples // 2

    theta1 = np.linspace(0, 3 * np.pi, n_per_spiral)
    r1 = theta1 / (3 * np.pi) * 3
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    theta2 = np.linspace(0, 3 * np.pi, n_samples - n_per_spiral)
    r2 = theta2 / (3 * np.pi) * 3
    x2 = r2 * np.cos(theta2 + np.pi)
    y2 = r2 * np.sin(theta2 + np.pi)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    data = np.stack([x, y], axis=1)

    data += noise * np.random.randn(*data.shape)

    return data.astype(np.float32)


class SpiralDataset(Dataset):
    """PyTorch dataset for two-spiral data."""

    def __init__(self, n_samples: int = 10000, noise: float = 0.1):
        self.data = torch.from_numpy(
            generate_spiral_data(n_samples, noise)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
