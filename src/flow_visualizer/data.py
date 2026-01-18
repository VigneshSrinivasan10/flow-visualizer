"""Dataset generation for flow matching models."""

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_trex_data(n_samples: int, noise: float = 0.02) -> np.ndarray:
    """
    Generate 2D T-Rex shaped dataset inspired by Google Chrome T-Rex runner game.

    The T-Rex has a pixelated, blocky appearance with:
    - Large rectangular head with an eye
    - Short thick neck
    - Rounded body
    - Tiny arms
    - Strong legs with feet
    - Horizontal tail

    Args:
        n_samples: Total number of samples to generate
        noise: Amount of Gaussian noise to add

    Returns:
        Array of shape (n_samples, 2) containing 2D points
    """
    points = []

    # Define Chrome T-Rex outline with blocky/pixelated body parts
    # Each part gets a proportion of the total samples

    # Head (blocky rectangular with rounded top) - 18%
    n_head = int(0.18 * n_samples)
    # Create blocky head shape
    head_top = int(n_head * 0.3)
    head_back = int(n_head * 0.2)
    head_front = int(n_head * 0.3)
    head_jaw = int(n_head * 0.2)

    head_x = np.concatenate([
        np.linspace(0.5, 0.8, head_top),      # top of head
        np.full(head_back, 0.5),              # back of head
        np.linspace(0.5, 0.9, head_front),    # front/snout
        np.linspace(0.9, 0.6, head_jaw)       # jaw line
    ])
    head_y = np.concatenate([
        np.full(head_top, 0.7),                           # top flat
        np.linspace(0.7, 0.4, head_back),                # back edge
        np.linspace(0.6, 0.5, head_front),               # snout slope
        np.full(head_jaw, 0.4)                           # jaw flat
    ])

    # Eye (small dot) - 2%
    n_eye = int(0.02 * n_samples)
    eye_x = 0.7 + 0.03 * np.cos(np.linspace(0, 2*np.pi, n_eye))
    eye_y = 0.62 + 0.03 * np.sin(np.linspace(0, 2*np.pi, n_eye))

    # Neck (short and thick) - 8%
    n_neck = int(0.08 * n_samples)
    neck_x = np.linspace(0.55, 0.35, n_neck)
    neck_y = np.linspace(0.4, 0.25, n_neck)

    # Body (rounded, egg-like) - 16%
    n_body = int(0.16 * n_samples)
    t_body = np.linspace(0, 2 * np.pi, n_body)
    body_x = 0.1 + 0.3 * np.cos(t_body)
    body_y = 0.05 + 0.35 * np.sin(t_body)

    # Tail (horizontal, tapering, slightly lifted) - 14%
    n_tail = int(0.14 * n_samples)
    t_tail = np.linspace(0, 1, n_tail)
    tail_x = -0.2 - 0.7 * t_tail
    # Tail curves up slightly and tapers
    tail_thickness = (1 - t_tail * 0.8)
    tail_y_upper = 0.15 + 0.1 * t_tail + 0.08 * tail_thickness
    tail_y_lower = 0.15 + 0.1 * t_tail - 0.08 * tail_thickness

    # Combine upper and lower tail edges
    tail_x = np.concatenate([tail_x, tail_x[::-1]])
    tail_y = np.concatenate([tail_y_upper, tail_y_lower[::-1]])

    # Back leg (thick, strong, bent) - 13%
    n_leg1 = int(0.13 * n_samples)
    leg1_upper = int(n_leg1 * 0.4)
    leg1_lower = int(n_leg1 * 0.4)
    leg1_foot = n_leg1 - leg1_upper - leg1_lower

    leg1_x = np.concatenate([
        np.linspace(-0.05, 0.0, leg1_upper),      # thigh
        np.linspace(0.0, 0.05, leg1_lower),       # shin/calf
        np.linspace(0.05, 0.15, leg1_foot)        # foot
    ])
    leg1_y = np.concatenate([
        np.linspace(0.0, -0.35, leg1_upper),
        np.linspace(-0.35, -0.65, leg1_lower),
        np.full(leg1_foot, -0.65)                  # foot flat on ground
    ])

    # Front leg (thick, strong, straighter) - 13%
    n_leg2 = int(0.13 * n_samples)
    leg2_upper = int(n_leg2 * 0.45)
    leg2_lower = int(n_leg2 * 0.35)
    leg2_foot = n_leg2 - leg2_upper - leg2_lower

    leg2_x = np.concatenate([
        np.linspace(0.25, 0.28, leg2_upper),
        np.linspace(0.28, 0.26, leg2_lower),
        np.linspace(0.26, 0.36, leg2_foot)
    ])
    leg2_y = np.concatenate([
        np.linspace(0.0, -0.4, leg2_upper),
        np.linspace(-0.4, -0.65, leg2_lower),
        np.full(leg2_foot, -0.65)
    ])

    # Tiny arms (very small, characteristic of T-Rex) - 8% each
    n_arm1 = int(0.08 * n_samples)
    arm1_x = np.linspace(0.28, 0.32, n_arm1)
    arm1_y = np.linspace(0.3, 0.15, n_arm1)

    n_arm2 = int(0.08 * n_samples)
    arm2_x = np.linspace(0.32, 0.36, n_arm2)
    arm2_y = np.linspace(0.28, 0.13, n_arm2)

    # Combine all parts
    all_x = np.concatenate([head_x, eye_x, neck_x, body_x, tail_x,
                            leg1_x, leg2_x, arm1_x, arm2_x])
    all_y = np.concatenate([head_y, eye_y, neck_y, body_y, tail_y,
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
