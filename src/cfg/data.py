"""Dataset generation for CFG flow matching - Face with left eye, right eye, and mouth."""

import torch
from torch.utils.data import Dataset, DataLoader


class FaceDataset(Dataset):
    """Face-shaped dataset with class labels: left eye (0), right eye (1), mouth (2)."""

    def __init__(
        self,
        n_samples: int = 10000,
        left_eye_center: tuple[float, float] = (-0.5, 0.5),
        right_eye_center: tuple[float, float] = (0.5, 0.5),
        eye_sigma: float = 0.15,
    ):
        n_per_class = n_samples // 3

        # Left eye (class 0)
        left_eye = torch.randn(n_per_class, 2) * eye_sigma
        left_eye[:, 0] += left_eye_center[0]
        left_eye[:, 1] += left_eye_center[1]
        left_eye_labels = torch.zeros(n_per_class, dtype=torch.long)

        # Right eye (class 1)
        right_eye = torch.randn(n_per_class, 2) * eye_sigma
        right_eye[:, 0] += right_eye_center[0]
        right_eye[:, 1] += right_eye_center[1]
        right_eye_labels = torch.ones(n_per_class, dtype=torch.long)

        # Mouth (class 2) - arc at bottom
        remaining = n_samples - 2 * n_per_class
        theta = torch.linspace(-0.8 * torch.pi, -0.2 * torch.pi, remaining)
        mouth_x = 0.6 * torch.cos(theta)
        mouth_y = 0.6 * torch.sin(theta) - 0.1
        mouth = torch.stack([mouth_x, mouth_y], dim=1)
        mouth += torch.randn_like(mouth) * 0.08
        mouth_labels = torch.full((remaining,), 2, dtype=torch.long)

        # Combine all
        self.data = torch.cat([left_eye, right_eye, mouth], dim=0)
        self.labels = torch.cat([left_eye_labels, right_eye_labels, mouth_labels], dim=0)

        # Shuffle
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def create(self, batch_size: int = 10000, shuffle: bool = True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @property
    def num_classes(self):
        return 3
