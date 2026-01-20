"""Flow Matching model for 2D data generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class ZeroToOneTimeEmbedding(nn.Module):
    """Time embedding for t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.arange(1, dim // 2 + 1) * torch.pi)

    def forward(self, t):
        emb = self.freqs * t[..., None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMLP(nn.Module):
    """Simple MLP network for flow matching."""

    def __init__(
        self,
        n_features: int = 2,
        width: int = 128,
        n_blocks: int = 5,
    ):
        super().__init__()

        self.n_features = n_features
        self.time_embed_dim = width

        # Time embedding
        self.time_embedding = ZeroToOneTimeEmbedding(self.time_embed_dim)

        # Build MLP layers
        layers = []
        input_dim = n_features + self.time_embed_dim

        for _ in range(n_blocks):
            layers.extend([
                nn.Linear(input_dim, width),
                nn.SiLU(),
            ])
            input_dim = width

        layers.append(nn.Linear(width, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, time: torch.Tensor = None) -> torch.Tensor:
        """
        Predict velocity at given position and time.

        Args:
            x: Data points of shape (batch_size, n_features)
            time: Time values of shape (batch_size,) or (batch_size, 1)

        Returns:
            Predicted velocity of shape (batch_size, n_features)
        """
        if time is None:
            time = torch.rand(x.shape[0], device=x.device)
        if time.dim() == 2:
            time = time.squeeze(-1)

        t_embed = self.time_embedding(time)
        h = torch.cat([x, t_embed], dim=-1)
        return self.net(h)


class FlowMatchingModel:
    """Flow Matching model for training and sampling."""

    def __init__(
        self,
        velocity_net: nn.Module,
        device: str = "cpu",
    ):
        self.velocity_net = velocity_net.to(device)
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_steps: int = 100,
        data_dim: int = 2,
    ) -> torch.Tensor:
        """
        Generate samples using Euler integration.

        Args:
            n_samples: Number of samples to generate
            n_steps: Number of integration steps
            data_dim: Dimension of the data

        Returns:
            Generated samples of shape (n_samples, data_dim)
        """
        self.velocity_net.eval()

        x = torch.randn(n_samples, data_dim, device=self.device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, time=t)
            x = x + v * dt

        return x

    @torch.no_grad()
    def sample_trajectory(
        self,
        n_samples: int,
        n_steps: int = 100,
        data_dim: int = 2,
    ) -> list[torch.Tensor]:
        """
        Generate samples and return full trajectory.

        Args:
            n_samples: Number of samples to generate
            n_steps: Number of integration steps
            data_dim: Dimension of the data

        Returns:
            List of tensors, each of shape (n_samples, data_dim)
        """
        self.velocity_net.eval()

        x = torch.randn(n_samples, data_dim, device=self.device)
        trajectory = [x.cpu().clone()]

        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, time=t)
            x = x + v * dt
            trajectory.append(x.cpu().clone())

        return trajectory
