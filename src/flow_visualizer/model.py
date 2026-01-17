"""Flow Matching model for 2D data generation."""

import torch
import torch.nn as nn


class MLPVelocityNet(nn.Module):
    """MLP network to predict velocity field for flow matching."""

    def __init__(
        self,
        data_dim: int = 2,
        time_embed_dim: int = 64,
        hidden_dims: list[int] = None,
    ):
        """
        Initialize the velocity network.

        Args:
            data_dim: Dimension of the data (2 for 2D)
            time_embed_dim: Dimension of time embedding
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]

        self.data_dim = data_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding layers
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Main network
        layers = []
        input_dim = data_dim + time_embed_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, data_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at given position and time.

        Args:
            x: Data points of shape (batch_size, data_dim)
            t: Time values of shape (batch_size, 1)

        Returns:
            Predicted velocity of shape (batch_size, data_dim)
        """
        # Embed time
        t_embed = self.time_mlp(t)

        # Concatenate x and time embedding
        h = torch.cat([x, t_embed], dim=-1)

        # Predict velocity
        v = self.net(h)

        return v


class FlowMatchingModel:
    """Flow Matching model for training and sampling."""

    def __init__(
        self,
        velocity_net: nn.Module,
        device: str = "cpu",
    ):
        """
        Initialize Flow Matching model.

        Args:
            velocity_net: Neural network to predict velocity
            device: Device to use for computation
        """
        self.velocity_net = velocity_net.to(device)
        self.device = device

    def compute_loss(
        self,
        x1: torch.Tensor,
        sigma_min: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute Flow Matching loss.

        Args:
            x1: Target data samples of shape (batch_size, data_dim)
            sigma_min: Minimum noise level

        Returns:
            Loss value
        """
        batch_size = x1.shape[0]

        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=self.device)

        # Sample x0 from standard Gaussian
        x0 = torch.randn_like(x1)

        # Interpolate: x_t = t * x1 + (1 - t) * x0
        x_t = t * x1 + (1 - t) * x0

        # Target velocity: v_t = x1 - x0
        v_target = x1 - x0

        # Predict velocity
        v_pred = self.velocity_net(x_t, t)

        # Compute MSE loss
        loss = torch.mean((v_pred - v_target) ** 2)

        return loss

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

        # Start from Gaussian noise
        x = torch.randn(n_samples, data_dim, device=self.device)

        dt = 1.0 / n_steps

        # Euler integration
        for step in range(n_steps):
            t = torch.ones(n_samples, 1, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, t)
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

        # Start from Gaussian noise
        x = torch.randn(n_samples, data_dim, device=self.device)
        trajectory = [x.cpu().clone()]

        dt = 1.0 / n_steps

        # Euler integration
        for step in range(n_steps):
            t = torch.ones(n_samples, 1, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, t)
            x = x + v * dt
            trajectory.append(x.cpu().clone())

        return trajectory
