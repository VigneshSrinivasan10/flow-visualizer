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


class RectifiedFlowModel(FlowMatchingModel):
    """
    Rectified Flow model that learns straighter trajectories.

    Rectified Flow improves upon standard flow matching by iteratively
    'straightening' the learned trajectories through a reflow operation.
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        device: str = "cpu",
    ):
        """
        Initialize Rectified Flow model.

        Args:
            velocity_net: Neural network to predict velocity
            device: Device to use for computation
        """
        super().__init__(velocity_net, device)
        self.reflow_data = None

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        n_samples: int,
        n_steps: int = 100,
        data_dim: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate paired (x0, x1) samples for reflow training.

        This samples initial noise x0 and flows it through the current
        velocity field to get x1. Training on these pairs helps straighten
        the trajectories.

        Args:
            n_samples: Number of pairs to generate
            n_steps: Number of integration steps
            data_dim: Dimension of the data

        Returns:
            Tuple of (x0, x1) tensors, each of shape (n_samples, data_dim)
        """
        self.velocity_net.eval()

        # Sample initial noise
        x0 = torch.randn(n_samples, data_dim, device=self.device)

        # Flow from x0 to x1 using current model
        x = x0.clone()
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.ones(n_samples, 1, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, t)
            x = x + v * dt

        x1 = x

        return x0, x1

    def compute_reflow_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rectified flow loss using paired samples.

        Unlike standard flow matching which uses random x0 and data x1,
        rectified flow uses paired samples generated from a previous model.

        Args:
            x0: Initial points of shape (batch_size, data_dim)
            x1: Target points of shape (batch_size, data_dim)

        Returns:
            Loss value
        """
        batch_size = x1.shape[0]

        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=self.device)

        # Interpolate: x_t = t * x1 + (1 - t) * x0
        x_t = t * x1 + (1 - t) * x0

        # Target velocity: v_t = x1 - x0 (straight line)
        v_target = x1 - x0

        # Predict velocity
        v_pred = self.velocity_net(x_t, t)

        # Compute MSE loss
        loss = torch.mean((v_pred - v_target) ** 2)

        return loss

    @torch.no_grad()
    def compute_trajectory_straightness(
        self,
        n_samples: int = 1000,
        n_steps: int = 100,
        data_dim: int = 2,
    ) -> float:
        """
        Compute a straightness metric for trajectories.

        Measures the ratio of path length to straight-line distance.
        A value of 1.0 means perfectly straight trajectories.

        Args:
            n_samples: Number of trajectories to sample
            n_steps: Number of integration steps
            data_dim: Dimension of the data

        Returns:
            Average straightness score (lower is straighter, 1.0 is perfect)
        """
        self.velocity_net.eval()

        x0 = torch.randn(n_samples, data_dim, device=self.device)
        x = x0.clone()

        dt = 1.0 / n_steps
        path_length = 0.0

        for step in range(n_steps):
            t = torch.ones(n_samples, 1, device=self.device) * (step / n_steps)
            v = self.velocity_net(x, t)
            dx = v * dt
            path_length += torch.sqrt(torch.sum(dx ** 2, dim=1)).mean().item()
            x = x + dx

        x1 = x

        # Compute straight-line distance
        straight_distance = torch.sqrt(torch.sum((x1 - x0) ** 2, dim=1)).mean().item()

        # Straightness ratio (1.0 is perfect)
        straightness = path_length / (straight_distance + 1e-8)

        return straightness
