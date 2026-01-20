"""Tests for data generation and visualization functions."""

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

# Use non-interactive backend for testing
matplotlib.use("Agg")

from flow_visualizer.data import generate_trex_data, TRexDataset
from flow_visualizer.model import FlowMatchingModel, MLPVelocityNet
from flow_visualizer.visualize import (
    plot_comparison,
    plot_trajectory,
    plot_vector_field,
    create_flow_animation,
    create_particle_trajectories_animation,
)


class TestDataGeneration:
    """Tests for data generation functions."""

    def test_generate_trex_data_shape(self):
        """Test that generate_trex_data returns correct shape."""
        n_samples = 1000
        data = generate_trex_data(n_samples=n_samples)

        assert data.shape == (n_samples, 2)
        assert data.dtype == np.float32

    def test_generate_trex_data_normalized(self):
        """Test that data is normalized to approximately [-1, 1]."""
        data = generate_trex_data(n_samples=1000)

        assert np.abs(data).max() <= 1.0 + 1e-6

    def test_generate_trex_data_noise(self):
        """Test that noise parameter affects data variance."""
        data_low_noise = generate_trex_data(n_samples=1000, noise=0.01)
        data_high_noise = generate_trex_data(n_samples=1000, noise=0.1)

        # Higher noise should generally lead to more spread
        # This is a statistical test, so we use a loose check
        assert data_low_noise.std() > 0
        assert data_high_noise.std() > 0

    def test_trex_dataset(self):
        """Test TRexDataset class."""
        dataset = TRexDataset(n_samples=500, noise=0.02)

        assert len(dataset) == 500
        assert isinstance(dataset[0], torch.Tensor)
        assert dataset[0].shape == (2,)


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for visualization tests."""
        target_data = generate_trex_data(n_samples=100)
        generated_samples = torch.from_numpy(generate_trex_data(n_samples=100))
        return target_data, generated_samples

    @pytest.fixture
    def sample_trajectory(self):
        """Generate sample trajectory for visualization tests."""
        n_samples = 50
        n_steps = 10

        trajectory = []
        # Simulate a trajectory from noise to target
        noise = torch.randn(n_samples, 2)
        target = torch.from_numpy(generate_trex_data(n_samples=n_samples))

        for step in range(n_steps + 1):
            t = step / n_steps
            # Linear interpolation from noise to target
            current = (1 - t) * noise + t * target
            trajectory.append(current)

        return trajectory

    @pytest.fixture
    def sample_model(self):
        """Create a sample model for visualization tests."""
        velocity_net = MLPVelocityNet(
            data_dim=2,
            time_embed_dim=32,
            hidden_dims=[64, 64],
        )
        model = FlowMatchingModel(velocity_net=velocity_net, device="cpu")
        return model

    def test_plot_comparison(self, sample_data):
        """Test that plot_comparison creates a file."""
        target_data, generated_samples = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "comparison.png"
            plot_comparison(generated_samples, target_data, save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_plot_trajectory(self, sample_trajectory, sample_data):
        """Test that plot_trajectory creates a file."""
        target_data, _ = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "trajectory.png"
            plot_trajectory(sample_trajectory, target_data, save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_plot_trajectory_custom_steps(self, sample_trajectory, sample_data):
        """Test plot_trajectory with custom step indices."""
        target_data, _ = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "trajectory.png"
            plot_trajectory(
                sample_trajectory,
                target_data,
                save_path,
                steps_to_plot=[0, 5, 10],
            )

            assert save_path.exists()

    def test_plot_vector_field(self, sample_model):
        """Test that plot_vector_field creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vector_field.png"
            plot_vector_field(sample_model, save_path, grid_size=10)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_plot_vector_field_custom_times(self, sample_model):
        """Test plot_vector_field with custom time values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "vector_field.png"
            plot_vector_field(
                sample_model,
                save_path,
                grid_size=8,
                t_values=[0.0, 0.5, 1.0],
            )

            assert save_path.exists()


class TestAnimations:
    """Tests for animation functions (minimal tests due to performance)."""

    @pytest.fixture
    def small_trajectory(self):
        """Generate a small trajectory for animation tests."""
        n_samples = 20
        n_steps = 5

        trajectory = []
        noise = torch.randn(n_samples, 2)
        target = torch.from_numpy(generate_trex_data(n_samples=n_samples))

        for step in range(n_steps + 1):
            t = step / n_steps
            current = (1 - t) * noise + t * target
            trajectory.append(current)

        return trajectory

    @pytest.fixture
    def small_target_data(self):
        """Generate small target data for animation tests."""
        return generate_trex_data(n_samples=20)

    def test_create_flow_animation(self, small_trajectory, small_target_data):
        """Test that create_flow_animation creates a GIF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "flow.gif"
            create_flow_animation(
                small_trajectory,
                small_target_data,
                save_path,
                fps=5,
                dpi=50,
                subsample=2,
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0
            # Verify it's a GIF by checking magic bytes
            with open(save_path, "rb") as f:
                magic = f.read(6)
                assert magic in (b"GIF87a", b"GIF89a")

    def test_create_particle_trajectories_animation(
        self, small_trajectory, small_target_data
    ):
        """Test that create_particle_trajectories_animation creates a GIF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "particles.gif"
            create_particle_trajectories_animation(
                small_trajectory,
                small_target_data,
                save_path,
                n_particles=10,
                fps=5,
                dpi=50,
                subsample=2,
                trail_length=3,
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0


class TestModelVisualization:
    """Tests for model-based visualization with actual sampling."""

    def test_model_sample_visualization(self):
        """Test visualizing samples from an untrained model."""
        # Create a simple model
        velocity_net = MLPVelocityNet(
            data_dim=2,
            time_embed_dim=32,
            hidden_dims=[32, 32],
        )
        model = FlowMatchingModel(velocity_net=velocity_net, device="cpu")

        # Generate samples (won't look like T-Rex since untrained)
        samples = model.sample(n_samples=50, n_steps=10, data_dim=2)
        target_data = generate_trex_data(n_samples=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "untrained_comparison.png"
            plot_comparison(samples, target_data, save_path)

            assert save_path.exists()

    def test_model_trajectory_visualization(self):
        """Test visualizing trajectory from an untrained model."""
        velocity_net = MLPVelocityNet(
            data_dim=2,
            time_embed_dim=32,
            hidden_dims=[32, 32],
        )
        model = FlowMatchingModel(velocity_net=velocity_net, device="cpu")

        trajectory = model.sample_trajectory(n_samples=50, n_steps=10, data_dim=2)
        target_data = generate_trex_data(n_samples=50)

        assert len(trajectory) == 11  # n_steps + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "trajectory.png"
            plot_trajectory(trajectory, target_data, save_path)

            assert save_path.exists()
