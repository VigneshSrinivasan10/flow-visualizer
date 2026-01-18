"""
Quick demonstration script for Rectified Flow.

This script shows how to:
1. Create a dataset
2. Train a rectified flow model
3. Visualize the results

Usage:
    python examples/rectified_flow_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from flow_visualizer.data import MoonsDataset
from flow_visualizer.model import RectifiedFlowModel, MLPVelocityNet

def main():
    print("Rectified Flow Quick Demo")
    print("=" * 60)

    # Configuration
    device = "cpu"
    n_samples = 1000
    n_epochs = 50
    batch_size = 128
    learning_rate = 0.003

    # Create dataset
    print("\n1. Creating Moons dataset...")
    dataset = MoonsDataset(n_samples=n_samples, noise=0.05)
    print(f"   Created {len(dataset)} samples")

    # Create model
    print("\n2. Creating Rectified Flow model...")
    velocity_net = MLPVelocityNet(
        data_dim=2,
        time_embed_dim=64,
        hidden_dims=[128, 256, 256, 128]
    )
    model = RectifiedFlowModel(velocity_net=velocity_net, device=device)
    optimizer = torch.optim.Adam(velocity_net.parameters(), lr=learning_rate)

    # Training
    print(f"\n3. Training for {n_epochs} epochs...")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

    # Compute straightness
    print("\n4. Computing trajectory straightness...")
    straightness = model.compute_trajectory_straightness(n_samples=500, n_steps=50)
    print(f"   Straightness metric: {straightness:.4f}")

    # Generate samples
    print("\n5. Generating samples...")
    samples = model.sample(n_samples=500, n_steps=50)
    print(f"   Generated {len(samples)} samples")

    # Reflow demonstration
    print("\n6. Demonstrating reflow process...")
    print("   Generating paired samples...")
    x0, x1 = model.generate_reflow_pairs(n_samples=500, n_steps=50)
    print(f"   Generated {len(x0)} paired samples for reflow training")

    print("\n" + "=" * 60)
    print("Demo complete! To run the full training pipeline, use:")
    print("  uv run python -m flow_visualizer.train_rectified_flow")
    print("  uv run python -m flow_visualizer.visualize_rectified_flow")
    print("=" * 60)


if __name__ == "__main__":
    main()
