"""Flow Visualizer - Simplified Diffusion Model Explorer"""

import sys


def main():
    """Main entry point with command selection."""
    if len(sys.argv) < 2:
        print("Flow Visualizer - Simplified Diffusion Model Explorer")
        print("\nUsage:")
        print("  uv run main.py train       - Train the flow matching model")
        print("  uv run main.py visualize   - Visualize trained model results")
        print("  uv run main.py all         - Train and visualize")
        print("\nYou can override config values with Hydra syntax:")
        print("  uv run main.py train training.n_epochs=2000")
        print("  uv run main.py train data.n_spirals=5")
        sys.exit(0)

    command = sys.argv[1]

    if command == "train":
        from flow_visualizer.train import main as train_main
        sys.argv.pop(1)  # Remove 'train' from args for Hydra
        train_main()

    elif command == "visualize":
        from flow_visualizer.visualize import main as visualize_main
        sys.argv.pop(1)  # Remove 'visualize' from args for Hydra
        visualize_main()

    elif command == "all":
        print("Training model...")
        from flow_visualizer.train import main as train_main
        sys.argv.pop(1)
        train_main()

        print("\nGenerating visualizations...")
        from flow_visualizer.visualize import main as visualize_main
        visualize_main()

    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, visualize, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
