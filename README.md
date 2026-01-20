# Flow Visualizer

Visualizations of flow-based generative models that learn to transform Gaussian noise into structured data distributions.

## Flow Matching

Flow Matching learns a velocity field that transports samples from a simple prior (Gaussian) to a target data distribution. The model is trained to predict the velocity at any point along the interpolation path between noise and data. During sampling, we integrate this learned velocity field using Euler steps to generate new samples.

![Trajectory Curvature](outputs/visualizations/trajectory_curvature.gif)

![Probability Path](outputs/visualizations/probability_path.gif)

## Rectified Flow

Rectified Flow straightens the trajectories learned by flow matching. It works by generating coupled pairs (noise, generated sample) from a trained base model, then retraining on these pairs to learn straighter paths. Straighter trajectories allow for faster sampling with fewer integration steps while maintaining sample quality.

![Rectified Flow Trajectory Curvature](outputs/visualizations/rectified_flow_trajectory_curvature.gif)

![Rectified Flow Probability Path](outputs/visualizations/rectified_flow_probability_path.gif)

## Usage

Train flow matching model:

```bash
uv run fv-train
uv run fv-visualize
```

Train rectified flow model (requires trained base model):

```bash
uv run rf-train
uv run rf-visualize
```

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
