"""Loss functions for CFG flow matching with class dropout."""

import torch


class CFGFlowMatchingLoss:
    """Conditional Flow Matching loss with classifier-free guidance training."""

    def __init__(self, sigma_min: float = 1e-4, class_dropout_prob: float = 0.2, null_class_idx: int = 3):
        self.sigma_min = sigma_min
        self.class_dropout_prob = class_dropout_prob
        self.null_class_idx = null_class_idx

    def __call__(self, flow_model, data):
        batch = next(iter(data))
        x, labels = batch

        t = torch.rand(x.shape[0], device=x.device)
        noise = torch.randn_like(x)

        x_t = (1 - (1 - self.sigma_min) * t[:, None]) * noise + t[:, None] * x
        optimal_flow = x - (1 - self.sigma_min) * noise

        # Apply class dropout: replace class labels with null class with probability class_dropout_prob
        dropout_mask = torch.rand(labels.shape[0], device=labels.device) < self.class_dropout_prob
        labels_with_dropout = labels.clone()
        labels_with_dropout[dropout_mask] = self.null_class_idx

        predicted_flow = flow_model(x_t, time=t, class_labels=labels_with_dropout)

        return (predicted_flow - optimal_flow).square().mean()
