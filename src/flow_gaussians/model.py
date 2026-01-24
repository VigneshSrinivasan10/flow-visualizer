"""Neural network model for flow matching (numpy implementation)."""

from typing import Dict, Tuple

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)"""
    return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))


def silu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
    sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return sig + x * sig * (1 - sig)


class AdamOptimizer:
    """Adam optimizer for manual gradient descent."""

    def __init__(
        self, lr: float = 5e-4, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[str, np.ndarray] = {}  # First moment
        self.v: Dict[str, np.ndarray] = {}  # Second moment
        self.t = 0  # Time step

    def step(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Perform one optimization step."""
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update parameters
            params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params


class SimpleFlowNetwork:
    """
    Simple Flow Network for 2D flow matching.

    Architecture:
    - Time embedding: MLP (1 → hidden_dim → hidden_dim) with SiLU activation
    - Class embedding: Learnable embedding matrix (3 classes: 0, 1, unconditional=-1)
    - Main network: 4-layer MLP with SiLU activations
    - Output: 2D velocity vector

    Input concatenation: [x (2D), time_emb (hidden_dim), class_emb (hidden_dim)]
    """

    def __init__(self, hidden_dim: int = 128, lr: float = 5e-4):
        self.hidden_dim = hidden_dim

        # Initialize parameters with He initialization
        scale = np.sqrt(2.0)

        # Time embedding MLP: 1 → hidden_dim → hidden_dim
        self.time_fc1_w = np.random.randn(1, hidden_dim) * scale / np.sqrt(1)
        self.time_fc1_b = np.zeros(hidden_dim)
        self.time_fc2_w = np.random.randn(hidden_dim, hidden_dim) * scale / np.sqrt(hidden_dim)
        self.time_fc2_b = np.zeros(hidden_dim)

        # Class embedding: 3 classes (0, 1, -1 for unconditional)
        # Map: -1 → 0, 0 → 1, 1 → 2
        self.class_emb = np.random.randn(3, hidden_dim) * 0.02

        # Main network: 4-layer MLP
        # Input: 2 + hidden_dim + hidden_dim
        input_dim = 2 + hidden_dim + hidden_dim

        self.fc1_w = np.random.randn(input_dim, hidden_dim) * scale / np.sqrt(input_dim)
        self.fc1_b = np.zeros(hidden_dim)

        self.fc2_w = np.random.randn(hidden_dim, hidden_dim) * scale / np.sqrt(hidden_dim)
        self.fc2_b = np.zeros(hidden_dim)

        self.fc3_w = np.random.randn(hidden_dim, hidden_dim) * scale / np.sqrt(hidden_dim)
        self.fc3_b = np.zeros(hidden_dim)

        self.fc4_w = np.random.randn(hidden_dim, 2) * 0.01  # Output layer, small init
        self.fc4_b = np.zeros(2)

        self.optimizer = AdamOptimizer(lr=lr)

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get all parameters as a dictionary."""
        return {
            "time_fc1_w": self.time_fc1_w,
            "time_fc1_b": self.time_fc1_b,
            "time_fc2_w": self.time_fc2_w,
            "time_fc2_b": self.time_fc2_b,
            "class_emb": self.class_emb,
            "fc1_w": self.fc1_w,
            "fc1_b": self.fc1_b,
            "fc2_w": self.fc2_w,
            "fc2_b": self.fc2_b,
            "fc3_w": self.fc3_w,
            "fc3_b": self.fc3_b,
            "fc4_w": self.fc4_w,
            "fc4_b": self.fc4_b,
        }

    def set_params(self, params: Dict[str, np.ndarray]):
        """Set all parameters from a dictionary."""
        self.time_fc1_w = params["time_fc1_w"]
        self.time_fc1_b = params["time_fc1_b"]
        self.time_fc2_w = params["time_fc2_w"]
        self.time_fc2_b = params["time_fc2_b"]
        self.class_emb = params["class_emb"]
        self.fc1_w = params["fc1_w"]
        self.fc1_b = params["fc1_b"]
        self.fc2_w = params["fc2_w"]
        self.fc2_b = params["fc2_b"]
        self.fc3_w = params["fc3_w"]
        self.fc3_b = params["fc3_b"]
        self.fc4_w = params["fc4_w"]
        self.fc4_b = params["fc4_b"]

    def forward(self, x: np.ndarray, t: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass of the flow network.

        Args:
            x: (batch_size, 2) - 2D positions
            t: (batch_size, 1) - time values in [0, 1]
            labels: (batch_size,) - class labels (0, 1, or -1 for unconditional)

        Returns:
            v: (batch_size, 2) - predicted velocity vectors
            cache: Dictionary of intermediate values for backprop
        """
        cache = {}

        # Time embedding
        cache["t_input"] = t
        t_h1_pre = t @ self.time_fc1_w + self.time_fc1_b
        cache["t_h1_pre"] = t_h1_pre
        t_h1 = silu(t_h1_pre)
        cache["t_h1"] = t_h1

        t_h2_pre = t_h1 @ self.time_fc2_w + self.time_fc2_b
        cache["t_h2_pre"] = t_h2_pre
        t_emb = silu(t_h2_pre)
        cache["t_emb"] = t_emb

        # Class embedding (map -1 → 0, 0 → 1, 1 → 2)
        label_idx = (labels + 1).astype(int)
        cache["label_idx"] = label_idx
        c_emb = self.class_emb[label_idx]
        cache["c_emb"] = c_emb

        # Concatenate inputs: [x, time_emb, class_emb]
        concat_input = np.concatenate([x, t_emb, c_emb], axis=1)
        cache["concat_input"] = concat_input
        cache["x"] = x

        # Main network forward pass
        h1_pre = concat_input @ self.fc1_w + self.fc1_b
        cache["h1_pre"] = h1_pre
        h1 = silu(h1_pre)
        cache["h1"] = h1

        h2_pre = h1 @ self.fc2_w + self.fc2_b
        cache["h2_pre"] = h2_pre
        h2 = silu(h2_pre)
        cache["h2"] = h2

        h3_pre = h2 @ self.fc3_w + self.fc3_b
        cache["h3_pre"] = h3_pre
        h3 = silu(h3_pre)
        cache["h3"] = h3

        v = h3 @ self.fc4_w + self.fc4_b

        return v, cache

    def backward(
        self, v_pred: np.ndarray, v_target: np.ndarray, cache: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients.

        Args:
            v_pred: (batch_size, 2) - predicted velocities
            v_target: (batch_size, 2) - target velocities
            cache: Dictionary of intermediate values from forward pass

        Returns:
            grads: Dictionary of gradients for all parameters
        """
        batch_size = v_pred.shape[0]
        grads = {}

        # Loss gradient: d(MSE)/d(v_pred) = 2 * (v_pred - v_target) / batch_size
        dv = 2 * (v_pred - v_target) / batch_size

        # Output layer gradients
        grads["fc4_w"] = cache["h3"].T @ dv
        grads["fc4_b"] = np.sum(dv, axis=0)

        # Backprop through h3
        dh3 = dv @ self.fc4_w.T
        dh3_pre = dh3 * silu_derivative(cache["h3_pre"])

        grads["fc3_w"] = cache["h2"].T @ dh3_pre
        grads["fc3_b"] = np.sum(dh3_pre, axis=0)

        # Backprop through h2
        dh2 = dh3_pre @ self.fc3_w.T
        dh2_pre = dh2 * silu_derivative(cache["h2_pre"])

        grads["fc2_w"] = cache["h1"].T @ dh2_pre
        grads["fc2_b"] = np.sum(dh2_pre, axis=0)

        # Backprop through h1
        dh1 = dh2_pre @ self.fc2_w.T
        dh1_pre = dh1 * silu_derivative(cache["h1_pre"])

        grads["fc1_w"] = cache["concat_input"].T @ dh1_pre
        grads["fc1_b"] = np.sum(dh1_pre, axis=0)

        # Backprop through concatenation
        d_concat = dh1_pre @ self.fc1_w.T
        d_t_emb = d_concat[:, 2 : 2 + self.hidden_dim]
        d_c_emb = d_concat[:, 2 + self.hidden_dim :]

        # Class embedding gradients
        grads["class_emb"] = np.zeros_like(self.class_emb)
        np.add.at(grads["class_emb"], cache["label_idx"], d_c_emb)

        # Time embedding gradients
        d_t_h2_pre = d_t_emb * silu_derivative(cache["t_h2_pre"])
        grads["time_fc2_w"] = cache["t_h1"].T @ d_t_h2_pre
        grads["time_fc2_b"] = np.sum(d_t_h2_pre, axis=0)

        d_t_h1 = d_t_h2_pre @ self.time_fc2_w.T
        d_t_h1_pre = d_t_h1 * silu_derivative(cache["t_h1_pre"])
        grads["time_fc1_w"] = cache["t_input"].T @ d_t_h1_pre
        grads["time_fc1_b"] = np.sum(d_t_h1_pre, axis=0)

        return grads

    def predict(self, x: np.ndarray, t: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Forward pass without caching (for inference)."""
        v, _ = self.forward(x, t, labels)
        return v

    def save(self, filepath: str):
        """Save model parameters to file."""
        params = self.get_params()
        np.savez(filepath, **params)

    def load(self, filepath: str):
        """Load model parameters from file."""
        data = np.load(filepath)
        params = {key: data[key] for key in data.files}
        self.set_params(params)
