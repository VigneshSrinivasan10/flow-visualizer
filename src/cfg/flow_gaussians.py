"""
Flow Matching with Classifier-Free Guidance (CFG) on 2D Overlapping Gaussians

A comprehensive implementation demonstrating:
- Flow matching training with label dropout
- Classifier-free guidance for improved class separation
- Deterministic Euler ODE sampling
- Visualization of CFG effects and flow evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
import os

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# Part 1: Data Generation
# =============================================================================

def generate_overlapping_gaussians(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate overlapping 2D Gaussian data for two classes.
    
    Class 0: mean = [-0.3, -0.3], std = 0.5
    Class 1: mean = [0.3, 0.3], std = 0.5
    
    The distributions overlap significantly, making CFG > 1 necessary for separation.
    
    Args:
        n_samples: Total number of samples (split evenly between classes)
        
    Returns:
        data: (n_samples, 2) array of 2D points
        labels: (n_samples,) array of class labels (0 or 1)
    """
    n_half = n_samples // 2
    mean1 = [-0.3, -0.3]
    mean2 = [0.3, 0.3]
    std = 0.5
    
    data1 = np.random.randn(n_half, 2) * std + mean1
    data2 = np.random.randn(n_samples - n_half, 2) * std + mean2
    
    data = np.vstack([data1, data2])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)])
    
    return data, labels


def generate_non_overlapping_gaussians(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate non-overlapping 2D Gaussian data for two classes - far apart.
    
    Class 0: mean = [-2, -2], std = 0.3
    Class 1: mean = [2, 2], std = 0.3
    
    The distributions are far apart with essentially no overlap.
    
    Args:
        n_samples: Total number of samples (split evenly between classes)
        
    Returns:
        data: (n_samples, 2) array of 2D points
        labels: (n_samples,) array of class labels (0 or 1)
    """
    n_half = n_samples // 2
    mean1 = [-2, -2]
    mean2 = [2, 2]
    std = 0.3
    
    data1 = np.random.randn(n_half, 2) * std + mean1
    data2 = np.random.randn(n_samples - n_half, 2) * std + mean2
    
    data = np.vstack([data1, data2])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)])
    
    return data, labels


def generate_multimodal_non_overlapping(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multimodal non-overlapping 2D data for two classes.
    
    Class 0: 3 modes at [-2, 0], [-1, -1.5], [-1.5, 1.5]
    Class 1: 3 modes at [2, 0], [1, 1.5], [1.5, -1.5]
    
    Each class has multiple clusters, positioned so they don't overlap with the other class.
    
    Args:
        n_samples: Total number of samples (split evenly between classes)
        
    Returns:
        data: (n_samples, 2) array of 2D points
        labels: (n_samples,) array of class labels (0 or 1)
    """
    n_half = n_samples // 2
    n_per_mode = n_half // 3
    std = 0.25
    
    # Class 0 modes (left side of space)
    modes_0 = [[-2, 0], [-1, -1.5], [-1.5, 1.5]]
    data0_list = []
    for i, mean in enumerate(modes_0):
        n = n_per_mode if i < 2 else n_half - 2 * n_per_mode
        data0_list.append(np.random.randn(n, 2) * std + mean)
    data0 = np.vstack(data0_list)
    
    # Class 1 modes (right side of space)
    n_class1 = n_samples - n_half
    n_per_mode_1 = n_class1 // 3
    modes_1 = [[2, 0], [1, 1.5], [1.5, -1.5]]
    data1_list = []
    for i, mean in enumerate(modes_1):
        n = n_per_mode_1 if i < 2 else n_class1 - 2 * n_per_mode_1
        data1_list.append(np.random.randn(n, 2) * std + mean)
    data1 = np.vstack(data1_list)
    
    data = np.vstack([data0, data1])
    labels = np.concatenate([np.zeros(n_half), np.ones(n_class1)])
    
    return data, labels


# Dataset configurations
DATASET_CONFIGS = {
    'overlapping': {
        'generator': generate_overlapping_gaussians,
        'title': 'Overlapping Gaussians',
        'class0_centers': [[-0.3, -0.3]],
        'class1_centers': [[0.3, 0.3]],
    },
    'non_overlapping': {
        'generator': generate_non_overlapping_gaussians,
        'title': 'Non-Overlapping Gaussians (Far Apart)',
        'class0_centers': [[-2, -2]],
        'class1_centers': [[2, 2]],
    },
    'multimodal': {
        'generator': generate_multimodal_non_overlapping,
        'title': 'Multimodal Non-Overlapping',
        'class0_centers': [[-2, 0], [-1, -1.5], [-1.5, 1.5]],
        'class1_centers': [[2, 0], [1, 1.5], [1.5, -1.5]],
    },
}


# =============================================================================
# Part 2: Model Architecture - Simple Flow Network
# =============================================================================

def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)"""
    return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))


def silu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
    sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return sig + x * sig * (1 - sig)


class AdamOptimizer:
    """Adam optimizer for manual gradient descent."""
    
    def __init__(self, lr: float = 5e-4, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return params


class SimpleFlowNetwork:
    """
    Simple Flow Network for 2D flow matching.
    
    Architecture:
    - Time embedding: MLP (1 → 128 → 128) with SiLU activation
    - Class embedding: Learnable embedding matrix (3 classes: 0, 1, unconditional=-1)
    - Main network: 4-layer MLP with SiLU activations
    - Output: 2D velocity vector
    
    Input concatenation: [x (2D), time_emb (128D), class_emb (128D)]
    Total input dimension: 2 + 128 + 128 = 258
    """
    
    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
        
        # Initialize parameters with He initialization
        scale = np.sqrt(2.0)
        
        # Time embedding MLP: 1 → 128 → 128
        self.time_fc1_w = np.random.randn(1, hidden_dim) * scale / np.sqrt(1)
        self.time_fc1_b = np.zeros(hidden_dim)
        self.time_fc2_w = np.random.randn(hidden_dim, hidden_dim) * scale / np.sqrt(hidden_dim)
        self.time_fc2_b = np.zeros(hidden_dim)
        
        # Class embedding: 3 classes (0, 1, -1 for unconditional)
        # Map: -1 → 0, 0 → 1, 1 → 2
        self.class_emb = np.random.randn(3, hidden_dim) * 0.02
        
        # Main network: 4-layer MLP
        # Input: 2 + 128 + 128 = 258
        input_dim = 2 + hidden_dim + hidden_dim
        
        self.fc1_w = np.random.randn(input_dim, hidden_dim) * scale / np.sqrt(input_dim)
        self.fc1_b = np.zeros(hidden_dim)
        
        self.fc2_w = np.random.randn(hidden_dim, hidden_dim) * scale / np.sqrt(hidden_dim)
        self.fc2_b = np.zeros(hidden_dim)
        
        self.fc3_w = np.random.randn(hidden_dim, hidden_dim) * scale / np.sqrt(hidden_dim)
        self.fc3_b = np.zeros(hidden_dim)
        
        self.fc4_w = np.random.randn(hidden_dim, 2) * 0.01  # Output layer, small init
        self.fc4_b = np.zeros(2)
        
        self.optimizer = AdamOptimizer(lr=5e-4)
        
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get all parameters as a dictionary."""
        return {
            'time_fc1_w': self.time_fc1_w, 'time_fc1_b': self.time_fc1_b,
            'time_fc2_w': self.time_fc2_w, 'time_fc2_b': self.time_fc2_b,
            'class_emb': self.class_emb,
            'fc1_w': self.fc1_w, 'fc1_b': self.fc1_b,
            'fc2_w': self.fc2_w, 'fc2_b': self.fc2_b,
            'fc3_w': self.fc3_w, 'fc3_b': self.fc3_b,
            'fc4_w': self.fc4_w, 'fc4_b': self.fc4_b,
        }
    
    def set_params(self, params: Dict[str, np.ndarray]):
        """Set all parameters from a dictionary."""
        self.time_fc1_w = params['time_fc1_w']
        self.time_fc1_b = params['time_fc1_b']
        self.time_fc2_w = params['time_fc2_w']
        self.time_fc2_b = params['time_fc2_b']
        self.class_emb = params['class_emb']
        self.fc1_w = params['fc1_w']
        self.fc1_b = params['fc1_b']
        self.fc2_w = params['fc2_w']
        self.fc2_b = params['fc2_b']
        self.fc3_w = params['fc3_w']
        self.fc3_b = params['fc3_b']
        self.fc4_w = params['fc4_w']
        self.fc4_b = params['fc4_b']
        
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
        batch_size = x.shape[0]
        
        # Time embedding
        cache['t_input'] = t
        t_h1_pre = t @ self.time_fc1_w + self.time_fc1_b
        cache['t_h1_pre'] = t_h1_pre
        t_h1 = silu(t_h1_pre)
        cache['t_h1'] = t_h1
        
        t_h2_pre = t_h1 @ self.time_fc2_w + self.time_fc2_b
        cache['t_h2_pre'] = t_h2_pre
        t_emb = silu(t_h2_pre)
        cache['t_emb'] = t_emb
        
        # Class embedding (map -1 → 0, 0 → 1, 1 → 2)
        label_idx = (labels + 1).astype(int)
        cache['label_idx'] = label_idx
        c_emb = self.class_emb[label_idx]
        cache['c_emb'] = c_emb
        
        # Concatenate inputs: [x, time_emb, class_emb]
        concat_input = np.concatenate([x, t_emb, c_emb], axis=1)
        cache['concat_input'] = concat_input
        cache['x'] = x
        
        # Main network forward pass
        h1_pre = concat_input @ self.fc1_w + self.fc1_b
        cache['h1_pre'] = h1_pre
        h1 = silu(h1_pre)
        cache['h1'] = h1
        
        h2_pre = h1 @ self.fc2_w + self.fc2_b
        cache['h2_pre'] = h2_pre
        h2 = silu(h2_pre)
        cache['h2'] = h2
        
        h3_pre = h2 @ self.fc3_w + self.fc3_b
        cache['h3_pre'] = h3_pre
        h3 = silu(h3_pre)
        cache['h3'] = h3
        
        v = h3 @ self.fc4_w + self.fc4_b
        
        return v, cache
    
    def backward(self, v_pred: np.ndarray, v_target: np.ndarray, cache: Dict) -> Dict[str, np.ndarray]:
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
        grads['fc4_w'] = cache['h3'].T @ dv
        grads['fc4_b'] = np.sum(dv, axis=0)
        
        # Backprop through h3
        dh3 = dv @ self.fc4_w.T
        dh3_pre = dh3 * silu_derivative(cache['h3_pre'])
        
        grads['fc3_w'] = cache['h2'].T @ dh3_pre
        grads['fc3_b'] = np.sum(dh3_pre, axis=0)
        
        # Backprop through h2
        dh2 = dh3_pre @ self.fc3_w.T
        dh2_pre = dh2 * silu_derivative(cache['h2_pre'])
        
        grads['fc2_w'] = cache['h1'].T @ dh2_pre
        grads['fc2_b'] = np.sum(dh2_pre, axis=0)
        
        # Backprop through h1
        dh1 = dh2_pre @ self.fc2_w.T
        dh1_pre = dh1 * silu_derivative(cache['h1_pre'])
        
        grads['fc1_w'] = cache['concat_input'].T @ dh1_pre
        grads['fc1_b'] = np.sum(dh1_pre, axis=0)
        
        # Backprop through concatenation
        d_concat = dh1_pre @ self.fc1_w.T
        d_x = d_concat[:, :2]
        d_t_emb = d_concat[:, 2:2+self.hidden_dim]
        d_c_emb = d_concat[:, 2+self.hidden_dim:]
        
        # Class embedding gradients
        grads['class_emb'] = np.zeros_like(self.class_emb)
        np.add.at(grads['class_emb'], cache['label_idx'], d_c_emb)
        
        # Time embedding gradients
        d_t_h2_pre = d_t_emb * silu_derivative(cache['t_h2_pre'])
        grads['time_fc2_w'] = cache['t_h1'].T @ d_t_h2_pre
        grads['time_fc2_b'] = np.sum(d_t_h2_pre, axis=0)
        
        d_t_h1 = d_t_h2_pre @ self.time_fc2_w.T
        d_t_h1_pre = d_t_h1 * silu_derivative(cache['t_h1_pre'])
        grads['time_fc1_w'] = cache['t_input'].T @ d_t_h1_pre
        grads['time_fc1_b'] = np.sum(d_t_h1_pre, axis=0)
        
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


# =============================================================================
# Part 3: Flow Matching Training
# =============================================================================

def train_flow_matching(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    epochs: int = 500,
    batch_size: int = 128,
    label_dropout: float = 0.3,
    verbose: bool = True
) -> List[float]:
    """
    Train the flow matching model with label dropout for CFG.
    
    Args:
        model: SimpleFlowNetwork instance
        data: (n_samples, 2) training data
        labels: (n_samples,) class labels
        epochs: Number of training epochs
        batch_size: Batch size
        label_dropout: Probability of dropping label (set to -1 for unconditional)
        verbose: Whether to print training progress
        
    Returns:
        losses: List of average losses per epoch
    """
    n_samples = data.shape[0]
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(n_samples)
        data_shuffled = data[perm]
        labels_shuffled = labels[perm]
        
        epoch_losses = []
        
        for i in range(0, n_samples, batch_size):
            # Get batch
            x1 = data_shuffled[i:i+batch_size]
            batch_labels = labels_shuffled[i:i+batch_size].copy()
            actual_batch_size = x1.shape[0]
            
            # Label dropout: randomly set labels to -1 (unconditional)
            dropout_mask = np.random.rand(actual_batch_size) < label_dropout
            batch_labels[dropout_mask] = -1
            
            # Sample random time t ∈ [0, 1]
            t = np.random.rand(actual_batch_size, 1)
            
            # Sample noise (starting point)
            x0 = np.random.randn(actual_batch_size, 2)
            
            # Straight-line interpolation: x_t = (1 - t) * x0 + t * x1
            x_t = (1 - t) * x0 + t * x1
            
            # Target velocity (constant along straight line): v = x1 - x0
            v_target = x1 - x0
            
            # Forward pass
            v_pred, cache = model.forward(x_t, t, batch_labels)
            
            # Compute MSE loss
            loss = np.mean((v_pred - v_target) ** 2)
            epoch_losses.append(loss)
            
            # Backward pass
            grads = model.backward(v_pred, v_target, cache)
            
            # Update parameters
            params = model.get_params()
            params = model.optimizer.step(params, grads)
            model.set_params(params)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


# =============================================================================
# Part 4 & 5: Classifier-Free Guidance (CFG) and Sampling
# =============================================================================

def sample_euler(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    cfg_scale: float = 0.0
) -> np.ndarray:
    """
    Sample using deterministic Euler ODE with classifier-free guidance.
    
    CFG Formula: v = v_uncond + cfg_scale * (v_cond - v_uncond)
    
    Args:
        model: Trained SimpleFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        num_steps: Number of Euler steps
        cfg_scale: CFG strength (0 = unconditional, 1 = conditional, >1 = extrapolation)
        
    Returns:
        x: (n_samples, 2) generated samples
    """
    # Start from noise (t=0)
    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps
    
    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)
    
    for step in range(num_steps):
        t = np.full((n_samples, 1), step * dt)
        
        # Compute conditional velocity
        v_cond = model.predict(x, t, labels_cond)
        
        # Compute unconditional velocity
        v_uncond = model.predict(x, t, labels_uncond)
        
        # Apply CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # Euler step (deterministic)
        x = x + v * dt
    
    return x


def sample_euler_with_trajectory(
    model: SimpleFlowNetwork,
    n_samples: int,
    label: int,
    num_steps: int = 100,
    cfg_scale: float = 0.0,
    save_times: List[float] = None
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """
    Sample with trajectory tracking at specified time points.
    
    Args:
        model: Trained SimpleFlowNetwork
        n_samples: Number of samples to generate
        label: Target class label (0 or 1)
        num_steps: Number of Euler steps
        cfg_scale: CFG strength
        save_times: List of time points to save (default: [0, 0.25, 0.5, 0.75, 1.0])
        
    Returns:
        x: Final samples
        trajectories: Dictionary mapping time to samples at that time
    """
    if save_times is None:
        save_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    x = np.random.randn(n_samples, 2)
    dt = 1.0 / num_steps
    
    labels_cond = np.full(n_samples, label, dtype=float)
    labels_uncond = np.full(n_samples, -1, dtype=float)
    
    trajectories = {0.0: x.copy()}
    
    for step in range(num_steps):
        t_val = step * dt
        t = np.full((n_samples, 1), t_val)
        
        v_cond = model.predict(x, t, labels_cond)
        v_uncond = model.predict(x, t, labels_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        x = x + v * dt
        
        # Save trajectory at specified times
        current_time = (step + 1) * dt
        for save_time in save_times:
            if abs(current_time - save_time) < dt / 2 and save_time not in trajectories:
                trajectories[save_time] = x.copy()
    
    # Ensure final time is saved
    trajectories[1.0] = x.copy()
    
    return x, trajectories


# =============================================================================
# Part 9: Visualizations
# =============================================================================

def plot_training_data(
    data: np.ndarray, 
    labels: np.ndarray, 
    save_path: Optional[str] = None,
    title: str = 'Training Data',
    class0_centers: List[List[float]] = None,
    class1_centers: List[List[float]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3)
):
    """
    Visualize the training data.
    """
    if class0_centers is None:
        class0_centers = [[-0.3, -0.3]]
    if class1_centers is None:
        class1_centers = [[0.3, 0.3]]
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot each class
    mask_0 = labels == 0
    mask_1 = labels == 1
    
    ax.scatter(data[mask_0, 0], data[mask_0, 1], c='gray', alpha=0.5, s=10, label='Class 0')
    ax.scatter(data[mask_1, 0], data[mask_1, 1], c='lightcoral', alpha=0.5, s=10, label='Class 1')
    
    # Mark the centers
    for i, center in enumerate(class0_centers):
        label = 'Mean 0' if i == 0 else None
        ax.scatter([center[0]], [center[1]], c='black', marker='x', s=200, linewidths=3, label=label)
    for i, center in enumerate(class1_centers):
        label = 'Mean 1' if i == 0 else None
        ax.scatter([center[0]], [center[1]], c='darkred', marker='x', s=200, linewidths=3, label=label)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Training Data: {title}', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_training_loss(losses: List[float], save_path: Optional[str] = None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.plot(losses, color='#2E86AB', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Flow Matching Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def classify_samples(
    samples: np.ndarray, 
    class0_centers: List[List[float]] = None,
    class1_centers: List[List[float]] = None
) -> np.ndarray:
    """
    Classify samples based on distance to nearest class center.
    
    Args:
        samples: (n_samples, 2) array of 2D points
        class0_centers: List of [x, y] centers for class 0
        class1_centers: List of [x, y] centers for class 1
        
    Returns:
        labels: (n_samples,) array of predicted class labels (0 or 1)
    """
    if class0_centers is None:
        class0_centers = [[-0.3, -0.3]]
    if class1_centers is None:
        class1_centers = [[0.3, 0.3]]
    
    class0_centers = np.array(class0_centers)
    class1_centers = np.array(class1_centers)
    
    # Compute minimum distance to any class 0 center
    dist_to_class0 = np.min([
        np.sqrt(np.sum((samples - center) ** 2, axis=1))
        for center in class0_centers
    ], axis=0)
    
    # Compute minimum distance to any class 1 center
    dist_to_class1 = np.min([
        np.sqrt(np.sum((samples - center) ** 2, axis=1))
        for center in class1_centers
    ], axis=0)
    
    # Classify based on nearest center
    return (dist_to_class1 < dist_to_class0).astype(int)


def plot_cfg_comparison(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    target_label: int = 0,
    cfg_scales: List[float] = None,
    n_samples: int = 500,
    save_path: Optional[str] = None,
    class0_centers: List[List[float]] = None,
    class1_centers: List[List[float]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3)
):
    """
    Create a grid visualization comparing different CFG scales.
    
    Shows how separation improves with increasing CFG scale.
    """
    if cfg_scales is None:
        cfg_scales = [0, 1, 3, 5, 7, 9]
    
    n_cfg = len(cfg_scales)
    fig, axes = plt.subplots(2, (n_cfg + 1) // 2, figsize=(4 * ((n_cfg + 1) // 2), 8))
    axes = axes.flatten()
    
    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1
    
    for idx, cfg_scale in enumerate(cfg_scales):
        ax = axes[idx]
        
        # Generate samples
        np.random.seed(123)  # Same starting noise for comparison
        samples = sample_euler(model, n_samples, target_label, num_steps=100, cfg_scale=cfg_scale)
        
        # Plot training data (faded)
        ax.scatter(data[mask_0, 0], data[mask_0, 1], c='gray', alpha=0.15, s=5)
        ax.scatter(data[mask_1, 0], data[mask_1, 1], c='lightcoral', alpha=0.15, s=5)
        
        # Plot generated samples
        ax.scatter(samples[:, 0], samples[:, 1], c='#2E86AB', edgecolors='black', 
                  linewidths=0.5, alpha=0.7, s=30)
        
        # Compute statistics
        predicted_classes = classify_samples(samples, class0_centers, class1_centers)
        n_class0 = np.sum(predicted_classes == 0)
        n_class1 = np.sum(predicted_classes == 1)
        target_count = n_class0 if target_label == 0 else n_class1
        ratio = target_count / n_samples
        std_x = np.std(samples[:, 0])
        std_y = np.std(samples[:, 1])
        
        # Statistics box
        stats_text = f"Target: {target_count}/{n_samples}\nRatio: {ratio:.2%}\nStd: ({std_x:.2f}, {std_y:.2f})"
        
        # Color code by effectiveness
        if ratio >= 0.9:
            box_color = '#90EE90'  # Light green
        elif ratio >= 0.7:
            box_color = '#FFFF99'  # Light yellow
        else:
            box_color = '#FFB6C1'  # Light pink
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'CFG Scale = {cfg_scale}', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Hide extra axes if odd number
    for idx in range(n_cfg, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'CFG Scale Comparison (Target: Class {target_label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_flow_evolution(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    cfg_scales: List[float] = None,
    target_labels: List[int] = None,
    time_steps: List[float] = None,
    n_samples: int = 300,
    save_path: Optional[str] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3)
):
    """
    Create a grid showing flow evolution over time for different CFG scales.
    
    Rows: CFG scales
    Columns: Time steps (t=0, 0.25, 0.5, 0.75, 1.0)
    """
    if cfg_scales is None:
        cfg_scales = [0, 1, 2, 4]
    if target_labels is None:
        target_labels = [0]
    if time_steps is None:
        time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    n_rows = len(cfg_scales)
    n_cols = len(time_steps)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Training data masks
    mask_0 = labels == 0
    mask_1 = labels == 1
    
    for row_idx, cfg_scale in enumerate(cfg_scales):
        # Generate samples with trajectory tracking
        np.random.seed(456)  # Same starting noise
        target_label = target_labels[0] if len(target_labels) == 1 else target_labels[row_idx % len(target_labels)]
        _, trajectories = sample_euler_with_trajectory(
            model, n_samples, target_label, num_steps=100, 
            cfg_scale=cfg_scale, save_times=time_steps
        )
        
        for col_idx, t in enumerate(time_steps):
            ax = axes[row_idx, col_idx]
            
            # Get samples at this time
            samples = trajectories.get(t, trajectories[min(trajectories.keys(), key=lambda x: abs(x - t))])
            
            # Plot training data (faded)
            ax.scatter(data[mask_0, 0], data[mask_0, 1], c='gray', alpha=0.1, s=3)
            ax.scatter(data[mask_1, 0], data[mask_1, 1], c='lightcoral', alpha=0.1, s=3)
            
            # Plot samples
            ax.scatter(samples[:, 0], samples[:, 1], c='#2E86AB', edgecolors='black',
                      linewidths=0.3, alpha=0.6, s=20)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Labels
            if row_idx == 0:
                ax.set_title(f't = {t:.2f}', fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'CFG = {cfg_scale}', fontsize=11, fontweight='bold')
    
    fig.suptitle(f'Flow Evolution (Target: Class {target_labels[0]})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_both_classes_cfg(
    model: SimpleFlowNetwork,
    data: np.ndarray,
    labels: np.ndarray,
    cfg_scales: List[float] = None,
    n_samples: int = 400,
    save_path: Optional[str] = None,
    class0_centers: List[List[float]] = None,
    class1_centers: List[List[float]] = None,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3)
):
    """
    Show CFG comparison for both classes side by side.
    """
    if cfg_scales is None:
        cfg_scales = [0, 1, 3, 5]
    
    n_cfg = len(cfg_scales)
    fig, axes = plt.subplots(2, n_cfg, figsize=(4 * n_cfg, 8))
    
    mask_0 = labels == 0
    mask_1 = labels == 1
    
    for row_idx, target_label in enumerate([0, 1]):
        for col_idx, cfg_scale in enumerate(cfg_scales):
            ax = axes[row_idx, col_idx]
            
            # Generate samples
            np.random.seed(789 + row_idx)
            samples = sample_euler(model, n_samples, target_label, num_steps=100, cfg_scale=cfg_scale)
            
            # Plot training data (faded)
            ax.scatter(data[mask_0, 0], data[mask_0, 1], c='gray', alpha=0.15, s=5)
            ax.scatter(data[mask_1, 0], data[mask_1, 1], c='lightcoral', alpha=0.15, s=5)
            
            # Color samples by target class
            color = '#3498db' if target_label == 0 else '#e74c3c'
            ax.scatter(samples[:, 0], samples[:, 1], c=color, edgecolors='black',
                      linewidths=0.5, alpha=0.7, s=30)
            
            # Compute statistics
            predicted_classes = classify_samples(samples, class0_centers, class1_centers)
            target_count = np.sum(predicted_classes == target_label)
            ratio = target_count / n_samples
            
            # Statistics
            # stats_text = f"Correct: {ratio:.1%}"
            # box_color = '#90EE90' if ratio >= 0.9 else '#FFFF99' if ratio >= 0.7 else '#FFB6C1'
            # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=15,
            #        verticalalignment='top', fontweight='bold',
            #        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f'CFG = {cfg_scale}', fontsize=25)
            if col_idx == 0:
                ax.set_ylabel(f'Class {target_label}', fontsize=25)
    
    #fig.suptitle('CFG Comparison: Both Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def run_experiment(
    dataset_name: str,
    config: Dict,
    base_output_dir: str = "flow_matching_outputs",
    force_retrain: bool = False
):
    """
    Run a complete flow matching + CFG experiment for a given dataset.
    
    Args:
        dataset_name: Name of the dataset (used for output folder)
        config: Dataset configuration dict with generator, title, centers
        base_output_dir: Base directory for outputs
        force_retrain: If True, retrain even if model exists
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {config['title']}")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract config
    class0_centers = config['class0_centers']
    class1_centers = config['class1_centers']
    
    # Determine axis limits based on data centers
    all_centers = class0_centers + class1_centers
    max_coord = max(max(abs(c[0]), abs(c[1])) for c in all_centers) + 1.5
    xlim = (-max_coord, max_coord)
    ylim = (-max_coord, max_coord)
    
    # -------------------------------------------------------------------------
    # Part 1: Generate Data
    # -------------------------------------------------------------------------
    print("\n[1/4] Generating data...")
    np.random.seed(42)  # Reset seed for reproducibility
    data, labels = config['generator'](n_samples=5000)
    print(f"    Generated {len(data)} samples")
    print(f"    Class 0: {np.sum(labels == 0)} samples, Class 1: {np.sum(labels == 1)} samples")
    
    # Visualize training data
    plot_training_data(
        data, labels, 
        save_path=os.path.join(output_dir, "1_training_data.png"),
        title=config['title'],
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim,
        ylim=ylim
    )
    
    # -------------------------------------------------------------------------
    # Part 2 & 3: Create and Train Model
    # -------------------------------------------------------------------------
    print("\n[2/4] Creating and training flow matching model...")
    model = SimpleFlowNetwork(hidden_dim=128)
    
    # Check for saved model
    model_path = os.path.join(output_dir, "flow_model.npz")
    if os.path.exists(model_path) and not force_retrain:
        print(f"    Loading saved model from {model_path}")
        model.load(model_path)
    else:
        print("    Training from scratch (500 epochs, batch_size=128, lr=5e-4)")
        print("    Label dropout: 30% (for CFG training)")
        np.random.seed(42)  # Reset seed for training reproducibility
        losses = train_flow_matching(
            model, data, labels,
            epochs=500,
            batch_size=128,
            label_dropout=0.3,
            verbose=True
        )
        
        # Save model
        model.save(model_path)
        print(f"    Model saved to {model_path}")
        
        # Plot training loss
        plot_training_loss(losses, save_path=os.path.join(output_dir, "2_training_loss.png"))
    
    # -------------------------------------------------------------------------
    # Part 4 & 5: CFG Sampling and Visualization
    # -------------------------------------------------------------------------
    print("\n[3/4] Generating samples with different CFG scales...")
    
    # CFG comparison for Class 0
    print("    CFG comparison for Class 0...")
    plot_cfg_comparison(
        model, data, labels,
        target_label=0,
        cfg_scales=[0, 1, 3, 5, 7, 9],
        n_samples=500,
        save_path=os.path.join(output_dir, "3_cfg_comparison_class0.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim,
        ylim=ylim
    )
    
    # CFG comparison for Class 1
    print("    CFG comparison for Class 1...")
    plot_cfg_comparison(
        model, data, labels,
        target_label=1,
        cfg_scales=[0, 1, 3, 5, 7, 9],
        n_samples=500,
        save_path=os.path.join(output_dir, "4_cfg_comparison_class1.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim,
        ylim=ylim
    )
    
    # Both classes comparison
    print("    Both classes comparison...")
    xlim = (-3.5, 3.5)
    ylim = (-3.5, 3.5)
    plot_both_classes_cfg(
        model, data, labels,
        cfg_scales=[0, 1, 3, 5, 7, 9],
        n_samples=400,
        save_path=os.path.join(output_dir, "5_both_classes_cfg.png"),
        class0_centers=class0_centers,
        class1_centers=class1_centers,
        xlim=xlim,
        ylim=ylim
    )
    
    # -------------------------------------------------------------------------
    # Flow Evolution Visualization
    # -------------------------------------------------------------------------
    print("\n[4/4] Visualizing flow evolution over time...")
    
    # Flow evolution for Class 0
    print("    Flow evolution for Class 0...")
    plot_flow_evolution(
        model, data, labels,
        cfg_scales=[0, 1, 2, 4, 6, 8, 10],
        target_labels=[0],
        time_steps=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_samples=300,
        save_path=os.path.join(output_dir, "6_flow_evolution_class0.png"),
        xlim=xlim,
        ylim=ylim
    )
    
    # Flow evolution for Class 1
    print("    Flow evolution for Class 1...")
    plot_flow_evolution(
        model, data, labels,
        cfg_scales=[0, 1, 2, 4, 6, 8, 10],
        target_labels=[1],
        time_steps=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_samples=300,
        save_path=os.path.join(output_dir, "7_flow_evolution_class1.png"),
        xlim=xlim,
        ylim=ylim
    )
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n[Summary] Final sampling quality:")
    for target_label in [0, 1]:
        print(f"\n  Class {target_label}:")
        for cfg_scale in [0, 1, 3, 5]:
            np.random.seed(999 + target_label)
            samples = sample_euler(model, 1000, target_label, num_steps=100, cfg_scale=cfg_scale)
            predicted = classify_samples(samples, class0_centers, class1_centers)
            accuracy = np.mean(predicted == target_label)
            print(f"    CFG={cfg_scale}: {accuracy:.1%} correct class assignments")
    
    print(f"\n    All outputs saved to: {output_dir}")
    
    return model, data, labels


def main():
    """Main function to run flow matching + CFG experiments on all dataset types."""
    
    print("=" * 70)
    print("Flow Matching with Classifier-Free Guidance (CFG)")
    print("Comparing Three Dataset Types:")
    print("  1. Overlapping Gaussians")
    print("  2. Non-Overlapping Gaussians (Far Apart)")
    print("  3. Multimodal Non-Overlapping")
    print("=" * 70)
    
    base_output_dir = "flow_matching_outputs"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Run experiments for all three dataset types
    results = {}
    for dataset_name, config in DATASET_CONFIGS.items():
        results[dataset_name] = run_experiment(
            dataset_name=dataset_name,
            config=config,
            base_output_dir=base_output_dir,
            force_retrain=False  # Set to True to force retraining
        )
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved in subdirectories of: {base_output_dir}/")
    print("  - overlapping/")
    print("  - non_overlapping/")
    print("  - multimodal/")


if __name__ == "__main__":
    main()

