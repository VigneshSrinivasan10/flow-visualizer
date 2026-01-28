"""
Rectified CFG++ for Flow Matching

Implements CFG++ and Rectified CFG++ methods that prevent off-manifold drift
at high guidance scales.

Methods:
1. Standard CFG: v = v_uncond + w*(v_cond - v_uncond)
2. CFG++: Rescales guided velocity to match unconditional magnitude
3. Rectified CFG++: Rescales guided velocity to match conditional magnitude

Reference: "CFG++: Manifold-constrained Classifier Free Guidance" (ICLR 2025)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal


GuidanceMethod = Literal['cfg', 'cfg++', 'rectified_cfg++']


class RectifiedCFGPlusPlusSampler:
    """
    Sampler implementing Standard CFG, CFG++, and Rectified CFG++ for flow matching.
    
    Methods:
    - 'cfg': Standard classifier-free guidance
    - 'cfg++': CFG++ (manifold-constrained, uses unconditional magnitude)
    - 'rectified_cfg++': Rectified CFG++ (uses conditional magnitude)
    """

    def __init__(self, velocity_net: nn.Module, device: str = "cpu"):
        self.velocity_net = velocity_net.to(device)
        self.device = device

    @torch.no_grad()
    def get_velocity_components(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get conditional and unconditional velocity components."""
        self.velocity_net.eval()
        
        # Conditional velocity
        v_cond = self.velocity_net(x, time=t, class_labels=class_labels)
        
        # Unconditional velocity
        null_labels = torch.full_like(class_labels, self.velocity_net.null_class_idx)
        v_uncond = self.velocity_net(x, time=t, class_labels=null_labels)
        
        return v_cond, v_uncond

    @torch.no_grad()
    def get_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
        guidance_scale: float = 2.0,
        method: GuidanceMethod = 'cfg',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get velocity with different guidance methods.
        
        Args:
            x: Current position (batch, 2)
            t: Current time (batch,)
            class_labels: Class condition (batch,)
            guidance_scale: Guidance scale (w)
            method: Guidance method
            
        Returns:
            v_guided: The guided velocity to use for stepping
            v_cond: Conditional velocity (for visualization)
            v_uncond: Unconditional velocity (for visualization)
            v_direction: Direction component v_cond - v_uncond
        """
        v_cond, v_uncond = self.get_velocity_components(x, t, class_labels)
        
        # Direction from unconditional to conditional
        v_direction = v_cond - v_uncond
        
        if method == 'cfg':
            # Standard CFG: v = v_uncond + w*(v_cond - v_uncond)
            v_guided = v_uncond + guidance_scale * v_direction
            
        elif method == 'cfg++':
            # CFG++: Use guided direction but unconditional magnitude
            v_cfg = v_uncond + guidance_scale * v_direction
            
            v_cfg_norm = torch.norm(v_cfg, dim=-1, keepdim=True).clamp(min=1e-8)
            v_uncond_norm = torch.norm(v_uncond, dim=-1, keepdim=True).clamp(min=1e-8)
            
            # Direction from CFG, magnitude from unconditional
            v_guided = v_cfg * (v_uncond_norm / v_cfg_norm)
            
        elif method == 'rectified_cfg++':
            # Rectified CFG++: Use guided direction but conditional magnitude
            v_cfg = v_uncond + guidance_scale * v_direction
            
            v_cfg_norm = torch.norm(v_cfg, dim=-1, keepdim=True).clamp(min=1e-8)
            v_cond_norm = torch.norm(v_cond, dim=-1, keepdim=True).clamp(min=1e-8)
            
            # Direction from CFG, magnitude from conditional
            v_guided = v_cfg * (v_cond_norm / v_cfg_norm)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return v_guided, v_cond, v_uncond, v_direction

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        class_labels: torch.Tensor,
        n_steps: int = 100,
        data_dim: int = 2,
        guidance_scale: float = 2.0,
        method: GuidanceMethod = 'cfg',
    ) -> torch.Tensor:
        """Sample from the model using specified guidance method."""
        self.velocity_net.eval()
        
        x = torch.randn(n_samples, data_dim, device=self.device)
        class_labels = class_labels.to(self.device)
        
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v_guided, _, _, _ = self.get_velocity(x, t, class_labels, guidance_scale, method)
            x = x + v_guided * dt
            
        return x.cpu()

    @torch.no_grad()
    def sample_trajectory(
        self,
        n_samples: int,
        class_labels: torch.Tensor,
        n_steps: int = 100,
        data_dim: int = 2,
        guidance_scale: float = 2.0,
        method: GuidanceMethod = 'cfg',
    ) -> Tuple[list, list, list, list]:
        """Sample and return full trajectory with velocity decomposition."""
        self.velocity_net.eval()
        
        x = torch.randn(n_samples, data_dim, device=self.device)
        class_labels = class_labels.to(self.device)
        
        trajectory = [x.cpu().clone()]
        velocities_guided = []
        velocities_cond = []
        velocities_uncond = []
        
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.ones(n_samples, device=self.device) * (step / n_steps)
            v_guided, v_cond, v_uncond, _ = self.get_velocity(
                x, t, class_labels, guidance_scale, method
            )
            
            velocities_guided.append(v_guided.cpu().clone())
            velocities_cond.append(v_cond.cpu().clone())
            velocities_uncond.append(v_uncond.cpu().clone())
            
            x = x + v_guided * dt
            trajectory.append(x.cpu().clone())
            
        return trajectory, velocities_guided, velocities_cond, velocities_uncond


def extend_cfg_model_with_methods(model_class):
    """
    Decorator to extend a CFG model class with CFG++ and Rectified CFG++ methods.
    """
    original_forward_cfg = model_class.forward_cfg
    
    def forward_cfg_extended(
        self,
        X: torch.Tensor,
        time: torch.Tensor,
        class_labels: torch.Tensor,
        guidance_scale: float = 1.0,
        method: GuidanceMethod = 'cfg',
    ) -> torch.Tensor:
        """Forward pass with configurable guidance method."""
        if guidance_scale == 1.0 or method == 'cfg':
            # Fall back to original for standard CFG
            if method == 'cfg':
                return original_forward_cfg(self, X, time, class_labels, guidance_scale)
            else:
                return self.forward(X, time, class_labels)
        
        # Conditional prediction
        v_cond = self.forward(X, time, class_labels)
        
        # Unconditional prediction
        null_labels = torch.full_like(class_labels, self.null_class_idx)
        v_uncond = self.forward(X, time, null_labels)
        
        # Direction
        v_direction = v_cond - v_uncond
        v_cfg = v_uncond + guidance_scale * v_direction
        
        if method == 'cfg++':
            v_cfg_norm = torch.norm(v_cfg, dim=-1, keepdim=True).clamp(min=1e-8)
            v_uncond_norm = torch.norm(v_uncond, dim=-1, keepdim=True).clamp(min=1e-8)
            return v_cfg * (v_uncond_norm / v_cfg_norm)
            
        elif method == 'rectified_cfg++':
            v_cfg_norm = torch.norm(v_cfg, dim=-1, keepdim=True).clamp(min=1e-8)
            v_cond_norm = torch.norm(v_cond, dim=-1, keepdim=True).clamp(min=1e-8)
            return v_cfg * (v_cond_norm / v_cfg_norm)
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    model_class.forward_cfg_extended = forward_cfg_extended
    return model_class
