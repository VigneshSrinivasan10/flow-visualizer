"""CFG Flow Model with class conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class ZeroToOneTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.arange(1, dim // 2 + 1) * torch.pi)

    def forward(self, t):
        emb = self.freqs * t[..., None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CFGFlowMLP(nn.Module):
    """Flow MLP with class conditioning for CFG."""

    def __init__(
        self,
        n_features: int = 2,
        width: int = 512,
        n_blocks: int = 5,
        num_classes: int = 3,
        class_emb_dim: int = 32,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.num_classes = num_classes
        # +1 for unconditional class (null class)
        self.class_embedding = nn.Embedding(num_classes + 1, class_emb_dim)
        self.null_class_idx = num_classes  # Index for unconditional

        self.time_embedding_size = width - n_features - class_emb_dim
        self.time_embedding = ZeroToOneTimeEmbedding(self.time_embedding_size)

        blocks = []
        for _ in range(self.n_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(width, width, bias=False),
                nn.SiLU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Linear(width, n_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self, base_std=0.02) -> None:
        for n, p in self.named_parameters():
            if 'class_embedding' in n:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'final' in n:
                nn.init.zeros_(p)
            elif len(p.shape) >= 2:
                fan_in = p.shape[1]
                p.data.normal_(mean=0.0, std=base_std * (fan_in) ** -0.5)

    def forward(self, X, time=None, class_labels=None):
        if time is None:
            time = torch.rand(X.shape[0], device=X.device)

        # Use null class if no labels provided (unconditional)
        if class_labels is None:
            class_labels = torch.full(
                (X.shape[0],), self.null_class_idx,
                dtype=torch.long, device=X.device
            )

        time_emb = self.time_embedding(time)
        class_emb = self.class_embedding(class_labels)

        X = torch.cat([X, time_emb, class_emb], dim=1)

        for block in self.blocks:
            X = X + block(X)
        X = self.final(X)
        return X

    def forward_cfg(self, X, time, class_labels, guidance_scale=1.0):
        """Forward pass with classifier-free guidance."""
        if guidance_scale == 1.0:
            return self.forward(X, time, class_labels)

        # Conditional prediction
        v_cond = self.forward(X, time, class_labels)

        # Unconditional prediction
        null_labels = torch.full_like(class_labels, self.null_class_idx)
        v_uncond = self.forward(X, time, null_labels)

        # CFG interpolation: v = v_uncond + guidance_scale * (v_cond - v_uncond)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        no_decay_name_list = ["bias", "norm"]

        param_groups = defaultdict(
            lambda: {"params": [], "weight_decay": None, "lr": None}
        )

        for n, p in self.named_parameters():
            if p.requires_grad:
                if any(ndnl in n for ndnl in no_decay_name_list):
                    lr_value = learning_rate * 0.1
                    per_layer_weight_decay_value = 0.0
                elif "class_embedding" in n:
                    lr_value = learning_rate * 0.3
                    per_layer_weight_decay_value = 0.0
                elif "time_embedding" in n:
                    lr_value = learning_rate * 0.3
                    per_layer_weight_decay_value = 0.0
                else:
                    hidden_dim = p.shape[-1]
                    lr_value = learning_rate * (32 / hidden_dim)
                    per_layer_weight_decay_value = weight_decay * hidden_dim / 1024

                group_key = (lr_value, per_layer_weight_decay_value)
                param_groups[group_key]["params"].append(p)
                param_groups[group_key]["weight_decay"] = per_layer_weight_decay_value
                param_groups[group_key]["lr"] = lr_value

        optimizer_grouped_parameters = [v for v in param_groups.values()]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=betas)

        return optimizer
