"""MaskedDinoV2: the core intervention of the ablation.

Wraps a frozen DINOv2 (with registers) backbone and drops a random subset of
patch tokens AFTER patch embedding + positional-encoding interpolation and
BEFORE the transformer blocks. CLS and register tokens are always kept.
Dropped tokens are removed entirely (no mask-token substitution) - ViTs handle
variable-length sequences natively.

Masks are deterministic per image: the kept indices depend only on
(mask_seed, sample_id), never on batch composition.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

_PER_SAMPLE_SEED_STRIDE = 1_000_003  # large prime; decorrelates mask_seed vs sample_id


def load_backbone(name: str = "dinov2_vits14_reg") -> nn.Module:
    """Load a frozen DINOv2 backbone from torch.hub."""
    backbone = torch.hub.load("facebookresearch/dinov2", name)
    backbone.eval().requires_grad_(False)
    return backbone


class MaskedDinoV2(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = int(getattr(backbone, "num_register_tokens", 0))
        self.num_prefix = 1 + self.num_register_tokens  # CLS + registers
        self.embed_dim = int(backbone.embed_dim)
        self.patch_size = int(backbone.patch_size)

    def sample_kept_indices(self, n_patches: int, keep_ratio: float,
                            mask_seed: int, sample_ids: torch.Tensor) -> torch.Tensor:
        """(B, k) sorted patch indices to keep; deterministic per (mask_seed, sample_id)."""
        k = max(1, int(round(keep_ratio * n_patches)))
        rows = []
        for sid in sample_ids.tolist():
            gen = torch.Generator().manual_seed(int(mask_seed) * _PER_SAMPLE_SEED_STRIDE + int(sid))
            rows.append(torch.randperm(n_patches, generator=gen)[:k])
        return torch.stack(rows).sort(dim=1).values

    @torch.no_grad()
    def forward(self, x: torch.Tensor, keep_ratio: float = 1.0, mask_seed: int = 0,
                sample_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Returns dict with:
            cls      (B, D)      normalized CLS feature
            patch    (B, k, D)   normalized features of the KEPT patch tokens
            kept_idx (B, k)      grid indices (row-major) of the kept patches
            grid     (gh, gw)    full patch-grid shape
        """
        B, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input {H}x{W} must be a multiple of patch size {self.patch_size}"
        gh, gw = H // self.patch_size, W // self.patch_size
        n = gh * gw

        tokens = self.backbone.prepare_tokens_with_masks(x, None)  # (B, prefix + n, D)
        assert tokens.shape[1] == self.num_prefix + n, \
            f"Token count {tokens.shape[1]} != prefix {self.num_prefix} + patches {n}"

        if keep_ratio >= 1.0:
            kept_idx = torch.arange(n, device=x.device).unsqueeze(0).expand(B, n)
        else:
            if sample_ids is None:
                sample_ids = torch.arange(B)
            kept_idx = self.sample_kept_indices(n, keep_ratio, mask_seed, sample_ids).to(x.device)
            patches = tokens[:, self.num_prefix:]
            gathered = torch.gather(
                patches, 1, kept_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))
            tokens = torch.cat([tokens[:, :self.num_prefix], gathered], dim=1)

        for blk in self.backbone.blocks:
            tokens = blk(tokens)
        tokens = self.backbone.norm(tokens)

        return {
            "cls": tokens[:, 0],
            "patch": tokens[:, self.num_prefix:],
            "kept_idx": kept_idx,
            "grid": (gh, gw),
        }
