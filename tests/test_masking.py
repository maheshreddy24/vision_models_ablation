"""Tests for MaskedDinoV2. Requires network access on first run (torch.hub).

    pytest tests/ -v
"""
import pytest
import torch

from src import utils
from src.models.masked_dinov2 import MaskedDinoV2, load_backbone
from src.tasks.depth import LinearDepthHead


@pytest.fixture(scope="module")
def encoder():
    backbone_name = utils.load_config("config.yaml")["backbone"]
    try:
        return MaskedDinoV2(load_backbone(backbone_name))
    except Exception as e:
        pytest.skip(f"could not load backbone (network?): {e}")


def test_full_budget_matches_stock_model(encoder):
    """keep_ratio=1.0 must reproduce the unmasked backbone exactly."""
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x, keep_ratio=1.0)
    ref = encoder.backbone.forward_features(x)
    assert torch.allclose(out["cls"], ref["x_norm_clstoken"], atol=1e-5)
    assert torch.allclose(out["patch"], ref["x_norm_patchtokens"], atol=1e-5)


def test_token_counts_and_determinism(encoder):
    x = torch.randn(2, 3, 224, 224)
    n = (224 // encoder.patch_size) ** 2
    kr = 0.2
    out1 = encoder(x, keep_ratio=kr, mask_seed=7, sample_ids=torch.tensor([10, 11]))
    out2 = encoder(x, keep_ratio=kr, mask_seed=7, sample_ids=torch.tensor([10, 11]))
    out3 = encoder(x, keep_ratio=kr, mask_seed=8, sample_ids=torch.tensor([10, 11]))
    k = max(1, round(kr * n))
    assert out1["patch"].shape == (2, k, encoder.embed_dim)
    assert torch.equal(out1["kept_idx"], out2["kept_idx"]), "mask must be deterministic"
    assert not torch.equal(out1["kept_idx"], out3["kept_idx"]), "seed must change mask"


def test_mask_independent_of_batch_composition(encoder):
    """A sample's mask depends on its sample_id, not on who shares its batch."""
    x = torch.randn(3, 3, 224, 224)
    full = encoder(x, keep_ratio=0.3, mask_seed=0, sample_ids=torch.tensor([0, 1, 2]))
    solo = encoder(x[1:2], keep_ratio=0.3, mask_seed=0, sample_ids=torch.tensor([1]))
    assert torch.equal(full["kept_idx"][1], solo["kept_idx"][0])


def test_depth_head_scatter():
    """Predictions must land on the grid cells of their kept indices."""
    dim, gh, gw = 8, 4, 4
    head = LinearDepthHead(dim)
    patch = torch.randn(1, 3, dim)
    kept_idx = torch.tensor([[1, 5, 10]])
    pred, vis = head(patch, kept_idx, (gh, gw), full_image=False)
    assert pred.shape == (1, 1, gh, gw) and vis.shape == (1, 1, gh, gw)
    assert vis.view(-1)[kept_idx.view(-1)].all()
    assert vis.sum() == kept_idx.numel()
