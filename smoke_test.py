"""Minimal CPU smoke test for ST-PGN v2."""

import torch
from models import STPGN, STPGNConfig


def test_v2_full():
    """Test with all v2 improvements enabled."""
    torch.manual_seed(42)
    cfg = STPGNConfig(
        in_channels=21,
        seq_len=30,
        patch_len=10,
        stride=5,
        d_model=32,
        prior_hidden_dim=48,
        prior_dim=16,
        graph_dim=8,
        graph_layers=2,
        learnable_filter=True,
        use_cross_attention=True,
        cross_attn_heads=4,
        graph_topk=5,
        graph_self_loop=True,
    )
    model = STPGN(cfg)
    x = torch.randn(4, 30, 21)
    prior = torch.randn(4, 21, 5, 48)
    y = torch.randn(4, 1)

    pred, aux = model(x, prior, return_aux=True)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    assert pred.shape == (4, 1), f"pred shape mismatch: {pred.shape}"
    assert aux["adjacency"].shape == (21, 21), f"adj shape mismatch: {aux['adjacency'].shape}"
    assert "rul_sequence" in aux, "rul_sequence missing from aux"

    n_patches = (30 - 10) // 5 + 1  # = 5
    assert aux["rul_sequence"].shape == (4, n_patches), \
        f"rul_sequence shape mismatch: {aux['rul_sequence'].shape}"

    print("  [v2 full] prediction:", tuple(pred.shape))
    print("  [v2 full] patches:", tuple(aux["patches"].shape))
    print("  [v2 full] adjacency:", tuple(aux["adjacency"].shape))
    print("  [v2 full] rul_sequence:", tuple(aux["rul_sequence"].shape))
    print("  [v2 full] PASSED")


def test_v1_compat():
    """Test backward compatibility with v1 config (all new features off)."""
    torch.manual_seed(42)
    cfg = STPGNConfig(
        in_channels=21,
        seq_len=30,
        patch_len=10,
        stride=5,
        d_model=32,
        prior_hidden_dim=48,
        prior_dim=16,
        graph_dim=8,
        graph_layers=2,
        learnable_filter=False,
        use_cross_attention=False,
        graph_topk=0,         # 0 = no sparsity (dense graph)
        graph_self_loop=False,
    )
    model = STPGN(cfg)
    x = torch.randn(4, 30, 21)
    prior = torch.randn(4, 21, 5, 48)
    y = torch.randn(4, 1)

    pred, aux = model(x, prior, return_aux=True)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    assert pred.shape == (4, 1), f"v1 pred shape: {pred.shape}"
    assert aux["adjacency"].shape == (21, 21), f"v1 adj shape: {aux['adjacency'].shape}"
    assert aux["patches"].shape == (4, 21, 5, 32), f"v1 patches shape: {aux['patches'].shape}"
    print("  [v1 compat] PASSED")


def test_no_prior():
    """Test without prior (zero-prior fallback)."""
    torch.manual_seed(42)
    cfg = STPGNConfig(
        in_channels=21, seq_len=30, patch_len=10, stride=5,
        d_model=32, prior_hidden_dim=48, prior_dim=16,
        graph_dim=8, graph_layers=2,
    )
    model = STPGN(cfg)
    x = torch.randn(4, 30, 21)
    y = torch.randn(4, 1)
    pred = model(x)  # no prior_hidden
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    assert pred.shape == (4, 1), f"no-prior pred shape: {pred.shape}"
    print("  [no-prior] PASSED")


def test_gradient_flow():
    """Verify gradients flow through all new components."""
    torch.manual_seed(42)
    cfg = STPGNConfig(
        in_channels=21, seq_len=30, patch_len=10, stride=5,
        d_model=32, prior_hidden_dim=48, prior_dim=16,
        graph_dim=8, graph_layers=2,
        learnable_filter=True, use_cross_attention=True,
        graph_topk=5, graph_self_loop=True,
    )
    model = STPGN(cfg)
    x = torch.randn(4, 30, 21)
    prior = torch.randn(4, 21, 5, 48)
    y = torch.randn(4, 1)

    pred, aux = model(x, prior, return_aux=True)
    # Include rul_sequence in loss to ensure mono_head gets gradients
    # (mirrors PhysicsInformedLoss behavior from train.py)
    rul_seq = aux["rul_sequence"]
    mono_penalty = torch.nn.functional.relu(rul_seq[:, 1:] - rul_seq[:, :-1]).mean()
    loss = torch.nn.functional.mse_loss(pred, y) + 0.1 * mono_penalty
    loss.backward()

    # Check learnable frequency weights have gradients
    assert model.dft.w_freq.grad is not None, "LearnableFreqFilter has no gradient"
    # Check graph self-loop weights have gradients
    assert model.graph.self_weight.grad is not None, "self_weight has no gradient"
    # Check cross-attention parameters have gradients
    for name, param in model.cross_attn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"cross_attn.{name} has no gradient"
    # Check mono_head has gradients
    assert model.mono_head.weight.grad is not None, "mono_head has no gradient"
    print("  [gradient flow] PASSED")


def main():
    print("=" * 50)
    print("ST-PGN v2 Smoke Tests")
    print("=" * 50)
    test_v2_full()
    test_v1_compat()
    test_no_prior()
    test_gradient_flow()
    print()
    print("All smoke tests passed!")


if __name__ == "__main__":
    main()
