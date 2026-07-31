"""Minimal CPU smoke test for ST-PGN."""

import torch

from models import STPGN, STPGNConfig


def main():
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
    )
    model = STPGN(cfg)
    x = torch.randn(4, 30, 21)
    prior = torch.randn(4, 21, 5, 48)
    y = torch.randn(4, 1)

    pred, aux = model(x, prior, return_aux=True)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    assert pred.shape == (4, 1)
    assert aux["patches"].shape == (4, 21, 5, 32)
    assert aux["adjacency"].shape == (21, 21)
    print("ST-PGN smoke test passed")
    print("prediction:", tuple(pred.shape))
    print("patches:", tuple(aux["patches"].shape))
    print("adjacency:", tuple(aux["adjacency"].shape))


if __name__ == "__main__":
    main()
