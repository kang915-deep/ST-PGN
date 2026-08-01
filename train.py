"""Train the first ST-PGN baseline on C-MAPSS."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import build_cmapss_datasets
from models import STPGN, STPGNConfig, UniTSPriorAdapter
from visualize_results import save_experiment_plots


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nasa_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """C-MAPSS asymmetric score; overestimation is penalized more strongly."""
    delta = pred - target
    score = torch.where(
        delta < 0,
        torch.exp(-delta / 13.0) - 1.0,
        torch.exp(delta / 10.0) - 1.0,
    )
    return float(score.sum().item())


@torch.no_grad()
def evaluate(model, loader, device, units=None, return_details=False):
    model.eval()
    predictions, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prior = units(x) if units is not None else None
        predictions.append(model(x, prior).squeeze(-1).cpu())
        targets.append(y.squeeze(-1).cpu())
    pred = torch.cat(predictions)
    target = torch.cat(targets)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    metrics = {"rmse": rmse, "mae": mae, "score": nasa_score(pred, target)}
    if return_details:
        dataset = loader.dataset
        metrics["details"] = {
            "prediction": pred.numpy(),
            "target": target.numpy(),
            "unit": np.asarray(dataset.sample_units, dtype=np.int64),
            "cycle": np.asarray(dataset.sample_cycles, dtype=np.int64),
        }
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, units=None):
    model.train()
    total_loss, total_count = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prior = units(x) if units is not None else None
        pred = model(x, prior)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/cmapss")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--units-repo", default=None)
    parser.add_argument("--units-checkpoint", default=None)
    parser.add_argument("--patch-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dft-keep-ratio", type=float, default=0.5)
    parser.add_argument("--padding", choices=["none", "replicate", "reflect"], default="none")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--val-all-windows", action="store_true",
        help="use all validation windows instead of one final window per engine",
    )
    args = parser.parse_args()

    if bool(args.units_repo) != bool(args.units_checkpoint):
        parser.error("--units-repo and --units-checkpoint must be provided together")
    if args.output_dir is None:
        variant = "units" if args.units_repo else "baseline"
        args.output_dir = f"checkpoints/stpgn_{args.subset.lower()}_{variant}"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = build_cmapss_datasets(
        root=args.data_root,
        subset=args.subset,
        window_size=args.window_size,
        seed=args.seed,
        val_last_only=not args.val_all_windows,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    units = None
    if args.units_repo:
        units = UniTSPriorAdapter(
            units_repo=args.units_repo,
            checkpoint=args.units_checkpoint,
            channels=21,
        ).to(device)

    model = STPGN(STPGNConfig(
        in_channels=21,
        seq_len=args.window_size,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=64,
        graph_dim=16,
        graph_layers=2,
        prior_dim=64,
        prior_hidden_dim=128,
        dft_keep_ratio=args.dft_keep_ratio,
        padding=args.padding,
        dropout=0.1,
        head_hidden=128,
    )).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_rmse = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, units)
        val_metrics = evaluate(model, val_loader, device, units)
        scheduler.step(val_metrics["rmse"])
        print(
            f"epoch={epoch:03d} loss={train_loss:.5f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_score={val_metrics['score']:.2f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            stale_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": vars(args),
            }, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch={epoch:03d}")
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, units, return_details=True)
    details = test_metrics.pop("details")
    print("best validation:", checkpoint["val_metrics"])
    print("test:", test_metrics)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    np.savez(
        output_dir / "history.npz",
        epoch=np.asarray([item["epoch"] for item in history]),
        train_loss=np.asarray([item["train_loss"] for item in history]),
        val_rmse=np.asarray([item["val_rmse"] for item in history]),
        val_mae=np.asarray([item["val_mae"] for item in history]),
        val_score=np.asarray([item["val_score"] for item in history]),
        lr=np.asarray([item["lr"] for item in history]),
    )
    np.savez(output_dir / "predictions.npz", **details)
    np.save(output_dir / "adjacency.npy", model.graph.adjacency().detach().cpu().numpy())
    save_experiment_plots(str(output_dir))


if __name__ == "__main__":
    main()
