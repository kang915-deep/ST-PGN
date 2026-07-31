"""Inspect the C-MAPSS loader on a local or AutoDL dataset directory."""

import argparse

from datasets import build_cmapss_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cmapss")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--window-size", type=int, default=30)
    args = parser.parse_args()

    datasets = build_cmapss_datasets(
        root=args.root,
        subset=args.subset,
        window_size=args.window_size,
    )
    for name, dataset in datasets.items():
        x, y = dataset[0]
        print(f"{name}: samples={len(dataset)}, x={tuple(x.shape)}, y={tuple(y.shape)}")
    print("sensor mean shape:", datasets["train"].stats.mean.shape)


if __name__ == "__main__":
    main()
