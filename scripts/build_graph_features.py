"""Build strictly-prior fan-in/fan-out features for the configured data scope."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.config import get_path, load_config
from argus.data.load import load_ibm_aml
from argus.data.preprocess import preprocess_transactions
from argus.features.graph import add_graph_history_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sampling = config["data"]["sampling"]
    max_rows = sampling.get("max_rows") if sampling.get("enabled") else None
    raw = load_ibm_aml(
        get_path(config, "raw_data"),
        sample_size=max_rows,
        random_seed=int(config["project"]["random_seed"]),
    )
    canonical = preprocess_transactions(raw, column_mapping=config["data"]["column_mapping"])
    featured = add_graph_history_features(canonical)
    output = get_path(config, "run_dir") / "tables" / "graph_history_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    graph_columns = [
        column
        for column in featured
        if "prior_fan_" in column or column == "pair_previous_transfer_count"
    ]
    featured[["transaction_id", "timestamp", *graph_columns]].to_csv(output, index=False)
    print(f"Graph-history features: PASS ({output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
