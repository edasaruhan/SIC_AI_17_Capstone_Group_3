"""Generate only Sprint 1 EDA artifacts from a configured real-data scope."""

from __future__ import annotations

import argparse
from pathlib import Path

from argus.config import get_path, load_config
from argus.data.load import load_ibm_aml
from argus.data.preprocess import preprocess_transactions
from argus.eda import generate_eda_artifacts
from argus.pipeline import _source_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sampling = config["data"]["sampling"]
    max_rows = sampling.get("max_rows") if sampling.get("enabled") else None
    source = get_path(config, "raw_data")
    raw = load_ibm_aml(
        source,
        sample_size=max_rows,
        random_seed=int(config["project"]["random_seed"]),
    )
    canonical = preprocess_transactions(raw, column_mapping=config["data"]["column_mapping"])
    eda_config = config.get("eda", {})
    output = get_path(config, "run_dir") / "eda"
    manifest = generate_eda_artifacts(
        canonical,
        output,
        random_seed=int(config["project"]["random_seed"]),
        graph_max_edges=eda_config.get("graph_max_edges", 50_000),
        top_n=int(eda_config.get("top_n", 20)),
        provenance={
            "dataset": "IBM AML-Data HI-Small",
            "dataset_is_synthetic": True,
            "locally_generated_fixture": False,
            "transactions": _source_fingerprint(source),
            "analysis_scope": raw.attrs["sampling_method"],
            "loaded_rows": len(raw),
        },
    )
    print(f"EDA: {manifest['status']} ({output})")
    print(f"Artifacts excluding manifest: {manifest['artifact_count_excluding_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
