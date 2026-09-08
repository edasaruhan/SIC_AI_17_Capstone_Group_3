"""Run Sprint 4 sampled GraphSAGE training and saved-artifact product export."""

from __future__ import annotations

import argparse

from argus.sprint4.pipeline import run_sprint4_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/sprint4.yaml",
        help="Sprint 4 YAML configuration",
    )
    args = parser.parse_args()
    result = run_sprint4_pipeline(args.config)
    print(f"Sprint 4 status: {result.manifest['status']}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Report: {result.report_path}")
    print(f"Runtime seconds: {result.manifest['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
