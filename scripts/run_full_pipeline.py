"""Run the out-of-core Sprint 1 pipeline against complete HI-Small files."""

from __future__ import annotations

from argus.full_pipeline import run_full_sprint1_pipeline


def main() -> int:
    result = run_full_sprint1_pipeline("configs/full.yaml")
    print(f"Full Sprint 1 pipeline: {result.manifest['status']}")
    print(f"Rows: {result.manifest['configuration']['loaded_transaction_rows']}")
    print(f"Runtime seconds: {result.manifest['runtime_seconds']:.3f}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
