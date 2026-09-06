"""Schema-aware loader for IBM AML-Data HI-Small transactions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_TRANSACTION_COLUMNS = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)


class DataLoadError(ValueError):
    """Raised when a source cannot be loaded as an IBM AML transaction file."""


def load_ibm_aml(
    path: str | Path,
    *,
    sample_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load HI-Small without losing zero-padded identifiers.

    ``sample_size`` selects the first N physical records. HI-Small is ordered by
    time, so this is deterministic and retains a valid history prefix. It is not
    a statistically representative sample and artifacts must label it as such.
    ``random_seed`` is recorded for configuration consistency; no random row
    operation is performed by this loader.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"IBM AML HI-Small transactions were not found. Expected a CSV at: {source.resolve()}"
        )
    if sample_size is not None and (
        isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0
    ):
        raise DataLoadError("sample_size must be null or a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise DataLoadError("random_seed must be a non-negative integer")

    try:
        frame = pd.read_csv(
            source,
            dtype="string",
            keep_default_na=False,
            nrows=sample_size,
            encoding="utf-8",
            on_bad_lines="error",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"Could not read IBM AML transactions at {source}: {exc}") from exc

    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != RAW_TRANSACTION_COLUMNS:
        missing = sorted(set(RAW_TRANSACTION_COLUMNS).difference(actual_columns))
        unexpected = sorted(set(actual_columns).difference(RAW_TRANSACTION_COLUMNS))
        raise DataLoadError(
            "Unexpected IBM AML transaction schema. The two raw 'Account' headers "
            "must load positionally as 'Account' and 'Account.1'. "
            f"missing={missing}, unexpected={unexpected}, actual={list(actual_columns)}"
        )
    if frame.empty:
        raise DataLoadError(f"IBM AML transaction file has no data rows: {source}")

    frame.insert(0, "source_row_number", range(2, len(frame) + 2))
    frame.attrs.update(
        {
            "source_path": str(source.resolve()),
            "source_name": source.name,
            "sampling_method": "chronological_prefix" if sample_size else "full_file",
            "sample_size_requested": sample_size,
            "random_seed": random_seed,
        }
    )
    return frame
