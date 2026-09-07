"""Deterministic, crash-safe artifact writing for Sprint 2 model runs.

The atomic writers create a unique temporary file beside the destination, close
it before replacement, and then use :func:`os.replace`.  Keeping both paths on
the same filesystem and closing the temporary handle first is important on
Windows.  A failed serialization or replacement leaves an existing destination
untouched and removes the partial temporary file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

_HASH_CHUNK_SIZE = 1024 * 1024
_DEFAULT_EXCLUDED_NAMES = (".gitkeep",)


def _json_default(value: object) -> object:
    """Convert common manifest scalar types without silently stringifying objects."""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value

    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError):
            pass
        else:
            if scalar is not value:
                return scalar

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_text_write(
    destination: str | Path,
    writer: Callable[[TextIO], None],
) -> Path:
    """Write text through a unique sibling file and atomically replace destination."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # The handle is deliberately closed before replacement for Windows.
        os.replace(temporary, output)
    finally:
        # After a successful replace the temporary path no longer exists.
        temporary.unlink(missing_ok=True)

    return output


def atomic_write_json(payload: Any, destination: str | Path) -> Path:
    """Serialize strict, key-sorted UTF-8 JSON and replace ``destination`` atomically."""

    def write(handle: TextIO) -> None:
        json.dump(
            payload,
            handle,
            allow_nan=False,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return _atomic_text_write(destination, write)


def atomic_write_csv(
    frame: pd.DataFrame,
    destination: str | Path,
    *,
    index: bool = False,
    **to_csv_kwargs: Any,
) -> Path:
    """Write a pandas frame as UTF-8 CSV and replace ``destination`` atomically.

    Column and row order are intentionally kept exactly as supplied by the
    caller.  Callers that require a deterministic order must sort explicitly;
    this helper standardizes only encoding and line endings.
    """

    options = dict(to_csv_kwargs)
    options.setdefault("lineterminator", "\n")

    def write(handle: TextIO) -> None:
        frame.to_csv(handle, index=index, **options)

    return _atomic_text_write(destination, write)


def sha256_file(path: str | Path, *, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Return the lowercase SHA-256 digest of one file using bounded memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Artifact file was not found: {source}")

    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_artifact_path(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact is outside inventory root: {resolved_path}") from exc
    return relative.as_posix()


def file_fingerprint(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    """Return a stable filename/path, size, and SHA-256 record for one file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Artifact file was not found: {source}")

    if relative_to is None:
        display_path = source.name
    else:
        root = Path(relative_to).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Fingerprint root is not a directory: {root}")
        display_path = _relative_artifact_path(source, root)

    return {
        "path": display_path,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _resolve_excluded_paths(root: Path, paths: Sequence[str | Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved.append(candidate.resolve())
    return tuple(resolved)


def _is_excluded(path: Path, excluded: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == item or item in resolved.parents for item in excluded)


def build_artifact_inventory(
    root: str | Path,
    *,
    exclude_paths: Sequence[str | Path] = (),
    exclude_names: Collection[str] = _DEFAULT_EXCLUDED_NAMES,
) -> list[dict[str, Any]]:
    """Hash files below ``root`` into a deterministically ordered inventory.

    An excluded path may identify either one file or a whole directory.  A
    relative exclusion is interpreted relative to ``root``.  Symlinks that
    resolve outside ``root`` are rejected instead of hashing an external file.
    """

    inventory_root = Path(root).resolve()
    if not inventory_root.is_dir():
        raise NotADirectoryError(f"Artifact inventory root is not a directory: {inventory_root}")

    excluded = _resolve_excluded_paths(inventory_root, exclude_paths)
    candidates = sorted(
        (candidate for candidate in inventory_root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(inventory_root).as_posix(),
    )

    inventory: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.name in exclude_names or _is_excluded(candidate, excluded):
            continue
        inventory.append(file_fingerprint(candidate, relative_to=inventory_root))
    return inventory


def _manifest_destination(run_dir: Path, manifest_path: str | Path) -> Path:
    destination = Path(manifest_path)
    if not destination.is_absolute():
        destination = run_dir / destination
    destination = destination.resolve()
    try:
        destination.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"Run manifest must be inside its run directory: {destination}") from exc
    return destination


def write_run_manifest(
    payload: Mapping[str, Any],
    run_dir: str | Path,
    manifest_path: str | Path = "run_manifest.json",
    *,
    exclude_paths: Sequence[str | Path] = (),
    exclude_names: Collection[str] = _DEFAULT_EXCLUDED_NAMES,
) -> dict[str, Any]:
    """Inventory a run and atomically write a deterministic run manifest.

    The manifest itself is always excluded.  Inventory fields supplied in
    ``payload`` are replaced with freshly computed values so a stale caller
    cannot create an internally inconsistent manifest.  No timestamp or random
    run identifier is injected; reproducibility metadata must be explicit in
    ``payload``.
    """

    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = _manifest_destination(root, manifest_path)

    exclusions = [*exclude_paths, destination]
    inventory = build_artifact_inventory(
        root,
        exclude_paths=exclusions,
        exclude_names=exclude_names,
    )
    manifest = dict(payload)
    manifest["artifacts"] = inventory
    manifest["artifact_count_excluding_manifest"] = len(inventory)
    atomic_write_json(manifest, destination)
    return manifest
