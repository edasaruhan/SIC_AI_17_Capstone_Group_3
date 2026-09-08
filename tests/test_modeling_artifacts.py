from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import argus.modeling.artifacts as artifact_helpers
from argus.modeling.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    build_artifact_inventory,
    file_fingerprint,
    sha256_file,
    write_run_manifest,
)
from argus.modeling.baseline import _source_snapshot


def _temporary_files(directory: Path) -> list[Path]:
    return list(directory.glob(".*.tmp"))


def test_atomic_json_is_utf8_deterministic_and_replaces_existing(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "result.json"
    destination.parent.mkdir()
    destination.write_text("previous", encoding="utf-8")

    first = {"z": 3, "name": "şüpheli aday", "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "name": "şüpheli aday", "z": 3}

    assert atomic_write_json(first, destination) == destination
    expected = destination.read_bytes()
    assert json.loads(expected) == first
    assert expected.endswith(b"\n")

    atomic_write_json(second, destination)
    assert destination.read_bytes() == expected
    assert not _temporary_files(destination.parent)


def test_failed_json_serialization_preserves_previous_file(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    destination.write_bytes(b"previous-valid-result\n")

    with pytest.raises(TypeError, match="not JSON serializable"):
        atomic_write_json({"partial": object()}, destination)

    assert destination.read_bytes() == b"previous-valid-result\n"
    assert not _temporary_files(tmp_path)


def test_failed_atomic_replace_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.json"
    destination.write_bytes(b"previous-valid-result\n")

    def fail_replace(_: object, __: object) -> None:
        raise PermissionError("simulated Windows replace failure")

    monkeypatch.setattr(artifact_helpers.os, "replace", fail_replace)

    with pytest.raises(PermissionError, match="simulated Windows replace failure"):
        atomic_write_json({"status": "PASS"}, destination)

    assert destination.read_bytes() == b"previous-valid-result\n"
    assert not _temporary_files(tmp_path)


def test_atomic_csv_has_stable_line_endings_and_no_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "comparison.csv"
    frame = pd.DataFrame(
        {
            "model": ["logistic_regression", "random_forest"],
            "pr_auc": [0.12, 0.34],
        }
    )

    atomic_write_csv(frame, destination)

    assert destination.read_text(encoding="utf-8") == (
        "model,pr_auc\nlogistic_regression,0.12\nrandom_forest,0.34\n"
    )
    assert not _temporary_files(tmp_path)


def test_failed_csv_write_preserves_previous_file(tmp_path: Path) -> None:
    destination = tmp_path / "comparison.csv"
    destination.write_bytes(b"previous-valid-result\n")

    class FailingFrame:
        def to_csv(self, handle: object, **_: object) -> None:
            handle.write("partial")  # type: ignore[attr-defined]
            raise RuntimeError("simulated CSV failure")

    with pytest.raises(RuntimeError, match="simulated CSV failure"):
        atomic_write_csv(FailingFrame(), destination)  # type: ignore[arg-type]

    assert destination.read_bytes() == b"previous-valid-result\n"
    assert not _temporary_files(tmp_path)


def test_hash_and_inventory_are_exact_sorted_and_excludable(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_bytes(b"z")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.bin").write_bytes(b"one")
    (tmp_path / "a" / ".gitkeep").write_bytes(b"")
    (tmp_path / "excluded").mkdir()
    (tmp_path / "excluded" / "secret.txt").write_bytes(b"not an artifact")

    assert sha256_file(tmp_path / "z.txt", chunk_size=1) == hashlib.sha256(b"z").hexdigest()
    assert file_fingerprint(tmp_path / "a" / "one.bin", relative_to=tmp_path) == {
        "path": "a/one.bin",
        "size_bytes": 3,
        "sha256": hashlib.sha256(b"one").hexdigest(),
    }

    inventory = build_artifact_inventory(tmp_path, exclude_paths=["excluded"])
    assert [entry["path"] for entry in inventory] == ["a/one.bin", "z.txt"]
    assert [entry["size_bytes"] for entry in inventory] == [3, 1]


def test_run_manifest_is_self_excluding_repeatable_and_overrides_stale_inventory(
    tmp_path: Path,
) -> None:
    (tmp_path / "metrics.json").write_text('{"pr_auc": 0.5}\n', encoding="utf-8")
    payload = {
        "status": "PASS",
        "protocol": "frozen_chronological",
        "artifacts": [{"path": "stale"}],
        "artifact_count_excluding_manifest": 99,
    }

    first = write_run_manifest(payload, tmp_path)
    first_bytes = (tmp_path / "run_manifest.json").read_bytes()
    second = write_run_manifest(payload, tmp_path)

    assert first == second
    assert (tmp_path / "run_manifest.json").read_bytes() == first_bytes
    assert first["artifact_count_excluding_manifest"] == 1
    assert [entry["path"] for entry in first["artifacts"]] == ["metrics.json"]
    assert not _temporary_files(tmp_path)


def test_manifest_cannot_be_written_outside_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="inside its run directory"):
        write_run_manifest({"status": "PASS"}, run_dir, outside)

    assert not outside.exists()


def test_source_snapshot_includes_app_and_excludes_generated_package_metadata(
    tmp_path: Path,
) -> None:
    for directory in ("configs", "scripts", "src/argus_ai.egg-info", "tests"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "app.py").write_text("# app\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pandas\n", encoding="utf-8")
    (tmp_path / "scripts" / "run.py").write_text("# run\n", encoding="utf-8")
    (tmp_path / "src" / "argus_ai.egg-info" / "PKG-INFO").write_text(
        "generated\n", encoding="utf-8"
    )

    snapshot = _source_snapshot(tmp_path)
    paths = {item["path"] for item in snapshot["files"]}

    assert "app.py" in paths
    assert "scripts/run.py" in paths
    assert not any(".egg-info" in path for path in paths)
