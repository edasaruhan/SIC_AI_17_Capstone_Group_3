from __future__ import annotations

from pathlib import Path

import pytest

from argus.modeling.refinement_quality import _safe_cleanup


def test_quality_cleanup_accepts_only_scoped_child(tmp_path: Path) -> None:
    run_dir = tmp_path / "sprint3"
    work_dir = run_dir / ".quality_1234"
    work_dir.mkdir(parents=True)
    (work_dir / "result.txt").write_text("temporary", encoding="utf-8")
    _safe_cleanup(work_dir, run_dir)
    assert not work_dir.exists()


def test_quality_cleanup_rejects_unexpected_child(tmp_path: Path) -> None:
    run_dir = tmp_path / "sprint3"
    work_dir = run_dir / "models"
    work_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="unexpected quality path"):
        _safe_cleanup(work_dir, run_dir)
