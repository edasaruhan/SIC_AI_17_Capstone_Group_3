from __future__ import annotations

from pathlib import Path

from argus.config import load_config


def test_quick_config_inherits_base_and_resolves_project_paths() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "quick.yaml")

    assert config["project"]["mode"] == "quick"
    assert config["project"]["random_seed"] == 42
    assert config["data"]["sampling"] == {
        "enabled": True,
        "method": "chronological_prefix",
        "max_rows": 10000,
    }
    assert (
        Path(config["paths"]["raw_data"])
        == (project_root / "data" / "raw" / "HI-Small_Trans.csv").resolve()
    )
    assert (
        Path(config["paths"]["raw_accounts"])
        == (project_root / "data" / "raw" / "HI-Small_accounts.csv").resolve()
    )


def test_full_config_declares_bounded_out_of_core_engine() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "full.yaml")

    assert config["project"]["mode"] == "full"
    assert config["data"]["sampling"]["enabled"] is False
    assert config["full_pipeline"] == {
        "engine": "duckdb",
        "memory_limit": "2GB",
        "threads": 1,
        "max_temp_directory_size": "100GB",
        "parquet_compression": "zstd",
        "eda_sample_rows": 100000,
    }
