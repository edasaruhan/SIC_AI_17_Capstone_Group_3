from __future__ import annotations

from copy import deepcopy

import pytest

from argus.config import ConfigError, load_config, validate_config


@pytest.fixture
def sprint4_config() -> dict:
    return load_config("configs/sprint4.yaml", resolve_paths=False)


def test_sprint4_config_loads_and_keeps_final_test_closed(sprint4_config: dict) -> None:
    settings = sprint4_config["sprint4"]
    assert settings["final_test_access"] is False
    assert settings["selection_partition"] == "validation"
    assert settings["model"]["unsupported_node_label_created"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "final_test_access", True),
        ("model", "unsupported_node_label_created", True),
        ("application", "train_on_page_load", True),
        ("graph_sampling", "full_graph_training_attempted", True),
    ],
)
def test_sprint4_config_rejects_scientific_protocol_violations(
    sprint4_config: dict,
    section: str | None,
    key: str,
    value: object,
) -> None:
    changed = deepcopy(sprint4_config)
    target = changed["sprint4"] if section is None else changed["sprint4"][section]
    target[key] = value
    with pytest.raises(ConfigError):
        validate_config(changed)


def test_sprint4_config_rejects_too_few_observed_evidence_items(
    sprint4_config: dict,
) -> None:
    changed = deepcopy(sprint4_config)
    changed["sprint4"]["case_builder"]["minimum_observed_evidence"] = 2
    with pytest.raises(ConfigError, match="at least three"):
        validate_config(changed)


def test_sprint4_config_requires_subset_disclosure(sprint4_config: dict) -> None:
    changed = deepcopy(sprint4_config)
    sampling = changed["sprint4"]["graph_sampling"]
    sampling["training_context_max_edges"] = sampling["training_context_population_rows"]
    with pytest.raises(ConfigError, match="must be a subset"):
        validate_config(changed)


def test_sprint4_threshold_contract_matches_sprint3(sprint4_config: dict) -> None:
    changed = deepcopy(sprint4_config)
    changed["sprint4"]["threshold_optimization"]["alert_budget"] = 4999
    with pytest.raises(ConfigError, match="must match Sprint 3"):
        validate_config(changed)
