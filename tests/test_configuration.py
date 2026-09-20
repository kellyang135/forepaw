from pathlib import Path

import pytest

from go2wm.config import (
    ConfigurationError,
    load_json,
    load_toml,
    validate_eval_scenarios,
    validate_experiment,
    validate_external_sources,
)

ROOT = Path(__file__).parents[1]


def test_checked_in_experiment_contract_is_internally_consistent() -> None:
    warnings = validate_experiment(load_toml(ROOT / "configs/experiment.toml"))
    assert warnings == ["experiment values are still provisional"]


def test_eval_manifest_reserves_balanced_scenarios() -> None:
    warnings = validate_eval_scenarios(load_json(ROOT / "configs/eval_scenarios.json"))
    assert warnings == ["evaluation scenarios are reserved but not materialized and frozen"]


def test_external_source_validator_accepts_retained_local_verification() -> None:
    warnings = validate_external_sources(load_toml(ROOT / "configs/external_sources.toml"))
    assert "lewm revision is not pinned" not in warnings
    assert "lewm has not been verified locally" not in warnings
    assert "dimos has not been verified locally" not in warnings


def test_mismatched_horizon_is_rejected() -> None:
    config = load_toml(ROOT / "configs/experiment.toml")
    config["timing"]["planning_horizon_s"] = 4.0
    with pytest.raises(ConfigurationError, match="planning horizon"):
        validate_experiment(config)
