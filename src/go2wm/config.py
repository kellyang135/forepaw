"""Load and validate the experiment's dependency-light configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, which LeWM requires
    import tomli as tomllib


class ConfigurationError(ValueError):
    """Raised when a configuration violates an experiment invariant."""


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return value


def validate_experiment(config: dict[str, Any]) -> list[str]:
    """Return warnings and raise for contradictions in the experiment contract."""

    timing = _table(config, "timing")
    observation = _table(config, "observation")
    action = _table(config, "action")
    model = _table(config, "model")
    collection = _table(config, "collection")
    runtime = _table(config, "runtime")

    block = _positive_float(timing, "block_duration_s")
    hz = _positive_float(timing, "observation_hz")
    history_frames = _positive_int(timing, "history_frames")
    history_span = _positive_float(timing, "history_span_s")
    planning_blocks = _positive_int(timing, "planning_blocks")
    horizon = _positive_float(timing, "planning_horizon_s")

    expected_period = 1.0 / hz
    if abs(expected_period - block) > 1e-9:
        raise ConfigurationError(
            "observation period must equal action block duration in the initial contract"
        )
    expected_history_span = (history_frames - 1) * expected_period
    if abs(history_span - expected_history_span) > 1e-9:
        raise ConfigurationError(
            f"history_span_s={history_span} but {history_frames} frames at {hz} Hz "
            f"span {expected_history_span} seconds"
        )
    if abs(horizon - planning_blocks * block) > 1e-9:
        raise ConfigurationError("planning horizon must equal planning_blocks * block_duration_s")
    if model.get("history_size") != history_frames:
        raise ConfigurationError("model.history_size must equal timing.history_frames")
    if model.get("candidate_count") != 64:
        raise ConfigurationError("the committed initial planner must contain exactly 64 candidates")
    if observation.get("channels") != 3:
        raise ConfigurationError("the committed observation contract is RGB")
    if action.get("fields") != ["forward_velocity_mps", "yaw_rate_rps"]:
        raise ConfigurationError(
            "action.fields must preserve the [forward velocity, yaw rate] order"
        )
    if action.get("lateral_velocity_mps") != 0.0:
        raise ConfigurationError("lateral velocity is outside the committed scope")
    if collection.get("split_unit") != "scenario_seed":
        raise ConfigurationError("splits must be made by complete scenario seeds")
    if set(collection.get("required_splits", [])) != {"train", "validation", "test"}:
        raise ConfigurationError("train, validation, and test splits are all required")
    if runtime.get("single_motion_owner") is not True:
        raise ConfigurationError("runtime must have exactly one motion-command owner")
    if runtime.get("stop_on_surprise") is not True:
        raise ConfigurationError("the committed surprise response is an immediate stop")

    warnings: list[str] = []
    if str(config.get("status", "")).startswith("PROVISIONAL"):
        warnings.append("experiment values are still provisional")
    return warnings


def validate_eval_scenarios(manifest: dict[str, Any]) -> list[str]:
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list):
        raise ConfigurationError("eval scenario manifest needs a scenarios list")

    identifiers: set[str] = set()
    behavior_counts = {"push": 0, "detour": 0}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ConfigurationError("each eval scenario must be an object")
        scenario_id = scenario.get("id")
        behavior = scenario.get("behavior")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ConfigurationError("each eval scenario needs a non-empty string id")
        if scenario_id in identifiers:
            raise ConfigurationError(f"duplicate eval scenario id: {scenario_id}")
        if behavior not in behavior_counts:
            raise ConfigurationError(f"unknown eval behavior for {scenario_id}: {behavior}")
        identifiers.add(scenario_id)
        behavior_counts[behavior] += 1

    if behavior_counts != {"push": 12, "detour": 12}:
        raise ConfigurationError(
            f"expected 12 push and 12 detour scenarios; found {behavior_counts}"
        )
    warnings: list[str] = []
    if manifest.get("status") != "FROZEN":
        warnings.append("evaluation scenarios are reserved but not materialized and frozen")
    return warnings


def validate_external_sources(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name in ("lewm", "dimos", "go2_model"):
        source = _table(config, name)
        revision = source.get("commit", source.get("revision"))
        if not revision or revision == "UNPINNED":
            warnings.append(f"{name} revision is not pinned")
        if source.get("verified_locally") is not True:
            warnings.append(f"{name} has not been verified locally")
    return warnings


def _table(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"missing [{key}] table")
    return value


def _positive_float(table: dict[str, Any], key: str) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive number")
    return float(value)


def _positive_int(table: dict[str, Any], key: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive integer")
    return value
