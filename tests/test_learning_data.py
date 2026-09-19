"""Person B data-consumer tests: strict loading, splits, windows, diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from _learning_fixtures import push_episode, written_episode

from go2wm.contracts import DatasetSplit
from go2wm.data.storage import write_episode
from go2wm.learning import (
    DatasetLoadError,
    SplitLeakError,
    build_split_manifest,
    check_leakage,
    classify_block,
    coverage_report,
    episode_windows,
    load_dataset,
    load_episode,
    read_split_manifest,
    write_split_manifest,
)


def _rewrite_jsonl(path: Path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_loader_round_trips_a_written_episode(tmp_path: Path) -> None:
    original = push_episode()
    directory = write_episode(tmp_path, original).episode_directory
    loaded = load_episode(directory)
    assert loaded == original


def test_loader_rejects_frame_checksum_mismatch(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)
    frame = directory / "frames" / "000004.rgb"
    payload = bytearray(frame.read_bytes())
    payload[0] ^= 0xFF
    frame.write_bytes(bytes(payload))
    with pytest.raises(DatasetLoadError, match="checksum"):
        load_episode(directory)


def test_loader_rejects_missing_frame_file(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)
    (directory / "frames" / "000003.rgb").unlink()
    with pytest.raises(DatasetLoadError, match="missing frame"):
        load_episode(directory)


def test_loader_rejects_missing_block(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)
    _rewrite_jsonl(directory / "blocks.jsonl", lambda rows: rows[:2] + rows[3:])
    with pytest.raises(DatasetLoadError, match="declares"):
        load_episode(directory)


def test_loader_rejects_duplicate_block_index(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)

    def duplicate(rows):
        rows[3]["block_index"] = 2
        return rows

    _rewrite_jsonl(directory / "blocks.jsonl", duplicate)
    with pytest.raises(DatasetLoadError, match="contiguous"):
        load_episode(directory)


def test_loader_rejects_nonfinite_label(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)

    def poison(rows):
        rows[1]["end_labels"]["robot_pose"]["x_m"] = math.nan
        return rows

    _rewrite_jsonl(directory / "blocks.jsonl", poison)
    with pytest.raises(DatasetLoadError, match="finite"):
        load_episode(directory)


def test_loader_rejects_action_duration_mismatch(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)

    def stretch(rows):
        rows[0]["applied_action"]["duration_s"] = 0.6
        return rows

    _rewrite_jsonl(directory / "blocks.jsonl", stretch)
    with pytest.raises(DatasetLoadError, match="duration"):
        load_episode(directory)


def test_loader_rejects_action_unit_mismatch(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)

    def change_units(rows):
        rows[0]["action_units"]["yaw_rate"] = "degrees/s"
        return rows

    _rewrite_jsonl(directory / "blocks.jsonl", change_units)
    with pytest.raises(DatasetLoadError, match="action units"):
        load_episode(directory)


def test_loader_checksum_covers_metadata_and_jsonl(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)
    meta = json.loads((directory / "episode.json").read_text())
    meta["metadata"]["tampered"] = "true"
    (directory / "episode.json").write_text(json.dumps(meta))
    with pytest.raises(DatasetLoadError, match="artifact checksum"):
        load_episode(directory)


def test_loader_rejects_shifted_label_timestamp(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)

    def shift(rows):
        rows[2]["start_labels"]["sim_time_s"] += 0.5
        return rows

    _rewrite_jsonl(directory / "blocks.jsonl", shift)
    with pytest.raises(DatasetLoadError, match="time-aligned"):
        load_episode(directory)


def test_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    directory = written_episode(tmp_path)
    meta = json.loads((directory / "episode.json").read_text())
    meta["schema_version"] = "go2wm.episode.v0"
    (directory / "episode.json").write_text(json.dumps(meta))
    with pytest.raises(DatasetLoadError, match="schema"):
        load_episode(directory)


def test_lenient_dataset_load_reports_every_rejection(tmp_path: Path) -> None:
    good = written_episode(tmp_path, episode_id="ep-a", seed=1)
    bad = written_episode(tmp_path, episode_id="ep-b", seed=2)
    (bad / "frames" / "000000.rgb").unlink()
    with pytest.raises(DatasetLoadError):
        load_dataset(tmp_path)
    dataset = load_dataset(tmp_path, strict=False)
    assert [e.episode_id for e in dataset.episodes] == [good.name]
    assert [name for name, _ in dataset.rejected] == ["ep-b"]


def test_split_assignment_is_deterministic_and_reserves_test_seeds() -> None:
    first = build_split_manifest(range(1, 301), salt="s1", reserved_test_seeds=[5, 6])
    second = build_split_manifest(range(300, 0, -1), salt="s1", reserved_test_seeds=[6, 5])
    assert first.split_id == second.split_id
    assert dict(first.assignments) == dict(second.assignments)
    assert first.split_of(5) is DatasetSplit.TEST
    counts = {split: len(first.seeds(split)) for split in DatasetSplit}
    assert all(count > 20 for count in counts.values())
    assert build_split_manifest(range(1, 301), salt="s2").split_id != first.split_id


def test_split_manifest_is_write_once_and_detects_hand_edits(tmp_path: Path) -> None:
    manifest = build_split_manifest(range(1, 50), salt="salt")
    path = tmp_path / "splits.json"
    write_split_manifest(path, manifest)
    assert read_split_manifest(path).split_id == manifest.split_id
    with pytest.raises(FileExistsError):
        write_split_manifest(path, manifest)
    payload = json.loads(path.read_text())
    seed = next(k for k, v in payload["assignments"].items() if v == "test")
    payload["assignments"][seed] = "train"
    path.write_text(json.dumps(payload))
    with pytest.raises(SplitLeakError, match="edited"):
        read_split_manifest(path)


def test_leak_check_flags_episode_in_wrong_split() -> None:
    manifest = build_split_manifest([7, 8], salt="x", reserved_test_seeds=[7])
    wrong = push_episode("ep-wrong", seed=7, split=DatasetSplit.TRAIN)
    report = check_leakage([wrong], manifest)
    assert not report.ok
    assert "frozen as test" in report.problems[0]
    unknown = push_episode("ep-unknown", seed=99, split=DatasetSplit.TRAIN)
    assert not check_leakage([unknown], manifest).ok


def test_windows_align_history_commands_and_future_labels() -> None:
    episode = push_episode()
    windows = episode_windows(episode, horizon=3)
    assert len(windows) == len(episode.blocks) - 3 + 1
    for window in windows:
        k = window.start_block
        block = episode.blocks[k].transition
        assert window.model_input.observations[-1] == block.start_observation
        assert len(window.model_input.observations) == 3
        assert len(window.model_input.command_history) == 2
        expected_last = (
            episode.history_actions[-1] if k == 0 else episode.blocks[k - 1].transition.action
        )
        assert (
            window.model_input.command_history[-1].forward_mps == expected_last.forward_velocity_mps
        )
        assert window.future_actions[0].forward_mps == block.action.forward_velocity_mps
        for step, labels in enumerate(window.future_labels, start=1):
            assert math.isclose(labels.sim_time_s, block.start_labels.sim_time_s + 0.5 * step)
        assert window.future_observations[0] == block.end_observation


def test_block_classification_and_coverage_report() -> None:
    episode = push_episode()
    kinds = [classify_block(block.transition) for block in episode.blocks]
    assert "push" in kinds
    assert kinds[-1] in {"free", "brief_contact"}
    report = coverage_report([episode])
    split = report["splits"]["train"]
    assert sum(split["interaction"].values()) == len(episode.blocks)
    assert 0.0 < split["interaction_fraction"] <= 1.0
    assert any("light:blue" in key for key in split["contacted_object"])
