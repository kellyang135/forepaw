"""LeWM training-cache contract: alignment, split purity, and tamper detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import numpy as np
from _learning_fixtures import push_episode

from go2wm.contracts import DatasetSplit
from go2wm.learning.lewm_cache import (
    CacheError,
    action_statistics,
    build_training_cache,
    check_disjoint,
    clip_starts,
    read_training_cache,
)
from go2wm.learning.windows import episode_sequence


def _cache(tmp_path: Path, name: str = "train", split: DatasetSplit = DatasetSplit.TRAIN, **kw):
    episodes = kw.pop("episodes", None) or [
        push_episode("ep-a", seed=1, split=split),
        push_episode("ep-b", seed=2, split=split),
    ]
    return build_training_cache(
        episodes,
        tmp_path / name,
        split=split,
        dataset_id="unit",
        split_id="splits-unit",
        image_size=32,
        **kw,
    ), episodes


def test_rows_pair_each_frame_with_its_outgoing_command(tmp_path: Path) -> None:
    cache, episodes = _cache(tmp_path)
    assert cache.manifest["episode_ids"] == ["ep-a", "ep-b"]
    for index, episode in enumerate(episodes):
        observations, actions = episode_sequence(episode)
        offset = int(cache.ep_offset[index])
        assert int(cache.ep_len[index]) == len(observations)
        for step, observation in enumerate(observations):
            row = offset + step
            expected = np.frombuffer(observation.rgb, dtype=np.uint8).reshape(32, 32, 3)
            assert np.array_equal(cache.pixels[row], expected)
            if step < len(actions):
                assert np.allclose(
                    cache.action[row], [actions[step].forward_mps, actions[step].yaw_rate_rps]
                )
            else:
                assert np.isnan(cache.action[row]).all()


def test_block_command_lands_on_the_block_start_frame(tmp_path: Path) -> None:
    cache, episodes = _cache(tmp_path)
    episode = episodes[0]
    history = len(episode.initial_history)
    for k, block in enumerate(episode.blocks):
        row = int(cache.ep_offset[0]) + history - 1 + k
        start = block.transition.start_observation
        assert np.array_equal(
            cache.pixels[row], np.frombuffer(start.rgb, dtype=np.uint8).reshape(32, 32, 3)
        )
        assert np.allclose(
            cache.action[row],
            [block.transition.action.forward_velocity_mps, block.transition.action.yaw_rate_rps],
        )


def test_labels_only_where_aligned_and_reset_frame_is_labeled(tmp_path: Path) -> None:
    cache, episodes = _cache(tmp_path)
    offset = int(cache.ep_offset[0])
    assert not np.isnan(cache.state[offset]).any()  # reset frame
    assert np.isnan(cache.state[offset + 1]).all()  # mid warm-up frame has no labels
    last = offset + int(cache.ep_len[0]) - 1
    end_labels = episodes[0].blocks[-1].transition.end_labels
    assert np.isclose(cache.state[last][0], end_labels.robot_pose.x_m)


def test_clips_never_cross_episodes(tmp_path: Path) -> None:
    cache, _ = _cache(tmp_path)
    starts = cache.clip_starts(4)
    lengths = cache.ep_len.tolist()
    assert len(starts) == sum(n - 3 for n in lengths)
    for start in starts:
        episode = int(np.searchsorted(cache.ep_offset, start, side="right")) - 1
        assert start + 4 <= cache.ep_offset[episode] + cache.ep_len[episode]
    assert len(clip_starts(np.array([2, 5]), np.array([0, 2]), 4)) == 2


def test_cache_refuses_mixed_splits_wrong_size_and_overwrite(tmp_path: Path) -> None:
    mixed = [
        push_episode("ep-a", seed=1, split=DatasetSplit.TRAIN),
        push_episode("ep-b", seed=2, split=DatasetSplit.VALIDATION),
    ]
    with pytest.raises(CacheError, match="not in split"):
        build_training_cache(
            mixed,
            tmp_path / "x",
            split=DatasetSplit.TRAIN,
            dataset_id="d",
            split_id="s",
            image_size=32,
        )
    with pytest.raises(CacheError, match="expected 224x224"):
        build_training_cache(
            mixed[:1],
            tmp_path / "y",
            split=DatasetSplit.TRAIN,
            dataset_id="d",
            split_id="s",
        )
    _cache(tmp_path)
    with pytest.raises(FileExistsError):
        _cache(tmp_path)


def test_tampered_cache_is_rejected(tmp_path: Path) -> None:
    cache, _ = _cache(tmp_path)
    action = np.load(cache.root / "action.npy")
    action[0, 0] += 0.1
    np.save(cache.root / "action.npy", action)
    with pytest.raises(CacheError, match="checksum"):
        read_training_cache(cache.root)


def test_final_frame_action_must_be_nan(tmp_path: Path) -> None:
    cache, _ = _cache(tmp_path)
    manifest = json.loads((cache.root / "manifest.json").read_text())
    action = np.load(cache.root / "action.npy")
    action[int(cache.ep_len[0]) - 1] = 0.0
    np.save(cache.root / "action.npy", action)
    with pytest.raises(CacheError, match="NaN action"):
        read_training_cache(cache.root, verify=False)
    assert manifest["action_alignment"].startswith("action[i] is applied from frame i")


def test_action_statistics_are_train_only_and_unbiased(tmp_path: Path) -> None:
    cache, _ = _cache(tmp_path)
    mean, std = action_statistics(cache)
    rows = cache.action[np.isfinite(cache.action).all(axis=1)]
    assert np.allclose(mean, rows.mean(axis=0), atol=1e-6)
    assert np.allclose(std, rows.std(axis=0, ddof=1), atol=1e-6)
    val, _ = _cache(
        tmp_path,
        name="val",
        split=DatasetSplit.VALIDATION,
        episodes=[
            push_episode("ep-c", seed=3, split=DatasetSplit.VALIDATION),
        ],
    )
    with pytest.raises(CacheError, match="train split only"):
        action_statistics(val)
    check_disjoint([cache, val])


def test_disjointness_catches_shared_episodes(tmp_path: Path) -> None:
    first, _ = _cache(tmp_path, name="a")
    second, _ = _cache(tmp_path, name="b")
    with pytest.raises(CacheError, match="more than one cache"):
        check_disjoint([first, second])


def test_synthetic_control_moves_square_by_the_command(tmp_path: Path) -> None:
    from go2wm.learning.lewm_cache import write_synthetic_control

    train, val = write_synthetic_control(tmp_path / "ctl", train_episodes=3, val_episodes=2)
    check_disjoint([train, val])
    assert train.manifest["rehearsal_only"] is True
    for row in range(int(train.ep_len[0]) - 1):
        before, after = train.state[row], train.state[row + 1]
        moved = (after[:2] - before[:2]) * 100.0
        expected = 40.0 * train.action[row]
        clipped = np.any(np.isclose(after[:2] * 100.0, [16.0, 208.0], atol=1e-3))
        assert clipped or np.allclose(moved, expected, atol=1e-3)
        ys, xs = np.nonzero(train.pixels[row + 1][..., 0])
        assert abs(xs.mean() - after[0] * 100.0) < 1.0
        assert abs(ys.mean() - after[1] * 100.0) < 1.0
