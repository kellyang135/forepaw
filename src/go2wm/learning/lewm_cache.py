"""Lossless, split-pure training caches for LeWM.

One cache directory holds one split of one dataset as flat, aligned columns::

    pixels.npy      uint8  (N, H, W, 3)   every frame of every episode, in order
    action.npy      float32 (N, 2)        command applied from row i to row i + 1;
                                          NaN on the last frame of each episode
    state.npy       float32 (N, K)        privileged labels for readout probes only;
                                          NaN where a frame has no aligned label
    ep_len.npy      int64  (E,)           frames per episode
    ep_offset.npy   int64  (E,)           first row of each episode
    manifest.json   provenance, shapes, and SHA-256 of every array file

The layout mirrors stable-worldmodel's ``ep_len``/``ep_offset`` convention and
LeWM's action alignment (``action[t]`` is applied from frame ``t``), but frames
are stored raw.  stable-worldmodel's Lance and folder writers JPEG-encode
frames, which would make training pixels differ from the raw RGB the deployed
controller sees.  Its ``train.py`` also splits windows at random, which lets
frames from one episode land in both train and validation; separate per-split
caches built from the frozen split manifest avoid both problems.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from go2wm.contracts import DatasetSplit, EpisodeRecord, StateLabels

from .readouts import Standardizer, TargetSpec
from .windows import episode_sequence

CACHE_SCHEMA = "go2wm.lewm-cache.v1"
ACTION_FIELDS = ("forward_velocity_mps", "yaw_rate_rps")
ARRAY_FILES = ("pixels.npy", "action.npy", "state.npy", "ep_len.npy", "ep_offset.npy")


class CacheError(ValueError):
    """Raised when a cache would violate, or does violate, the training contract."""


@dataclass(frozen=True, slots=True)
class TrainingCache:
    root: Path
    manifest: dict[str, Any]
    pixels: np.ndarray
    action: np.ndarray
    state: np.ndarray
    ep_len: np.ndarray
    ep_offset: np.ndarray

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest["episode_ids"])

    @property
    def frame_count(self) -> int:
        return int(self.pixels.shape[0])

    def clip_starts(self, num_steps: int) -> np.ndarray:
        """Global row index of every ``num_steps``-frame window inside one episode."""

        return clip_starts(self.ep_len, self.ep_offset, num_steps)


def clip_starts(ep_len: np.ndarray, ep_offset: np.ndarray, num_steps: int) -> np.ndarray:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    starts = [
        np.arange(offset, offset + length - num_steps + 1, dtype=np.int64)
        for length, offset in zip(ep_len.tolist(), ep_offset.tolist(), strict=True)
        if length >= num_steps
    ]
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def episode_content_digest(episode: EpisodeRecord) -> str:
    """Order-sensitive digest of an episode's frames, commands, and ids."""

    digest = hashlib.sha256()
    observations, actions = episode_sequence(episode)
    digest.update(episode.episode_id.encode())
    digest.update(str(episode.scenario_seed).encode())
    for observation in observations:
        digest.update(observation.sha256.encode())
        digest.update(f"{observation.sim_time_s:.9f}".encode())
    for action in actions:
        digest.update(f"{action.forward_mps:.9g},{action.yaw_rate_rps:.9g}".encode())
    return digest.hexdigest()


def _labels_by_frame(episode: EpisodeRecord) -> dict[str, StateLabels]:
    labels: dict[str, StateLabels] = {
        episode.reset_report.final_observation.observation_id: episode.reset_report.final_labels
    }
    for block in episode.blocks:
        transition = block.transition
        labels[transition.start_observation.observation_id] = transition.start_labels
        labels[transition.end_observation.observation_id] = transition.end_labels
    return labels


def build_training_cache(
    episodes: Iterable[EpisodeRecord],
    out_dir: str | Path,
    *,
    split: DatasetSplit,
    dataset_id: str,
    split_id: str,
    image_size: int = 224,
    block_duration_s: float = 0.5,
    spec: TargetSpec | None = None,
) -> TrainingCache:
    """Write one split's cache atomically; refuse mixed splits, sizes, or cameras."""

    records = tuple(episodes)
    if not records:
        raise CacheError("no episodes to cache")
    if not dataset_id or not split_id:
        raise CacheError("dataset_id and split_id are required for provenance")
    wrong = [e.episode_id for e in records if e.split != split]
    if wrong:
        raise CacheError(f"episodes {wrong[:5]} are not in split {split.value}")
    ids = [e.episode_id for e in records]
    if len(set(ids)) != len(ids):
        raise CacheError("duplicate episode ids")
    cameras = {e.camera_id for e in records}
    if len(cameras) != 1:
        raise CacheError(f"a cache must use one camera, found {sorted(cameras)}")
    spec = spec or TargetSpec(
        tuple(item.object_id for item in records[0].reset_report.final_labels.objects)
    )

    sequences = [episode_sequence(e) for e in records]
    total = sum(len(obs) for obs, _ in sequences)
    for episode, (observations, actions) in zip(records, sequences, strict=True):
        for observation in observations:
            if (observation.width_px, observation.height_px) != (image_size, image_size):
                raise CacheError(
                    f"{episode.episode_id}: frame {observation.observation_id} is "
                    f"{observation.width_px}x{observation.height_px}, expected "
                    f"{image_size}x{image_size}; resize in the simulator, never here"
                )
        for action in actions:
            if not math.isclose(action.duration_s, block_duration_s, abs_tol=1e-9):
                raise CacheError(f"{episode.episode_id}: command duration {action.duration_s}")
        if len(actions) != len(observations) - 1:
            raise CacheError(f"{episode.episode_id}: expected one command between frames")

    target = Path(out_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite cache {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        pixels = np.lib.format.open_memmap(
            staging / "pixels.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total, image_size, image_size, 3),
        )
        action = np.full((total, len(ACTION_FIELDS)), np.nan, dtype=np.float32)
        state = np.full((total, spec.dim), np.nan, dtype=np.float32)
        ep_len = np.zeros(len(records), dtype=np.int64)
        ep_offset = np.zeros(len(records), dtype=np.int64)
        row = 0
        digests: dict[str, str] = {}
        labeled = 0
        for index, (episode, (observations, actions)) in enumerate(
            zip(records, sequences, strict=True)
        ):
            labels = _labels_by_frame(episode)
            ep_offset[index] = row
            ep_len[index] = len(observations)
            for step, observation in enumerate(observations):
                pixels[row] = np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
                    image_size, image_size, 3
                )
                if step < len(actions):
                    action[row] = (actions[step].forward_mps, actions[step].yaw_rate_rps)
                label = labels.get(observation.observation_id)
                if label is not None:
                    state[row] = spec.encode(label)
                    labeled += 1
                row += 1
            digests[episode.episode_id] = episode_content_digest(episode)
        pixels.flush()
        del pixels
        np.save(staging / "action.npy", action)
        np.save(staging / "state.npy", state)
        np.save(staging / "ep_len.npy", ep_len)
        np.save(staging / "ep_offset.npy", ep_offset)
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_id": dataset_id,
            "split_id": split_id,
            "split": split.value,
            "camera_id": cameras.pop(),
            "image_size": image_size,
            "block_duration_s": block_duration_s,
            "frame_count": int(total),
            "labeled_frame_count": int(labeled),
            "episode_count": len(records),
            "scene_ids": sorted({e.scene_id for e in records}),
            "rehearsal_only": any(e.scene_id.startswith("fake") for e in records),
            "episode_ids": ids,
            "scenario_seeds": [e.scenario_seed for e in records],
            "episode_content_digests": digests,
            "action_fields": list(ACTION_FIELDS),
            "action_alignment": "action[i] is applied from frame i to frame i+1; NaN on the "
            "last frame of each episode",
            "state_names": list(spec.names),
            "state_note": "privileged labels for readout probes only; never a model input",
            "files": {name: _sha256(staging / name) for name in ARRAY_FILES},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_training_cache(target, verify=False)


def read_training_cache(root: str | Path, *, verify: bool = True) -> TrainingCache:
    path = Path(root)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise CacheError(f"{path} has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise CacheError(f"unsupported cache schema {manifest.get('schema_version')!r}")
    if verify:
        for name, expected in manifest["files"].items():
            if _sha256(path / name) != expected:
                raise CacheError(f"{path}: checksum mismatch for {name}")
    pixels = np.load(path / "pixels.npy", mmap_mode="r")
    cache = TrainingCache(
        root=path,
        manifest=manifest,
        pixels=pixels,
        action=np.load(path / "action.npy"),
        state=np.load(path / "state.npy"),
        ep_len=np.load(path / "ep_len.npy"),
        ep_offset=np.load(path / "ep_offset.npy"),
    )
    _validate_cache(cache)
    return cache


def _validate_cache(cache: TrainingCache) -> None:
    n = cache.frame_count
    size = cache.manifest["image_size"]
    if cache.pixels.shape != (n, size, size, 3) or cache.pixels.dtype != np.uint8:
        raise CacheError(f"pixels have shape {cache.pixels.shape} {cache.pixels.dtype}")
    if cache.action.shape != (n, len(ACTION_FIELDS)):
        raise CacheError("action rows do not match frames")
    if cache.state.shape[0] != n:
        raise CacheError("state rows do not match frames")
    if n != int(cache.manifest["frame_count"]) or int(cache.ep_len.sum()) != n:
        raise CacheError("episode lengths do not cover every frame")
    expected_offsets = np.concatenate([[0], np.cumsum(cache.ep_len)[:-1]])
    if not np.array_equal(cache.ep_offset, expected_offsets):
        raise CacheError("episode offsets are not contiguous")
    last_rows = cache.ep_offset + cache.ep_len - 1
    if not np.isnan(cache.action[last_rows]).all():
        raise CacheError("the last frame of every episode must have a NaN action")
    interior = np.ones(n, dtype=bool)
    interior[last_rows] = False
    if not np.isfinite(cache.action[interior]).all():
        raise CacheError("a non-final frame has a missing or non-finite action")


def action_statistics(cache: TrainingCache) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-field mean and sample std (ddof=1) over the finite rows of a TRAIN cache.

    Matches upstream ``get_column_normalizer`` (``torch.std`` is unbiased).
    """

    if cache.split != DatasetSplit.TRAIN.value:
        raise CacheError("normalization statistics must come from the train split only")
    rows = cache.action[np.isfinite(cache.action).all(axis=1)].astype(np.float64)
    if len(rows) < 2:
        raise CacheError("need at least two actions for statistics")
    std = rows.std(axis=0, ddof=1)
    if (std < 1e-8).any():
        raise CacheError(
            f"an action field has ~zero variance {std.tolist()}; the collection "
            "never varied it, so the model cannot learn its effect"
        )
    return tuple(rows.mean(axis=0).tolist()), tuple(std.tolist())


def state_statistics(cache: TrainingCache) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Train-only normalization for the optional supervised state auxiliary.

    Only completely aligned label rows participate.  Constant targets such as
    the no-fall indicator receive the same guarded scale as the frozen readout
    path instead of producing a divide-by-zero.  Validation labels never
    influence these statistics.
    """

    if cache.split != DatasetSplit.TRAIN.value:
        raise CacheError("state normalization statistics must come from the train split only")
    rows = cache.state[np.isfinite(cache.state).all(axis=1)].astype(np.float64)
    if len(rows) < 2:
        raise CacheError("need at least two aligned state rows for statistics")
    normalizer = Standardizer.fit(rows)
    return tuple(normalizer.mean.tolist()), tuple(normalizer.scale.tolist())


def check_disjoint(caches: Sequence[TrainingCache]) -> None:
    """Caches used together must share dataset/split ids and never share episodes or seeds."""

    if len({c.manifest["dataset_id"] for c in caches}) != 1:
        raise CacheError("caches come from different datasets")
    if len({c.manifest["split_id"] for c in caches}) != 1:
        raise CacheError("caches were built with different split manifests")
    if len({c.manifest["image_size"] for c in caches}) != 1:
        raise CacheError("caches use different image sizes")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for cache in caches:
        ids = set(cache.episode_ids)
        seeds = set(cache.manifest["scenario_seeds"])
        if ids & seen_ids or seeds & seen_seeds:
            raise CacheError("episodes or scenario seeds appear in more than one cache")
        seen_ids |= ids
        seen_seeds |= seeds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_synthetic_control(
    out_dir: str | Path,
    *,
    train_episodes: int = 120,
    val_episodes: int = 20,
    length: int = 12,
    square_px: int = 28,
    step_px: float = 40.0,
    seed: int = 0,
    image_size: int = 224,
) -> tuple[TrainingCache, TrainingCache]:
    """Known-answer positive control for the LeWM training path.

    A white square on black moves by exactly ``step_px * action`` each block,
    with a fresh random command every block, so the next frame is predictable
    from the command and not from the history alone.  A correctly wired model
    that trains long enough must show validation prediction loss well below
    both the copy-last and the shuffled-command references.  State columns hold
    the square's centre in units of 100 px (robot x, y) for readout checks.
    Caches are marked ``rehearsal_only``.
    """

    root = Path(out_dir)
    caches = []
    for split, episodes, split_seed in (
        (DatasetSplit.TRAIN, train_episodes, seed * 2 + 1),
        (DatasetSplit.VALIDATION, val_episodes, seed * 2 + 2),
    ):
        caches.append(
            _write_synthetic_split(
                root / f"cache-{split.value}",
                split=split,
                episodes=episodes,
                length=length,
                square_px=square_px,
                step_px=step_px,
                seed=split_seed,
                image_size=image_size,
            )
        )
    return caches[0], caches[1]


def _write_synthetic_split(
    target: Path,
    *,
    split: DatasetSplit,
    episodes: int,
    length: int,
    square_px: int,
    step_px: float,
    seed: int,
    image_size: int,
) -> TrainingCache:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite cache {target}")
    rng = np.random.default_rng(seed)
    total = episodes * length
    half = square_px // 2
    margin = half + 2
    spec = TargetSpec(())
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        pixels = np.lib.format.open_memmap(
            staging / "pixels.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total, image_size, image_size, 3),
        )
        action = np.full((total, 2), np.nan, dtype=np.float32)
        state = np.full((total, spec.dim), np.nan, dtype=np.float32)
        row = 0
        for _ in range(episodes):
            x, y = rng.uniform(margin, image_size - margin, size=2)
            for t in range(length):
                xi, yi = round(x), round(y)
                pixels[row] = 0
                pixels[row, yi - half : yi + half, xi - half : xi + half] = 255
                state[row] = (x / 100.0, y / 100.0, 0.0, 1.0, 0.0)
                if t < length - 1:
                    command = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                    action[row] = command
                    x = float(np.clip(x + step_px * command[0], margin, image_size - margin))
                    y = float(np.clip(y + step_px * command[1], margin, image_size - margin))
                row += 1
        pixels.flush()
        del pixels
        np.save(staging / "action.npy", action)
        np.save(staging / "state.npy", state)
        np.save(staging / "ep_len.npy", np.full(episodes, length, dtype=np.int64))
        np.save(staging / "ep_offset.npy", np.arange(episodes, dtype=np.int64) * length)
        ids = [f"synthetic-{split.value}-{i:04d}" for i in range(episodes)]
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_id": "synthetic-control",
            "split_id": "synthetic-control",
            "split": split.value,
            "camera_id": "synthetic",
            "image_size": image_size,
            "block_duration_s": 0.5,
            "frame_count": total,
            "labeled_frame_count": total,
            "episode_count": episodes,
            "episode_ids": ids,
            "scenario_seeds": [seed * 1_000_000 + i for i in range(episodes)],
            "episode_content_digests": {},
            "scene_ids": ["fake-synthetic-square"],
            "rehearsal_only": True,
            "action_fields": list(ACTION_FIELDS),
            "action_alignment": "action[i] is applied from frame i to frame i+1; NaN on the "
            "last frame of each episode",
            "state_names": list(spec.names),
            "state_note": "square centre in units of 100 px; positive control only",
            "synthetic": {"square_px": square_px, "step_px": step_px, "seed": seed},
            "files": {name: _sha256(staging / name) for name in ARRAY_FILES},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_training_cache(target, verify=False)
