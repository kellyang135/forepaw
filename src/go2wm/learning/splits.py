"""Deterministic scenario-seed splits, frozen before any model fitting.

The split of a seed depends only on ``(salt, seed)`` through SHA-256, so the
assignment is reproducible on any machine and never depends on which episodes
happen to exist yet.  The manifest is written once and then only read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from go2wm.contracts import DatasetSplit, EpisodeRecord

SPLIT_SCHEMA = "go2wm.splits.v1"


class SplitLeakError(ValueError):
    """Raised when an episode's split disagrees with the frozen manifest."""


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0 for value in values):
            raise ValueError("every split fraction must be positive")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    split_id: str
    salt: str
    fractions: SplitFractions
    assignments: Mapping[int, DatasetSplit]
    reserved_test_seeds: tuple[int, ...] = ()
    schema_version: str = SPLIT_SCHEMA
    notes: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def split_of(self, seed: int) -> DatasetSplit:
        try:
            return self.assignments[seed]
        except KeyError:
            raise SplitLeakError(
                f"scenario seed {seed} is not in frozen split manifest {self.split_id!r}"
            ) from None

    def seeds(self, split: DatasetSplit) -> tuple[int, ...]:
        return tuple(sorted(seed for seed, value in self.assignments.items() if value == split))

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "salt": self.salt,
            "fractions": {
                "train": self.fractions.train,
                "validation": self.fractions.validation,
                "test": self.fractions.test,
            },
            "assignments": {
                str(seed): split.value for seed, split in sorted(self.assignments.items())
            },
            "reserved_test_seeds": list(self.reserved_test_seeds),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def hash_split(seed: int, *, salt: str, fractions: SplitFractions) -> DatasetSplit:
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < fractions.train:
        return DatasetSplit.TRAIN
    if unit < fractions.train + fractions.validation:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def build_split_manifest(
    seeds: Iterable[int],
    *,
    salt: str,
    fractions: SplitFractions | None = None,
    reserved_test_seeds: Iterable[int] = (),
    notes: str = "",
) -> SplitManifest:
    """Assign seeds by hash; reserved seeds (final-evaluation scenarios) are forced to test."""

    fractions = fractions or SplitFractions()
    if not salt:
        raise ValueError("salt must be non-empty")
    reserved = tuple(sorted(set(reserved_test_seeds)))
    all_seeds = sorted(set(seeds) | set(reserved))
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in all_seeds):
        raise TypeError("scenario seeds must be integers")
    assignments = {
        seed: DatasetSplit.TEST
        if seed in reserved
        else hash_split(seed, salt=salt, fractions=fractions)
        for seed in all_seeds
    }
    body = json.dumps(
        {
            "salt": salt,
            "fractions": [fractions.train, fractions.validation, fractions.test],
            "assignments": {str(k): v.value for k, v in assignments.items()},
            "reserved": list(reserved),
        },
        sort_keys=True,
    )
    split_id = "splits-" + hashlib.sha256(body.encode()).hexdigest()[:12]
    return SplitManifest(
        split_id=split_id,
        salt=salt,
        fractions=fractions,
        assignments=assignments,
        reserved_test_seeds=reserved,
        notes=notes,
    )


def write_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen split manifest {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json(), encoding="utf-8")


def read_split_manifest(path: str | Path) -> SplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SPLIT_SCHEMA:
        raise ValueError(f"unsupported split schema {payload.get('schema_version')!r}")
    fractions = SplitFractions(**payload["fractions"])
    manifest = build_split_manifest(
        (int(seed) for seed in payload["assignments"]),
        salt=payload["salt"],
        fractions=fractions,
        reserved_test_seeds=payload.get("reserved_test_seeds", ()),
        notes=payload.get("notes", ""),
    )
    stored = {int(k): DatasetSplit(v) for k, v in payload["assignments"].items()}
    if stored != dict(manifest.assignments) or manifest.split_id != payload["split_id"]:
        raise SplitLeakError(
            "split manifest on disk does not match its salt/fractions; it was edited by hand"
        )
    return manifest


@dataclass(frozen=True, slots=True)
class LeakReport:
    episode_count: int
    counts_by_split: Mapping[str, int]
    seeds_by_split: Mapping[str, int]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def check_leakage(episodes: Iterable[EpisodeRecord], manifest: SplitManifest) -> LeakReport:
    """Every episode must use the frozen split of its seed; ids and seeds never straddle splits."""

    problems: list[str] = []
    counts: dict[str, int] = {split.value: 0 for split in DatasetSplit}
    seeds: dict[str, set[int]] = {split.value: set() for split in DatasetSplit}
    seen_ids: dict[str, DatasetSplit] = {}
    episode_count = 0
    for episode in episodes:
        episode_count += 1
        counts[episode.split.value] += 1
        seeds[episode.split.value].add(episode.scenario_seed)
        if episode.episode_id in seen_ids:
            problems.append(f"duplicate episode id {episode.episode_id!r}")
        seen_ids[episode.episode_id] = episode.split
        expected = manifest.assignments.get(episode.scenario_seed)
        if expected is None:
            problems.append(
                f"episode {episode.episode_id!r} uses seed {episode.scenario_seed} "
                "that is not in the frozen split manifest"
            )
        elif expected != episode.split:
            problems.append(
                f"episode {episode.episode_id!r} is labeled {episode.split.value} but seed "
                f"{episode.scenario_seed} is frozen as {expected.value}"
            )
    for left in DatasetSplit:
        for right in DatasetSplit:
            if left.value < right.value:
                shared = seeds[left.value] & seeds[right.value]
                if shared:
                    problems.append(
                        f"seeds {sorted(shared)} appear in both {left.value} and {right.value}"
                    )
    return LeakReport(
        episode_count=episode_count,
        counts_by_split=counts,
        seeds_by_split={key: len(value) for key, value in seeds.items()},
        problems=tuple(problems),
    )
