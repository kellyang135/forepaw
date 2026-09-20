"""Strict acceptance checks for the direct-MjLab G1A timing packet.

G1A proves the simulator/data contract through the production MjLab adapter.
dimOS transport is intentionally not part of this gate; it remains pending for
the G5/L7 deployment gate and may not be substituted with dimOS's Go1 alias.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

from go2wm.contracts import ActionCommand, DatasetSplit, EpisodeRecord
from go2wm.learning.dataset import load_dataset
from go2wm.learning.splits import check_leakage, read_split_manifest
from go2wm.sim.controller_handoff import verify_controller_handoff

G1_PACKET_SCHEMA = "go2wm.g1a-packet.v1"
G1_REPORT_SCHEMA = "go2wm.g1a-verification.v1"
G1_SCOPE = "direct_mjlab_data_contract; dimos_pending_g5_l7"
G1_SPLIT_SALT = "go2wm-real-v1-20260920"
G1_SPLIT_ID = "splits-7ae472b8826f"
G1_ROLE_SEEDS = {"alignment": 3, "push": 5, "resist": 18}
G1_ROLE_MODES = {"alignment": "free", "push": "push", "resist": "resist"}
G1_RESERVED_TEST_SEEDS = tuple(range(377, 401))
G1_ALL_SEEDS = frozenset(range(1, 401))
EXPECTED_CONTROLLER_DT_S = 0.02
EXPECTED_PHYSICS_DT_S = 0.005
EXPECTED_BLOCK_DURATION_S = 0.5
EXPECTED_PHYSICS_SAMPLES = 100


def _alignment_schedule() -> tuple[ActionCommand, ...]:
    values = (
        [(0.0, 0.0)] * 2
        + [(0.4, 0.0)] * 4
        + [(0.0, 0.0)] * 2
        + [(0.0, 0.6)] * 4
        + [(0.0, 0.0)] * 2
        + [(0.0, -0.6)] * 2
        + [(0.0, 0.0)] * 2
        + [(0.2, 0.0)]
        + [(0.0, 0.0)]
    )
    return tuple(ActionCommand(forward, yaw, EXPECTED_BLOCK_DURATION_S) for forward, yaw in values)


ALIGNMENT_SCHEDULE = _alignment_schedule()


def alignment_schedule_payload() -> list[dict[str, float]]:
    """Return the JSON form of the frozen alignment schedule."""

    return [
        {
            "forward_velocity_mps": action.forward_velocity_mps,
            "yaw_rate_rps": action.yaw_rate_rps,
            "duration_s": action.duration_s,
        }
        for action in ALIGNMENT_SCHEDULE
    ]


class G1VerificationError(ValueError):
    """Raised when a packet does not satisfy every direct-MjLab G1A invariant."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_g1_packet(packet_root: str | Path) -> dict[str, Any]:
    """Strictly reload and verify a complete direct-MjLab G1A packet."""

    root = Path(packet_root)
    metadata = _read_json(root / "packet.json")
    _equal(metadata.get("schema_version"), G1_PACKET_SCHEMA, "packet schema")
    _equal(metadata.get("scope"), G1_SCOPE, "packet scope")
    _equal(metadata.get("dimos_status"), "pending_g5_l7", "dimOS status")
    _equal(metadata.get("split_id"), G1_SPLIT_ID, "split id")
    _equal(metadata.get("seeds"), G1_ROLE_SEEDS, "role seeds")
    _equal(
        metadata.get("alignment_schedule"),
        alignment_schedule_payload(),
        "alignment schedule metadata",
    )
    _equal(metadata.get("backend"), "mujoco-mjlab-direct", "packet backend")

    handoff = _mapping(metadata.get("controller_handoff"), "controller_handoff")
    handoff_root = Path(_string(handoff.get("path"), "controller_handoff.path"))
    summary = verify_controller_handoff(handoff_root)
    _equal(handoff.get("controller_id"), summary.controller_id, "controller id")
    _equal(summary.physics_dt_s, EXPECTED_PHYSICS_DT_S, "handoff physics dt")
    _equal(summary.controller_dt_s, EXPECTED_CONTROLLER_DT_S, "handoff controller dt")
    referenced = {
        "manifest_sha256": handoff_root / "manifest.json",
        "policy_sha256": summary.policy_path,
        "robot_xml_sha256": handoff_root / "robot" / "go2.xml",
        "deploy_yaml_sha256": handoff_root / "config" / "deploy.yaml",
    }
    for key, path in referenced.items():
        if not path.is_file():
            raise G1VerificationError(f"missing referenced controller file {path}")
        _equal(handoff.get(key), sha256_file(path), key)

    split_manifest = read_split_manifest(root / "splits.json")
    verify_master_split(split_manifest)

    dataset = load_dataset(root / "data", strict=True)
    leakage = check_leakage(dataset.episodes, split_manifest)
    if not leakage.ok:
        raise G1VerificationError("split leakage: " + "; ".join(leakage.problems))
    if len(dataset.episodes) != len(G1_ROLE_SEEDS):
        raise G1VerificationError(
            f"expected {len(G1_ROLE_SEEDS)} episodes, found {len(dataset.episodes)}"
        )

    by_role: dict[str, EpisodeRecord] = {}
    for episode in dataset.episodes:
        role = episode.metadata.get("g1_role", "")
        if role not in G1_ROLE_SEEDS:
            raise G1VerificationError(f"episode {episode.episode_id} has unknown G1 role {role!r}")
        if role in by_role:
            raise G1VerificationError(f"duplicate G1 role {role!r}")
        by_role[role] = episode

    packet_policy_sha = _string(handoff.get("policy_sha256"), "policy_sha256")
    expected_adapter_controller = _string(
        metadata.get("adapter_controller_id"), "adapter_controller_id"
    )
    expected_scene = _string(metadata.get("scene_id"), "scene_id")
    expected_camera = _string(metadata.get("camera_id"), "camera_id")
    episode_reports: dict[str, Any] = {}
    transient_contacts: list[dict[str, Any]] = []
    for role in G1_ROLE_SEEDS:
        try:
            episode = by_role[role]
        except KeyError:
            raise G1VerificationError(f"missing G1 role {role!r}") from None
        _equal(episode.scenario_seed, G1_ROLE_SEEDS[role], f"{role} seed")
        _equal(episode.split, DatasetSplit.TRAIN, f"{role} split")
        _equal(episode.scene_id, expected_scene, f"{role} scene")
        _equal(episode.camera_id, expected_camera, f"{role} camera")
        _equal(episode.metadata.get("g1_scope"), G1_SCOPE, f"{role} G1 scope")
        _equal(episode.metadata.get("g1_role"), role, f"{role} role")
        _equal(episode.metadata.get("backend"), "mujoco-mjlab-direct", f"{role} backend")
        _equal(episode.metadata.get("dimos_status"), "pending_g5_l7", f"{role} dimOS status")
        _equal(
            episode.metadata.get("collection_mode"),
            G1_ROLE_MODES[role],
            f"{role} collection mode",
        )
        _equal(
            episode.metadata.get("controller_id"),
            summary.controller_id,
            f"{role} controller id",
        )
        _equal(
            episode.metadata.get("adapter_controller_id"),
            expected_adapter_controller,
            f"{role} adapter controller id",
        )
        _equal(
            episode.metadata.get("policy_sha256"),
            packet_policy_sha,
            f"{role} policy digest",
        )
        _equal(
            episode.metadata.get("collection_policy"),
            (
                "g1a-alignment-schedule-v1"
                if role == "alignment"
                else "scripted-collection-policy-v1"
            ),
            f"{role} collection policy",
        )
        _verify_episode_blocks(episode, role, transient_contacts)
        episode_reports[role] = {
            "episode_id": episode.episode_id,
            "seed": episode.scenario_seed,
            "blocks": len(episode.blocks),
            "contact_blocks": sum(
                bool(block.transition.events.contacted_object_ids) for block in episode.blocks
            ),
            "transient_contact_blocks": [
                item["block_index"]
                for item in transient_contacts
                if item["episode_id"] == episode.episode_id
            ],
            "termination": episode.blocks[-1].termination_reason,
        }

    if not transient_contacts:
        raise G1VerificationError(
            "packet lacks a transient contact: aggregate contact true but endpoint contact empty"
        )
    if episode_reports["push"]["contact_blocks"] == 0:
        raise G1VerificationError("push episode contains no contact")
    if episode_reports["resist"]["contact_blocks"] == 0:
        raise G1VerificationError("resist episode contains no contact")

    montage = _mapping(metadata.get("montage"), "montage")
    _equal(montage.get("path"), "montage.html", "montage path")
    _equal(montage.get("block_count"), 10, "montage block count")
    montage_path = root / "montage.html"
    if not montage_path.is_file():
        raise G1VerificationError("packet lacks its self-contained 10-block montage")
    _equal(montage.get("sha256"), sha256_file(montage_path), "montage checksum")
    montage_text = montage_path.read_text(encoding="utf-8")
    if montage_text.count('data-block-row="true"') != 10:
        raise G1VerificationError("montage does not contain exactly ten block rows")
    if montage_text.count("data:image/png;base64,") != 20:
        raise G1VerificationError("montage does not embed twenty start/end PNG images")

    return {
        "schema_version": G1_REPORT_SCHEMA,
        "passed": True,
        "packet_id": _string(metadata.get("packet_id"), "packet_id"),
        "scope": G1_SCOPE,
        "dimos_status": "pending_g5_l7",
        "controller_id": summary.controller_id,
        "policy_sha256": packet_policy_sha,
        "scene_id": expected_scene,
        "camera_id": expected_camera,
        "split_id": split_manifest.split_id,
        "episode_reports": episode_reports,
        "transient_contact_examples": transient_contacts,
        "checks": {
            "strict_loader_and_episode_checksums": True,
            "split_leakage": True,
            "three_predeclared_train_seeds": True,
            "twenty_consecutive_valid_blocks_each": True,
            "exact_alignment_schedule": True,
            "half_second_blocks": True,
            "one_hundred_5ms_samples_per_block": True,
            "no_fall_or_out_of_bounds": True,
            "terminal_fixed_length_only_at_19": True,
            "requested_equals_applied": True,
            "stable_controller_scene_camera_policy": True,
            "transient_contact_preserved": True,
            "self_contained_ten_block_montage": True,
        },
    }


def verify_master_split(split_manifest: Any) -> None:
    """Require the already-frozen 1..400 split, including held-out seeds."""

    _equal(split_manifest.split_id, G1_SPLIT_ID, "frozen split id")
    _equal(split_manifest.salt, G1_SPLIT_SALT, "frozen split salt")
    _equal(set(split_manifest.assignments), G1_ALL_SEEDS, "frozen split seed universe")
    _equal(
        split_manifest.reserved_test_seeds,
        G1_RESERVED_TEST_SEEDS,
        "reserved test seeds",
    )
    for role, seed in G1_ROLE_SEEDS.items():
        if split_manifest.split_of(seed) is not DatasetSplit.TRAIN:
            raise G1VerificationError(f"{role} seed {seed} is not frozen as train")


def build_g1_montage(packet_root: str | Path) -> Path:
    """Write a deterministic, self-contained ten-block visual audit artifact."""

    from go2wm.telemetry import encode_png

    root = Path(packet_root)
    output = root / "montage.html"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    dataset = load_dataset(root / "data", strict=True)
    by_role = {episode.metadata.get("g1_role", ""): episode for episode in dataset.episodes}
    if set(by_role) != set(G1_ROLE_SEEDS):
        raise G1VerificationError("cannot build montage without exactly the three G1 roles")

    chosen: list[tuple[str, EpisodeRecord, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add(role: str, index: int) -> None:
        key = (role, index)
        if key not in seen:
            seen.add(key)
            chosen.append((role, by_role[role], by_role[role].blocks[index]))

    for index in (0, 2, 8, 14):
        add("alignment", index)
    for role in ("push", "resist"):
        contact = next(
            (
                block
                for block in by_role[role].blocks
                if block.transition.events.contacted_object_ids
            ),
            None,
        )
        if contact is None:
            raise G1VerificationError(f"cannot build montage: {role} has no contact block")
        add(role, contact.block_index)
    for role in ("push", "resist", "alignment"):
        for block in by_role[role].blocks:
            transition = block.transition
            if (
                transition.events.contacted_object_ids
                and not transition.physics_samples[-1].contacts
            ):
                add(role, block.block_index)
                break
    for role, index in (
        ("push", 0),
        ("resist", 0),
        ("push", 10),
        ("resist", 10),
        ("alignment", 18),
        ("push", 19),
        ("resist", 19),
    ):
        if len(chosen) == 10:
            break
        add(role, index)
    if len(chosen) != 10:
        raise G1VerificationError(f"could only select {len(chosen)} montage blocks")

    rows = []
    for role, episode, block in chosen:
        transition = block.transition
        requested = transition.requested_action
        applied = transition.action
        event = transition.events
        rows.append(
            '<tr data-block-row="true"><td><img alt="start" src="data:image/png;base64,'
            + encode_png(transition.start_observation)
            + '"></td><td><img alt="end" src="data:image/png;base64,'
            + encode_png(transition.end_observation)
            + '"></td><td><b>'
            + html.escape(role)
            + "</b> &middot; "
            + html.escape(episode.episode_id)
            + f" block {block.block_index}<br>requested: {requested.forward_velocity_mps:.3f} m/s, "
            + f"{requested.yaw_rate_rps:+.3f} rad/s<br>applied: "
            + f"{applied.forward_velocity_mps:.3f} m/s, {applied.yaw_rate_rps:+.3f} rad/s<br>"
            + f"t={transition.start_observation.sim_time_s:.3f}&rarr;"
            + f"{transition.end_observation.sim_time_s:.3f}s; "
            + f"samples={len(transition.physics_samples)}; dt={transition.physics_dt_s:.3f}s<br>"
            + "contacts="
            + html.escape(", ".join(event.contacted_object_ids) or "none")
            + f"; fell={event.fell}; out_of_bounds={event.out_of_bounds}</td></tr>"
        )
    document = (
        "<!doctype html><meta charset=utf-8><title>G1A direct-MjLab montage</title>"
        "<style>body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#161616}"
        "table{border-collapse:collapse}td{vertical-align:top;padding:8px;"
        "border-bottom:1px solid #ccc}"
        "img{width:224px;image-rendering:pixelated;border:1px solid #aaa}</style>"
        "<h1>G1A direct-MjLab 10-block montage</h1>"
        "<p>Each row embeds the retained start/end frames and their aligned command and events.</p>"
        "<table><thead><tr><th>Start</th><th>End</th><th>Evidence</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(document)
    return output


def write_g1_verification(packet_root: str | Path, report: dict[str, Any]) -> tuple[Path, Path]:
    """Write the verification report and a checksum inventory exactly once."""

    root = Path(packet_root)
    report_path = root / "verification.json"
    checksums_path = root / "checksums.sha256"
    if report_path.exists() or checksums_path.exists():
        raise FileExistsError("refusing to overwrite G1A verification evidence")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path != checksums_path
        )
        rows = [
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths
        ]
        with checksums_path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    except BaseException:
        report_path.unlink(missing_ok=True)
        raise
    return report_path, checksums_path


def _verify_episode_blocks(
    episode: EpisodeRecord,
    role: str,
    transient_contacts: list[dict[str, Any]],
) -> None:
    if len(episode.blocks) != 20:
        raise G1VerificationError(
            f"{role} episode has {len(episode.blocks)} blocks; expected exactly 20"
        )
    if tuple(block.block_index for block in episode.blocks) != tuple(range(20)):
        raise G1VerificationError(f"{role} block indexes are not 0..19")
    for index, block in enumerate(episode.blocks):
        transition = block.transition
        if not block.valid:
            raise G1VerificationError(f"{role} block {index} is marked invalid")
        expected_termination = "fixed_length" if index == 19 else None
        _equal(block.termination_reason, expected_termination, f"{role} block {index} termination")
        if transition.events.fell or transition.events.out_of_bounds:
            raise G1VerificationError(f"{role} block {index} has a fall/out-of-bounds event")
        if not math.isclose(
            transition.action.duration_s, EXPECTED_BLOCK_DURATION_S, abs_tol=1e-12
        ):
            raise G1VerificationError(f"{role} block {index} is not 0.5 seconds")
        if not math.isclose(transition.physics_dt_s, EXPECTED_PHYSICS_DT_S, abs_tol=1e-12):
            raise G1VerificationError(f"{role} block {index} physics dt is not 0.005")
        if len(transition.physics_samples) != EXPECTED_PHYSICS_SAMPLES:
            raise G1VerificationError(f"{role} block {index} does not have 100 samples")
        if transition.requested_action != transition.action:
            raise G1VerificationError(f"{role} block {index} requested/applied actions differ")
        if role == "alignment":
            expected = ALIGNMENT_SCHEDULE[index]
            if transition.requested_action != expected or transition.action != expected:
                raise G1VerificationError(f"alignment block {index} differs from frozen schedule")
        if transition.events.contacted_object_ids and not transition.physics_samples[-1].contacts:
            transient_contacts.append(
                {
                    "episode_id": episode.episode_id,
                    "role": role,
                    "block_index": index,
                    "contacted_object_ids": list(transition.events.contacted_object_ids),
                    "contact_sample_count": transition.events.contact_sample_count,
                    "last_contact_time_s": transition.events.last_contact_time_s,
                    "block_end_time_s": transition.end_observation.sim_time_s,
                }
            )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise G1VerificationError(f"cannot read {path}: {error}") from error
    return _mapping(value, str(path))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise G1VerificationError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise G1VerificationError(f"{name} must be a non-empty string")
    return value


def _equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise G1VerificationError(f"{name}: {actual!r} != {expected!r}")
