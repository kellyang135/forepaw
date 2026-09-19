"""Coverage diagnostics so data gaps are found before training, not after."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

from go2wm.contracts import BlockTransition, EpisodeRecord

STALL_PROGRESS_M = 0.02
DISPLACED_OBJECT_M = 0.02


def classify_block(transition: BlockTransition) -> str:
    """Name the interaction type of one block from privileged labels (analysis only).

    ``free``: no contact.  ``push``: contact and a contacted object moved.
    ``resisted``: contact, commanded forward motion, no object moved and the
    robot barely progressed.  ``brief_contact``: any other contact.
    """

    events = transition.events
    if not events.contacted_object_ids:
        return "free"
    start = {item.object_id: item.pose for item in transition.start_labels.objects}
    end = {item.object_id: item.pose for item in transition.end_labels.objects}
    moved = any(
        math.hypot(end[oid].x_m - start[oid].x_m, end[oid].y_m - start[oid].y_m)
        > DISPLACED_OBJECT_M
        for oid in events.contacted_object_ids
        if oid in start and oid in end
    )
    if moved:
        return "push"
    robot_start = transition.start_labels.robot_pose
    robot_end = transition.end_labels.robot_pose
    progress = math.hypot(robot_end.x_m - robot_start.x_m, robot_end.y_m - robot_start.y_m)
    if transition.action.forward_velocity_mps > 0.05 and progress < STALL_PROGRESS_M:
        return "resisted"
    return "brief_contact"


def action_family(forward_mps: float, yaw_rate_rps: float) -> str:
    if abs(forward_mps) < 0.05 and abs(yaw_rate_rps) < 0.05:
        return "stop"
    if abs(forward_mps) < 0.05:
        return "turn_in_place"
    if abs(yaw_rate_rps) < 0.15:
        return "straight"
    return "arc_left" if yaw_rate_rps > 0 else "arc_right"


def speed_bin(forward_mps: float) -> str:
    edges = (0.05, 0.2, 0.4)
    labels = ("0-0.05", "0.05-0.2", "0.2-0.4", "0.4+")
    for edge, label in zip(edges, labels, strict=False):
        if forward_mps < edge:
            return label
    return labels[-1]


def coverage_report(episodes: Iterable[EpisodeRecord]) -> dict[str, Any]:
    """JSON-safe counts by split, interaction type, action family, speed, object, termination."""

    per_split: dict[str, dict[str, Counter[str]]] = {}
    totals = Counter[str]()
    command_transitions = Counter[str]()
    for episode in episodes:
        split = episode.split.value
        bucket = per_split.setdefault(
            split,
            {
                "interaction": Counter(),
                "action_family": Counter(),
                "speed": Counter(),
                "contacted_object": Counter(),
                "termination": Counter(),
                "scene": Counter(),
            },
        )
        totals[f"{split}.episodes"] += 1
        bucket["scene"][episode.scene_id] += 1
        bucket["termination"][episode.metadata.get("termination_reason", "unrecorded")] += 1
        previous_family: str | None = None
        for block in episode.blocks:
            transition = block.transition
            totals[f"{split}.blocks"] += 1
            bucket["interaction"][classify_block(transition)] += 1
            family = action_family(
                transition.action.forward_velocity_mps, transition.action.yaw_rate_rps
            )
            bucket["action_family"][family] += 1
            bucket["speed"][speed_bin(transition.action.forward_velocity_mps)] += 1
            for object_id in transition.events.contacted_object_ids:
                appearance = next(
                    (
                        item.appearance_class
                        for item in transition.start_labels.objects
                        if item.object_id == object_id
                    ),
                    "unknown",
                )
                bucket["contacted_object"][f"{object_id}:{appearance}"] += 1
            if transition.events.fell:
                totals[f"{split}.fall_blocks"] += 1
            if previous_family is not None and previous_family != family:
                command_transitions[f"{previous_family}->{family}"] += 1
            previous_family = family

    report: dict[str, Any] = {"totals": dict(sorted(totals.items())), "splits": {}}
    warnings: list[str] = []
    for split, bucket in sorted(per_split.items()):
        blocks = totals[f"{split}.blocks"]
        interaction = bucket["interaction"]
        contact_blocks = blocks - interaction["free"]
        fraction = contact_blocks / blocks if blocks else 0.0
        report["splits"][split] = {
            name: dict(sorted(counter.items())) for name, counter in bucket.items()
        }
        report["splits"][split]["interaction_fraction"] = round(fraction, 4)
        if blocks and not 0.20 <= fraction <= 0.30:
            warnings.append(
                f"{split}: interaction fraction {fraction:.2f} is outside the 0.20-0.30 target"
            )
        if blocks and interaction["push"] == 0:
            warnings.append(f"{split}: no push blocks")
        if blocks and interaction["resisted"] == 0:
            warnings.append(f"{split}: no resisted-push blocks")
        if blocks and interaction["resisted"] > 2 * max(interaction["push"], 1):
            warnings.append(
                f"{split}: resisted blocks dominate pushes; check for repeated stall frames"
            )
    report["command_transitions"] = dict(sorted(command_transitions.items()))
    report["warnings"] = warnings
    return report
