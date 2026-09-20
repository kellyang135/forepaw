"""G1 visual check: an HTML montage of ten blocks from a collected wave.

    python scripts/g1_montage.py --data ~/go2wm-data/wave1 --out ~/go2wm-data/g1_montage.html

Each row shows the start and end frames of one 0.5 s block next to the requested
and applied command, the boundary times, the robot and box pose deltas, and the
event flags. The eye test: the motion between the two frames must match the
command and the pose delta in the same row. If the frames look one block ahead
of or behind the numbers, the data has an off-by-one shift and G1 stops.
"""

from __future__ import annotations

import argparse
import html
import math
import random
from pathlib import Path

from go2wm.learning.dataset import load_episode
from go2wm.learning.diagnostics import action_family, classify_block
from go2wm.telemetry import encode_png

WANTED = (
    ("push", 2),
    ("resisted", 2),
    ("brief_contact", 2),
    ("free", 2),
    ("stop", 1),
    ("turn_in_place", 1),
)


def pose_delta(start, end) -> str:
    dyaw = math.degrees(
        math.atan2(math.sin(end.yaw_rad - start.yaw_rad), math.cos(end.yaw_rad - start.yaw_rad))
    )
    return f"dx {end.x_m - start.x_m:+.3f} m, dy {end.y_m - start.y_m:+.3f} m, dyaw {dyaw:+.1f} deg"


def box_deltas(transition) -> str:
    start = {o.object_id: o for o in transition.start_labels.objects}
    rows = []
    for obj in transition.end_labels.objects:
        before = start[obj.object_id].pose
        moved = math.dist((before.x_m, before.y_m), (obj.pose.x_m, obj.pose.y_m))
        rows.append(f"{obj.object_id} ({obj.appearance_class}): moved {moved:.3f} m")
    return "<br>".join(rows)


def pick(episodes: list[Path], seed: int):
    rng = random.Random(seed)
    rng.shuffle(episodes)
    need = dict(WANTED)
    chosen = []
    for directory in episodes:
        if not any(need.values()):
            break
        record = load_episode(directory)
        for block in record.blocks:
            t = block.transition
            kinds = [
                classify_block(t),
                action_family(t.action.forward_velocity_mps, t.action.yaw_rate_rps),
            ]
            for kind in kinds:
                if need.get(kind, 0) > 0:
                    need[kind] -= 1
                    chosen.append((kind, record, block))
                    break
            else:
                continue
            break  # at most one block per episode keeps the sample spread out
    return chosen


def row(kind: str, record, block) -> str:
    t = block.transition
    req, app = t.requested_action, t.action
    span = t.end_observation.sim_time_s - t.start_observation.sim_time_s
    ev = t.events
    flags = [
        f"contact: {', '.join(ev.contacted_object_ids) or 'none'}"
        f" ({ev.contact_sample_count} samples)",
        f"fell: {ev.fell}",
        f"out of bounds: {ev.out_of_bounds}",
    ]
    info = [
        f"<b>{html.escape(kind)}</b> &middot; {html.escape(record.episode_id)}"
        f" block {block.block_index}",
        f"requested: fwd {req.forward_velocity_mps:.2f} m/s, yaw {req.yaw_rate_rps:+.2f} rad/s",
        f"applied: fwd {app.forward_velocity_mps:.2f} m/s, yaw {app.yaw_rate_rps:+.2f} rad/s",
        f"t {t.start_observation.sim_time_s:.3f} &rarr; {t.end_observation.sim_time_s:.3f} s "
        f"(span {span:.3f} s, {len(t.physics_samples)} physics samples, dt {t.physics_dt_s:g})",
        "robot: " + pose_delta(t.start_labels.robot_pose, t.end_labels.robot_pose),
        box_deltas(t),
        " &middot; ".join(flags),
    ]
    img = '<figure><img src="data:image/png;base64,{}"><figcaption>{}</figcaption></figure>'
    return (
        "<tr><td>"
        + img.format(encode_png(t.start_observation), "start")
        + "</td><td>"
        + img.format(encode_png(t.end_observation), "end")
        + "</td><td>"
        + "<br>".join(info)
        + "</td></tr>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    episodes = sorted(p for p in args.data.iterdir() if (p / "episode.json").exists())
    chosen = pick(episodes, args.seed)
    rows = "\n".join(row(*item) for item in chosen)
    args.out.write_text(
        "<!doctype html><meta charset=utf-8><title>G1 montage</title><style>"
        "body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:24px;color:#1a1a1a}"
        "td{vertical-align:top;padding:8px;border-bottom:1px solid #ddd}"
        "img{width:224px;image-rendering:pixelated;border:1px solid #ccc}"
        "figure{margin:0}figcaption{font-size:12px;color:#666}</style>"
        f"<h1>G1 montage</h1><p>{len(chosen)} blocks from {html.escape(str(args.data))}. "
        "Check that each frame pair moves the way its numbers say.</p>"
        f"<table>{rows}</table>",
        encoding="utf-8",
    )
    print(f"{len(chosen)} blocks -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
