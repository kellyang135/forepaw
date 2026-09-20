"""Record closed-loop telemetry and serve the browser viewer.

    python -m go2wm.ui record --out runs/ui-<id> [--scenario push|detour|anomaly] [--bundle PATH]
    python -m go2wm.ui record --sim mujoco --controller mjlab --policy P --robot-xml X ...
    python -m go2wm.ui serve [--port 8765]

``record`` without ``--bundle`` first builds the fake-data linear baseline bundle
(the same code path as ``go2wm.learning smoke``), so the whole path runs today.
Fake-backend runs are software evidence only, and the log says so.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import http.server
import json
import math
import sys
import webbrowser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VIEWER = ROOT / "ui" / "viewer"


def _fake_world(image_size: int):
    from go2wm.contracts import CameraConfig
    from go2wm.sim import DeterministicFakeSimulator, FakeSimulatorConfig

    return DeterministicFakeSimulator(
        FakeSimulatorConfig(camera=CameraConfig("overhead", image_size, image_size))
    )


def _mujoco_world(args: argparse.Namespace):
    """The Go2 adapter with the chosen 50 Hz gait (draft adapter, docs/SIM_ADAPTER_DRAFT.md)."""

    from go2wm.sim.locomotion import (
        MJLAB_GO2_FLAT,
        RL_SAR_ROBOT_LAB_GO2,
        OnnxPolicy,
        check_mjlab_deploy_yaml,
        check_mjlab_onnx_metadata,
        load_rl_sar_policy,
    )
    from go2wm.sim.mujoco_go2 import MujocoGo2Simulator
    from go2wm.sim.task_scene import menagerie_scene_xml

    if args.policy is None:
        raise SystemExit("--sim mujoco needs --policy")
    if args.controller == "mjlab":
        if args.robot_xml is None:
            raise SystemExit("--controller mjlab needs --robot-xml (MjLab go2.xml with assets/)")
        check_mjlab_onnx_metadata(Path(args.policy))
        if args.deploy_yaml:
            check_mjlab_deploy_yaml(Path(args.deploy_yaml))
        spec = MJLAB_GO2_FLAT
        policy = OnnxPolicy(Path(args.policy), spec.observation_size)
        robot_xml = Path(args.robot_xml)
    else:
        spec = RL_SAR_ROBOT_LAB_GO2
        policy = load_rl_sar_policy(Path(args.policy))
        robot_xml = menagerie_scene_xml(Path(args.menagerie_cache))
    return MujocoGo2Simulator(spec, policy, robot_xml)


# Scenarios per backend: the fake arena is 3 m with 0.32 m objects; the MuJoCo
# arena is 4 m with 0.40 m boxes and a 0.7 m robot.  Object ids must stay
# "light" and "resistant" (the reference encoder reads them by id).
_SCENARIOS = {
    "fake": {
        "push": ((-1.0, -0.1, 0.0), (-0.35, -0.05), (0.45, -0.85), (0.9, 0.35, 0.2)),
        "detour": ((-1.0, 0.0, 0.0), (0.45, -0.9), (-0.1, 0.0), (0.9, 0.0, 0.2)),
    },
    "mujoco": {
        "push": ((-1.4, -0.1, 0.0), (-0.45, -0.05), (0.6, -1.1), (1.3, 0.35, 0.25)),
        "detour": ((-1.4, 0.0, 0.0), (0.6, -1.2), (0.0, 0.0), (1.3, 0.0, 0.25)),
    },
}


def _scenario(name: str, sim: str = "fake"):
    from go2wm.contracts import Goal2D, ObjectState, Pose2D, ResetRequest

    layout = "push" if name == "anomaly" else name
    robot, light, resistant, goal = _SCENARIOS[sim][layout]
    reset = ResetRequest(
        episode_id=f"ui-{name}" if sim == "fake" else f"ui-{sim}-{name}",
        scenario_seed=9001,
        robot_pose=Pose2D(*robot),
        objects=(
            ObjectState("light", "blue", Pose2D(*light), True),
            ObjectState("resistant", "red", Pose2D(*resistant), False),
        ),
    )
    return reset, Goal2D(*goal)


def _anomaly(after_block: int, heavy_kg: float = 20.0):
    """Make the light box immovable without telling the model.

    MuJoCo: the blue box silently takes the resistant box's mass.  Fake backend:
    the box's movable flag flips.
    """

    def perturb(block: int, simulator: Any) -> None:
        if block != after_block:
            return
        if hasattr(simulator, "set_object_mass"):
            simulator.set_object_mass("light", heavy_kg)
            return
        if not hasattr(simulator, "_objects"):
            return
        simulator._objects = [
            dataclasses.replace(item, movable=False) if item.object_id == "light" else item
            for item in simulator._objects
        ]

    return perturb


def record(args: argparse.Namespace) -> int:
    from go2wm.learning.bundle_io import load_bundle
    from go2wm.learning.pipeline import run_fake_smoke
    from go2wm.planning import RecedingHorizonPlanner, RolloutScorer
    from go2wm.runtime import SurpriseStopGuard
    from go2wm.runtime.loop import ClosedLoopRunner, LoopConfig
    from go2wm.telemetry import JsonlTelemetrySink

    out = Path(args.out)
    mujoco = args.sim == "mujoco"
    if mujoco and args.model == "baseline" and not args.bundle:
        raise SystemExit("--sim mujoco needs --model reference or a real --bundle")
    simulator = _mujoco_world(args) if mujoco else _fake_world(args.image_size)
    world = f"MuJoCo Go2 ({simulator.controller_id}, draft adapter)" if mujoco else "fake simulator"
    if args.model == "reference" and not args.bundle:
        from go2wm.ui.reference import LABEL, reference_bundle

        threshold = args.surprise_threshold or (0.10 if mujoco else 0.05)
        contact = 0.55 if mujoco else 0.31
        bundle, encoder, readout, calibration = reference_bundle(
            simulator, threshold=threshold, contact_m=contact
        )
        planner = RecedingHorizonPlanner(bundle)
        model_label = LABEL
        scope = (
            f"UI rehearsal only: {world} and a privileged kinematic reference, not a learned model"
        )
    else:
        if args.bundle:
            bundle_path = Path(args.bundle)
            model_label = f"bundle {bundle_path.name}"
        else:
            report = run_fake_smoke(out / "model", episode_count=args.episodes)
            bundle_path = Path(report["bundle_path"])
            model_label = report["model"]
        loaded = load_bundle(bundle_path)
        planner = RecedingHorizonPlanner(
            loaded.model_bundle,
            candidates=loaded.candidates,
            scorer=RolloutScorer(loaded.scoring),
            safety_policy=loaded.safety,
        )
        calibration = loaded.surprise
        encoder = loaded.model_bundle.backend.encoder
        readout = loaded.readout
        scope = (
            f"{world} with bundle {bundle_path.name}"
            if mujoco
            else "software contracts only: fake simulator, fake-data model"
        )
    stops: list[str] = []
    guard = SurpriseStopGuard(calibration, stop_callback=stops.append)
    reset, goal = _scenario(args.scenario, args.sim)
    config = LoopConfig(
        max_blocks=args.max_blocks,
        run_id=out.name,
        evidence_scope=scope,
        fallback_level=(
            "F3 rehearsal (MuJoCo Go2, reference model)"
            if mujoco and args.model == "reference" and not args.bundle
            else "F3 rehearsal (fake backend)"
            if not mujoco
            else "unassessed (MuJoCo Go2, published bundle)"
        ),
        model_label=model_label,
        render_hints=(
            {"robot_scale": 1.0, "box_size_m": 0.40, "robot_body_m": 0.7}
            if mujoco
            else {"robot_scale": 0.45, "box_size_m": 0.28, "robot_body_m": 0.3}
        ),
    )
    runner = ClosedLoopRunner(
        simulator,
        planner,
        guard,
        encode=encoder.encode,
        locate=lambda latent: readout.decode(tuple(latent), step=1).robot.xy,
        sink=JsonlTelemetrySink(out / "ui.jsonl"),
        config=config,
    )
    perturb = _anomaly(args.anomaly_block) if args.scenario == "anomaly" else None
    try:
        summary = runner.run(reset, goal, perturb=perturb)
    finally:
        runner.sink.close()
        if hasattr(simulator, "close"):
            simulator.close()
    p50 = summary.planning_latency_p50_s
    print(
        json.dumps(
            {
                "log": str(out / "ui.jsonl"),
                "reason": summary.reason,
                "blocks": len(summary.blocks),
                "planning_latency_p50_s": None if p50 is None or not math.isfinite(p50) else p50,
                "stop_reasons": stops,
            },
            indent=2,
        )
    )
    return 0


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/ui/viewer/")
            self.end_headers()
            return
        if self.path.startswith("/api/runs"):
            logs = sorted(
                (p for p in (ROOT / "runs").glob("*/ui.jsonl")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            body = json.dumps([str(p.relative_to(ROOT)) for p in logs]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        allowed = ("/ui/", "/runs/")
        if not self.path.startswith(allowed):
            self.send_error(404)
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(args: argparse.Namespace) -> int:
    if not (VIEWER / "index.html").exists():
        print(f"viewer not found at {VIEWER}", file=sys.stderr)
        return 1
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    url = f"http://127.0.0.1:{args.port}/ui/viewer/"
    print(f"serving {url}  (Ctrl-C to stop)")
    if args.open:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m go2wm.ui")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="run the closed loop and write runs/<id>/ui.jsonl")
    rec.add_argument("--out", required=True)
    rec.add_argument("--model", choices=("baseline", "reference"), default="baseline")
    rec.add_argument(
        "--bundle", help="published bundle directory; default builds the fake baseline"
    )
    rec.add_argument("--scenario", choices=("push", "detour", "anomaly"), default="push")
    rec.add_argument("--sim", choices=("fake", "mujoco"), default="fake")
    rec.add_argument("--controller", choices=("mjlab", "rl_sar"), default="mjlab")
    rec.add_argument("--policy", help="--sim mujoco: policy.onnx (mjlab) or policy.pt (rl_sar)")
    rec.add_argument("--robot-xml", help="--controller mjlab: MjLab go2.xml with its assets/")
    rec.add_argument("--deploy-yaml", help="--controller mjlab: deploy.yaml to check")
    rec.add_argument("--menagerie-cache", default="/tmp/go2wm-menagerie-cache")
    rec.add_argument("--surprise-threshold", type=float, default=None)
    rec.add_argument("--anomaly-block", type=int, default=2)
    rec.add_argument("--max-blocks", type=int, default=30)
    rec.add_argument("--episodes", type=int, default=60)
    rec.add_argument("--image-size", type=int, default=32)
    srv = sub.add_parser("serve", help="serve the viewer and run logs on localhost")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    return record(args) if args.command == "record" else serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
