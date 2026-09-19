#!/usr/bin/env python3
"""Probe the official Go2 MJCF without claiming a locomotion controller.

On macOS, use ``mjpython`` when ``--render`` is requested.  The JSON report
separates model compilation, uncontrolled physics stepping, and RGB rendering
so a partial success cannot be mistaken for Gate G0 locomotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_png(path: Path, pixels: Any) -> None:
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError(f"expected RGB pixels, got shape {pixels.shape}")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("steps and render dimensions must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import mujoco
    import mujoco_menagerie as menagerie
    import numpy as np

    robot = menagerie.get("unitree_go2")
    cache = menagerie.Cache(args.cache_dir)
    xml_path = robot.xml(cache=cache)
    started = time.perf_counter()
    model = robot.model(cache=cache)
    compile_wall_s = time.perf_counter() - started

    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_base_z = float(data.qpos[2]) if model.nq > 2 else float("nan")
    min_base_z = initial_base_z
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
        if model.nq > 2:
            min_base_z = min(min_base_z, float(data.qpos[2]))

    render: dict[str, Any] = {
        "requested": args.render,
        "status": "NOT_REQUESTED",
    }
    artifacts: list[Path] = []
    exit_code = 0
    if args.render:
        try:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [0.0, 0.0, 0.25]
            camera.distance = 2.3
            camera.azimuth = 90
            camera.elevation = -80
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
            renderer.update_scene(data, camera)
            pixels = renderer.render()
            renderer.close()
            image_path = args.output_dir / "overhead.png"
            _write_png(image_path, pixels)
            artifacts.append(image_path)
            unique_colors = int(np.unique(pixels.reshape(-1, 3), axis=0).shape[0])
            render = {
                "requested": True,
                "status": "PASS" if unique_colors > 1 else "FAIL",
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                "min_channel": int(pixels.min()),
                "max_channel": int(pixels.max()),
                "unique_rgb_colors": unique_colors,
                "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "artifact": image_path.name,
            }
            if unique_colors <= 1:
                exit_code = 2
        except Exception as error:  # evidence must survive a platform rendering failure
            render = {
                "requested": True,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            exit_code = 2

    finite_state = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    if not finite_state:
        status = "FAIL_NONFINITE_PHYSICS"
    elif render["status"] == "PASS":
        status = "PASS_MODEL_AND_RENDER"
    elif args.render:
        status = "PARTIAL_RENDER_FAILED"
    else:
        status = "PASS_MODEL_ONLY"
    report = {
        "schema_version": "go2wm.go2-model-probe.v1",
        "recorded_at_utc": _utc_now(),
        "status": status,
        "scope": "official Go2 model compile/step/render probe; no gait controller",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "executable": sys.executable,
            "mujoco_version": mujoco.__version__,
        },
        "source": {
            "name": robot.name,
            "display_name": robot.display_name,
            "license": robot.license,
            "menagerie_commit": menagerie.commit(),
            "model_tree_oid": robot.oid,
            "archive_sha256": robot.sha256,
            "xml_path": str(xml_path),
            "xml_sha256": _sha256(xml_path),
        },
        "model": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nbody": model.nbody,
            "ngeom": model.ngeom,
            "ncam": model.ncam,
            "nkey": model.nkey,
            "physics_dt_s": model.opt.timestep,
            "compile_wall_s": compile_wall_s,
        },
        "uncontrolled_step_probe": {
            "steps": args.steps,
            "simulated_duration_s": float(data.time),
            "finite_state": finite_state,
            "initial_base_z_m": initial_base_z,
            "minimum_base_z_m": min_base_z,
            "final_base_z_m": float(data.qpos[2]) if model.nq > 2 else float("nan"),
            "qpos_l2_change": float(np.linalg.norm(data.qpos - initial_qpos)),
            "final_contact_count": int(data.ncon),
            "controller_attached": False,
            "gait_verified": False,
        },
        "render": render,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifacts.append(report_path)
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in sorted(artifacts))
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
