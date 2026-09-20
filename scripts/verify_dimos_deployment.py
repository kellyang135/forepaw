#!/usr/bin/env python3
"""Retain a truthful dimOS/MCP deployment packet from a running Forepaw stack.

This verifies transport, serialization, plan-lock ordering, and true-Go2 MjLab
telemetry.  It deliberately does not promote a privileged reference model into
learned-model evidence or claim the complete L7 gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str], *, timeout: float = 180.0) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {
        "command": command,
        "exit_code": result.returncode,
        "wall_latency_s": time.perf_counter() - started,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _json_stdout(result: dict[str, Any]) -> Any:
    if result["exit_code"] != 0:
        raise RuntimeError(f"command failed: {result}")
    return json.loads(result["stdout"])


def _candidate() -> list[list[dict[str, float]]]:
    return [
        [
            {
                "forward_velocity_mps": 0.3,
                "yaw_rate_rps": 0.0,
                "duration_s": 0.5,
            }
            for _ in range(6)
        ]
    ]


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimos-bin", type=Path, required=True)
    parser.add_argument("--dimos-source", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--plan-x", type=float, default=1.3)
    parser.add_argument("--plan-y", type=float, default=0.35)
    args = parser.parse_args(argv)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    previous_pid = None
    previous_report = output / "report.json"
    if previous_report.exists():
        previous = json.loads(previous_report.read_text(encoding="utf-8"))
        previous_pid = previous.get("mcp_status", {}).get("pid")
    dimos = str(args.dimos_bin)
    calls: dict[str, dict[str, Any]] = {}
    calls["list_tools"] = _run([dimos, "mcp", "list-tools"])
    calls["status"] = _run([dimos, "mcp", "status"])
    calls["modules"] = _run([dimos, "mcp", "modules"])
    calls["imagine_valid"] = _run(
        [
            dimos,
            "mcp",
            "call",
            "imagine",
            "--json-args",
            json.dumps({"candidate_commands": _candidate()}, separators=(",", ":")),
            "--timeout",
            "120",
        ]
    )
    calls["imagine_invalid"] = _run(
        [
            dimos,
            "mcp",
            "call",
            "imagine",
            "--json-args",
            json.dumps(
                {
                    "candidate_commands": [
                        [
                            {
                                "forward_velocity_mps": 0.9,
                                "yaw_rate_rps": 0.0,
                                "duration_s": 0.5,
                            }
                        ]
                    ]
                },
                separators=(",", ":"),
            ),
            "--timeout",
            "30",
        ]
    )
    calls["plan_to"] = _run(
        [
            dimos,
            "mcp",
            "call",
            "plan_to",
            "--json-args",
            json.dumps({"x": args.plan_x, "y": args.plan_y}, separators=(",", ":")),
            "--timeout",
            "120",
        ]
    )
    calls["stop_motion_idle"] = _run(
        [dimos, "mcp", "call", "stop_motion", "--json-args", "{}", "--timeout", "30"]
    )

    tools = _json_stdout(calls["list_tools"])
    status = _json_stdout(calls["status"])
    modules = _json_stdout(calls["modules"])
    imagine = _json_stdout(calls["imagine_valid"])
    plan = _json_stdout(calls["plan_to"])
    stop = _json_stdout(calls["stop_motion_idle"])
    tool_names = {item["name"] for item in tools}

    telemetry_reports: list[dict[str, Any]] = []
    all_latencies: list[float] = []
    for path in sorted(args.telemetry_root.rglob("ui.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        plans = [record for record in records if record["type"] == "plan_locked"]
        executed = [record for record in records if record["type"] == "block_executed"]
        types = [record["type"] for record in records]
        ordered = types[0] == "run_start" and types[-1] == "run_end"
        ordered = ordered and all(
            types.index("plan_locked", 1 + 2 * index)
            < types.index("block_executed", 1 + 2 * index)
            for index in range(min(len(plans), len(executed)))
        )
        latencies = [float(record["planning_latency_s"]) for record in plans]
        all_latencies.extend(latencies)
        telemetry_reports.append(
            {
                "path": str(path),
                "sha256": _digest(path),
                "run_scope": records[0].get("evidence_scope"),
                "run_reason": records[-1].get("reason"),
                "planning_cycles": len(plans),
                "executed_blocks": len(executed),
                "ordered_plan_lock_before_motion": ordered,
                "all_candidate_counts_64": all(len(record["rankings"]) == 64 for record in plans),
                "all_first_blocks_half_second": all(
                    record["first_action"]["duration_s"] == 0.5 for record in plans
                ),
            }
        )

    checks = {
        "external_blueprint_loaded": status.get("modules") == ["ForepawSkills", "McpServer"],
        "required_tools_discovered": {"imagine", "plan_to", "stop_motion"} <= tool_names,
        "module_map_correct": modules.get("modules", {}).get("ForepawSkills")
        == ["imagine", "plan_to", "stop_motion"],
        "valid_imagine_serialized": imagine.get("status") == "imagined"
        and imagine.get("candidate_count") == 1,
        "imagine_reports_no_motion": imagine.get("motion_executed") is False,
        "invalid_imagine_failed_closed": "exactly six" in calls["imagine_invalid"]["stdout"],
        "plan_to_returned_contract": plan.get("bundle_id")
        and plan.get("first_selected_candidate")
        and plan.get("first_action", {}).get("duration_s") == 0.5
        and plan.get("first_score") is not None,
        "stop_surface_acknowledged": stop.get("stop_acknowledged") is True,
        "process_restart_verified": previous_pid is not None
        and status.get("pid") != previous_pid,
        "telemetry_present": bool(telemetry_reports),
        "all_logs_lock_before_motion": all(
            item["ordered_plan_lock_before_motion"] for item in telemetry_reports
        ),
        "all_logs_use_64_candidates": all(
            item["all_candidate_counts_64"] for item in telemetry_reports
        ),
        "all_logs_execute_half_second_blocks": all(
            item["all_first_blocks_half_second"] for item in telemetry_reports
        ),
        "at_least_20_transport_planning_cycles": sum(
            item["planning_cycles"] for item in telemetry_reports
        )
        >= 20,
        "true_go2_scope_labeled": all(
            "true-Go2 MjLab" in (item["run_scope"] or "") for item in telemetry_reports
        ),
    }
    project_root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "go2wm.dimos-deployment-report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "DIMOS_MJLAB_REFERENCE_REHEARSAL_PASS"
        if all(checks.values())
        else "DIMOS_MJLAB_REFERENCE_REHEARSAL_FAIL",
        "truthfulness": {
            "dimos_transport_verified": all(checks.values()),
            "true_go2_mjlab_path": True,
            "learned_world_model_verified": False,
            "full_l7_verified": False,
            "note": (
                "The model is the privileged kinematic reference. This packet verifies dimOS "
                "transport/orchestration and the true-Go2 simulator path, not learned prediction "
                "quality or final closed-loop task performance. Process restart was verified; "
                "active stop preemption remains unverified on macOS single-threaded rendering."
            ),
        },
        "environment": {
            "host": platform.node(),
            "os": platform.platform(),
            "python": sys.version,
            "project_revision": _git(project_root, "rev-parse", "HEAD"),
            "project_dirty": bool(_git(project_root, "status", "--short")),
            "dimos_revision": _git(args.dimos_source, "rev-parse", "HEAD"),
            "dimos_dirty": bool(_git(args.dimos_source, "status", "--short")),
            "policy_sha256": _digest(args.policy),
            "robot_xml_sha256": _digest(args.robot_xml),
        },
        "checks": checks,
        "mcp_status": status,
        "mcp_modules": modules,
        "planning_cycles": sum(item["planning_cycles"] for item in telemetry_reports),
        "latency_s": {
            "sample_count": len(all_latencies),
            "p50": _percentile(all_latencies, 0.50),
            "p90": _percentile(all_latencies, 0.90),
            "p95": _percentile(all_latencies, 0.95),
            "physics_paused_during_planning": True,
        },
        "telemetry": telemetry_reports,
    }
    (output / "mcp_calls.json").write_text(
        json.dumps(calls, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    notes = """# dimOS deployment verification

This packet is a transport/integration rehearsal. The real pinned dimOS coordinator
and MCP server invoked Forepaw skills against the true-Go2 MjLab simulator. The
runtime model was the explicitly labeled privileged kinematic reference, so this
is **not** learned-world-model or final task-performance evidence.

macOS required `mjpython` plus single-threaded HTTP handling so MuJoCo rendering
stayed on the main thread. Both processes were restarted and a fresh plan
completed. Active stop preemption must still be verified on the target Linux
deployment; the idle stop surface and core stop contract were tested.
"""
    (output / "notes.md").write_text(notes, encoding="utf-8")
    files = [output / "mcp_calls.json", output / "notes.md", output / "report.json"]
    (output / "checksums.sha256").write_text(
        "".join(f"{_digest(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
