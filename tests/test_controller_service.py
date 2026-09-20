from __future__ import annotations

import json
import threading

from go2wm.integration import ControllerRuntimeConfig, RunnerControllerBackend
from go2wm.integration.controller_service import HttpControllerClient, make_server


def _candidate(forward: float = 0.2, yaw: float = 0.0) -> list[dict]:
    return [
        {
            "forward_velocity_mps": forward,
            "yaw_rate_rps": yaw,
            "duration_s": 0.5,
        }
        for _ in range(6)
    ]


def test_runner_backend_imagines_without_nonzero_motion(tmp_path) -> None:
    backend = RunnerControllerBackend(
        ControllerRuntimeConfig(output_root=tmp_path, max_blocks=2)
    )
    response = backend.imagine([_candidate(), _candidate(yaw=0.4)])
    assert response["status"] == "imagined"
    assert response["motion_executed"] is False
    assert response["candidate_count"] == 2
    assert len(response["candidates"][0]["states"]) == 6


def test_runner_backend_plan_response_has_l7_fields_and_telemetry(tmp_path) -> None:
    backend = RunnerControllerBackend(
        ControllerRuntimeConfig(output_root=tmp_path, max_blocks=2)
    )
    response = backend.plan_to(0.8, 0.35)
    assert response["bundle_id"] == "reference-kinematic-v1"
    assert response["blocks_executed"] >= 1
    assert response["first_selected_candidate"]
    assert response["first_action"]["duration_s"] == 0.5
    assert set(response["first_score"]) == {"total", "parts"}
    from pathlib import Path

    telemetry = Path(response["telemetry_path"])
    records = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()]
    assert records[0]["type"] == "run_start"
    assert records[-1]["type"] == "run_end"
    assert any(record["type"] == "plan_locked" for record in records)


def test_loopback_http_client_round_trip(tmp_path) -> None:
    backend = RunnerControllerBackend(
        ControllerRuntimeConfig(output_root=tmp_path, max_blocks=1)
    )
    server = make_server(backend, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpControllerClient(f"http://127.0.0.1:{server.server_port}", timeout_s=5.0)
    try:
        assert client.health()["status"] == "ready"
        assert client.imagine([_candidate()])["candidate_count"] == 1
        assert client.stop("test")["status"] == "idle"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
