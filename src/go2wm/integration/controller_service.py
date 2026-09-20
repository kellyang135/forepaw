"""Local JSON controller service used by the dimOS Forepaw blueprint.

Run the service in the simulator/model environment, then run dimOS in its own
environment.  This keeps the version boundary from D-016 explicit while all
motion remains owned by one ``RunnerControllerBackend`` process.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .controller import ControllerRuntimeConfig, RunnerControllerBackend


class ControllerServiceError(RuntimeError):
    """A controller service request failed or returned an invalid response."""


class HttpControllerClient:
    """Small dependency-free client safe to construct inside a dimOS worker."""

    def __init__(self, base_url: str = "http://127.0.0.1:8787", *, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def imagine(self, candidate_commands: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self._request("POST", "/imagine", {"candidate_commands": candidate_commands})

    def plan_to(self, x_m: float, y_m: float) -> dict[str, Any]:
        return self._request("POST", "/plan_to", {"x_m": x_m, "y_m": y_m})

    def stop(self, reason: str) -> dict[str, Any]:
        return self._request("POST", "/stop", {"reason": reason})

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ControllerServiceError(
                f"controller service returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise ControllerServiceError(f"controller service unavailable: {error}") from error
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ControllerServiceError("controller service returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise ControllerServiceError("controller service response must be an object")
        return decoded


def serve(
    backend: RunnerControllerBackend,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    threaded: bool = True,
) -> None:
    """Serve until interrupted.  Binding defaults to loopback only."""

    server = make_server(backend, host=host, port=port, threaded=threaded)
    print(
        json.dumps(
            {
                "status": "ready",
                "url": f"http://{host}:{server.server_port}",
                "controller": backend.health(),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def make_server(
    backend: RunnerControllerBackend,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    threaded: bool = True,
) -> HTTPServer:
    """Construct the loopback server without starting its request loop."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(HTTPStatus.OK, backend.health())
                return
            self._json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})

        def do_POST(self) -> None:
            try:
                payload = self._payload()
                if self.path == "/imagine":
                    result = backend.imagine(payload["candidate_commands"])
                elif self.path == "/plan_to":
                    result = backend.plan_to(float(payload["x_m"]), float(payload["y_m"]))
                elif self.path == "/stop":
                    result = backend.stop(str(payload.get("reason", "skill_requested")))
                else:
                    self._json(
                        HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"}
                    )
                    return
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "status": "error",
                        "error": str(error),
                        "motion_executed": False,
                    },
                )
                return
            self._json(HTTPStatus.OK, result)

        def _payload(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if length <= 0 or length > 2_000_000:
                raise ValueError("request body must contain at most 2 MB of JSON")
            raw = self.rfile.read(length)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError("request JSON must be an object")
            return value

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server_type = ThreadingHTTPServer if threaded else HTTPServer
    return server_type((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m go2wm.integration.controller_service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--sim", choices=("fake", "mujoco"), default="fake")
    parser.add_argument("--controller", choices=("mjlab", "rl_sar"), default="mjlab")
    parser.add_argument("--model", choices=("reference", "bundle"), default="reference")
    parser.add_argument("--scenario", choices=("push", "detour", "anomaly"), default="push")
    parser.add_argument("--output-root", default="runs/dimos")
    parser.add_argument("--bundle")
    parser.add_argument("--policy")
    parser.add_argument("--robot-xml")
    parser.add_argument("--deploy-yaml")
    parser.add_argument("--menagerie-cache", default="/tmp/go2wm-menagerie-cache")
    parser.add_argument("--lewm-repo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-blocks", type=int, default=30)
    parser.add_argument("--goal-radius", type=float, default=0.25)
    parser.add_argument("--surprise-threshold", type=float)
    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help="run handlers on the main thread (required for macOS MuJoCo rendering)",
    )
    args = parser.parse_args(argv)
    config = ControllerRuntimeConfig(
        simulator=args.sim,
        controller=args.controller,
        model=args.model,
        scenario=args.scenario,
        output_root=Path(args.output_root),
        bundle_path=None if args.bundle is None else Path(args.bundle),
        policy_path=None if args.policy is None else Path(args.policy),
        robot_xml_path=None if args.robot_xml is None else Path(args.robot_xml),
        deploy_yaml_path=None if args.deploy_yaml is None else Path(args.deploy_yaml),
        menagerie_cache=Path(args.menagerie_cache),
        lewm_repo=None if args.lewm_repo is None else Path(args.lewm_repo),
        device=args.device,
        max_blocks=args.max_blocks,
        goal_radius_m=args.goal_radius,
        surprise_threshold=args.surprise_threshold,
    )
    serve(
        RunnerControllerBackend(config),
        host=args.host,
        port=args.port,
        threaded=not args.single_threaded,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
