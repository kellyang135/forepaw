# Forepaw through dimOS

Status: **transport rehearsal verified; full L7 and learned-model deployment remain open**.

## Architecture

```text
dimOS MCP server
  -> ForepawSkills external blueprint
    -> loopback JSON controller client
      -> RunnerControllerBackend (sole motion owner)
        -> ClosedLoopRunner
          -> true-Go2 MjLab simulator
```

The split is intentional. It lets dimOS run in its own environment while the
controller service uses the pinned MuJoCo/ONNX or LeWM environment. Only the
controller service owns the simulator and publishes motion. The dimOS module
never issues a simulator command directly.

## Install

Use the pinned dimOS revision from `configs/external_sources.toml`. In the dimOS
environment, install Forepaw editable without replacing dimOS dependencies:

```bash
uv pip install --no-deps -e /absolute/path/to/hackmit
```

The installed distribution registers the external blueprint as:

```bash
dimos list | grep go2-world-model.forepaw
```

## Start the controller service

For Linux, run the service normally. Replace `C` with the retained controller
artifact directory:

```bash
C=artifacts/runs/20260920T034139Z-go2-controller-training
python -m go2wm.integration.controller_service \
  --sim mujoco --model reference --controller mjlab \
  --policy "$C/policy/policy.onnx" \
  --robot-xml "$C/source/go2.xml" \
  --deploy-yaml "$C/config/deploy.yaml" \
  --scenario push --max-blocks 20 \
  --output-root artifacts/runs/<run_id>
```

On macOS, MuJoCo rendering must stay on the main thread. Use `mjpython` and add
`--single-threaded`:

```bash
mjpython -m go2wm.integration.controller_service ... --single-threaded
```

macOS single-threaded mode cannot accept a concurrent HTTP stop call while a
plan is running. It is therefore suitable for transport rehearsal, not the
active-stop portion of L7. Verify preemption on the target Linux deployment.

`--model reference` is a privileged integration rehearsal and must always be
labeled that way. Replace it with `--model bundle --bundle /path/to/bundle`
only after a published bundle passes its own compatibility and reference replay.

## Start dimOS and call Forepaw

In the dimOS environment:

```bash
dimos run go2-world-model.forepaw
dimos mcp list-tools
dimos mcp call plan_to --json-args '{"x":1.3,"y":0.35}' --timeout 120
dimos mcp call stop_motion --json-args '{}'
```

`plan_to` runs the existing receding-horizon controller: each cycle evaluates
64 six-block candidates, locks the predictions and scores, executes only the
first 0.5-second block, observes, and replans. `stop_motion` is used instead of
`stop` because dimOS reserves `Module.stop` for process lifecycle shutdown.

## Retain verification evidence

With both processes running:

```bash
python scripts/verify_dimos_deployment.py \
  --dimos-bin /absolute/path/to/dimos \
  --dimos-source /absolute/path/to/pinned/dimos \
  --telemetry-root artifacts/runs/<run_id> \
  --output-dir artifacts/runs/<run_id>/verification \
  --policy "$C/policy/policy.onnx" \
  --robot-xml "$C/source/go2.xml"
```

The retained 2026-09-20 rehearsal verified blueprint discovery, MCP tool
discovery, valid and invalid `imagine` serialization, the `plan_to` response
contract, 45 true-Go2 planning cycles, process restart, 64 candidates per
cycle, lock-before-motion ordering, 0.5-second first blocks, and an idle stop
acknowledgement. Planner-only p50/p90/p95 latency was 2.661/3.015/8.233 ms with physics paused
during planning. These measurements do not include CLI startup and are not
learned-model latency.
