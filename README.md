# Forepaw

Forepaw is an agentic robotics system for the Dimensional track. It connects a
dimOS skill interface to a closed-loop planner and a simulated Unitree Go2,
then records every prediction, decision, action, and safety response for replay.

The demo answers a practical question: can an agent inspect several possible
physical futures, choose a short safe action, observe what actually happened,
and replan without hiding the decision process?

## What I built

- **A reusable dimOS blueprint** registered as `go2-world-model.forepaw` with
  three MCP skills: `imagine`, `plan_to`, and `stop_motion`.
- **A receding-horizon planner** that evaluates exactly 64 six-block action
  candidates, executes only the first 0.5-second block, observes the result,
  and replans.
- **A true-Go2 simulation path** using the MjLab Go2 controller and MuJoCo
  scene, kept outside dimOS behind a versioned loopback controller service.
- **Single-owner motion control**: dimOS requests plans, but only the controller
  service can publish simulator actions.
- **Fail-closed safety behavior** for malformed requests, controller loss,
  surprise detection, and explicit stop requests.
- **Append-only evidence logs** that persist the selected plan before motion and
  keep planner-visible inputs separate from privileged simulator labels.
- **A browser replay UI** for push, detour, and surprise-stop runs, packaged as
  a static Vercel site for a reliable hackathon demo.

## How it works

```text
dimOS MCP
  -> ForepawSkills: imagine | plan_to | stop_motion
    -> loopback JSON controller client
      -> controller service (sole motion owner)
        -> 64-candidate planner
          -> true-Go2 MjLab simulation
            -> lock decision -> execute 0.5 s -> observe -> replan
```

Each planning cycle receives three 2 Hz RGB observations and the two commands
between them. Candidate actions are `[forward_velocity_mps, yaw_rate_rps]`;
lateral velocity is always zero. The planner scores six-block futures but sends
only the first block to the robot controller.

The telemetry viewer is deliberately downstream of control. It can replay or
follow the append-only log, but it cannot change a plan or command motion.

## Verified demo results

| Area | Retained result |
| --- | --- |
| dimOS integration | External blueprint and all three MCP skills discovered; valid and invalid request paths exercised |
| Closed-loop execution | Three goal-reaching dimOS-to-true-Go2 push runs; 45 planning cycles across the goal and restart checks |
| Planner contract | 64 candidates per cycle, six 0.5 s blocks per candidate, first block only executed, plan locked before motion |
| Reliability | Controller process restart verified; malformed `imagine` input failed closed; idle stop acknowledged |
| Planner latency | 2.661 ms p50, 3.015 ms p90, 8.233 ms p95 over 45 cycles with physics paused during planning |
| Go2 scene behavior | 5/5 movable-box trials moved at least 1.2292 m; 5/5 resistant-box trials stayed at or below 0.0448 m; zero falls |
| Data/timing packet | Three predeclared seeds completed 20 consecutive valid 0.5 s blocks with strict reload, split, timing, and checksum checks |
| Software checks | 201 passed, 6 skipped; Ruff, CLI, protocol, and focused dimOS checks passed |

The retained dimOS rehearsal is
[`artifacts/runs/20260920T090000Z-dimos-mjlab/`](artifacts/runs/20260920T090000Z-dimos-mjlab/).
The formal scene and timing packets are
[`artifacts/runs/20260920T073000Z-g0-formal-mjlab/`](artifacts/runs/20260920T073000Z-g0-formal-mjlab/)
and
[`artifacts/runs/20260920T073500Z-g1a-mjlab/`](artifacts/runs/20260920T073500Z-g1a-mjlab/).
Each published packet includes checksums or links to its machine-readable report.

## Run the demo

### Static evidence replay

Live demo:
[vercel-two-dun-11.vercel.app](https://vercel-two-dun-11.vercel.app)

The fastest demo path has no build step:

```bash
python3 -m http.server 8788 --directory deploy/vercel/public
```

Open [http://127.0.0.1:8788](http://127.0.0.1:8788). The replay contains a
dimOS-to-Go2 run plus direct push/detour and surprise-stop rehearsals. It is a
recorded evidence viewer, not a browser-hosted simulator.

### Software verification

```bash
uv sync --extra dev
make check
make dimos-smoke
```

The core scaffold uses Python 3.10.12+. If a cloud-managed folder evicts the
virtual environment, place it on a local volume:

```bash
UV_PROJECT_ENVIRONMENT=/path/on-local-disk/go2wm-venv uv sync --extra dev --extra sim
UV_PROJECT_ENVIRONMENT=/path/on-local-disk/go2wm-venv make check
```

### dimOS and true-Go2 path

Install this package into the pinned dimOS environment without replacing its
dependencies:

```bash
uv pip install --no-deps -e /absolute/path/to/hackmit
dimos list | grep go2-world-model.forepaw
```

Start the controller service in the MuJoCo/MjLab environment, then start the
blueprint and call its skills from the dimOS environment:

```bash
dimos run go2-world-model.forepaw
dimos mcp list-tools
dimos mcp call imagine --json-args '{"actions":[[0.4,0.0],[0.4,0.0]]}'
dimos mcp call plan_to --json-args '{"x":1.3,"y":0.35}' --timeout 120
dimos mcp call stop_motion --json-args '{}'
```

The exact controller command, pinned artifacts, platform notes, and retained
verification procedure are in
[`docs/DIMOS_DEPLOYMENT.md`](docs/DIMOS_DEPLOYMENT.md).

## Evidence and safety design

Forepaw makes its control loop auditable rather than showing only the final
robot trajectory:

1. Candidate predictions, costs, selection, model identity, and timestamp are
   written and fsynced before motion.
2. The applied command is recorded separately from the requested command.
3. The next observation is compared with the locked prediction.
4. A surprise threshold can command a stop; it does not silently adapt or
   rewrite the prior prediction.
5. Privileged poses, contacts, and mobility labels live under `ground_truth`
   telemetry and never enter the runtime planner interface.

See [`docs/UI_INTEGRATION.md`](docs/UI_INTEGRATION.md) for the log schema and
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) for the evidence rules.

## Scope

The deployed hackathon replay and retained dimOS rehearsal use an explicitly
labeled **privileged reference predictor** to verify integration, planning,
motion ownership, logging, restart behavior, and the user-facing workflow. The
learned LeWM experiments did not pass the frozen G3 predictive-control gate, so
the learned model is not connected to motion and no learned closed-loop result
is claimed.

Active stop preemption still needs verification on the target Linux deployment;
macOS single-threaded MuJoCo can acknowledge a stop only when the active plan
returns. The formal G0 technical checks passed, with the required second-owner
sign-off still pending. These are acceptance items, not blockers for the
recorded Dimensional-track MVP.

## Repository map

```text
src/go2wm/integration/  dimOS blueprint, skills, and controller service
src/go2wm/planning/     candidate library, scoring, and MPC selection
src/go2wm/runtime/      closed-loop execution and surprise-stop behavior
src/go2wm/sim/          simulator protocol and backends
src/go2wm/data/         collection and manifest validation
src/go2wm/model/        model/readout interfaces and bundle metadata
deploy/vercel/          public-safe static evidence replay
artifacts/runs/         retained reports, telemetry, media, and checksums
tests/                  executable contract and integration checks
docs/                   deployment, verification, decisions, and runbooks
```

The two-person ownership and handoff plan is in
[`docs/TWO_PERSON_PLAN.md`](docs/TWO_PERSON_PLAN.md), and experiment-boundary
decisions are recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Pitch assets

- [Final presentation](.codex-output/Forepaw_HackMIT_Dimensional_Pitch_FINAL_v2.pptx)
- [Three-minute script and demo runbook](docs/PITCH.md)
