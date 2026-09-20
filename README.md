# Go2 World Model

Predictive control for a simulated Unitree Go2: use a compact visual world model
to compare push, detour, adjustment, and stop futures, execute one 0.5-second
block, observe the result, and replan.

> **Current status:** Gate G0 remains blocked pending checkpoint and ONNX
> simulator playback plus the required motion trials. CUDA training/export at
> the pinned upstream revision completed successfully; the final checkpoint,
> self-contained ONNX export, configs, XML, and logs are retained in
> [`artifacts/runs/20260920T034139Z-go2-controller-training/`](artifacts/runs/20260920T034139Z-go2-controller-training/).
> This proves the training/export path, not gait quality, G0, task physics, or
> dimOS deployment.

Latest full-tree local software verification (2026-09-20): **161 tests passed,
17 skipped**; Ruff and all CLI/protocol smoke checks passed. The skips include
unavailable optional dependencies, headless OpenGL, and real-controller paths.
These figures are software-contract evidence, not a passing Go2 simulator gate.

## The two-person split

- **Person A — simulator, data, and deployment:** Go2/MuJoCo scene, action
  timing, camera, reset behavior, collection, dataset integrity, dimOS command
  ownership, and the live demo path.
- **Person B — learning, planning, and evaluation:** LeWM adaptation, training,
  readouts, rollout metrics, candidate scoring, surprise calibration,
  baselines, and frozen evaluation.
- **Shared integration windows:** hours 2, 6, 10, 12, 18, and 22. Neither
  person changes a shared contract without recording the decision and getting
  the other person's acknowledgement.

The detailed ownership and handoff plan is in
[`docs/TWO_PERSON_PLAN.md`](docs/TWO_PERSON_PLAN.md). Start every work session
with [`docs/RUNBOOK.md`](docs/RUNBOOK.md), and do not call a milestone complete
until its evidence is recorded according to
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

The retained Person A findings, exact upstream revisions, passed sub-checks,
blockers, and recovery sequence are in
[`docs/PERSON_A_EXECUTION.md`](docs/PERSON_A_EXECUTION.md).
The exact Linux/NVIDIA controller-training, playback, handoff, and integration
procedure is in
[`docs/GO2_CONTROLLER_RECOVERY.md`](docs/GO2_CONTROLLER_RECOVERY.md).
The completed CUDA training/export result and its remaining acceptance work are
summarized in
[`artifacts/runs/20260920T034139Z-go2-controller-training/notes.md`](artifacts/runs/20260920T034139Z-go2-controller-training/notes.md).

## Quick start

The core scaffold has no runtime dependencies beyond Python 3.10.12+. Development
uses pytest and Ruff.

```bash
uv sync --extra dev
make check
```

If the repository is in a cloud-managed folder that evicts `.venv` files, keep
the environment on a local volume and use the same path for setup and checks:

```bash
UV_PROJECT_ENVIRONMENT=/path/on-local-disk/go2wm-venv uv sync --extra dev --extra sim
UV_PROJECT_ENVIRONMENT=/path/on-local-disk/go2wm-venv make check
```

For the pinned true-Go2 model/render probe:

```bash
uv sync --extra dev --extra sim
.venv/bin/mjpython scripts/verify_go2_model.py \
  --cache-dir /path/to/menagerie-cache \
  --output-dir artifacts/runs/<run_id> \
  --render
```

To audit a checkout of the pinned controller-training source:

```bash
PYTHONPATH=src python scripts/audit_go2_controller_source.py \
  --checkout /path/to/unitree_rl_mjlab \
  --output-dir artifacts/runs/<run_id>
```

On the intended Linux/NVIDIA training machine, run the read-only host gate
before installing or training:

```bash
PYTHONPATH=src python scripts/check_go2_training_host.py
```

Run a deterministic, non-robot smoke test:

```bash
make smoke
make candidates
```

The fake backend exists to catch contract bugs early. Passing it is necessary,
but never evidence that Go2 pushing or the learned model works.

## Repository map

```text
configs/                 experiment contract and provisional gates
docs/                    two-person plan, verification, runbook, decisions
src/go2wm/contracts.py   shared boundary types
src/go2wm/sim/           simulator protocol and deterministic fake
src/go2wm/data/          collector and manifest validation
src/go2wm/model/         world-model/readout interfaces and bundle metadata
src/go2wm/planning/      64 candidates, scoring, and MPC selection
src/go2wm/runtime/       surprise and stop-oriented runtime behavior
tests/                   executable contract tests
scripts/                 repository and protocol verification helpers
```

## Viewer and closed loop

`make ui-demo && make ui-serve` records a closed-loop run on the fake backend
and replays it in the 3D viewer. The runner locks each plan to
`runs/<id>/ui.jsonl` before any motion (D-010, D-027). See
[`docs/UI_INTEGRATION.md`](docs/UI_INTEGRATION.md).

## Non-negotiable experiment rules

1. Split by complete scenario/episode seeds before any model training.
2. Store the observation before a command, the command actually applied, and
   the observation after it, all on simulator time.
3. Aggregate transient contact and fall events across each action block.
4. Never expose simulator poses, mobility flags, contacts, or oracle rollouts
   to the deployed planner.
5. Test readouts on predicted latents, not only encoded real frames.
6. Lock predictions before executing the corresponding command.
7. Treat contact as valid during pushing; it is not itself a failure.
8. Freeze thresholds, score weights, candidate definitions, checkpoint, and
   seed list before final evaluation.
9. Report raw counts, exclusions, latency, false stops, and failure cases.
10. If the complete learned controller misses its gate, present the honest
    prediction/surprise fallback and label it below the original criterion.

## External integrations

The project intentionally does not vendor LeWM or dimOS. Pin reviewed commits
in `configs/external_sources.toml` before installation. The upstream LeWM
instructions currently use Python 3.10 and `stable-worldmodel[train,env]`;
dimOS exposes skills through `@skill` methods. Treat both integrations as
unverified until the repository's real-path gates pass.

## First checkpoint

Before training anything, produce one tiny real-path recording and prove:

- every block lasts 0.5 simulated seconds within the agreed tolerance;
- images are 224 x 224 RGB from the pinned overhead camera;
- start/end frames, commands, poses, and event summaries share one timeline;
- episode boundaries and split IDs survive serialization;
- replay through the deployment path uses the same units and timing.

If this checkpoint fails, stop and repair collection. More data will only make
the defect more expensive.
