# UI integration: closed loop, telemetry, viewer

Status: **PROPOSED** (D-027). Everything below runs today on the fake backend.
Fake-backend runs are software evidence only, and every log says so.

## Pieces

| Piece | File | Role |
| --- | --- | --- |
| Closed-loop runner | `src/go2wm/runtime/loop.py` | plan, arm monitor, write and fsync the lock, execute block 0, observe, check surprise, repeat |
| Telemetry | `src/go2wm/telemetry.py` | append-only `runs/<id>/ui.jsonl`, schema `go2wm.ui.v1` |
| CLI | `src/go2wm/ui/__main__.py` | `record` a run, `serve` the viewer on localhost |
| Rehearsal model | `src/go2wm/ui/reference.py` | privileged kinematic reference, **not a learned model**, UI rehearsal only |
| Viewer | `ui/viewer/` | Three.js replay or live follow of a log; never plans |
| Design prototypes | `ui/prototype/` | standalone demos with an in-browser toy model; not connected |

## Commands

```bash
make ui-demo        # reference model, anomaly scenario -> runs/ui-reference-<ts>/ui.jsonl
make ui-baseline    # fake-data linear baseline bundle, same loop
make ui-serve       # http://127.0.0.1:8765/ui/viewer/  (newest run first; ?log=/runs/<id>/ui.jsonl)
# a published bundle, once one exists:
python -m go2wm.ui record --bundle runs/<id>/bundles/<bundle-id> --out runs/ui-<id>
```

The viewer follows a log that is still being written, so start `serve`, then
`record`, and the browser shows each block as the controller locks it.

## Log records (one JSON object per line)

| `type` | When | Key fields |
| --- | --- | --- |
| `run_start` | after reset and history warm-up | scene size and frame, camera, goal, bundle id and label, score weights, surprise threshold, 3 history PNGs, 2 history actions, `ground_truth`, `evidence_scope`, `fallback_level`, `render_hints` |
| `plan_locked` | before motion, fsynced | `locked_at_utc`, `selected_id`, `selection_reason`, `first_action`, `selected_states`, `predicted_next_latent`, `current_estimate` (readout of the current latent), `planning_latency_s`, all `rankings` with weighted cost parts and predicted paths |
| `block_executed` | after the block | requested and applied action, observation PNG, `observed_latent`, `surprise` (status, discrepancy, threshold, stop_commanded), `ground_truth` start/end/events and robot prediction error |
| `run_end` | once | `reason`: `goal_reached`, `surprise_stop`, `stalled`, `fall`, `out_of_bounds`, `max_blocks` |

Runtime-visible values and privileged labels are kept apart: only
`ground_truth` keys carry simulator state, and the planner never reads them.

## What changes for the real Go2 path

1. Done (ML, 2026-09-20, draft adapter): `record --sim mujoco` runs the loop on
   the MuJoCo Go2 (`--controller mjlab|rl_sar`), with MuJoCo-scale scenarios
   `push`, `detour` and `anomaly` (the blue box silently becomes 20 kg after
   block 2) and `render_hints` of `robot_scale: 1.0`, `box_size_m: 0.40`.
   With SIM's trained MjLab policy (its `source/go2.xml` needs MjLab's 16
   meshes in `source/assets/`, now present locally):

   ```bash
   C=artifacts/runs/20260920T034139Z-go2-controller-training
   uv run --no-sync python -m go2wm.ui record --sim mujoco --model reference \
     --controller mjlab --policy $C/policy/policy.onnx --robot-xml $C/source/go2.xml \
     --deploy-yaml $C/config/deploy.yaml --scenario push --out runs/ui-mujoco-push-$(date -u +%Y%m%dT%H%M%SZ)
   ```
2. ML's published bundle replaces `--model reference` via `--bundle`.
3. The dimOS `plan_to` skill should call `ClosedLoopRunner`, so the demo and
   the skill share one code path and one log.
4. D-028 is fixed (see DECISIONS.md).

## First MuJoCo rehearsal runs (2026-09-20)

SIM's MjLab policy, reference model, surprise threshold 0.10 (hand-set for the
reference model only; a learned bundle calibrates its own on validation data).

| Scenario | Result | Evidence |
| --- | --- | --- |
| push | goal reached, 13 blocks; blue box contacted for 6 blocks and moved 1.08 m; red untouched | `runs/ui-mujoco-push/ui.jsonl` |
| detour | goal reached, 14 blocks; no contact with either box | `runs/ui-mujoco-detour/ui.jsonl` |
| anomaly | surprise stop at block 3 (0.122 > 0.10) on first contact with the silently heavy box | `runs/ui-mujoco-anomaly/ui.jsonl` |

Normal runs peaked at 0.075. The rl_sar gait under the same reference model
false-stopped at 0.103 on its first push, so this threshold does not transfer
between gaits. These are rehearsals: the reference encoder reads simulator
poses (D-004), so none of this is evidence about a learned model.
