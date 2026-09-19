# Decision Log and Assumption Register

## 1. Rules

This file records project-level decisions that affect interfaces, claims, evaluation, or reproducibility. It must not be used to claim verification; link to run evidence for that.

Decision statuses:

- **Proposed** — default from the build specification; not yet frozen.
- **Accepted** — team agreed and implementation should conform.
- **Superseded** — replaced by a later decision, with a backlink.
- **Rejected** — considered and deliberately not used.

Every new decision should include date/time, both-person acknowledgment, rationale, consequences, and evidence or measurement that informed it.

Template:

```markdown
### D-XXX — Title

- Status:
- Date / gate:
- Owners acknowledging:
- Decision:
- Rationale:
- Consequences:
- Evidence:
- Supersedes / superseded by:
```

## 2. Initial decisions

### D-001 — Two-person ownership split

- Status: Proposed
- Date / gate: project start
- Owners acknowledging: pending SIM, pending ML
- Decision: SIM owns simulator/data/runtime integration and ML owns learning/readouts/planner/metrics. Shared interfaces require both signatures.
- Rationale: minimizes simultaneous edits on the critical path while making integration responsibilities explicit.
- Consequences: schema, model bundle, and planner response require formal handoffs; neither person may silently change them.
- Evidence: planning choice only.
- Supersedes / superseded by: none.

### D-002 — Fixed 0.5-second block temporal contract

- Status: Proposed
- Date / gate: freeze at G1
- Owners acknowledging: pending
- Decision: render at block start, apply `[forward_velocity_mps, yaw_rate_radps]` for 0.5 simulated seconds, aggregate events across all physics substeps, then render block end. Observations are 2 Hz; three frames are oldest-to-newest; planner rolls six blocks and executes one.
- Rationale: unifies collection, model training, evaluation, and runtime control.
- Consequences: any change requires a schema/config version and regeneration or explicit migration of dependent data.
- Evidence: must be verified by G1 timing packet.
- Supersedes / superseded by: none.

### D-003 — Fixed overhead RGB is the sole runtime observation

- Status: Proposed
- Date / gate: freeze at G1
- Owners acknowledging: pending
- Decision: begin with one fixed 224×224 overhead RGB camera and a visible front marker.
- Rationale: keeps heading and object position observable within the hackathon scope.
- Consequences: privileged simulator state remains training/evaluation-only; camera changes invalidate model/data comparability.
- Evidence: camera coverage/visibility test required at G0/G1.
- Supersedes / superseded by: none.

### D-004 — Labels are privileged, runtime inputs are not

- Status: Proposed
- Date / gate: accept at kickoff
- Owners acknowledging: pending
- Decision: robot/box poses, contacts, mass/mobility, fall flags, and oracle futures may be used for labels and evaluation but cannot enter runtime planning.
- Rationale: preserves the project’s central visually grounded prediction claim.
- Consequences: runtime interfaces and logs must make input provenance auditable; simple baseline must also disclose its visual estimator.
- Evidence: deployment interface/input audit at G5/G6.
- Supersedes / superseded by: none.

### D-005 — Same geometry, visually distinct behavior-associated boxes

- Status: Proposed
- Date / gate: decide at G0
- Owners acknowledging: pending
- Decision: first attempt equal geometry and friction, distinct colors, and empirically selected masses. If resistant behavior cannot be made reliable, use a fixed object and disclose it.
- Rationale: produces a bounded, legible push-versus-detour task without claiming hidden mass inference.
- Consequences: results demonstrate an appearance/behavior association in this environment, not general physical reasoning.
- Evidence: repeated push trials required at G0.
- Supersedes / superseded by: none.

### D-006 — Episode/seed-level split before fitting

- Status: Proposed
- Date / gate: freeze at G1
- Owners acknowledging: pending
- Decision: train, validation, and final test are separated by complete episode and scenario seed. Test manifest is immutable before training.
- Rationale: prevents temporal and scenario leakage.
- Consequences: sparse test classes are reported, not repaired by moving episodes after results are seen.
- Evidence: deterministic split/leak report.
- Supersedes / superseded by: none.

### D-007 — Immutable composite model bundle

- Status: Proposed
- Date / gate: first bundle at G3, final freeze at G6
- Owners acknowledging: pending
- Decision: encoder, predictor, preprocessing, readouts, normalizers, candidates, planner weights, and surprise threshold share one versioned bundle manifest.
- Rationale: each component changes the meaning of downstream representations and scores.
- Consequences: an encoder update invalidates readouts/calibration; mixed bundles fail closed.
- Evidence: bundle compatibility and reference-output tests.
- Supersedes / superseded by: none.

### D-008 — Structured 64-candidate, six-block planner

- Status: Proposed
- Date / gate: freeze at G6
- Owners acknowledging: pending
- Decision: use 64 deterministic candidates across push, left/right detour, adjustment, slow/stop families, each six 0.5-second blocks. Execute only the first block before replanning.
- Rationale: coherent maneuvers use the small search budget better than arbitrary jitter.
- Consequences: candidate feasibility must be tested independently with evaluation-only oracle rollouts.
- Evidence: candidate manifest, bounds test, oracle coverage report.
- Supersedes / superseded by: none.

### D-009 — No blanket contact penalty

- Status: Proposed
- Date / gate: accept before planner tuning
- Owners acknowledging: pending
- Decision: planner cost includes goal distance, undesirable risk, sustained lack of progress, effort, and command change; contact alone has no universal penalty.
- Rationale: intended pushing requires contact.
- Consequences: failure/stall definitions must distinguish safe successful pushing from dangerous/stuck behavior.
- Evidence: synthetic cost tests and validation outcome analysis.
- Supersedes / superseded by: none.

### D-010 — Predictions lock before execution

- Status: Proposed
- Date / gate: accept before G5
- Owners acknowledging: pending
- Decision: persist candidate predictions, scores, selection, bundle ID, and timestamp before sending the first motion command. UI never retroactively changes that record.
- Rationale: makes the demonstration falsifiable and auditable.
- Consequences: the runtime path must prioritize durable evidence write before execution and show missing-lock as an error.
- Evidence: ordered runtime trace and screen capture.
- Supersedes / superseded by: none.

### D-011 — Surprise means detect and stop only

- Status: Proposed
- Date / gate: freeze at G6
- Owners acknowledging: pending
- Decision: compare previously predicted next latent to the next observation latent; threshold breach issues stop. No online model update or autonomous recovery is claimed.
- Rationale: calibration and stopping are achievable and measurable within scope; adaptation is not.
- Consequences: behavior-change examples must be excluded from threshold calibration; false stops and delay are reported.
- Evidence: surprise validation report.
- Supersedes / superseded by: none.

### D-012 — Approximate settled resets, not exact counterfactuals

- Status: Proposed
- Date / gate: accept at G0
- Owners acknowledging: pending
- Decision: reset/settle and collect a fresh observation history for every trial; measure state variance and use repeats.
- Rationale: exact restoration of internal gait/controller state is outside scope.
- Consequences: oracle and paired comparisons include repeat variance and are not described as identical counterfactual branches.
- Evidence: reset repeatability report.
- Supersedes / superseded by: none.

### D-013 — Validation tuning, untouched final test

- Status: Proposed
- Date / gate: accept at kickoff; freeze artifacts at G6
- Owners acknowledging: pending
- Decision: checkpoint choice, planner weights, threshold, and candidate changes use training/validation only. Approximately 24 held-out scenarios remain untouched until final evaluation.
- Rationale: preserves credibility of final outcomes.
- Consequences: after G6, test failures are reported rather than tuned away; only predeclared infrastructure failures can be rerun.
- Evidence: access/manifest audit and evaluation ledger.
- Supersedes / superseded by: none.

### D-014 — Honest fallback ladder

- Status: Proposed
- Date / gate: accept at kickoff
- Owners acknowledging: pending
- Decision: use F0 full predictive control, F1 learned prediction plus declared baseline control, F2 open-loop prediction/surprise, or F3 dataset/instrumentation. Always label the active level.
- Rationale: preserves a defensible deliverable when a sequential dependency fails.
- Consequences: no route-selection claim is made at F1–F3; presentation and UI language change accordingly.
- Evidence: gate records and final sign-off.
- Supersedes / superseded by: none.

### D-015 — Physics-paused planning is acceptable if disclosed

- Status: Proposed
- Date / gate: decide from latency at G5/G6
- Owners acknowledging: pending
- Decision: initial demo may pause/block simulator progression while planning. Report median/p95 wall-clock latency and label the execution mode.
- Rationale: preserves temporal semantics when inference exceeds the 0.5-second real-time budget.
- Consequences: do not claim real-time asynchronous control unless p95 and scheduling behavior support it.
- Evidence: deployment latency report.
- Supersedes / superseded by: none.

### D-016 — Prove Python compatibility or isolate model inference

- Status: Proposed
- Date / gate: resolve before G2
- Owners acknowledging: pending
- Decision: first test whether the pinned LeWM stack runs inside the pinned dimOS Python environment. If it does not, keep training/inference in a Python 3.10 environment and expose a versioned local controller service to the Python 3.12 dimOS skill module. Do not attempt an unplanned dependency upgrade during final integration.
- Rationale: the reviewed LeWM instructions currently create a Python 3.10 environment, while the reviewed current dimOS setup instructions use Python 3.12. This is an integration risk, not proof of incompatibility.
- Consequences: the model bundle/request/response schema and stop behavior must be identical in-process and across the service boundary; deployment latency includes serialization and RPC time.
- Evidence: clean environment installation logs, import smoke tests, one reference inference, and dimOS-path parity output.
- Supersedes / superseded by: none.

### D-017 — Current dimOS `unitree-go2` simulator path is not Go2 evidence

- Status: Proposed
- Date / gate: 2026-09-19 / G0 source audit
- Owners acknowledging: SIM executed and recorded; ML/human co-owner pending
- Decision: do not use the pinned dimOS `unitree-go2` simulation command as evidence for a Go2 model or gait. At revision `c1c3cdc9d2ee54ca72259465688395699d7d99a2`, the simulator rewrites `unitree_go2` to `unitree_go1`; the loader includes Go1 and G1 assets/controllers but no Go2 controller case.
- Rationale: accepting the CLI label would silently substitute a different robot and invalidate the task's physical and visual claims.
- Consequences: dimOS may still be tested later as an interface/skill transport using an explicitly labeled surrogate. Gate G0 remains blocked until a true-Go2 controller is integrated and passes motion trials.
- Evidence: `artifacts/runs/20260919T191500Z-person-a-preflight/preflight.json`; audited source hashes and revisions are embedded in that report.
- Supersedes / superseded by: none.

### D-018 — Pin the official Menagerie Go2 model and its Python floor

- Status: Proposed
- Date / gate: 2026-09-19 / G0 model-render probe
- Owners acknowledging: SIM executed and recorded; ML/human co-owner pending
- Decision: use MuJoCo `3.3.4` with `mujoco-menagerie` `2026.9.0` as the initial true-Go2 model source, and declare Python `>=3.10.12,<3.13` because that Menagerie release does not support earlier 3.10 patches.
- Rationale: the pinned model compiled and rendered reproducibly from the project environment; the tighter Python floor is the resolver-proven compatibility intersection.
- Consequences: this establishes model and camera infrastructure only. It does not provide a Go2 gait policy, front marker, task scene, or dimOS transport.
- Evidence: `artifacts/runs/20260919T191000Z-go2-project-env-render/report.json` and its checksum manifest.
- Supersedes / superseded by: none.

### D-019 — Train and validate the gait on its exact Unitree RL MjLab model

- Status: Proposed
- Date / gate: 2026-09-19 / G0 controller recovery
- Owners acknowledging: SIM source-audited and recorded; ML/human co-owner pending
- Decision: train `Unitree-Go2-Flat` from Unitree RL MjLab revision `1425b15f73bd4095f0df53709d7c389c3eb9e790`, then validate the checkpoint and ONNX policy against that revision's bundled Go2 XML before project integration. Do not insert the policy directly into the stock Menagerie scene.
- Rationale: the training XML checksum differs from the verified Menagerie XML, and the training stack uses a 5 ms physics step, four-step control decimation, and scaled joint-position targets rather than the stock scene's actuator assumptions.
- Consequences: the first real adapter uses the training model and 50 Hz controller. A Menagerie port, if desired, is a separate compatibility experiment. Each 0.5-second project block contains 25 policy updates and 100 physics steps, and project lateral velocity is always zero.
- Evidence: `artifacts/runs/20260919T202600Z-go2-controller-source-audit/report.json`, the procedure in `docs/GO2_CONTROLLER_RECOVERY.md`, and the requirement that controller packets pass `scripts/verify_controller_handoff.py`.
- Supersedes / superseded by: narrows D-018 for controller execution; D-018 still covers the independently verified render/model probe.

## 3. Assumption register

Every row begins unverified. Change status only with a direct evidence link.

| ID | Assumption | Risk if false | Owner | Must resolve by | Status / evidence |
| --- | --- | --- | --- | --- | --- |
| A-001 | Existing Go2 asset and gait controller load in the target MuJoCo environment | Entire interaction task blocked | SIM | G0 | **BLOCKED** — official Go2 asset compiles/renders, but no Go2 gait is present in pinned dimOS or the inspected Unitree RL Gym pretrained artifacts; see `artifacts/runs/20260919T191500Z-person-a-preflight/preflight.json` |
| A-002 | Go2 can safely walk/turn under bounded commands for repeated trials | Data/task reliability fails | SIM | G0 | ASSUMPTION |
| A-003 | Equal-geometry boxes can be tuned into repeatable movable/resistant behaviors | Core comparison becomes unreliable | SIM | G0 | ASSUMPTION |
| A-004 | Overhead 224×224 RGB exposes robot heading and both boxes accurately enough | Readouts/planning fail | Both | G2 | ASSUMPTION |
| A-005 | Settled resets are repeatable enough for comparative trials | Candidate/evaluation variance overwhelms effects | SIM | G0/G4 | ASSUMPTION |
| A-006 | Production collection throughput is sufficient for training waves | Schedule slips or data is inadequate | SIM | G0 | ASSUMPTION |
| A-007 | Linux CUDA machine is available, compatible, and has storage | Model training blocked or delayed | ML | H0:30 | ASSUMPTION |
| A-008 | LeWM source adapts cleanly to 2-D actions and the selected history convention | Architecture work expands | ML | G2 | ASSUMPTION |
| A-009 | Available data yields action-conditioned, task-relevant predictions | Predictive-control claim fails | ML | G3 | ASSUMPTION |
| A-010 | Frozen/simple readouts decode robot/box states from generated latents | Planner cannot score useful futures | ML | G3 | ASSUMPTION |
| A-011 | Useful push/detour behavior fits within a 3-second horizon | Candidate ranking lacks signal | Both | G0/G4 | ASSUMPTION |
| A-012 | A 64-candidate batch fits memory and latency constraints | Deployment may require batching/reduction | ML | G5 | ASSUMPTION |
| A-013 | dimOS can serialize the proposed skill request/response fields | Integration contract must narrow | SIM | G5 | ASSUMPTION |
| A-014 | A single motion publisher can be enforced | Stop and plan may conflict unsafely | SIM | G5 | ASSUMPTION |
| A-015 | Latent residual separates behavior changes from ordinary contacts/turns | Surprise demo false-alarms or misses | ML | G5/G6 | ASSUMPTION |
| A-016 | Approximately 24 held-out scenarios are operationally feasible | Final evaluation must be reduced and qualified | Both | G6 | ASSUMPTION |
| A-017 | Pinned LeWM and dimOS dependencies can share one Python environment | A local model-service boundary is required | Both | G2 | ASSUMPTION |

## 4. Decisions required from measurements

Fill these no later than the named gate:

| ID | Decision needed | Inputs | Deadline |
| --- | --- | --- | --- |
| P-001 | Final forward/yaw action bounds | stable locomotion trials | G0 |
| P-002 | Movable/resistant box parameters or fixed-object fallback | repeated displacement trials | G0 |
| P-003 | Arena/camera dimensions and final resolution | coverage and maneuverability tests | G0/G1 |
| P-004 | Reset settle duration and acceptance tolerances | 10+ reset repeats | G1 |
| P-005 | Exact history/action tensor convention | synthetic timing packet and upstream model API | G1 |
| P-006 | Goal radius, time budget, fall/out-of-bounds, stall rules | manual task trials and noise floor | G1 |
| P-007 | Absolute prediction error targets | displacement scales, reset noise, baselines | G1 |
| P-008 | Dataset wave sizes and class/interaction targets | throughput and learning curves | each wave |
| P-009 | Readout capacity or supervised auxiliary loss | real/predicted latent probe results | G3 |
| P-010 | Multistep fine-tuning | horizon error curve | G3 |
| P-011 | Final candidate values/family allocation | oracle feasibility | G4/G6 |
| P-012 | Planner weights | validation ranking and task results | G6 |
| P-013 | Surprise metric/threshold | normal validation residuals | G6 |
| P-014 | Real-time versus paused/block-step demo | p95 deployment latency | G6 |
| P-015 | Final fallback level | gates G3–G6 | G6 |
| P-016 | Unified Python environment versus local model service | pinned 3.10/3.12 install and inference smoke tests | G2 |

## 5. Explicitly deferred choices

These remain deferred unless every core gate passes early:

- eight-block/four-second planning horizon;
- CEM or other online trajectory optimization;
- egocentric second visual model;
- photorealistic future-frame generation;
- online adaptation after surprise;
- new locomotion policy training;
- physical-robot/sim-to-real deployment;
- unseen-category or appearance-independent mass generalization.

Reactivating one requires a recorded decision explaining which core risk is already closed and why the addition will not jeopardize G6/G7.
