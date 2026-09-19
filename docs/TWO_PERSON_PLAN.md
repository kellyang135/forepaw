# Go2 World Model — Two-Person Delivery Plan

## 1. Purpose and truthfulness contract

This document turns the build specification into a two-person execution plan. It is a plan, not evidence that the system works.

Status vocabulary used throughout the project:

- **ASSUMPTION** — plausible, but not tested in this repository/environment.
- **PROPOSED** — a target or design choice that has not been accepted by experiment.
- **VERIFIED** — backed by a named artifact from a reproducible run.
- **FAILED** — tested against a frozen criterion and did not pass.
- **DEFERRED** — explicitly removed from the committed build.

No item becomes VERIFIED because code exists, a unit test passes, or a developer saw it work once. A verified behavior must have a run ID, configuration snapshot, logs, and the evidence required by [VERIFICATION.md](VERIFICATION.md).

## 2. People and ownership

Use role names until real names are assigned.

| Role | Primary ownership | Secondary responsibility | Must not silently change |
| --- | --- | --- | --- |
| **Person A — Simulation & Integration (SIM)** | MuJoCo scene, Go2/gait adapter, reset/settling, action timing, camera/rendering, collector, schema, dimOS transport, UI/recording | Candidate execution harness, deployment latency, packaging | Temporal contract, action units, camera, scene physics, label definitions |
| **Person B — Learning & Control (ML)** | Dataset loader/splits, LeWM adaptation, training, checkpoints, readouts, rollout inference, candidate generation/scoring, surprise calibration, metrics | Data coverage analysis, evaluation analysis, controller packaging | Preprocessing, representation/checkpoint bundle, splits, planner weights, surprise threshold |

Shared responsibilities:

- Both inspect the first 20 collected blocks and sign the data-integrity gate.
- Both approve any interface or metric change after Gate G1.
- Both rehearse the live and backup demonstrations.
- Neither tunes on the final test set.
- The person who did not implement a gate performs the acceptance replay whenever practical.

Single-writer rules:

- SIM is the only owner of simulator motion commands at runtime.
- ML is the only owner of model-bundle publication.
- Only the current release captain may mark a gate VERIFIED. The release captain rotates at hour 12: SIM before hour 12, ML after hour 12.

## 3. Fixed interfaces between the two people

These are **PROPOSED** until Gate G1 and then frozen.

### 3.1 Temporal contract

- Observation frequency: 2 Hz.
- One action block: 0.5 simulated seconds.
- Observation history: 3 frames, ordered oldest to newest.
- The action stored at block `k` is the command actually applied from `t_k` (inclusive) to `t_{k+1}` (exclusive).
- The image stored at block `k` is rendered at `t_k` before applying action `k`.
- The next image is rendered at `t_{k+1}` after all physics steps for the action block.
- Planning horizon: 6 action blocks / 3.0 simulated seconds.
- Execution: apply only block 0, observe, then replan.
- Dataset `frameskip=1` means the source is already block-aggregated; no further temporal subsampling occurs in the loader.

The exact relationship between the three observation frames and prior commands must be encoded in the schema version and demonstrated with a synthetic timing test. Do not rely on array position alone.

### 3.2 Action contract

Command vector: `[forward_velocity_mps, yaw_rate_radps]`; lateral velocity is always zero. SIM establishes safe bounds during Gate G0 and writes them into the scene/config snapshot. Clipping happens exactly once at the simulator boundary, and both requested and applied actions are logged.

### 3.3 Required block record

Each block must contain or reference:

- `schema_version`, `run_id`, `episode_id`, `block_index`;
- simulation start/end timestamps and wall-clock collection timestamp;
- start and end observation paths plus checksums;
- requested action, applied action, units, and commanded duration;
- actual physics-step count and simulator timestep;
- robot pose at both boundaries;
- both box poses at both boundaries, with stable IDs;
- block-aggregated contact/fall/out-of-bounds flags and counts;
- scene, camera, gait/controller, code revision, and seed identifiers;
- termination reason and validity flag.

Runtime inference receives only images, action history, goal, and public bundle/config metadata. Privileged pose, contact, physics, mass, or mobility values are labels/evaluation data only.

### 3.4 Model bundle contract

One immutable bundle ID covers:

- image preprocessing and normalization;
- encoder and predictor weights/configuration;
- readout weights/configuration;
- target normalization statistics;
- candidate-library version;
- planner cost weights;
- surprise normalization and threshold;
- source revision and training dataset/split IDs.

Changing any member creates a new bundle ID. Updating the encoder invalidates the dependent readouts and surprise calibration until reverified.

### 3.5 Handoff packets

Every handoff is a directory or archived artifact with a short manifest. A Slack message or verbal statement is not a handoff.

| Producer → Consumer | Packet | Minimum contents | Consumer acceptance |
| --- | --- | --- | --- |
| SIM → ML | `data-contract` | schema, 20-block smoke episode, config, checksums, timing plot/table, contact example, reset example | Loader reads all blocks; visual/label spot-check passes; no timestamp/index mismatch |
| ML → SIM | `model-bundle` | immutable bundle, manifest, tiny CPU/GPU smoke input/output, expected shapes/ranges, latency note | Integration smoke run reproduces outputs within tolerance and rejects incompatible schema |
| ML → SIM | `planner-contract` | candidate IDs/families, score components, selected index, first action, predicted paths, status/error schema | UI renders all fields; controller executes only returned first action; stop status dominates motion |
| SIM → ML | `oracle-rollouts` | evaluation-only realized candidate outcomes from declared starts, never exposed at runtime | Candidate-family coverage report identifies at least one feasible action where expected |
| Both → release | `evaluation-evidence` | frozen bundle/config, seed list, raw logs, summary tables, video, failure inventory | Third-party command replays summary from raw logs |

## 4. Work breakdown

### Person A — Simulation & Integration

1. **Repository and environment smoke check**
   - Record OS, Python, MuJoCo, Go2 asset/controller, CUDA machine access, and dimOS versions.
   - Prove the exact Go2 asset and gait checkpoint load.
   - Produce one no-motion render before modifying physics.
2. **Manual physics feasibility**
   - Establish safe forward/yaw bounds.
   - Measure Go2 displacement/heading response for zero, forward, left, and right commands.
   - Tune equal-geometry boxes so one is displaced and the other reliably resists under allowed commands.
   - Test at multiple approach angles and at least 5 repeats per representative command.
3. **Adapter and temporal tests**
   - Implement one authoritative step/reset/render API.
   - Add a synthetic action schedule whose boundaries are obvious in logs.
   - Aggregate contacts/falls over every physics substep.
4. **Collector and reset protocol**
   - Capture atomic per-block records and episode manifests.
   - Settle, warm observation history, and record actual settled state.
   - Add rejection/termination reasons rather than silently dropping bad episodes.
5. **Collection campaign**
   - Collect free motion, turns, approaches, pushes, resisted pushes, departures, detours, stops, and command transitions.
   - Monitor counts by scenario, box class, contact, and command family; do not meet the interaction target by repeated stall frames.
6. **Candidate execution and oracle harness**
   - Replay all structured candidates from comparable settled starts for evaluation only.
   - Record which families are actually capable of push/detour/stop outcomes.
7. **dimOS and UI**
   - Expose one motion publisher and the `imagine`, `plan_to`, and `stop` surfaces.
   - Render predictions before motion, lock them, then overlay actual outcomes.
   - Surface bundle ID, latency, score components, and surprise/stop state.
8. **Packaging and demo operations**
   - Build one-command local smoke/evaluation/demo entry points once source exists.
   - Capture a clearly labeled backup recording with the frozen bundle.

### Person B — Learning & Control

1. **Data contract consumer**
   - Implement strict schema/version validation.
   - Reject duplicate/missing block indexes, non-monotonic timestamps, missing frames, nonfinite labels, and action-duration mismatches.
   - Build episode-level train/validation/test manifests before model fitting.
2. **Data diagnostics**
   - Produce counts and distributions by split, interaction type, action, speed, box, layout, and termination.
   - Detect leakage by episode ID and scenario seed.
3. **Model adaptation**
   - Pin the LeWM source revision and dependencies.
   - Adapt two-dimensional actions, three-frame history, 0.5-second blocks, and 192-dimensional representations.
   - Save early and periodic checkpoints with training configuration and data ID.
4. **Readouts**
   - Fit robot `[x, y, sin(yaw), cos(yaw)]`, box-position, and defined failure-risk readouts.
   - Evaluate on real encoded latents and predicted latents at every horizon.
   - Try linear probes first; diagnose alignment/visibility before increasing capacity.
5. **Prediction verification and ablations**
   - Report robot/box errors at 0.5, 1, 2, and 3 seconds, separated into free and interaction windows.
   - Compare matched actions to shuffled actions.
   - Compare the learned predictor to declared kinematic/simple task baselines.
6. **Planner**
   - Define exactly 64 coherent candidates with stable IDs and family labels.
   - Score goal progress, risk, stall, effort, and command changes; do not globally penalize contact.
   - Tune only on validation, then freeze weights.
7. **Surprise detector**
   - Define latent discrepancy and normalization.
   - Calibrate on held-out normal transitions containing pushes and turns.
   - Freeze the threshold before behavior-swap anomaly trials.
8. **Evaluation analysis and bundle publication**
   - Publish the immutable model bundle and machine-readable manifest.
   - Generate raw and summary metrics without excluding failures unless the exclusion rule was predeclared.

## 5. Twenty-four-hour schedule and handoffs

The schedule assumes the environment and source dependencies are obtainable. Those are **ASSUMPTIONS** until G0. Hours are elapsed project hours, not clock times.

| Window | Person A — SIM | Person B — ML | Joint checkpoint / deliverable |
| --- | --- | --- | --- |
| H0–H0:30 | Inventory simulator/assets; boot static scene | Pin source/dependency candidates; prepare schema/loader tests | Kickoff: scope, role assignment, stop conditions, evidence root |
| H0:30–H2 | Manual gait/box/camera/reset feasibility; measure throughput | Formalize schema, split policy, metrics; inspect upstream model API | **G0 Physics:** usable locomotion, camera, reset, two box behaviors; or invoke fallback |
| H2–H3 | Implement adapter/collector; create 20-block smoke packet including dimOS path | Implement strict loader and synthetic fixtures | **G1 Data:** both sign timing/alignment packet; schema and camera freeze |
| H3–H5 | Collect wave 1 with coverage dashboard; repair only data blockers | Start first training run; fit/read out real observations; establish baselines | **G2 Learnability:** first checkpoint plus held-out diagnostics, not merely training loss |
| H5–H7 | Collect targeted wave 2 based on coverage gaps; build candidate executor | Autoregressive rollout; predicted-latent probes; matched-vs-shuffled action test | **G3 Prediction:** go/no-go for learned planning versus prediction-only fallback |
| H7–H10 | Oracle-check candidate families; integrate bundle contract and UI skeleton | Define 64 candidates, planner costs; validation ranking study; multistep tuning only if warranted | **G4 Planning:** candidate feasibility separated from model ranking quality |
| H10–H12 | Closed-loop controller integration; locked overlays; dimOS calls | Surprise calibration; controller debugging; publish integration bundle | **G5 End-to-end:** repeatable run for each box behavior and stop path |
| H12–H15 | Improve deployment reliability/latency; recording path | Target largest measured error; compare baselines | Release captain transfers SIM → ML; no new major features after H15 |
| H15–H18 | UI/evidence polish; deployment-path validation | Freeze candidate library, costs, thresholds; prepare evaluation scripts | **G6 Freeze:** bundle/config/seed list immutable; checksum them |
| H18–H21 | Execute untouched held-out scenarios and capture raw logs/video | Monitor protocol compliance; compute results after runs complete | **G7 Final evaluation:** no tuning, all failures retained |
| H21–H23 | Rehearse live demo, failure injection, restart, backup video | Audit claims, metrics, plots, manifest reproducibility | **G8 Release candidate:** clean-machine/shell smoke replay |
| H23–H24 | Preserve working system; package demo | Final report/results/limitations | Final sign-off and handoff |

### Checkpoint meeting format (maximum 10 minutes)

At G0–G8, answer in order:

1. What exact artifact was produced, and what is its run ID?
2. Which criterion passed or failed? Show the measurement.
3. What changed since the previous checkpoint?
4. What is the single largest current risk?
5. Continue, repair, or invoke fallback?
6. Who owns the next blocking deliverable, and by what hour?

Record answers in the run ledger described in [RUNBOOK.md](RUNBOOK.md).

## 6. Gate criteria and fallback ladder

Numeric thresholds below are **PROPOSED starting thresholds**. G0 may revise task tolerances once, based on measured locomotion/reset variability. Any revision must occur before G1 and be logged in [DECISIONS.md](DECISIONS.md). Final test criteria cannot change after G6.

| Gate | Minimum pass evidence | If it fails by deadline |
| --- | --- | --- |
| **G0 Physics, H2** | 5/5 walk and turn trials complete without fall; light box measurably displaces in ≥4/5 intended push trials; resistant box remains below the predeclared displacement tolerance in ≥4/5; camera keeps robot, boxes, goal in frame; reset settle variance measured | Spend at most 45 additional minutes on physics. If two behaviors remain unreliable, use a fixed obstacle and disclose it; if gait itself is unreliable, switch to prediction/surprise of free motion only |
| **G1 Data, H3** | 20 consecutive valid blocks; exact start/action/end ordering; monotonically increasing times; correct event aggregation; both people visually check at least 10 blocks; one packet traverses dimOS | Stop all model work. Repair collector/schema first. Never train around known misalignment |
| **G2 Learnability, H5** | Real-frame robot/box readout beats constant-mean baseline on validation; data split leak check passes; training loss finite and checkpoint reload deterministic within declared tolerance | Inspect visibility, normalization, units, labels, and coverage. Do not add complex model capacity first |
| **G3 Prediction, H7** | One-step prediction beats persistence/kinematic reference on at least one task-relevant measure; matched actions outperform shuffled actions; rollout does not become nonfinite through 6 steps | Try targeted data or short multistep loss. If still failed at H9, freeze prediction-only evidence and do not claim predictive control |
| **G4 Planning, H10** | Oracle runs show useful push and detour candidates; learned ranking is better than random and shows positive validation goal-progress association | Fix candidate geometry/library before model if oracle fails. If model ranking fails but candidates work, use simple baseline for task demo and present learned model as prediction experiment |
| **G5 End-to-end, H12** | At least 3/5 repeated completions in one push-favoring and one detour-favoring validation scenario; predictions locked before execution; surprise causes stop; dimOS path executes | Freeze feature work. Use the highest honest fallback level below |
| **G6 Freeze, H18** | Bundle/config/criteria/seed list checksummed; no test seed used for training/tuning; restart smoke passes; backup video path tested | Delay final evaluation until freeze integrity is restored; never evaluate a moving target |
| **G7 Test, H21** | All declared test starts attempted; raw logs complete; exclusions match predeclared rules; metrics regenerated from raw logs | Rerun only trials invalidated by a logged infrastructure failure; do not rerun ordinary behavioral failures |
| **G8 Release, H23** | Live path starts from documented commands; demo can recover from one process restart; backup video, evidence bundle, license/attribution, and limitations present | Deliver the last verified bundle; remove broken optional features rather than patching core behavior |

Fallback levels, highest to lowest:

1. **F0 — Full predictive controller:** learned rollouts choose actions, replan, show predictions/actuals, and surprise-stop.
2. **F1 — Learned prediction + baseline controller:** a simple declared controller moves the robot; learned model predictions remain locked and evaluated. Never claim the learned model selected the route.
3. **F2 — Open-loop prediction evaluator:** show matched candidate predictions and actual outcomes, action ablation, and surprise detection without autonomous goal completion.
4. **F3 — Dataset/instrumentation demonstration:** show aligned Go2 interaction collection, quantitative baselines, and clearly labeled preliminary model results.

The team must announce the active fallback level at every checkpoint after G3. A fallback is a successful scope decision, not permission to blur claims.

## 7. Critical dependencies and communication

Critical path:

```text
gait/physics feasibility
  → temporal contract + aligned smoke data
    → episode splits + model checkpoint
      → predicted-latent readouts
        → oracle-valid candidates
          → learned ranking + closed-loop integration
            → frozen evaluation + demo
```

Parallel work is useful only off the critical path. Examples:

- While SIM collects wave 1, ML trains and reports targeted coverage gaps.
- While ML trains, SIM builds oracle candidate execution and UI with a fake deterministic predictor.
- While SIM integrates the current bundle, ML calibrates on already frozen validation data.

Communication cadence:

- Start-of-window: state the one blocking outcome for each person.
- Every 60–90 minutes: write one ledger entry even if no gate is reached.
- Immediately: report schema, timing, camera, action-bound, or bundle incompatibilities.
- No surprise interface changes: announce them before code changes and add a decision record.
- Keep troubleshooting notes factual: observed, expected, evidence path, hypothesis, next discriminating test.

## 8. Definition of done

The project is done only when the selected fallback level has:

- a frozen, checksummed artifact set;
- a reproducible start/evaluation command;
- evidence appropriate to every public claim;
- raw test results including failures;
- measured latency and compute/data accounting;
- a live demonstration plan plus labeled backup recording;
- license/source attribution;
- an explicit limitations statement;
- two-person sign-off that no privileged runtime state or test-set tuning occurred.

