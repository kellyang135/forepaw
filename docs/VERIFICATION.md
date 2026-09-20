# Verification and Evidence Protocol

## 1. Scope

This protocol distinguishes “implemented” from “verified.” Thresholds remain **PROPOSED** until the required two-owner freeze unless otherwise marked. The repository retains the completed CUDA controller training/export, checkpoint playback, ONNX adapter playback, cross-simulator comparison, and strict controller handoff. A clean-source formal G0 trial suite passed all technical criteria at [`artifacts/runs/20260920T073000Z-g0-formal-mjlab/`](../artifacts/runs/20260920T073000Z-g0-formal-mjlab/), but its status is `FORMAL_TRIALS_PASS_PENDING_COOWNER_SIGNOFF`. The direct-MjLab G1A timing/data packet at [`artifacts/runs/20260920T073500Z-g1a-mjlab/`](../artifacts/runs/20260920T073500Z-g1a-mjlab/) passed strict verification and outer checksums. Full G1, LeWM-on-Go2 quality, learned planning, CUDA inference latency, and dimOS deployment remain unverified.

The final report must separate:

- unit/component correctness;
- model prediction quality;
- candidate-library feasibility;
- controller decision quality;
- closed-loop task performance;
- deployment-path reliability and latency.

Passing one category does not imply another.

## 2. Evidence rules

Every reported run requires:

- unique `run_id` and UTC timestamp;
- source revision and dirty-tree flag;
- host, OS, Python, accelerator, dependency lock identifier;
- scene/camera/gait/data/model/controller configuration hashes;
- seed list and split ID;
- stdout/stderr log;
- machine-readable per-episode/per-block results;
- a short human-readable summary;
- artifact checksums;
- exit code and completion state.

Evidence directory convention:

```text
artifacts/runs/<run_id>/
  manifest.json
  config/
  logs/
  metrics/
  media/
  checksums.sha256
  notes.md
```

Do not hand-edit generated metric files. If analysis is corrected, preserve the prior output and create a new analysis run referencing the same raw input.

## 3. Verification layers

### L0 — Static and unit checks

Required tests:

- configurations parse and reject unknown/invalid fields;
- action clipping and unit conversion have boundary tests;
- pose/yaw conversions correctly handle wraparound;
- candidate count is exactly 64 and every candidate is within bounds;
- score components and total score are finite;
- stop overrides any queued motion command;
- bundle and schema versions reject incompatible inputs;
- split generator is deterministic and episode-disjoint;
- checksum and manifest generation are reproducible.

Pass condition: all required tests pass twice from a clean process. These tests establish software behavior only, not simulator/model performance.

### L1 — Simulator and timing verification

| Test | Procedure | Proposed pass threshold | Evidence |
| --- | --- | --- | --- |
| Static stability | Reset and hold zero command for 10 seconds, 5 seeds | 5/5 no fall/out-of-bounds; drift measured and below G0-declared tolerance | pose trace, video, reset manifest |
| Basic motion | 5 repeats each: forward, left turn, right turn, stop | Correct qualitative direction in 5/5; no falls; response distribution recorded | trajectories and applied commands |
| Block duration | Apply a known alternating action schedule for ≥20 blocks | `end-start` equals 0.5 simulated seconds within one physics timestep; step count exact | raw timing table |
| Observation ordering | Use a visually/kinematically obvious action transition | start image precedes action; end image follows all steps in 20/20 inspected blocks | montage and timestamps |
| Applied-action logging | Request in-bound and clipped commands | logged applied command equals simulator-boundary value in all cases | unit test + episode log |
| Event aggregation | Cause a brief contact/fall indicator within a block | block label remains true even if endpoint flag is false | high-rate trace and aggregate |
| Camera coverage | Sample declared start/goal/box extrema | all required task entities visible with margin in every declared setup | image contact sheet |
| Reset variability | Repeat one scenario ≥10 times using settled reset | mean/std/max pose/heading/box deviations reported; acceptance tolerance frozen from measurement | table and trajectories |
| Box behavior | ≥5 intended pushes per class across representative angles | movable box exceeds declared displacement in ≥80%; resistant box stays below tolerance in ≥80%; robot safety reported | paired trials/video |
| Collection throughput | Collect ≥100 blocks using production path | simulated-seconds/wall-second, render time, serialization time, and failures reported | profiler/timing log |

The exact box displacement and reset tolerances are intentionally not invented here. SIM must set them from arena scale, sensor resolution, and repeated no-contact noise before G1. Record the decision in [DECISIONS.md](DECISIONS.md).

### L2 — Dataset integrity and coverage

Hard validity checks for every accepted episode:

- unique `(episode_id, block_index)`;
- contiguous block indexes unless an explicit gap record exists;
- monotonic simulation time;
- frame file exists, decodes, has expected shape, and checksum matches;
- finite commands and labels;
- duration/step count match temporal contract;
- stable object IDs and coordinate convention;
- no post-termination blocks;
- declared scene/camera/schema versions present;
- complete episode-level seed and split fields.

Leakage checks:

- no episode ID in more than one split;
- no scenario seed in more than one split;
- held-out test manifests generated before training and immutable after G1;
- normalization and calibration use training/validation only;
- no final-test outcome informs candidate weights, thresholds, or checkpoint selection.

Coverage report must contain:

- valid/rejected blocks and rejection reasons;
- episode/block counts by split;
- free-motion versus interaction counts;
- contacts and successful displacement by box class;
- action histogram/joint distribution;
- scenario family, approach-angle, start, and goal bins;
- termination reason counts;
- longest repeated-stall runs;
- representative observation montages.

Proposed dataset gate: 20–30% interaction blocks overall with both box classes represented, but this is a collection heuristic, not a proof of sufficiency. Learning curves determine whether more data helps.

### L3 — Model and readout verification

#### Evaluation sets

Report metrics separately for:

1. free motion;
2. approach/no contact;
3. movable-box interaction;
4. resistant-box interaction;
5. turns/command changes;
6. stops.

Never report only an aggregate dominated by free motion.

#### Metrics

At horizons 0.5, 1.0, 2.0, and 3.0 simulated seconds:

- robot position error in meters: median, mean, p90;
- yaw error in degrees using circular difference: median, mean, p90;
- per-box position error in meters: median, mean, p90;
- box displacement-vector error and motion/no-motion classification;
- fall-risk AUROC/AUPRC plus thresholded precision/recall when both classes exist;
- rollout nonfinite/invalid rate;
- sample count and bootstrap 95% confidence interval.

Readout ladder:

1. constant-mean readout;
2. linear readout on encoded real frames;
3. same readout evaluated on one-step predicted latents;
4. same readout through six-step rollouts;
5. small MLP only after an alignment/visibility audit.

Proposed minimum gate:

- real-latent readout beats constant mean on held-out position MAE;
- one-step learned prediction beats persistence on at least robot position or task-relevant box displacement;
- matched-action predictions outperform action-shuffled predictions with a positive effect and bootstrap interval reported;
- six-step rollout invalid rate is 0%;
- all horizon errors are reported even when they miss target.

Do not invent a meter-error threshold before G0 locomotion/reset scale is measured. At G1, freeze absolute targets plus relative requirements. Recommended relative requirement: at 3 seconds, learned robot-position median error at most 80% of persistence baseline and interaction box-displacement error at most 90% of the simple contact baseline. Mark these as **PROPOSED** until frozen.

#### Action-conditioning ablation

For the same observations and candidate set:

- matched condition: predict with each candidate’s real actions;
- shuffled condition: randomly permute candidate action sequences while preserving observations;
- execute/evaluate the original candidates;
- repeat with at least 5 fixed permutation seeds.

Report error difference, candidate rank correlation, and selected-action outcome. This establishes dependence on action, not general model superiority.

### L4 — Candidate and planner verification

#### Candidate-library feasibility (oracle diagnosis only)

From a set of validation starts, execute each candidate in separate settled resets. The realized future may be used only to diagnose candidate coverage.

Pass condition:

- every candidate is safe and within command bounds;
- push, left-detour, right-detour, adjustment, and stop families are all represented;
- at least one candidate makes declared useful progress in ≥80% of manually classified representative validation scenarios;
- when no candidate succeeds, label the scenario “library-infeasible” rather than a model error.

Because settled resets are approximate, repeat representative candidates at least 3 times and report outcome variance.

#### Ranking quality

Freeze model predictions and scores before simulator execution. Compare predicted ranks with realized outcomes using:

- Spearman rank correlation where outcomes are sufficiently continuous;
- top-1 and top-5 “useful candidate” precision;
- pairwise accuracy for push versus best detour;
- regret: realized best score minus selected candidate’s realized score;
- predicted/actual goal-progress plot;
- failure rate of selected versus oracle-best candidate.

Proposed validation gate: positive median rank correlation, top-5 useful precision above random candidate prevalence, and pairwise push/detour accuracy above 60% with raw counts. Small samples must be disclosed.

#### Cost verification

Unit-test every cost term independently. For a fixed synthetic prediction:

- decreasing goal distance cannot worsen goal cost;
- increasing failure risk cannot improve risk cost;
- sustained zero progress under nonzero forward command increases stall cost;
- zero command is not falsely classified as a stall failure;
- successful object contact is not automatically penalized;
- total score decomposition sums exactly to total score.

### L5 — Closed-loop task verification

Freeze before final evaluation:

- goal region radius;
- simulated-time budget;
- fall/out-of-bounds definition;
- stall speed/displacement threshold and duration;
- candidate library and score weights;
- model bundle and surprise threshold;
- approximately 24 held-out seeds balanced across push- and detour-favoring cases;
- allowed infrastructure-failure rerun rules.

Outcome labels:

- `SUCCESS`: goal region reached within budget, no fall/out-of-bounds;
- `TIMEOUT`: budget exhausted;
- `STALL`: declared sustained-lack-of-progress criterion met;
- `FALL`, `OUT_OF_BOUNDS`, `SAFETY_STOP`, `INFRA_FAILURE`;
- successful box contact/push is not a failure.

Report:

- raw successes / attempted scenarios and 95% interval;
- completion simulated time and blocks;
- falls, stalls, out-of-bounds, surprise stops;
- push-favoring and detour-favoring subsets separately;
- repeats/variance for representative paired starts;
- learned controller, simple color/motion baseline, and relevant ablation on the same seed list;
- every exclusion with reason.

Proposed integration gate before final test: ≥3/5 successes for one validation push scenario and ≥3/5 for one validation detour scenario. This is not the final scientific claim; it is a readiness threshold.

### L6 — Surprise verification

Normal calibration data must contain free walking, stops, turns, ordinary contacts, and successful pushes. The anomalous behavior-swap episodes are excluded from calibration.

Tests:

- false-stop rate per block and per episode on normal held-out validation;
- detection delay in blocks/seconds after controlled behavior change;
- detection rate across behavior-change trials;
- residual traces for true/false positives and missed detections;
- stop-command latency and distance traveled after alarm.

Proposed demo target: ≤5% normal episodes falsely stopped, ≥80% behavior-change trials detected within 2 blocks (1 second), and stop command issued in the same control cycle as the alarm. Freeze exact threshold at G6 and report achieved results honestly.

### L7 — dimOS/deployment verification

Verify through the actual deployment path, not only direct Python calls:

- `imagine` accepts a valid candidate set and returns serializable predictions/status;
- invalid action shape/range returns a safe error without motion;
- `plan_to` reports bundle ID, chosen candidate, first action, scores, and latency;
- `stop` preempts/cancels motion and returns observed stopped status;
- only one process publishes motion;
- checkpoint/config mismatch fails closed;
- simulator and deployed paths share action units, block time, preprocessing, and camera;
- 20 consecutive planning cycles complete without leak/crash;
- process restart and reconnection procedure works.

Latency report:

- warm-up method;
- hardware and batch/candidate count;
- at least 100 planning calls where feasible;
- median, p90, and p95 total latency;
- preprocessing, model rollout, scoring, serialization, and transport breakdown;
- whether physics is paused during planning.

If p95 exceeds the 0.5-second action block in live real-time mode, the demo must explicitly run in paused/block-stepping mode or reduce complexity and be reverified.

### L8 — UI and claim verification

Record a screen capture proving this order:

1. current frame and bundle ID appear;
2. candidate predictions and selected plan appear;
3. predictions are visibly locked;
4. motion begins;
5. actual path overlays without modifying locked prediction;
6. residual updates;
7. behavior change triggers alarm and stop, if claimed.

Before presentation, audit each slide/spoken claim:

| Claim | Minimum required evidence |
| --- | --- |
| “Predicts dynamics” | held-out multi-horizon error versus declared baselines |
| “Uses actions” | matched-versus-shuffled ablation |
| “Plans with predictions” | locked scores before execution and runtime trace showing selected action came from them |
| “Better decisions” | same-seed outcome comparison with uncertainty/raw counts |
| “Detects change” | normal false-stop and behavior-change delay results |
| “Runs through dimOS” | deployment-path log/video and latency |

Forbidden claims without additional evidence: general physical reasoning, visual mass inference, unseen-object generalization, exact counterfactuals, sim-to-real transfer, online adaptation, or guaranteed safety.

## 4. Failure triage matrix

| Symptom | First discriminating checks | Likely owner | Do not do first |
| --- | --- | --- | --- |
| Training loss decreases but rollouts fail | timestamp/action alignment; predicted-latent probe; horizon curve | ML with SIM data audit | increase model size |
| Readout fails on real frames | camera visibility; coordinates; normalization; label timing | both | add nonlinear layers immediately |
| Real-frame probe works, predicted probe fails | latent drift by horizon; train/test mode; encoder/predictor version | ML | blame labels without evidence |
| Planner always chooses stop | score-term scale/decomposition; goal units; risk calibration | ML | remove stop candidate |
| Planner never pushes | contact inadvertently penalized; candidate feasibility; box prediction | ML/SIM | hard-code color route silently |
| Candidate rankings poor but oracle succeeds | world-model/readout/ranking issue | ML | alter task seeds |
| Oracle candidates all fail | action library, arena geometry, gait/physics | SIM | retrain model |
| Surprise fires on every push | calibration lacks contacts; latent normalization; bundle mismatch | ML | raise threshold using anomaly test |
| Direct path works, dimOS path fails | units, preprocessing, serialization, stale bundle, competing publisher | SIM | demo only direct path while claiming dimOS |
| Test performance unexpectedly collapses | preserve raw results; check distribution and infrastructure validity | both | tune on test or delete failures |

## 5. Final evidence manifest checklist

- [ ] Final source revision and dirty status.
- [ ] Environment/dependency lock and hardware description.
- [ ] Scene/camera/action/temporal configurations.
- [ ] Dataset manifest, schema, split/seed lists, and coverage report.
- [ ] Model bundle manifest and checksums.
- [ ] Multi-horizon metrics with free/interaction strata.
- [ ] Matched/shuffled action ablation.
- [ ] Baseline definitions and results.
- [ ] Candidate oracle feasibility and ranking results.
- [ ] Closed-loop raw outcomes and exclusions.
- [ ] Surprise false-stop/detection-delay results.
- [ ] Deployment latency and 20-cycle stability run.
- [ ] Locked-prediction demo video plus backup.
- [ ] Known failures, limitations, licenses, and source attribution.
- [ ] SIM and ML sign-off.
