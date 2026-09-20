# Build, Training, Evaluation, and Demo Runbook

## 1. How to use this runbook

This is the operational procedure for a two-person build. Command names below are interface targets for the scaffold; they are **not yet verified to exist**. Replace a placeholder only when the repository supplies the corresponding command, and record the actual command in the run ledger.

Never proceed past a failed data-integrity check merely to save time. Prefer the fallback ladder in [TWO_PERSON_PLAN.md](TWO_PERSON_PLAN.md).

## 2. Session start (both people)

1. Synchronize on the same source revision. Record revision and whether the tree is dirty.
2. Record each machine’s hostname, OS, Python, accelerator/driver, and dependency environment ID.
3. Confirm the current role assignment and release captain.
4. Create a unique run/session ID using UTC time plus a short purpose slug.
5. Review unresolved blocking decisions in [DECISIONS.md](DECISIONS.md).
6. State the current fallback level (F0–F3) and the next gate.
7. Confirm available disk space, expected artifact destination, and checkpoint copy path.
8. Do not delete or overwrite previous evidence directories.

Suggested future command surface:

```bash
make doctor
make test
make smoke
```

The eventual `doctor` command should print versions, hardware, asset availability, configuration hashes, disk capacity, and camera/model loading status without changing simulator or data state.

## 3. Run ledger

Maintain one append-only ledger entry per checkpoint and every 60–90 minutes. Suggested template:

```markdown
## <UTC timestamp> — <gate or activity> — <run_id>

- Owner:
- Source revision / dirty:
- Config and bundle IDs:
- Dataset/split IDs:
- Command:
- Expected criterion:
- Observed result:
- Evidence path:
- Status: ASSUMPTION | PROPOSED | VERIFIED | FAILED | DEFERRED
- Known confounds:
- Decision / next discriminating test:
- Next owner and deadline:
```

Do not write “looks good.” Include a number, trace, image reference, or explicit pass/fail observation.

### 2026-09-19T19:14:59Z — G0 preflight — `20260919T191500Z-person-a-preflight`

- Owner: Person A / SIM
- Source revision / dirty: `NO_COMMIT`; scaffold is an uncommitted working tree
- Config and bundle IDs: MuJoCo `3.3.4`; Menagerie `2026.9.0`; dimOS source `c1c3cdc9d2ee54ca72259465688395699d7d99a2`
- Dataset/split IDs: none; collection correctly not started
- Commands: `scripts/verify_go2_model.py` through project `.venv/bin/mjpython`, then `scripts/person_a_preflight.py`
- Expected criterion: true Go2 model plus gait/controller, 224×224 RGB render, and an installable production path
- Observed result: official Go2 model compiled (`nu=12`, `dt=0.002`) and rendered 224×224 RGB with 2,965 unique colors; gait remained unverified. Pinned dimOS aliases Go2 selection to Go1 and has no Go2 policy case. Inspected Unitree RL Gym revision ships no Go2 `motion.pt`.
- Evidence path: `artifacts/runs/20260919T191000Z-go2-project-env-render/` and `artifacts/runs/20260919T191500Z-person-a-preflight/`
- Status: FAILED (G0 BLOCKED; model/render sub-check passed)
- Known confounds: macOS requires `mjpython` for rendering; no NVIDIA/CUDA device; no dimOS install; only about 18.6 GB free during final preflight; required task scene/front marker not implemented
- Decision / next discriminating test: obtain or train a compatible true-Go2 gait, then execute the complete G0 motion protocol. Do not substitute the dimOS Go1 surrogate for Go2 evidence.
- Next owner and deadline: Person A; before G1 or any real-data collection

### 2026-09-19T20:21:42Z — Go2 controller source audit — `20260919T202600Z-go2-controller-source-audit`

- Owner: Person A / SIM
- Source revision / dirty: project `NO_COMMIT`; Unitree RL MjLab `1425b15f73bd4095f0df53709d7c389c3eb9e790`
- Config and bundle IDs: `Unitree-Go2-Flat`; training XML SHA-256 `077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912`
- Dataset/split IDs: none
- Command: `scripts/audit_go2_controller_source.py` against the pinned checkout
- Expected criterion: exact source pin, XML, 5 ms physics step, four-step decimation, 20 ms deployment step, 0.25 joint-position action scale, ONNX exporter, and low-command publisher guard are present
- Observed result: all audited declarations and eight source-file hashes passed; a 0.5-second project block maps to 25 policy calls and 100 physics steps
- Evidence path: `artifacts/runs/20260919T202600Z-go2-controller-source-audit/`
- Status: VERIFIED for source compatibility only; G0 remains BLOCKED
- Known confounds: no training GPU on this host, no checkpoint, no ONNX playback, no locomotion trials
- Decision / next discriminating test: train the two-iteration smoke on a conforming Linux/NVIDIA host, then inspect its checkpoint and export before starting the full run
- Next owner and deadline: Person A; next active G0 step

### 2026-09-19T20:29:35Z — Local training-host gate — `20260919T205000Z-local-training-host-preflight`

- Owner: Person A / SIM
- Source revision / dirty: project `NO_COMMIT`; dirty initial scaffold
- Config and bundle IDs: Go2 training-host preflight v1
- Dataset/split IDs: none
- Command: `scripts/check_go2_training_host.py`
- Expected criterion: Linux, Python 3.11, visible NVIDIA GPU, and driver 550 or newer
- Observed result: FAIL — macOS arm64, Python 3.12.4, no `nvidia-smi`, approximately 18.9 GiB free
- Evidence path: `artifacts/runs/20260919T205000Z-local-training-host-preflight/`
- Status: FAILED as expected; this machine is integration-only
- Known confounds: Ubuntu and GPU-specific dependency installation cannot be tested locally
- Decision / next discriminating test: rerun the same preflight on the supplied Linux/NVIDIA host; proceed only on PASS
- Next owner and deadline: Person A; immediately after training-host access is supplied

### 2026-09-20 — G0 formal technical trials — `20260920T073000Z-g0-formal-mjlab`

- Owner: Person A / SIM execution; second human owner sign-off pending
- Source revision / dirty: `756cb3c0547da3568ddf4ec93440d00ba3664c33`; clean sparse checkout
- Config and bundle IDs: `configs/g0_acceptance.toml` SHA-256 `786692931733ff71016db323ee5eb89b79dadda1aa0246e844046b172ccfdf04`; MjLab policy SHA-256 `c1f66f9b85aa1ea4231c0025fe90d8d4636288657ccee6a490fde274edcaa9bc`
- Dataset/split IDs: none
- Command: clean-source `scripts/run_g0_trials.py --controller mjlab ... --acceptance configs/g0_acceptance.toml`
- Expected criterion: all six frozen technical checks true; artifact must remain pending sign-off unless both owners acknowledge
- Observed result: PASS — 5/5 movable pushes (1.2292–1.6029 m), 5/5 resistant holds (0.0327–0.0448 m), zero falls, 10/10 settled resets, 2,696 camera/goal-region points checked with zero outside, 50.65 blocks/wall-second
- Evidence path: `artifacts/runs/20260920T073000Z-g0-formal-mjlab/`
- Status: `FORMAL_TRIALS_PASS_PENDING_COOWNER_SIGNOFF`
- Known confounds: resistant-box worst case is only 5.2 mm below the accepted limit; goal coverage is geometric because the public goal input is not rendered into model pixels
- Decision / next discriminating test: second owner reviews the exact thresholds and retained images, then acknowledges or rejects without editing this run
- Next owner and deadline: second human owner; before bulk collection is accepted

### 2026-09-20 — G1A direct-MjLab data contract — `20260920T073500Z-g1a-mjlab`

- Owner: Person A / SIM producer; Person B and second human visual sign-off pending
- Source revision / dirty: `756cb3c0547da3568ddf4ec93440d00ba3664c33`; clean
- Config and bundle IDs: controller handoff `20260920T045422Z-go2-controller-handoff`; scene `go2wm-push-detour-draft-v0-7b54a57edcf0`; camera `overhead_v1`
- Dataset/split IDs: `splits-7ae472b8826f`; role seeds alignment 3, push 5, resist 18; test seeds 377–400 reserved
- Command: `scripts/collect_g1_packet.py` through `mjpython`, followed by independent `scripts/verify_g1_packet.py`
- Expected criterion: three exact 20-block episodes; 0.5 seconds and 100×5 ms samples per block; requested/applied equality; no fall/OOB; push/resist contact; transient contact aggregation; strict reload, montage, and outer checksums
- Observed result: PASS for direct MjLab — all thirteen verifier checks true; push has 11 contact blocks, resist has 12, and five transient-contact examples were retained
- Evidence path: `artifacts/runs/20260920T073500Z-g1a-mjlab/`
- Status: VERIFIED for direct-MjLab G1A only; full G1 not verified
- Known confounds: dimOS is explicitly `pending_g5_l7`; human montage sign-off has not been recorded
- Decision / next discriminating test: review the ten-block montage, obtain both approvals, then use the copied split for a canary and fresh MjLab wave on non-iCloud storage
- Next owner and deadline: both owners; before training on collected Go2 data

### 2026-09-20 — Pinned LeWM local compatibility — `20260920T072900Z-lewm-local-verify`

- Owner: Person B / ML
- Source revision / dirty: `85d9bca7fcb5044bd03c903634e58113c233d576`; clean sparse checkout
- Config and bundle IDs: le-wm `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`; Python 3.10.19; torch 2.14.0; stable-worldmodel 0.1.1
- Dataset/split IDs: synthetic test fixtures only; no Go2 dataset
- Command: isolated `/private/tmp` environment, `pytest -q tests/test_lewm_torch.py tests/test_lewm_cache.py` with `GO2WM_LEWM_REPO` pinned
- Expected criterion: upstream architecture/API loads; exact loss and rollout parity; cache, train, checkpoint, resume, evaluation, and bundle reload tests pass
- Observed result: 17 passed, 0 skipped in 39.32 seconds; MPS built and available outside the sandbox
- Evidence path: `artifacts/runs/20260920T072900Z-lewm-local-verify/`
- Status: VERIFIED for pinned local software compatibility only
- Known confounds: tiny synthetic end-to-end fixture emits numerical readout warnings; no Go2 data or CUDA benchmark was used
- Decision / next discriminating test: strict-load a fresh MjLab wave, run baseline/cache and a two-batch overfit before any paid main training
- Next owner and deadline: Person B after fresh data acceptance

### 2026-09-20 — Direct-MjLab wave 120 G2/G3 — `20260920T084850Z-g2g3-mjlab-wave120`

- Owner: Person B / ML
- Source revision / dirty: plain run from clean `f3463e6`; auxiliary run from clean `343e7b4`
- Config and bundle IDs: baseline `go2wm-bb43d84ca9973ca9`; plain LeWM `go2wm-cf7d5fc5e1b30062`; auxiliary diagnostic `go2wm-6f4a5b1da045ddb0`
- Dataset/split IDs: 92 training and 19 validation episodes from the frozen `splits-7ae472b8826f`; reserved test seeds were not accessed
- Command: strict audit and lossless caches, pooled-pixel baseline, 1,150-step plain LeWM training/evaluation, then one three-epoch training-only state-auxiliary diagnostic
- Expected criterion: real-latent readouts beat constant mean, learned one-step prediction beats persistence, matched actions beat shuffled actions, and all rollouts remain finite
- Observed result: data checks passed; plain G2 passed marginally (robot 1.281 m vs 1.307 m mean, object 1.086 m vs 1.182 m), but G3 failed (robot one-step 1.236 m vs 0.082 m persistence and matched-vs-shuffled +0.022%). The auxiliary diagnostic worsened robot readout to 1.302 m and action dependence to -0.99%; it was rejected.
- Evidence path: `artifacts/runs/20260920T084850Z-g2g3-mjlab-wave120/`
- Status: G3 failed; fallback F3 is active; no learned controller is approved for execution
- Known confounds: full G1 and both-owner sign-off remain open; raw data, caches, and weight files remain on local non-iCloud storage and must be copied to durable external storage
- Decision / next discriminating test: run only the predeclared short multistep predictor fine-tune (P-010); if it fails, freeze model work and complete the F3 dataset/instrumentation delivery
- Next owner and deadline: Person B for P-010; both owners before any learned-control claim

### 2026-09-20 — Final P-010 multistep diagnostic — `20260920T093200Z-g3-multistep-final`

- Owner: Person B / ML
- Source revision / dirty: clean `3adf69e153eff1247beb9c1a829da4ef4d4ac5fc`
- Config and bundle IDs: initialized from plain SHA-256 `269500460733838bfb7c94e0c52b36dad46e1b3b9400ff9bd0f8c15c0f951423`; evidence-only bundle `go2wm-b9a94a218341bf98`
- Dataset/split IDs: `mjlab-wave120-20260920T081500Z`; `splits-7ae472b8826f`; 92 train/19 validation episodes; test untouched
- Command: six-step recursive latent loss, frozen encoder/projector, predictor-side components only, 300 MPS updates followed by unchanged `lewm-eval`
- Expected criterion: beat persistence in state space, matched actions beat shuffled by at least 15%, and all six-step rollouts remain finite
- Observed result: latent MSE improved 0.029653 to 0.023526 but remained above copy-last 0.008428; action gap +0.095%. Robot one-step was 1.243 m versus 0.082 m persistence; matched six-block robot error was 1.336 m versus 1.332 m shuffled; zero nonfinite rollouts.
- Evidence path: `artifacts/runs/20260920T093200Z-g3-multistep-final/`
- Status: COMPLETE; G3 FAILED; F3 FROZEN; learned bundle prohibited from motion
- Known confounds: full G1 and both-owner sign-off remain open; large checkpoints/caches/raw data remain local outside Git
- Decision / next discriminating test: none before release; stop model work under the hard cutoff and preserve the negative result
- Next owner and deadline: both human owners acknowledge fallback F3 and copy large artifacts to durable storage before reboot

## 4. G0 simulator feasibility procedure

Owner: SIM. Witness: ML for at least one complete pass.

1. Start the simulator with the intended Go2 asset, gait/controller, scene, and overhead camera.
2. Run zero command for 10 simulated seconds; record drift, contacts, and fall state.
3. Run safe low-amplitude forward, left-yaw, and right-yaw commands.
4. Increase within controller-safe ranges only after observing stable response.
5. Freeze provisional action limits from measured stable behavior.
6. Place the movable box and perform at least five intended pushes across representative approach angles.
7. Repeat for the resistant box using identical geometry and documented physical difference.
8. Measure box displacement from labels and visually inspect video.
9. Sample all intended arena start/goal/box extrema; verify camera coverage.
10. Execute at least ten settled resets of one scenario and report initial-state variance.
11. Measure wall-clock throughput for at least 100 production-path blocks if possible.
12. Write a G0 summary and choose: pass, 45-minute repair, fixed-obstacle fallback, or free-motion fallback.

Safety/quality stop conditions:

- recurring fall at nominal commands;
- box launches, penetrates, or behaves numerically unstably;
- required entities leave camera frame;
- reset produces task-significant variation;
- actual command units/timing cannot be established.

Do not begin a long collection run while any of these is unresolved.

## 5. G1 timing and data-contract procedure

Owners: SIM produces, ML consumes. Both sign.

1. Create an episode with at least 20 blocks and a conspicuous sequence such as stop → forward → stop → left turn → stop.
2. Preserve high-frequency physics/event trace for this episode.
3. Validate for every block:
   - start timestamp and image;
   - requested versus applied action;
   - 0.5-second duration and physics-step count;
   - end timestamp and image;
   - boundary poses;
   - contact/fall aggregate.
4. Build a contact example where a brief event does not persist to the endpoint.
5. Produce a montage containing start/end frames, action, robot pose delta, both box deltas, and event flags.
6. SIM inspects all metadata and ten visual examples.
7. ML loads the packet through the production dataset loader, independently checks indexes/times/shapes, and inspects ten examples.
8. Send a short packet through the dimOS collection path and compare fields/configuration to direct collection.
9. Freeze schema version, temporal convention, camera config, coordinate convention, action bounds, and data acceptance rules.
10. Checksum the packet and attach the two-person sign-off.

If any sample is shifted by one block, stop. Repair the producer/consumer contract and regenerate the entire smoke packet.

## 6. Dataset collection procedure

Owner: SIM. ML reviews coverage after each wave.

### Before a wave

1. Select only training seeds from the frozen split manifest.
2. Name the wave and create a manifest before collection.
3. Declare desired scenario/behavior counts.
4. Verify free disk space and write a tiny canary episode.
5. Run integrity checks on the canary before scaling.

### During a wave

1. Log rejected resets/episodes with reasons.
2. Monitor rate of invalid images, timing mismatches, falls, and serialization errors.
3. Sample visual montages periodically.
4. Track interaction blocks, but distinguish approach, brief contact, displacement, and prolonged stall.
5. Stop the wave if camera/schema/controller versions change.

### After a wave

1. Close the manifest atomically and compute checksums.
2. Run the full integrity validator.
3. Generate coverage by action, scenario, box, contact, displacement, and termination.
4. Quarantine invalid episodes; never silently edit records in place.
5. ML identifies the highest-value coverage gaps.
6. Add only accepted episodes to a new immutable dataset ID.
7. Record blocks/hour and storage/compute cost.

Do not move validation or test episodes into training because a class is sparse. Collect new training seeds instead.

## 7. Training procedure

Owner: ML. SIM keeps collection compatible with the frozen schema.

### Preflight

1. Select immutable dataset and split IDs.
2. Run leak, integrity, and coverage checks.
3. Record upstream LeWM revision/license, local source revision, environment, config, seed, device, and expected resource use.
4. Fit preprocessing/target-normalization statistics on training data only.
5. Run a tiny overfit test on a few batches; verify finite loss and checkpoint reload.
6. Run one forward and six-step autoregressive shape/range test.

### Main run

1. Start with the published/self-supervised objective adapted only for the explicit action/history contract.
2. Save periodic checkpoints and optimizer/config state.
3. Log training and validation curves, throughput, device memory, and wall time.
4. Evaluate early checkpoints rather than waiting for a promised duration.
5. Fit linear readouts without updating encoder/predictor.
6. Evaluate readouts on encoded real latents and predicted latents by horizon.
7. Run persistence, kinematic/contact, and action-shuffle comparisons.
8. Select checkpoints on validation metrics only.

### Escalation order when learning fails

1. Re-audit alignment, coordinate units, preprocessing, and split integrity.
2. Inspect coverage and visibility.
3. Verify train/eval mode and bundle component compatibility.
4. Add targeted training data if a clear gap exists.
5. Add a small readout MLP if the representation is usable but not linearly decoded.
6. Add short multistep predictor loss if one-step quality is adequate but rollout drift dominates.
7. Add a modest supervised auxiliary pose/object loss only after recording the architectural change.

Change one major component at a time and keep the comparison run.

## 8. Candidate-library and planner procedure

Owner: ML defines; SIM executes oracle checks.

1. Publish a table of exactly 64 candidate IDs with family, six commands, and bounds.
2. Required families: approach/push, left detour, right detour, turning adjustment, reduced-speed/stop.
3. Unit-test count, duration, bounds, serialization, and deterministic generation.
4. Select representative validation starts.
5. SIM executes candidates in separate settled resets, with at least three repeats for representative family comparisons.
6. Label scenarios where the library contains no useful action as library-infeasible.
7. ML verifies score decomposition on synthetic trajectories.
8. Tune score weights on validation outcomes only.
9. Freeze candidate library and weights at G6.

Oracle outcomes are diagnostic artifacts. They must never be supplied to `plan_to` during deployment.

## 9. Integration procedure

Owners: SIM integrates; ML publishes bundle and reference outputs.

1. ML publishes an immutable bundle plus a tiny reference input/output packet.
2. SIM verifies bundle hash, schema compatibility, shapes, numeric ranges, and reference output tolerance.
3. Integrate `imagine` without enabling motion.
4. Verify that predictions serialize and UI overlays use the declared coordinate projection.
5. Integrate `plan_to`; log all candidate IDs, score terms, chosen candidate, first applied action, and timing.
6. Ensure only the first 0.5-second block is executed before a fresh observation/replan.
7. Integrate `stop` and verify it preempts queued commands.
8. Test invalid inputs, missing/stale bundle, nonfinite predictions, timeout, and process failure; all must fail closed to stop/no motion.
9. Run 20 consecutive planning cycles through dimOS.
10. Restart each involved process once and repeat a smoke scenario from documented commands.

No competing navigation publisher may remain active.

## 10. Surprise calibration and behavior-change test

Owner: ML calibrates; SIM runs trials.

1. Define exactly which previously predicted next latent is compared to which newly encoded observation.
2. Normalize residuals using training/validation-normal statistics only.
3. Include free motion, stops, turns, pushes, and ordinary contacts in calibration.
4. Choose threshold against the predeclared normal false-stop target.
5. Freeze threshold before any behavior-change test.
6. Run normal held-out episodes and behavior-change episodes with fresh histories.
7. On threshold breach, record residual, detection timestamp, stop issue/acknowledgment, and subsequent motion.
8. Report false stops, detection rate, and delay, including misses.

If surprise is unreliable, remove it from the live critical path and present it as an offline result. Do not tune the threshold on the anomaly demo.

## 11. Release freeze and final evaluation

At G6:

1. Stop feature development.
2. Freeze and checksum source revision, environment, scene/camera/action configs, dataset/splits, bundle, candidate library, planner weights, surprise threshold, metric definitions, seed list, and rerun/exclusion rules.
3. Archive a clean smoke-run artifact.
4. Confirm test seeds have never been inspected for performance or used for selection.
5. Assign SIM as test operator and ML as protocol observer/analyst, or swap if that preserves blindness better.

For each test episode:

1. Allocate the next predeclared scenario/seed; do not cherry-pick order.
2. Capture actual settled state and fresh observation history.
3. Record the prediction/score packet before movement.
4. Lock the packet and execute through production path.
5. Record every block until declared termination.
6. Assign outcome using frozen rules.
7. If infrastructure fails, preserve the failed artifact; rerun only when the frozen rule permits it.
8. Never rerun an ordinary behavioral failure to replace it.

After all episodes, run analysis once against raw logs. Corrections create a new analysis run, never rewritten output.

## 12. Demo-day runbook

### T minus 60 minutes

- Verify power/network/GPU access and free disk.
- Confirm frozen bundle/config hashes.
- Run doctor, unit tests, and one smoke scenario—not a final test scenario.
- Open UI and confirm camera/overlay alignment.
- Check one-motion-publisher condition and physical pause/block mode.
- Preload model if warm-up is part of the declared latency protocol.
- Locate backup video locally and verify playback.

### T minus 15 minutes

- Reset processes using the documented order.
- Start a new demo run ID and logging.
- Confirm goal choices are within supported task distribution.
- Display current bundle ID and active fallback level.

### Live sequence

1. Explain the constrained learned appearance/behavior association.
2. Let the judge choose a supported goal/layout.
3. Call `plan_to`; show multiple robot and box futures and scores.
4. Lock predictions visibly before execution.
5. Execute/replan and overlay actual outcomes.
6. Run a matched situation with the other box behavior.
7. If G6-verified, run the controlled behavior-change surprise/stop test.
8. Show held-out raw counts, latency, baseline, and one failure case.

### Recovery rules

- UI-only failure: preserve controller log, restart UI, or use recorded evidence.
- Model-service failure: issue stop, restart once, verify bundle hash, resume with a new demo run ID.
- Simulator instability: stop, preserve failed run, use one predeclared reset attempt; then use backup video.
- GPU unavailable: use a previously verified lower-performance path only if it exists; otherwise show backup recording and evidence.
- Never swap to an unverified checkpoint or silently change physics/configuration live.

### Shutdown

- Issue stop and confirm no motion command remains queued.
- Close manifests, flush logs, compute checksums, and copy evidence to the backup destination.
- Record whether the demo was live, partially recovered, or recorded.

## 13. Daily/end-of-shift checklist

- [ ] All active runs have manifests and terminal status.
- [ ] Checkpoints/data/evidence are checksummed and copied.
- [ ] Current verified/failed assumptions updated in the ledger.
- [ ] Interface changes recorded as decisions.
- [ ] No test data entered training/tuning paths.
- [ ] Next gate, blocker, owner, and deadline are explicit.
- [ ] Active fallback level is recorded.
- [ ] Working live command and latest known-good bundle are documented.
