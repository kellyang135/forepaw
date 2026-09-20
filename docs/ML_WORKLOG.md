# Person B (Learning & Control) worklog

Owner: Person B / ML. This file tracks the ML work items from
[TWO_PERSON_PLAN.md](TWO_PERSON_PLAN.md) §4. It is a status log, not
evidence: nothing below is VERIFIED until a real-path run artifact exists
(see [VERIFICATION.md](VERIFICATION.md)). The LeWM training procedure is in
[LEWM_TRAINING.md](LEWM_TRAINING.md).

## Status by work item

| # | Work item | Code | Status |
| --- | --- | --- | --- |
| 1 | Data-contract consumer (strict loader, rejects) | `learning/dataset.py` | PROPOSED: unit-tested on fake data; SIM upgraded it to schema v2; awaiting the G1 packet |
| 1 | Episode-level splits frozen before fitting | `learning/splits.py` | PROPOSED |
| 2 | Coverage diagnostics and leak check | `learning/diagnostics.py`, `splits.check_leakage` | PROPOSED |
| 3 | LeWM adaptation (2-D action, 3 frames, 0.5 s, 192-D) | `learning/lewm_model.py`, `lewm_adapter.py` | VERIFIED for local software compatibility; Go2 prediction quality remains unverified |
| 3 | Lossless split-pure training caches | `learning/lewm_cache.py` | PROPOSED |
| 3 | LeWM training loop (MPS/CUDA/CPU) | `learning/lewm_train.py` | PROPOSED: runs end to end on fake data; resume tested |
| 3 | Optional training-only state auxiliary | `learning/lewm_train.py` | SOFTWARE-VERIFIED; first real diagnostic rejected because robot readout and action conditioning did not improve |
| 3 | Six-step predictor-only fine-tune | `learning/lewm_train.py` | COMPLETE/REJECTED: latent loss improved, but persistence and shuffled-action G3 controls failed; F3 frozen |
| 4 | Linear readouts on real and predicted latents | `learning/readouts.py` | PROPOSED: ridge penalty chosen by episode-grouped CV on train only |
| 5 | Horizon metrics, free vs interaction, shuffled actions, baselines | `learning/prediction_eval.py`, `report.py` | PROPOSED |
| 6 | Planner (64 candidates, scoring) | `planning/` (pre-existing) | PROPOSED; weights still defaults, not validation-tuned |
| 7 | Surprise calibration | `learning/surprise_fit.py` + `runtime/surprise.py` | PROPOSED |
| 8 | Bundle publication (ML -> SIM handoff) | `learning/bundle_io.py` | PROPOSED: LeWM weights inside the bundle, hashed into the bundle id |
| - | Linear latent baseline | `learning/baseline_model.py` | PROPOSED: the floor LeWM must beat |

## Evidence against upstream LeWM (software, not task performance)

`tests/test_lewm_torch.py`, run with torch 2.14.0 and le-wm `8edfeb3`:

- architecture: 18.03M parameters, action encoder input 2 (upstream config with our contract);
- `lejepa_losses` equals upstream `train.py::lejepa_forward` on the same batch
  (pred, SIGReg, and total loss identical to float precision);
- `LeWMWorldModel.rollout_batch` equals upstream `JEPA.rollout` for two
  6-block candidates (max abs diff below 1e-4), confirming D-022;
- batched and per-frame encoding, and batched and per-candidate rollouts, agree;
- cache -> train -> resume -> `lewm-eval` -> bundle reload works end to end.

Clean-source verification on 2026-09-20 used Python 3.10.19 and project
revision `85d9bca`: 17 tests passed with no skips. The retained environment,
logs, failure-preserving rerun, package inventory, and checksums are in
`artifacts/runs/20260920T072900Z-lewm-local-verify/`. This verifies the
software integration only; it does not establish Go2 prediction quality.

## Commands

```bash
python -m go2wm.learning check-data  --data data/<wave> --splits artifacts/splits.json
python -m go2wm.learning make-splits --seeds 1-400 --reserve-test <ids> --salt <salt> --out artifacts/splits.json
python -m go2wm.learning baseline    --data data/<wave> --splits artifacts/splits.json --out runs/<id>/eval-baseline --dataset-id <id>
python -m go2wm.learning lewm-cache  --data data/<wave> --splits artifacts/splits.json --out-dir runs/<id> --dataset-id <id>
python -m go2wm.learning.lewm_train  --train-cache runs/<id>/cache-train --val-cache runs/<id>/cache-validation --out runs/<id>/lewm
python -m go2wm.learning lewm-eval   --run runs/<id>/lewm --data data/<wave> --splits artifacts/splits.json --out runs/<id>/eval-lewm --dataset-id <id> --device mps
python -m go2wm.learning compare     runs/<id>/eval-baseline/ml_report.json runs/<id>/eval-lewm/ml_report.json
python -m go2wm.learning fake-data   --out runs/rehearsal --episodes 60 --image-size 224   # rehearsal only
```

## Compute measurement: Apple M4, MPS (2026-09-19)

Run by Person B in `le-wm` @ `8edfeb3`, torch 2.14.0, device `mps`, random
tensors, our exact contract (224x224, 3 history frames + 1 target, 2-D
actions, 192-D latents, 18.0M params, SIGReg on). Median of 5 timed steps.

| Batch | ms/step | clips/s |
| --- | --- | --- |
| 16 | 576 | 27.8 |
| 32 | 1127 | 28.4 |
| 64 | 2483 | 25.8 |

Planning, 64 candidates x 6 blocks, model only: 34 ms (2 runs; not p50/p95).
Decision: train on the Mac. About 6 min per epoch per 10k clips.

## Fake-data results (software-contract evidence only)

32 x 32 fake smoke (`python -m go2wm.learning smoke`, 60 episodes), baseline
with grouped-CV penalties: readout beats the constant mean (robot 0.21 m vs
0.67 m; boxes 0.20 m vs 0.68 m), so G2 readout checks pass; G3 fails (one-step
robot 0.23 m vs 0.02 m persistence; matched vs shuffled commands no different).

### LeWM rehearsal on fake data

Software rehearsal only (fake scenes, `rehearsal_only` everywhere; see
GO2_CONTROLLER_RECOVERY.md section 2). Cloud CPU, 2 cores, batch 16, about
4.5 clips/s (the Mac measured about 28 clips/s).

- 40 episodes at 224 x 224, 492 train / 80 validation clips, 20 epochs
  (600 steps, 41 min): train loss 0.53 -> 0.12, validation prediction loss
  0.118 -> 0.019 with precise BN, embeddings did not collapse (per-dim std
  about 0.46).
- The model did **not** learn to use the commands: validation loss with
  shuffled commands was within 1% of the real ones at every epoch, and
  `lewm-eval` G3 matched-vs-shuffled improvement was -0.1%. Linear readouts
  of its latents were no better than the constant mean (robot 0.68 m vs
  0.69 m). G2 object and all G3 prediction checks failed.
- Same result on fake data with a new random command every block (5 epochs).
- **Known-answer control** (`python -m go2wm.learning synthetic-control`): a
  28-pixel square moves by exactly 40 px x the command, fresh random command
  every block. Default training settings, 540 train clips (60 episodes), CPU. After 14 epochs (about
  460 steps) validation prediction loss was 0.24 vs 0.43 for "copy the last
  latent", so the predictor learns change; but shuffled commands raised the
  loss by only 1-2%. The trained predictor's output does move with the
  commands (23% relative change under shuffling), it just has not yet learned
  the right mapping. Upstream AdaLN-zero conditioning starts with exactly zero
  action influence (checked by test), so action use must be learned, and a
  few hundred CPU steps did not get there.

**Cloud CPU run of the default control (2026-09-20): PASS.**
1,080 train / 180 validation clips, batch 16, lr 5e-5, resumed after container
restarts from `state_last.pt`. Gap between shuffled and real commands by epoch:
+0.6%, +1.5%, +3.1%, +3.8%, +5.8%, +5.0%, then **+30.9% at epoch 7** and
**+59.5% at epoch 8** (val pred 0.059 vs shuffled 0.145; copy-last 0.123), and
**+67.5% at epoch 9** (val pred 0.045 vs shuffled 0.139; copy-last 0.108); the
AdaLN action-pathway norm grew every epoch (0.71 to 3.16). The unchanged
training path (default lr 5e-5) does learn action-conditioned dynamics once it
has had about 500 steps; the earlier near-zero gaps were too little training,
not a pipeline fault. Verdict against the rule set before the run (gap in the
tens of percent for several epochs, prediction well below copy-last, action
pathway growing): **PASS** at epoch 9, 603 steps. The run was stopped there;
checkpoints are cloud-only rehearsal artifacts and are not evidence about the
Go2 task. Real data still has to show its own gap at G3.

**Open question, to answer on the Mac before real data:** run the synthetic
control for about 2,000 steps (30 epochs of the default 1,080 clips, roughly
20-30 min on MPS). If the gap becomes
clearly positive (tens of percent) and prediction loss drops well below
copy-last, the training path can learn action-conditioned dynamics and the
remaining risk is data. If it stays near zero, the problem is in training
length, learning rate, or the pipeline, and must be solved before G3. On real
data, a gap near zero after a full run means G3's shuffled-action check fails
and the plan's fallback applies.

## Findings while building the training path

1. **BatchNorm lag made validation meaningless mid-training.** LeWM's two MLP
   projectors use BatchNorm. With the upstream loop, eval-mode validation loss
   on fake data went 0.53, 3.5, 42.2, 0.21 over four epochs while the training
   loss fell smoothly: the running statistics lag the weights. Planning also
   runs in eval mode. `lewm_train` now recomputes BatchNorm statistics over 20
   training batches before each validation and checkpoint ("precise BN"); the
   same run then gave 0.106, 0.060, 0.060, 0.061. Weights and optimizer are
   unaffected. `--bn-recalibration-batches 0` restores upstream behavior.
2. **Readouts overfit badly with a fixed ridge penalty.** Train error 0.03 m vs
   validation 0.25 m for the baseline. Penalties are now chosen by
   episode-grouped 5-fold CV on training episodes only, and the latent
   standardizer no longer divides by near-zero column spreads (a column that
   never varied in training could turn tiny drift into kilometre-scale errors).
3. **The baseline dynamics diverge at 3 s.** Its 6-block rollout can explode on
   224 x 224 data; its penalty is now chosen on 6-step rollout error, not one
   step. This is the baseline's weakness, reported, not hidden.
4. **Memory.** On CPU, batch 16 needs about 4 GB in the main process; each
   DataLoader worker imports torch (hundreds of MB). Validation now uses no
   workers.

## Decisions to raise with SIM (not yet accepted)

Numbered after SIM's D-017 to D-019 in DECISIONS.md; D-020 is the gait
proposal in [GAIT_OPTIONS.md](GAIT_OPTIONS.md).


- **D-021 (proposed): NumPy is an ML-only dependency.**
- **D-022 (proposed): LeWM action alignment.** `action[i]` is the command
  applied from frame `i`; the first predicted latent consumes the two history
  commands plus candidate command 0. Now checked against upstream
  `JEPA.rollout` by test, not just by reading.
- **D-023 (proposed): action normalization belongs to the bundle.**
- **D-024 (proposed): LeWM validation uses the frozen episode split** and
  lossless per-split caches, not upstream's random window split and JPEG
  datasets.
- **D-025 (proposed): fp32 training and precise-BN before every checkpoint.**
- **D-026 (proposed): published LeWM bundles carry their weights** (about
  72 MB) and accept a 2e-3 reference-replay tolerance across devices.

## Fixes to shared files (flag to SIM)

- `src/go2wm/config.py`: `tomllib` fallback to `tomli` on Python 3.10.
- `src/go2wm/data/collector.py`: `from datetime import UTC` exists only on
  Python 3.11+, so the collector (and everything in `go2wm.learning` that
  imports it) failed to import in the Python 3.10 LeWM environment. Replaced
  with `timezone.utc`; behavior is unchanged. Several Person A scripts
  (`scripts/*.py`) still use `datetime.UTC`; they run in the 3.12 project
  environment, but `pyproject.toml` declares `>=3.10.12`, so SIM should decide.
- `src/go2wm/model/components.py`: uses a predictor's `rollout_batch` when it
  has one (64 candidates in one forward pass per step). Output is unchanged;
  a test checks it.

## Blockers outside ML

- **G0 is blocked on a true-Go2 gait controller** (see
  [PERSON_A_EXECUTION.md](PERSON_A_EXECUTION.md)). No real training data can
  exist until a Go2 controller walks and G1 passes. Person B found and probed a
  pretrained Go2 policy that walks and pushes on the Menagerie Go2 on CPU;
  proposal and measurements in [GAIT_OPTIONS.md](GAIT_OPTIONS.md) (D-020,
  pending SIM). Person B also drafted a MuJoCo `SimulatorAdapter` and task
  scene that take either gait (rl_sar or SIM's MjLab policy) for SIM review:
  [SIM_ADAPTER_DRAFT.md](SIM_ADAPTER_DRAFT.md).
- **The repo lives in iCloud-synced `~/Documents` with Optimize Mac Storage.**
  Files were being evicted mid-session. Move the repo, datasets, and runs off
  iCloud before collecting data.

## Next ML steps, in gate order

1. Mac: run `tests/test_lewm_torch.py`, then the fake-data rehearsal in
   `LEWM_TRAINING.md` §3 on MPS; record epoch time and plan latency.
2. G1: `check-data` on SIM's 20-block smoke packet; inspect 10 blocks; sign.
3. Freeze `artifacts/splits.json` with the 24 evaluation seeds reserved.
4. G2/G3: baseline, then LeWM (overfit preflight, main run, `lewm-eval`,
   `compare`).
