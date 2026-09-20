# MjLab checkpoint playback acceptance (predeclared)

Status: **PROPOSED controller-validation procedure**, not a G0 sign-off. These
criteria were written before running the trained checkpoint through upstream
MjLab. They do not change the fixed experiment contract.

## Purpose

The training packet proves that a checkpoint and ONNX export were produced. It
does not yet prove that the learned controller can stand, walk, or turn in the
upstream simulator. `scripts/run_mjlab_checkpoint_playback.py` closes that gap
using the pinned `unitree_rl_mjlab` checkout and its native checkpoint loader.

The run is deliberately deterministic and nominal: one flat-terrain Go2,
observation corruption disabled, no pushes or domain randomization, zero reset
pose/velocity ranges, lateral command fixed to zero, and command resampling
disabled for the duration of a trial. Illegal-contact termination remains
enabled so a fall cannot be hidden.

## Pinned inputs

- Repository: `https://github.com/unitreerobotics/unitree_rl_mjlab`
- Revision: `1425b15f73bd4095f0df53709d7c389c3eb9e790`
- Task: `Unitree-Go2-Flat`
- Checkpoint SHA-256:
  `2dc44f0c9cddf488c008d513cd7a41745c65aedac0e781491304f13f43fa5dc4`
- MjLab: `1.2.0` (as pinned by the upstream checkout)

The harness refuses a wrong checkout revision or checkpoint digest and records
the relevant source hashes, package versions, device, seed, raw trajectories,
report, and checksums.

## Trial matrix

Five resets per command:

| Case | Command `[vx, vy, wz]` | Duration | Scored after |
| --- | --- | ---: | ---: |
| stand | `[0, 0, 0]` | 10 s | 1 s |
| forward | `[0.2, 0, 0]` | 4 s | 1 s |
| forward | `[0.4, 0, 0]` | 4 s | 1 s |
| forward | `[0.6, 0, 0]` | 4 s | 1 s |
| turn | `[0, 0, -0.96]` | 4 s | 1 s |
| turn | `[0, 0, +0.96]` | 4 s | 1 s |

Each trial stores every 20 ms sample. A run fails immediately if actions or
state become non-finite, if upstream MjLab terminates it, if base height falls
below 0.15 m, or if base tilt exceeds 60 degrees.

## Acceptance thresholds

All five repeats of all six cases must avoid a fall/non-finite result.

- Stand: median measured `|vx| <= 0.10 m/s`, `|wz| <= 0.15 rad/s`, and net
  displacement `<= 0.50 m` over 10 seconds.
- Forward: median measured `vx` must be within `0.15 m/s` of its command and
  median `|wz| <= 0.15 rad/s`.
- Turn: median measured `wz` must be within `0.18 rad/s` of its command and
  median `|vx| <= 0.15 m/s`.

These are controller-feasibility tolerances, not final action bounds (P-001).
They are intentionally looser than the adapter results so small MjLab-versus-
MuJoCo version differences do not create a false failure.

## Cross-simulator decision rule

After upstream checkpoint playback passes, rerun the same command matrix using
the exported ONNX policy in the project adapter. For each nonzero command,
compare the median scored velocity:

- absolute forward-speed difference `<= 0.08 m/s`;
- absolute yaw-rate difference `<= 0.12 rad/s`.

Any larger gap blocks G0 sign-off and triggers a physics/observation audit. It
must not be explained away by widening the threshold after seeing the result.

## Truthfulness boundary

A PASS here means only: the retained checkpoint ran through the pinned
upstream MjLab path and met the predeclared nominal command-response checks. It
does not verify the project adapter, ONNX deployment path, boxes, camera,
dimOS, G0, G1, or task performance.

## Recorded result (after criteria were frozen)

- Upstream run `20260920T044245Z-mjlab-checkpoint-playback`: **PASS**, all 12
  checks true and all 30 trials safe.
- Adapter run `20260920T045030Z-mjlab-adapter-playback`: **PASS**, same command
  matrix and repeat count.
- Comparison `20260920T045143Z-mjlab-playback-comparison`: **PASS**. Maximum
  forward-speed difference was 0.0052 m/s and maximum yaw-rate difference was
  0.0017 rad/s.

These results validate checkpoint playback and ONNX adapter equivalence within
the criteria above. They do not expand the truthfulness boundary.
