# Person A Execution Status

This is the audited status of the Simulation & Integration work through the 2026-09-20 clean-source G0/G1A runs. It distinguishes measured technical passes from the human sign-offs and deployment evidence still required.

## Outcome

**The formal G0 technical trial suite passed; formal sign-off is pending the second owner.** The trained MjLab Go2 checkpoint and ONNX adapter passed playback and cross-simulator comparison. From clean revision `756cb3c`, the formal suite recorded 5/5 movable-box pushes, 5/5 resistant-box holds, no falls, 10/10 settled resets, and zero out-of-frame points across 2,696 robot/box/goal-region camera projections. Direct-MjLab G1A also passed. Full G1 remains open because no true-Go2 dimOS packet exists.

| Area | Result | Evidence |
| --- | --- | --- |
| Project Python environment | PASS | Python 3.12.4; locked floor corrected to 3.10.12; NumPy and simulator extras resolved |
| Official Go2 MJCF compile | PASS | 19 positions, 18 velocities, 12 actuators, 2 ms physics timestep |
| 224×224 overhead-style RGB render | PASS | shape `[224, 224, 3]`, 2,965 unique RGB colors, retained PNG/checksum |
| True-Go2 locomotion gait | PASS for MjLab adapter | checkpoint and ONNX playback plus clean formal trials; exact policy SHA retained |
| Current dimOS Go2 simulator path | FAIL for Go2 evidence | pinned source rewrites `unitree_go2` to `unitree_go1`; loader supports Go1/G1 policies only |
| Official Unitree RL Gym pretrained candidate | BLOCKED | pinned checkout contains G1, H1, and H1_2 `motion.pt` files, but no Go2 policy |
| G0 motion/push/reset/throughput trials | TECHNICAL PASS, SIGN-OFF PENDING | `20260920T073000Z-g0-formal-mjlab`; all six checks true; source clean |
| G1 20-block timing packet | DIRECT-MJLAB PASS | `20260920T073500Z-g1a-mjlab`; three 20-block episodes; strict reload/checksums pass |
| G1 dimOS traversal | BLOCKED | pinned dimOS Go2 selection remains a Go1 surrogate |

## Retained evidence

- `artifacts/runs/20260919T191000Z-go2-project-env-render/report.json` — project-environment model/render report.
- `artifacts/runs/20260919T191000Z-go2-project-env-render/overhead.png` — exact inspected render.
- `artifacts/runs/20260919T191000Z-go2-project-env-render/checksums.sha256` — artifact hashes.
- `artifacts/runs/20260919T191500Z-person-a-preflight/preflight.json` — environment and upstream source audit, with exact commits and source hashes.
- `artifacts/runs/20260919T191500Z-person-a-preflight/checksums.sha256` — preflight hash.
- `artifacts/runs/20260919T193100Z-software-gate/` — raw outputs, JSON coverage, report, and hashes for the 94-test/Ruff/CLI/protocol pass.
- `artifacts/runs/20260919T202600Z-go2-controller-source-audit/` — pinned Unitree RL MjLab source declarations and file hashes; source compatibility only.
- `artifacts/runs/20260919T204000Z-controller-recovery-final/` — final controller-recovery software gate: 125 tests, Ruff, CLI/protocol checks, and 85.51% line coverage.
- `artifacts/runs/20260919T205000Z-local-training-host-preflight/` — explicit local-host rejection: macOS arm64, Python 3.12, no NVIDIA device, and under 19 GiB free.
- `artifacts/runs/20260919T210000Z-pre-baseline-final/` — expanded pre-baseline software gate: 133 tests, Ruff, CLI/protocol checks, and 85.67% line coverage.
- `artifacts/runs/20260920T045422Z-go2-controller-handoff/` — strict controller packet with checkpoint, ONNX policy, exact XML/configs, playback reports, and checksums.
- `artifacts/runs/20260920T073000Z-g0-formal-mjlab/` — clean-source formal G0 measurements; all technical checks pass, second-owner sign-off pending.
- `artifacts/runs/20260920T073500Z-g1a-mjlab/` — direct-MjLab G1A packet, self-contained ten-block montage, independent verification, and outer checksums.

The `artifacts/` tree is intentionally gitignored; copy these directories to durable team storage before cleaning this machine.

## Upstream audit

The dimOS source audit pinned revision `c1c3cdc9d2ee54ca72259465688395699d7d99a2`. Its legacy MuJoCo process explicitly maps `unitree_go2` to `unitree_go1`. Its model loader imports Go1 and G1 assets and chooses only `Go1OnnxController` or `G1OnnxController`. This path can later be useful for transport/interface experiments only if it is labeled as a Go1 surrogate.

The official Unitree RL Gym audit pinned revision `276801e46c5d433564f24658bac64f254b7d2d4b`. It defines a Go2 training configuration but the inspected `deploy/pre_train` tree contains policies only for G1, H1, and H1_2.

The follow-up source audit identified Unitree's official `unitree_rl_mjlab` at
revision `1425b15f73bd4095f0df53709d7c389c3eb9e790` as the controlled recovery
route. It supports `Unitree-Go2-Flat` training and ONNX export but does not ship
a ready checkpoint. Its 5 ms training XML has SHA-256
`077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912` and
uses position-actuator semantics. That XML differs from the rendered Menagerie
XML, so direct policy substitution is prohibited. The executable procedure and
handoff contract are in `docs/GO2_CONTROLLER_RECOVERY.md`.

The verified model source is MuJoCo Menagerie commit `32224735cf004df068cd29ba099459f71bcd7d21`, Go2 tree `98d14ab27a56c3623e29d852a997a422bbfefa19`, BSD-3-Clause. Model compilation and rendering are verified; locomotion is not.

## Contract repairs completed

The Person A data boundary was upgraded to episode/block schema v2 before any real collection:

- every episode has a non-empty `run_id`;
- every block stores UTC collection time and duplicate simulation start/end times;
- requested and actually applied actions are separate, with explicit units;
- boundary clipping is exercised exactly once by the fake simulator test;
- physics timestep and step count are stored and validated against every sample timestamp;
- fall and out-of-bounds events retain counts and first-event times across substeps;
- validity and terminal reason are first-class fields;
- collection stops after a fall or out-of-bounds event, preventing post-termination blocks;
- all episode files, including JSON/JSONL records, are covered by a write-once SHA-256 inventory;
- the loader reconstructs the strict contracts and checks the complete artifact inventory.

The expanded software gate now contains 133 passing tests, including 31 focused
controller-handoff cases. These are software-contract results, not Go2 physics
results.

## Environment risks

- The host is Apple Silicon macOS without NVIDIA/CUDA. It is suitable for CPU model and simulator integration, not the planned CUDA training path.
- The final preflight measured roughly 18.6 GB free. Avoid full dimOS/data downloads or long replay collection until space is reclaimed or a separate artifact volume is configured.
- MuJoCo rendering on macOS must run through `mjpython`; ordinary headless Python failed to create a CoreGraphics context.
- Formal G0/G1A evidence was generated from clean revision `756cb3c0547da3568ddf4ec93440d00ba3664c33`. Other dashboard/planner documentation edits remain in the working tree and must not be conflated with that evidence revision.

## Recovery path

1. Obtain the second owner's acknowledgement of `configs/g0_acceptance.toml` and visual review of the G1A montage; do not edit generated evidence in place.
2. Keep full G1 open until a true-Go2 dimOS traversal exists. A Go1 surrogate demo must not be labeled Go2.
3. Materialize and freeze evaluation starts and absolute G2/G3 targets before bulk data collection.
4. Collect deployment-identical MjLab data with the retained split on non-iCloud storage, strict-load it, and back it up before training.
5. Build the pinned LeWM environment outside iCloud, run the real-LeWM tests and tiny overfit, then benchmark one epoch before spending additional GPU credit.
6. Continue through oracle candidate feasibility, learned ranking, held-out surprise, dimOS integration, and untouched final evaluation. Until those gates pass, the defensible fallback remains F3.
