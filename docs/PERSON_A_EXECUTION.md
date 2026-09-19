# Person A Execution Status

This is the audited status of the Simulation & Integration work after the 2026-09-19 execution pass. It distinguishes verified subcomponents from the still-blocked G0 gate.

## Outcome

**Gate G0 is blocked.** The official Unitree Go2 MuJoCo model and a 224×224 RGB render work in the pinned project environment. No compatible true-Go2 gait/controller has been found or executed, so motion, box interaction, reset repeatability, collection throughput, and the dimOS production path cannot honestly pass yet.

| Area | Result | Evidence |
| --- | --- | --- |
| Project Python environment | PASS | Python 3.12.4; locked floor corrected to 3.10.12; NumPy and simulator extras resolved |
| Official Go2 MJCF compile | PASS | 19 positions, 18 velocities, 12 actuators, 2 ms physics timestep |
| 224×224 overhead-style RGB render | PASS | shape `[224, 224, 3]`, 2,965 unique RGB colors, retained PNG/checksum |
| True-Go2 locomotion gait | BLOCKED | no controller attached; uncontrolled model loses 0.177 m base height over the 1 s model-only probe |
| Current dimOS Go2 simulator path | FAIL for Go2 evidence | pinned source rewrites `unitree_go2` to `unitree_go1`; loader supports Go1/G1 policies only |
| Official Unitree RL Gym pretrained candidate | BLOCKED | pinned checkout contains G1, H1, and H1_2 `motion.pt` files, but no Go2 policy |
| G0 motion/push/reset/throughput trials | NOT RUN | prohibited by the missing true-Go2 gait prerequisite |
| G1 20-block timing/dimOS handoff | NOT RUN | G0 is not passed and current dimOS path is a surrogate |

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
- The repository has no commit yet. Evidence records therefore say `NO_COMMIT`, and the entire scaffold is an uncommitted working tree. Create a reviewed baseline commit before a team handoff.

## Recovery path

1. Train and play back `Unitree-Go2-Flat` at the pinned Unitree RL MjLab revision on a Linux/NVIDIA host, using its exact bundled Go2 XML.
2. Build a small controller-only probe first: zero command for 10 simulated seconds, then bounded forward/left/right trials. Log base pose, fall state, requested/applied commands, and all substeps.
3. Add the fixed task scene, two equal-geometry boxes, and a visible front marker. Repeat the 224×224 coverage inspection at all planned extrema.
4. Complete all G0 repeats and thresholds from the runbook. Only measured stable bounds may enter configuration.
5. Wire the validated adapter to dimOS. If dimOS cannot host the true model/controller, use its skill/transport layer around the external simulator and disclose that architecture.
6. Produce the 20-block G1 packet and require Person B to load, visually inspect, and sign it before training begins.

Until step 2 passes, the defensible fallback is F3: dataset/instrumentation and model/render infrastructure only. A Go1 surrogate demo must not be labeled Go2.
