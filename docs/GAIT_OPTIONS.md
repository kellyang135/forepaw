# Unblocking G0: a pretrained Go2 gait that runs on the Mac

Prepared by Person B for Person A (SIM owns this decision). Status: **PROPOSED**.
Nothing here passes G0; it is evidence for choosing the controller route.

## 1. The blocker, as recorded by SIM

- G0 is blocked because no true-Go2 walking controller exists in the project
  (`PERSON_A_EXECUTION.md`, A-001 BLOCKED).
- The recorded recovery route (D-019, `GO2_CONTROLLER_RECOVERY.md`) trains
  `Unitree-Go2-Flat` with Unitree RL MjLab. That needs Ubuntu + an NVIDIA GPU.
  The local-host preflight (`artifacts/runs/20260919T205000Z-local-training-host-preflight/`)
  rejected this Mac, and the team has not recorded access to such a host.
- Consequence: without a controller, the defensible level is F3.

## 2. What exists publicly (researched 2026-09-19)

| Candidate | Go2 weights shipped? | Runs on the Mac? | Notes |
| --- | --- | --- | --- |
| Unitree RL MjLab `1425b15` | No (Go2 has only `deploy.yaml`) | Train: no (mjlab: "requires an NVIDIA GPU for training; macOS evaluation only") | SIM's current route |
| Unitree RL Gym `276801e` | No (G1/H1/H1_2 only) | - | SIM already audited |
| dimOS `c1c3cdc9` | No (maps go2 to go1) | - | D-017 |
| MuJoCo Playground | Go1 only | Go1 ONNX, yes | would be a Go1 surrogate |
| Genesis `examples/locomotion/go2_*` | No (train your own) | train on Metal: unverified | |
| **fan-ziqi/rl_sar `376d42c`** | **Yes**: `policy/go2/robot_lab/policy.pt`, `policy/go2/himloco/himloco.pt` | **Yes, CPU** | Apache-2.0; trained in Isaac Lab with robot_lab (Apache-2.0) |
| walk-these-ways-go2 | Yes | port from Isaac Gym is non-trivial | |
| DIAL-MPC, Quadruped-PyMPC | training-free | JAX/CUDA or Linux-only build; not practical here | |

## 3. What was measured

`scripts/probe_pretrained_go2_gait.py` runs the rl_sar `robot_lab/policy.pt`
on the **unmodified** Menagerie `unitree_go2/scene.xml` (SHA-256
`b56123ea...`, the exact file SIM verified), MuJoCo 3.3.4, CPU only, no torch
(weights read with a restricted unpickler; checked against torch: max
difference 7.6e-6). Run: `artifacts/runs/20260919T233000Z-pretrained-gait-probe/`
(cloud Linux CPU; the whole probe took 65 s of wall time, so simulation runs
far faster than real time).

Results at the policy's native 5 ms physics step (the 2 ms Menagerie step gave
the same picture):

| Trial (5 repeats each, randomized start) | Falls | Measured |
| --- | --- | --- |
| Stand, zero command, 10 s | 0/5 | drift 4-7 cm; **yaw drift +0.13 rad/s, not measured at first (see section 8)** |
| Forward 0.3 / 0.6 / 1.0 m/s | 0/15 | vx 0.17 / 0.54 / 0.93; yaw drift +0.09 to +0.14 rad/s |
| Turn in place +-0.8 rad/s | 0/10 | wz +0.80 / -0.79 |
| Arc 0.4 m/s with +-0.6 rad/s | 0/10 | vx 0.26-0.36, wz 0.58 / -0.64 |
| Command step 0 -> 0.6 m/s | 0/5 | first 0.5 s block 0.29 m/s, second 0.44 m/s, 90% in 0.7-1.0 s |
| **0.40 m cube, 1 kg**, aimed push at 0.5 m/s, approaches -30 to +30 deg | 0/5 | **displaced 1.38-1.50 m in 5/5** |
| **0.40 m cube, 20 kg**, same geometry and approaches | 0/5 | **displaced at most 2 mm in 5/5**; robot stalls against it, tilt at most 6 deg |
| 0.30 m cube, 1 kg / 20 kg | 0/10 | light box moves only 1-6 cm; do not use this size |

"Aimed" means the scripted collector steers toward the box using privileged
position (allowed for collection, never at runtime); open-loop straight
commands miss because of the yaw drift.

Against the G0 box criterion ("light box displaces in at least 4/5 intended
pushes; resistant box stays below tolerance in at least 4/5"), the 0.40 m pair
is 5/5 and 5/5, with no falls in 90 trials in total.

## 4. What this does not show (and the risks)

- **Not G0.** No project adapter, task scene, overhead camera, front marker,
  reset repeatability, collection throughput, or dimOS path was tested.
- **Cross-simulator transfer.** The policy was trained in Isaac Lab on Unitree's
  URDF, not in MuJoCo. It works on the Menagerie model in these trials; rl_sar's
  own MuJoCo model differs only in friction and joint damping.
- **Command tracking is imperfect.** Low forward commands under-track (0.3 ->
  0.17 m/s) and straight walking yaws about 0.1 rad/s. The world model learns
  the real response, and the planner replans every 0.5 s, so this is a
  data-coverage issue, not a blocker. Project action bounds must come from
  these measurements (P-001).
- **Not trained with body contact.** Pushing worked here, but it is outside the
  policy's training distribution; watch for failures in the full G0 trials.
- **Honesty label.** Report it as "pretrained Go2 velocity policy (rl_sar
  robot_lab, Isaac Lab), run sim-to-sim in MuJoCo", not as a controller the
  team trained.

## 5. Proposed decision (needs SIM and ML acknowledgement)

**D-020 (proposed) — Use the pretrained rl_sar robot_lab Go2 policy as the G0
controller on the Menagerie Go2.** Pin rl_sar `376d42c` `policy/go2/robot_lab/policy.pt`
(SHA-256 `9f14cb95...`) and Menagerie `32224735`. Keep D-019's MjLab route as a
fallback only if G0 trials with this policy fail. Each 0.5 s project block is
25 policy calls (50 Hz) with lateral velocity fixed at 0. Supersedes the
D-019 requirement for a GPU training host; keeps D-018's model pin.

## 6. How SIM can reproduce this on the Mac (about 5 minutes)

```bash
curl -L -o /tmp/robot_lab_policy.pt \
  https://raw.githubusercontent.com/fan-ziqi/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/robot_lab/policy.pt
shasum -a 256 /tmp/robot_lab_policy.pt   # must be 9f14cb95e74ac9e5e30954da0fc0c33eaa41e337852ebed19fcee869279ade0b
uv run --no-sync python scripts/probe_pretrained_go2_gait.py \
  --policy /tmp/robot_lab_policy.pt \
  --output-dir artifacts/runs/$(date -u +%Y%m%dT%H%M%SZ)-pretrained-gait-probe
```

(The probe does not render, so plain `python` works; `mjpython` is only needed
for rendering on macOS.) The script refuses to run if either checksum differs.

## 7. If accepted, SIM's next steps (unchanged G0 procedure)

1. Wrap the policy in the project `SimulatorAdapter` (25 policy calls per block,
   clip once, log requested and applied actions, every substep checked).
2. Add the task scene: the two 0.40 m cubes (1 kg blue, 20 kg red, same
   geometry and friction), the overhead 224 x 224 camera, and a front marker.
3. Run the full G0 trials, reset repeats, and the 100-block throughput test.
4. Produce the 20-block G1 packet for Person B to load and sign.

## 8. Correction and follow-up (2026-09-19, later)

The first probe measured only position drift at stand. On zero command the
policy also turns in place at about +0.13 rad/s (0.75 rad in 6 s). The draft
adapter in [SIM_ADAPTER_DRAFT.md](SIM_ADAPTER_DRAFT.md) removes this with an
integral yaw-rate trim on the body gyro (an onboard sensor). With the trim,
stand yaw rate is +0.003 to +0.007 rad/s and straight walking is within
+-0.005 rad/s. The adapter's G0-style trials pass all six feasibility checks:
`artifacts/runs/20260919T231507Z-g0-trials-rl_sar-draft/`.
