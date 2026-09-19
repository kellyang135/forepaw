# Go2 Controller Recovery Runbook

This runbook closes the current G0 blocker: produce, validate, and hand off a
true-Go2 velocity controller without silently changing the robot model or action
semantics. It is a training-and-verification procedure, not evidence that G0 has
already passed.

## 1. Fixed source and compatibility boundary

Use Unitree's official `unitree_rl_mjlab` repository at revision
`1425b15f73bd4095f0df53709d7c389c3eb9e790` and task
`Unitree-Go2-Flat`. The audited revision has:

- MuJoCo physics timestep `0.005 s`;
- control decimation `4`, hence a `0.020 s` or 50 Hz policy interval;
- a three-value command ordered as forward velocity, lateral velocity, yaw rate;
- 12 scaled joint-position target actions with scale `0.25`;
- the Go2 XML SHA-256
  `077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912`.

The project-level action remains `[forward_velocity_mps, yaw_rate_rps]`.
The adapter must insert `lateral_velocity_mps = 0.0`; it must not expose lateral
motion to the planner.

The previously verified Menagerie `go2.xml` has a different checksum and
different actuator semantics. Do not place the exported ONNX policy into that
scene and call it compatible. First validate and integrate the policy against
the exact XML and actuator configuration used for training. A later port to the
Menagerie scene is a separate measured compatibility gate.

## 2. Division of work while G0 is blocked

Person A owns the training-host preflight, controller training/export, playback,
artifact handoff, simulator adapter, motion trials, task scene, and G0 evidence.
Person B can continue deterministic learning/planning unit work, but must not
train the visual world model on fake data or treat fake-backend results as robot
evidence. Person B witnesses one complete controller playback and later signs
the 20-block G1 packet.

## 3. Obtain the training host

The audited Mac has no NVIDIA/CUDA device. Use a separate machine matching the
upstream recommendation:

- Ubuntu 22.04;
- an NVIDIA GPU;
- NVIDIA driver 550 or later;
- Python 3.11 in an isolated Conda environment;
- enough free storage for the repository, environment, logs, checkpoints, and
  two retained run copies.

Record this before installation:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ
uname -a
python3 --version
nvidia-smi
df -h .
git --version
```

If the project checkout is already on that machine, run the executable
preflight as well:

```bash
PYTHONPATH=src python scripts/check_go2_training_host.py
```

If `nvidia-smi` fails, the driver is below the intended version, or Python is not
3.11, stop and repair the host. Do not start a long run in an unrecorded or
partly working environment.

## 4. Clone and install the exact revision

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
cd unitree_rl_mjlab
git checkout 1425b15f73bd4095f0df53709d7c389c3eb9e790
git rev-parse HEAD
git status --short

conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev \
  libspdlog-dev libfmt-dev
python -m pip install -e .
```

Preserve the complete install output. Confirm the task exists and inspect the
actual command-line surface before training:

```bash
python scripts/list_envs.py
python scripts/train.py Unitree-Go2-Flat --help
python scripts/play.py Unitree-Go2-Flat --help
sha256sum src/assets/robots/unitree_go2/xmls/go2.xml
```

The XML checksum must equal the pinned value above. If an option in the commands
below differs from the checked-out help text, use the help text and record the
exact command; do not guess.

From this project checkout, retain an automated source audit as well:

```bash
PYTHONPATH=src python scripts/audit_go2_controller_source.py \
  --checkout /path/to/unitree_rl_mjlab \
  --output-dir artifacts/runs/<unique-source-audit-run-id>
```

## 5. Two-stage training

First run a cheap pipeline smoke test. Its purpose is only to prove environment
creation, stepping, optimization, checkpointing, and ONNX export:

```bash
python scripts/train.py Unitree-Go2-Flat \
  --gpu-ids 0 \
  --env.scene.num-envs=64 \
  --agent.seed=42 \
  --agent.max-iterations=2 2>&1 | tee training-smoke.log
```

Do not call the resulting policy a gait. The smoke passes only if the process
exits cleanly, losses are finite, and a non-empty checkpoint plus ONNX export
are present.

Then start the real run. The upstream example uses 4096 environments and the
pinned Go2 PPO config defaults to 10001 iterations:

```bash
python scripts/train.py Unitree-Go2-Flat \
  --gpu-ids 0 \
  --env.scene.num-envs=4096 \
  --agent.seed=42 2>&1 | tee training-full-seed42.log
```

Treat 4096 as a starting point, not a guarantee that it fits every GPU. If an
out-of-memory error occurs, reduce only `num-envs`, record the new value and the
error, then restart from a clean run directory. Do not lower the iteration count
just to obtain a nominally complete artifact.

Expected output lives under
`logs/rsl_rl/go2_velocity/<timestamp>/`. Retain the selected `model_*.pt`,
`policy.onnx`, any matching `policy.onnx.data`, environment and agent YAML,
TensorBoard/event logs, console log, exact source revision, and the training XML.
The upstream README describes an external `.data` file, but the pinned exporter
does not explicitly force external-data mode. Inspect the actual export: record
`onnx_data_mode` as `external` and include the data file when one exists, or as
`self_contained` and omit that file when ONNX validation proves it is embedded.
Never create a dummy or empty `.data` file.

## 6. Playback acceptance before integration

Select a checkpoint using training diagnostics and playback, never a file name
alone. Run the checked-out playback command, for example:

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint_file=logs/rsl_rl/go2_velocity/<run>/model_<iteration>.pt
```

Save the console log and a screen recording. Begin with zero and low-amplitude
commands. The candidate is acceptable for handoff only when all of these are
recorded:

1. zero command remains upright for 10 simulated seconds;
2. low forward, left-yaw, and right-yaw commands each complete five repeats;
3. no repeat falls, produces non-finite state, or visibly maps a command to the
   wrong axis;
4. lateral command is held at exactly zero in project-mode trials;
5. simulation deployment of the ONNX export passes the same qualitative trials
   as checkpoint playback;
6. requested and observed control/physics rates are 50 Hz and 200 Hz.

These are controller handoff checks, not the complete G0 gate. Final project
action bounds are still chosen from repeated measured motion trials.

## 7. Build the immutable handoff packet

Create one directory containing these non-empty artifacts:

| Manifest key | Required artifact |
| --- | --- |
| `policy_onnx` | selected `policy.onnx` |
| `policy_onnx_data` | matching `policy.onnx.data`, required only in `external` mode |
| `checkpoint` | selected `model_*.pt` |
| `environment_config` | resolved environment YAML |
| `agent_config` | resolved PPO/runner YAML |
| `deployment_config` | exact Go2 `deploy.yaml` used for ONNX simulation deployment |
| `robot_xml` | training `src/assets/robots/unitree_go2/xmls/go2.xml` |
| `training_log` | complete console or event-derived training record |

Copy `docs/GO2_CONTROLLER_HANDOFF_TEMPLATE.json` to `manifest.json`, replace
every placeholder, and compute each digest with `sha256sum`. Do not include
absolute machine-specific paths. Set all four validation flags to `true` only
after their checks actually pass.
For a self-contained ONNX export, change `onnx_data_mode` to `self_contained`
and remove the `policy_onnx_data` entry entirely.

For the ONNX-specific validation, follow the checked-out upstream simulation
deployment path rather than assuming checkpoint playback proves export parity:

1. set `simulate/config.yaml` to `robot: "go2"` and
   `robot_scene: "src/assets/robots/unitree_go2/xmls/scene_go2.xml"`;
2. place the ONNX file and, when external-data mode is used, its matching data
   file in
   `deploy/robots/go2/config/policy/velocity/v0/exported/`;
3. build `simulate/` and `deploy/robots/go2/` according to the upstream README;
4. launch `./simulate/build/unitree_mujoco` and then
   `./deploy/robots/go2/build/go2_ctrl --network=lo`;
5. preserve the build logs, controller log, and screen recording, and run the
   zero/forward/left/right acceptance sequence again.

The upstream simulator path currently requires a gamepad. Its controller also
checks for another low-command publisher; treat a publisher-conflict warning as
a hard failure, not something to bypass.

On the receiving machine, run:

```bash
PYTHONPATH=src python scripts/verify_controller_handoff.py /path/to/packet
```

The verifier fails closed on a wrong source revision, task, robot XML, joint or
command order, action semantics, timing, path escape, empty file, or checksum
mismatch. A PASS means the packet is internally compatible and intact; it does
not mean locomotion quality or G0 passed.

## 8. Implement the project adapter

Use the exact training model first. The adapter has two nested clocks:

- one policy call every 4 physics steps (`0.020 s`);
- 25 policy calls, or 100 physics steps, per project block (`0.500 s`).

At each 0.5-second boundary:

1. validate and clip the requested two-dimensional action exactly once;
2. store both requested and applied action;
3. construct `[applied_forward, 0.0, applied_yaw]`;
4. hold that command for all 25 policy calls;
5. apply each 12-value scaled joint-position output for four physics substeps;
6. inspect every physics substep for contact, fall, out-of-bounds, and non-finite
   state, while storing the RGB observation only at the required boundary;
7. stop the block immediately on a terminal safety event.

The sole motion publisher rule remains mandatory. The adapter must have one
owner for the low-level targets, and stop must preempt queued motion.

## 9. Resume G0 in this order

After the adapter passes focused timing, axis, and fail-closed tests:

1. repeat zero/forward/left/right trials in the adapter;
2. measure and freeze provisional forward/yaw bounds;
3. add the fixed scene and visible front marker;
4. add two equal-geometry boxes with only the documented physical property
   changed;
5. inspect 224×224 camera coverage at every planned extremum;
6. run five pushes per box, ten reset repeats, and the 100-block throughput test;
7. create the retained G0 report with raw logs, video, configuration, source and
   artifact checksums;
8. only after G0 passes, produce the 20-block G1 timing packet.

Stop and report immediately if the policy falls at nominal commands, command
axes are reversed, the ONNX and checkpoint paths disagree, the XML checksum
changes, timing is not exact, or scene changes destabilize locomotion. Do not
paper over one of those failures by widening a threshold.
