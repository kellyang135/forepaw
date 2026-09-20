# Draft MuJoCo Go2 adapter and task scene (for SIM review)

Written by Person B for Person A. Status: **PROPOSED**. SIM owns `sim/`, the
scene, and G0; nothing here changes an existing SIM file, and none of it
is G0 evidence until SIM reviews it, reruns it on the Mac, and records the run.

## 1. What it is

Three new modules in `src/go2wm/sim/`, not yet exported from `sim/__init__.py`:

| File | Purpose |
| --- | --- |
| `locomotion.py` | `LocomotionSpec` (one per gait source), policy loaders, and `LocomotionController` (50 Hz policy, 200 Hz actuation) |
| `task_scene.py` | Builds the push/detour scene around the robot model: floor, arena border, two equal-geometry box slots, yellow front marker, fixed overhead camera `overhead_v1` |
| `mujoco_go2.py` | `MujocoGo2Simulator`, which satisfies the existing `SimulatorAdapter` protocol |

Plus `scripts/run_g0_trials.py` (the G0 trial runner) and
`tests/test_mujoco_go2.py` (13 tests; they skip without MuJoCo, the cached
Menagerie model, or an OpenGL renderer).

The gait is pluggable, so either controller route drops into the same adapter:

| Spec | Route | Robot model | Actuation | Policy inputs | State |
| --- | --- | --- | --- | --- | --- |
| `RL_SAR_ROBOT_LAB_GO2` | proposed D-020 | Menagerie `scene.xml` (`b56123ea...`) | explicit PD, kp 20, kd 0.5, 23.5 N m | 45 | run and measured |
| `MJLAB_GO2_FLAT` | D-019 | MjLab `go2.xml` (`077fc7f7...`) plus MjLab's actuators and collision settings | position actuators, kp 20/20/40, kd 1/1/2, 23.5/23.5/45 N m | 47 (adds a 0.6 s gait clock) | run and measured with SIM's trained policy (section 8) |

The MjLab settings come from the audited source at `1425b15`: `deploy.yaml`,
`go2_constants.py`, `velocity_env_cfg.py`, and mjlab 1.2.0. Its policy joint
order is FL, FR, RL, RR. rl_sar uses FR, FL, RR, RL.
`check_mjlab_deploy_yaml` refuses a packet whose gains, defaults, action scale,
observation order or gait period differ from the audit.

## 2. Contract behavior

- One block = 25 policy updates x 4 physics steps of 5 ms = 100 `PhysicsSample`s.
- The requested command is clipped once to the bounds in `MujocoGo2Config`
  (currently the PROPOSED `configs/experiment.toml` values: forward 0 to 0.6 m/s,
  yaw rate within +-1.2 rad/s). Requested and applied actions are both returned.
  Lateral velocity is always 0.
- Contacts: robot-to-box normal impulse per physics step, attributed to the
  box's `object_id`. Fall = base below 0.15 m or tilt above 60 deg (sticky).
  Out of bounds = base outside the arena.
- Simulated time is an integer step count times 5 ms, so block timestamps are exact.
- Reset: place robot and boxes, add a small joint perturbation seeded by
  `scenario_seed`, then settle under the gait with zero forward command until
  speed, yaw rate and heading error are quiet for 0.5 s (minimum 1 s, maximum 5 s).
  While settling, the reset holds the requested heading using the true yaw. This
  is simulator bookkeeping, like teleporting, and never a runtime input.
- Box class decides colour and mass (`blue` 1 kg movable, `red` 20 kg resistant;
  both 0.40 m cubes, friction 1.0). A reset whose `movable` flag disagrees with
  the class is refused, as is a box within 0.6 m of the robot or overlapping boxes.

## 3. Measured with the rl_sar policy (cloud Linux CPU, MuJoCo 3.3.4)

Run: `artifacts/runs/20260919T231507Z-g0-trials-rl_sar-draft/` (report,
frames, checksums). All six feasibility checks pass:

| Trial (5 repeats unless noted) | Falls | Result |
| --- | --- | --- |
| Stand, 10 s | 0 | yaw rate +0.003 to +0.007 rad/s, drift under 1.5 cm |
| Walk at 0.6 m/s, 4 s | 0 | 0.55 m/s, yaw rate within +-0.005 rad/s |
| Turn +-0.96 rad/s | 0 | +0.96 / -0.99 rad/s |
| Command grid, 20 cells (for P-001) | 0 | forward 0.2 / 0.4 / 0.6 gives 0.11 / 0.30 / 0.54 m/s; yaw within 0.075 rad/s of command |
| Aimed push, 1 kg box, approaches -30 to +30 deg | 0 | displaced 0.96 to 1.04 m, 5/5 |
| Aimed push, 20 kg box | 0 | displaced at most 2 mm with contact, 5/5 |
| Reset repeatability, 10 seeds, same request | - | 10/10 settle, 1.4 s mean; spread 5 mm, 0.004 rad; mean offset 3.8 cm, 0.03 rad |
| Camera | - | 2,680 projected robot-footprint and box-corner points, all inside the 224 x 224 frame |
| Throughput | - | 9.1 blocks per wall second including rendering |

Also checked end to end: `EpisodeCollector` -> `write_episode` ->
`learning.dataset.load_dataset` -> `check-data` (leak check passes) ->
`lewm-cache` on 8 rl_sar episodes. That rehearsal data was not kept; it is
not a G1 packet.

## 4. Findings SIM should know

1. **The rl_sar policy spins at stand.** On zero command it yaws at about
   +0.13 rad/s (0.75 rad in 6 s); it also drifts about +0.11 rad/s when walking
   straight. `GAIT_OPTIONS.md` reported only position drift at stand, so it
   missed this. `RL_SAR_ROBOT_LAB_GO2` now adds an integral yaw-rate trim from
   the body gyro (gain 1.0 /s, limit +-0.6 rad/s). The gyro is an onboard
   sensor, not privileged state. The trim carries over between blocks within an
   episode and resets with the episode. It is part of the controller and
   should be disclosed the same way.
2. **The controller handoff template cannot compile `go2.xml` by itself.** MjLab's
   `go2.xml` references 16 meshes in `xmls/assets/`, and the template's
   `files` list has no entry for them. Add the assets directory (with
   checksums) to the packet, or record the MjLab checkout path and revision
   used for playback.
3. **The MjLab route has physics the Menagerie scene does not.** Non-foot
   collisions are frictionless (`condim 1`), feet use friction 0.6, the
   solver runs 10 iterations with a pyramidal cone, and the actuators are
   position servos with armature. `_apply_mjlab_physics` reproduces these
   from source, but only an `onnx_sim_playback` comparison against MjLab's own
   play script can confirm they match.
4. **Forward commands under-track at low speed** (0.2 -> 0.11 m/s). Set P-001
   bounds from the command grid, not from the commanded values.
5. **Random-command collection leaves the arena often.** In the rehearsal, 3 of
   8 episodes ended out of bounds within 20 blocks. The collection policy
   needs a boundary-aware command sampler.

## 5. Not done (SIM decisions)

- Export from `sim/__init__.py` and the `sim` extra's dependencies
  (`onnxruntime` for the MjLab route; `pyyaml` for the deploy check).
- Goal marker in the image, final arena and camera size (P-003), box masses
  and tolerances (P-002), settle tolerances (P-004).
- Controller and scene identifiers in episode metadata; the collector passes
  `EpisodeRequest.metadata` through, and `scene_id` already hashes the scene
  config, robot XML, controller id and MuJoCo version.
- dimOS transport, the scripted collection policy, and the 20-block G1 packet.

## 6. How to run it on the Mac

```bash
# once: the pinned policy (checksum-verified by the loader)
curl -L -o /tmp/robot_lab_policy.pt \
  https://raw.githubusercontent.com/fan-ziqi/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/robot_lab/policy.pt

# tests (the Menagerie cache from verify_go2_model.py is reused)
GO2WM_RL_SAR_POLICY=/tmp/robot_lab_policy.pt uv run --no-sync pytest -q tests/test_mujoco_go2.py

# trials, about 1 minute
uv run --no-sync python scripts/run_g0_trials.py --controller rl_sar \
  --policy /tmp/robot_lab_policy.pt --menagerie-cache /tmp/go2wm-menagerie-cache \
  --output-dir artifacts/runs/$(date -u +%Y%m%dT%H%M%SZ)-g0-trials-rl_sar
```

When the MjLab packet exists, run the same script with `--controller mjlab
--policy <packet>/policy.onnx --robot-xml <checkout>/src/assets/robots/unitree_go2/xmls/go2.xml
--deploy-yaml <packet>/deploy.yaml`. The same checks then compare the two gaits.

On macOS, offscreen rendering may need `mjpython` (see `verify_go2_model.py`).
If `mujoco.Renderer` fails, the tests skip and the trial script stops at its
first reset.

## 7. Pre-run verification checklist (historical)

At 2026-09-20 00:20 UTC, nothing from the RunPod run was in the repo. The
following checklist was recorded before retrieval; section 9 records its final
execution and results.

1. **On the pod, before the long run:** the 2-iteration smoke from
   `GO2_CONTROLLER_RECOVERY.md` section 5 exits cleanly and writes a non-empty
   `model_*.pt` and `policy.onnx`. mjlab 1.2.0 pins `mujoco-warp==3.5.0` and
   needs `torch>=2.7`; a CUDA 12.8 torch build on a pod image whose driver
   shipped CUDA 12.4 must pass `torch.cuda.is_available()` and the smoke,
   not just install.
2. **Training health:** mean reward rising and episode length reaching the
   20 s cap in the TensorBoard log; checkpoint chosen by playback, not by
   file name. Copy the log back with the packet.
3. **Packet integrity:** `scripts/verify_controller_handoff.py` passes, and the
   packet also carries `xmls/assets/` (16 meshes; see finding 2 above).
4. **Interface match (new, automatic):** `check_mjlab_onnx_metadata` reads the
   metadata mjlab's exporter embeds in `policy.onnx` (joint names, stiffness,
   damping, default pose, actor observation names, action scale) and refuses
   any mismatch with the audited `MJLAB_GO2_FLAT` spec. `run_g0_trials.py`
   calls it before loading an MjLab policy.
5. **Cross-simulator check:** the policy was trained in MuJoCo Warp 3.5; the
   project pins MuJoCo 3.3.4. Run MjLab's own `play.py` and this adapter on
   the same commands and compare speed and yaw rate; a large gap means the
   `_apply_mjlab_physics` reproduction or the MuJoCo version differs.
6. **Same G0 trials as rl_sar:** `run_g0_trials.py --controller mjlab ...`, so
   both gaits are judged by identical checks. If both pass, D-019 and the
   proposed D-020 can be decided on measured results.

**One command for steps 3 to 6:**
`python scripts/verify_mjlab_packet.py <packet_dir> --output-dir artifacts/runs/<id>-mjlab-packet-check`
runs SIM's handoff verifier, a mesh-compile check, the ONNX metadata check,
the deploy.yaml check, a CPU inference check, and the G0 trials, and writes
`verification.json`. Tested on a synthetic packet built from MjLab's real
`go2.xml`: it passes a well-formed packet, catches missing meshes (SIM's
verifier alone does not), and fails a policy that stands but never pushes a box.

## 8. SIM's trained MjLab policy in this adapter (2026-09-20)

Policy: `artifacts/runs/20260920T034139Z-go2-controller-training/policy/policy.onnx`
(SHA-256 `c1f66f9b...`, commit `ee325d6`). Its `source/go2.xml` is byte-identical
to the pinned MjLab XML (`077fc7f7...`); the packet has no meshes, so this run
used the meshes from a checkout of `unitree_rl_mjlab@1425b15`.

- `check_mjlab_onnx_metadata`: PASS. Joint order, gains, default pose, the seven
  actor observations in order, and action scale all match the audited spec. The
  exporter writes the action scale as one shared value (`0.25`); the check now
  accepts that form.
- `check_mjlab_deploy_yaml` on the packet's `deploy.yaml`: PASS.
- An initial `run_g0_trials.py --controller mjlab` execution reported all six
  feasibility checks passing with zero falls, but its five PNG files no longer
  match the retained checksum manifest. Run
  `artifacts/runs/20260920T041026Z-g0-trials-mjlab-draft/` is therefore
  **INVALIDATED and non-evidentiary**. The table below is historical diagnostic
  output only; a fresh immutable run is required before citing these numbers.

- Fresh rerun:
  `artifacts/runs/20260920T043709Z-g0-trials-mjlab-draft-rerun/`. Its checksum
  manifest verifies the report and all five PNGs. It reproduced all six
  feasibility passes with zero falls, 5/5 light-box pushes, and 5/5 resistant-
  box holds. This repairs the evidence-integrity problem, but remains a draft
  feasibility run rather than joint G0 sign-off.

| Trial (5 repeats unless noted) | Fresh MjLab rerun | rl_sar (section 3) |
| --- | --- | --- |
| Stand, 10 s | yaw rate 0.000; no trim needed | +0.003 to +0.007 with gyro trim |
| Walk at 0.6 m/s | 0.54 m/s, yaw -0.015 rad/s | 0.55 m/s, yaw within +-0.005 |
| Turn +-0.96 rad/s | +0.92 / -0.89 | +0.96 / -0.99 |
| Forward 0.2 / 0.4 / 0.6 | 0.16 / 0.36 / 0.54 m/s | 0.11 / 0.30 / 0.54 m/s |
| Push 1 kg box | 1.23 to 1.60 m, 5/5 | 0.96 to 1.04 m, 5/5 |
| Push 20 kg box | 3.3 to 4.5 cm with contact, 5/5 | at most 2 mm, 5/5 |
| Reset, 10 seeds | 10/10; spread 2 mm, 0.007 rad; offset 1.2 cm | 10/10; spread 5 mm; offset 3.8 cm |
| Throughput | 10.2 blocks/s | 9.1 blocks/s |

Read with care:

1. **The 20 kg box moves 3 to 4.5 cm** under this policy, inside the PROPOSED
   5 cm tolerance but with little margin. Before G1, raise the resistant mass or
   the tolerance deliberately (P-002), not after seeing collection data.
2. **This is the adapter's reproduction of MjLab physics on MuJoCo 3.3.4.**
   Section 9 records the now-completed upstream MjLab 3.5 playback and the
   cross-simulator comparison.
3. **Lighting differs by route.** MjLab's `go2.xml` has no light, so frames are
   darker than with the Menagerie scene. Pick one route before collecting data;
   the camera appearance is part of the dataset (D-003).
4. The light box tips onto an edge during some pushes. Harmless for G0; watch
   it in the box-position readout.

## 9. Upstream playback and strict handoff (2026-09-20)

The remaining controller-playback gap is now closed with retained artifacts:

- Upstream checkpoint playback:
  `artifacts/runs/20260920T044245Z-mjlab-checkpoint-playback/` — PASS for all
  30 predeclared trials in MjLab 1.2.0 / MuJoCo Warp 3.5.0 on an NVIDIA A40.
- Project ONNX adapter playback:
  `artifacts/runs/20260920T045030Z-mjlab-adapter-playback/` — PASS for the same
  six commands and five resets each on MuJoCo 3.3.4.
- Cross-simulator comparison:
  `artifacts/runs/20260920T045143Z-mjlab-playback-comparison/` — PASS. Maximum
  absolute gap was 0.0052 m/s forward and 0.0017 rad/s yaw, below the
  predeclared 0.08 m/s and 0.12 rad/s limits.
- Strict controller packet:
  `artifacts/runs/20260920T045422Z-go2-controller-handoff/` — 30 manifest files,
  including all 16 meshes and raw playback trajectories.
- End-to-end packet verification:
  `artifacts/runs/20260920T045603Z-mjlab-packet-verification/` — handoff, meshes,
  metadata, deploy config, inference, and fresh trials all PASS; the outer and
  nested evidence checksums verify.

The upstream C++ `unitree_mujoco` deployment application is still unverified.
Its documented FSM transitions and velocity commands require a connected
gamepad; the headless RunPod did not expose `/dev/input/js0`. No virtual-input
workaround was used or counted as evidence. dimOS and physical-robot paths also
remain unverified.
