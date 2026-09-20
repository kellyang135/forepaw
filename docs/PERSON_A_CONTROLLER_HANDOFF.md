# Person A controller handoff status — 2026-09-20

Owner: Person A / SIM. This records what is ready for the next work block
without upgrading draft feasibility evidence into a formal gate.

## Outcome

The trained MjLab Go2 controller is no longer the G0 blocker. Its checkpoint
runs in the pinned upstream MjLab environment, its exported ONNX policy runs in
the project MuJoCo adapter, the two paths agree within predeclared tolerances,
and the strict handoff packet verifies end to end.

| Evidence | Run ID | Result |
| --- | --- | --- |
| Upstream checkpoint playback | `20260920T044245Z-mjlab-checkpoint-playback` | PASS, 30/30 safe trials |
| Project ONNX playback | `20260920T045030Z-mjlab-adapter-playback` | PASS, 30/30 safe trials |
| Cross-simulator comparison | `20260920T045143Z-mjlab-playback-comparison` | PASS, max deltas 0.0052 m/s and 0.0017 rad/s |
| Draft scene feasibility rerun | `20260920T043709Z-g0-trials-mjlab-draft-rerun` | All six checks PASS; 0 falls |
| Strict controller packet | `20260920T045422Z-go2-controller-handoff` | PASS, 30 hashed files and 16 meshes |
| Packet verification | `20260920T045603Z-mjlab-packet-verification` | All six stages PASS |

The earlier `20260920T041026Z-g0-trials-mjlab-draft` directory is invalidated
because its PNGs do not match its checksum file. It must never be cited as
evidence; the fresh rerun replaces it.

## Measured controller response

Median upstream / adapter response after a one-second warmup:

| Command | Upstream | Adapter | Absolute gap |
| --- | ---: | ---: | ---: |
| forward 0.2 m/s | 0.1614 | 0.1611 | 0.0003 m/s |
| forward 0.4 m/s | 0.3579 | 0.3572 | 0.0006 m/s |
| forward 0.6 m/s | 0.5444 | 0.5440 | 0.0004 m/s |
| yaw -0.96 rad/s | -0.8938 | -0.8921 | 0.0017 rad/s |
| yaw +0.96 rad/s | +0.9190 | +0.9177 | 0.0013 rad/s |

The fresh scene trial again moved the 1 kg blue box in 5/5 attempts. The 20 kg
red box held in 5/5 under the provisional 5 cm rule, but its worst displacement
was 4.48 cm. That small margin must remain visible when P-002 is finalized.

## Verified scope

- Exact upstream revision and checkpoint, ONNX, XML, and mesh digests.
- Policy shape, joint order, gains, default pose, gait phase, and deploy map.
- Native checkpoint stand/walk/turn behavior on the A40.
- ONNX project-adapter behavior and cross-simulator equivalence.
- Draft scene pushes, camera coverage, reset settling, and throughput.

## Not verified

- Formal G0 joint sign-off; the final goal marker and P-001 through P-004
  decisions are not frozen.
- C++ `unitree_mujoco` deployment playback; upstream requires a connected
  gamepad for FSM transitions and commands, and none was present on RunPod.
- dimOS, the sole motion publisher, physical Go2 deployment, G1, or task-level
  success.

## Next executable step

1. Add the goal marker and freeze scene/camera identifiers.
2. Decide P-001 bounds and P-002 resistance, explicitly acknowledging the
   4.48 cm worst-case red-box displacement.
3. Freeze P-003 camera/arena and P-004 settle tolerances.
4. Rerun formal G0 into a new immutable directory.
5. After G0 passes, collect the 20-block G1 packet through the sole motion
   publisher and real dimOS transport; do not substitute the fake path.
