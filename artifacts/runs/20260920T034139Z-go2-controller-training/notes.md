# Go2 controller CUDA training/export result

## Result

The pinned `Unitree-Go2-Flat` training job completed on an NVIDIA A40 with exit
code 0. The final iteration-10000 checkpoint, self-contained ONNX export,
resolved environment/agent/deployment configs, exact training XML, and complete
logs are retained in this directory. The transfer archive and every payload
file matched their RunPod SHA-256 digests after local extraction.

The final checkpoint deserialized successfully with the expected training-state
keys, and its local PyTorch ZIP container passed a complete integrity test. ONNX
1.23.0 loaded and checked the export successfully; it has IR version 8, input
`obs`, and output `actions`. A separate local ONNX Runtime 1.22.1 CPU inference
accepted a finite zero input of shape `[1, 47]` and returned 12 finite actions.
The complete log reached iteration 10000/10001, reports 983,138,304 steps, ends
with `FULL_EXIT=0`, and passed a case-insensitive scan for traceback, OOM, CUDA
error, `Error:`, NaN, and Inf.

## What this proves

- The pinned upstream source, exact Go2 XML, CUDA stack, environment creation,
  PPO update path, checkpoint writer, and ONNX exporter completed together.
- The retained files are byte-for-byte identical to the files audited on the
  RunPod volume.
- The ONNX export is structurally valid and self-contained.

## What this does not prove

- The checkpoint has not passed zero/forward/left/right simulator playback.
- The ONNX policy has not passed the upstream simulator/deployment path.
- Training reward and termination diagnostics do not establish gait quality,
  project-scene compatibility, pushing ability, or Gate G0.
- No dimOS, physical robot, visual-world-model, planner, or closed-loop task
  performance claim follows from this run.

For that reason this is a training-evidence packet, not a
`go2wm.go2-controller-handoff.v1` packet. The strict handoff verifier should not
be run against it as though the two required playback flags were true.

## Next required acceptance work

1. Play `checkpoint/model_10000.pt` through the pinned upstream checkpoint
   path and retain zero/forward/left/right trials plus requested/observed rates.
2. Exercise `policy/policy.onnx` with `config/deploy.yaml` through the pinned
   upstream simulator deployment path and retain the same trials.
3. Only after both paths pass, create the immutable strict controller handoff
   manifest and run `scripts/verify_controller_handoff.py`.
4. Integrate the accepted controller into the project adapter and execute the
   full G0 protocol without changing the fixed project action contract.

## Recovery note

The stopped GPU pod initially could not resume because its original host had no
free GPU. It was started in RunPod's CPU-only transfer mode, the 6,466,397-byte
archive was downloaded and verified, and the recovery pod was stopped at
2026-09-20T04:00:01Z. Two issues were caught without compromising any payload:
the ONNX file was at the run root rather than an `exported/` subdirectory, and
the first generated checksum list included a stale self-hash. The ONNX was
resolved by its recorded digest; the self-entry was removed; every substantive
file then passed `sha256sum -c`.
