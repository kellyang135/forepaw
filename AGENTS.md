# Repository working agreement

Read `README.md`, `docs/TWO_PERSON_PLAN.md`, `docs/VERIFICATION.md`, and
`docs/DECISIONS.md` before changing an experiment boundary.

## Truthfulness

- Passing fake-backend or unit tests is software-contract evidence only.
- Never describe MuJoCo, Go2, LeWM, CUDA, dimOS, or task performance as verified
  without a retained real-path run artifact.
- Keep provisional thresholds and unverified assumptions visibly labeled.

## Fixed initial contract

- Three 2 Hz RGB frames span one second and have exactly two connecting prior
  commands.
- Actions are `[forward_velocity_mps, yaw_rate_rps]`, held for 0.5 simulated
  seconds; lateral velocity is zero.
- Planning uses 64 six-block candidates and executes only the first block.
- Privileged simulator poses, contacts, mass/mobility flags, and oracle futures
  never enter runtime planning.
- Surprise response is detection and stop, not adaptation.

Changing one of these requires a decision-log entry, a schema/bundle version
change where applicable, and acknowledgement from both human owners.

## Ownership

- Person A/SIM owns `sim`, `data`, real dimOS transport, camera/scene physics,
  and the sole motion publisher.
- Person B/ML owns `model`, `planning`, `runtime`, training, readouts, metrics,
  planner weights, and bundle publication.
- `contracts.py`, `integration`, configs, and final evaluation protocol are
  shared interfaces. Coordinate changes before editing them.

## Required checks

Run `make check` before a handoff. Add a focused test for every repaired
contract bug. Do not weaken a failing gate to make it pass. Final-test failures
remain in the reported result unless they match a predeclared infrastructure
exclusion.

