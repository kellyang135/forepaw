# Forepaw pitch guide

Use the final deck at
[`Forepaw_HackMIT_Dimensional_Pitch_FINAL_v2.pptx`](../.codex-output/Forepaw_HackMIT_Dimensional_Pitch_FINAL_v2.pptx).
The deck contains the same script in its speaker notes.

## Core sentence

Forepaw is a dimOS robotics agent that compares short physical futures,
records its choice before motion, executes one half-second action, and replans
from the next observation.

## Three-minute script

### Slide 1, 10 seconds

Forepaw gives a simulated Unitree Go2 a short forecast before every movement.
It compares possible futures, records its choice, moves for half a second, then
observes and replans.

### Slide 2, 20 seconds

The task has one goal and two useful behaviors. The blue one-kilogram box can
move, so pushing can clear the route. The red box is fixed, so the robot should
plan around it. The colors identify behavior in this environment. This is not a
claim about hidden mass inference.

### Slide 3, 25 seconds

dimOS discovers Forepaw as an external blueprint with three skills. `imagine`
returns predicted outcomes without motion. `plan_to` runs the closed loop.
`stop_motion` exposes the safety surface. A separate controller service is the
only process allowed to publish motion to the true-Go2 MjLab simulation.

If time allows, switch to the live replay here for 45 seconds.

### Slide 4, 25 seconds

Each cycle uses three RGB frames spanning one second plus the two applied
commands between them. Forepaw scores 64 candidate sequences with six
half-second blocks, which is 384 candidate blocks. It locks the selected
forecast, executes only block zero, and starts a new plan from the next
observation. The current integration demo uses the labeled reference predictor,
not LeWM.

### Slide 5, 25 seconds

The retained evidence shows 45 dimOS planning cycles with 64 candidates each
and lock-before-motion ordering. In the formal Go2 scene, every blue-box trial
moved at least 1.2292 meters, every fixed red-box trial stayed below 0.0448
meters, and no robot fell. The repository passes 201 tests with 6
environment-gated skips. G0 still awaits the second owner's sign-off.

### Slide 6, 20 seconds

The product idea is inspectable control for physical agents. One service owns
motion. Every selected future is saved before execution. Operators can review
what the agent expected and compare it with what happened. The workflow is
verified in simulation. Broader business impact remains a hypothesis until
deployment with users.

### Slide 7, 15 seconds

Today I verified the dimOS skill boundary, true-Go2 MjLab execution, restart
behavior, and the evidence pipeline. The learned predictor did not pass its
gate, so I kept it out of control. The next gates are learned multi-step
prediction, held-out task evaluation, and active stop preemption on Linux. The
MVP is the working, auditable dimOS control loop you just saw.

## Live demo, 45 seconds

Open [the public replay](https://vercel-two-dun-11.vercel.app).

1. Start with `dimos-go2-push.jsonl` and point out the three RGB inputs.
2. Show the 64 ranked candidates and the saved first action.
3. State that only the first 0.5-second block executes.
4. Switch to `go2-surprise-stop.jsonl`.
5. Let it reach block 3 and show the `Stop latched` banner.

If the venue network fails, use the same site locally:

```bash
python3 -m http.server 8788 --directory deploy/vercel/public
```

## Claim boundaries

Say:

- External dimOS blueprint and MCP skills verified.
- True-Go2 MjLab execution verified with the privileged reference predictor.
- Three goal-reaching dimOS push runs and 45 total planning cycles retained.
- Prediction lock, restart handling, and surprise-stop replay verified.

Do not say:

- The learned LeWM controls the robot.
- Learned closed-loop task performance passed.
- Linux active stop preemption passed.
- The fixed red object demonstrates hidden mass inference.
