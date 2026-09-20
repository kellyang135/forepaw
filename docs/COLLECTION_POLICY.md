# Collection policy

`scripts/collect_wave.py` drives the simulator with `go2wm.data.scripted_policy`, a
scripted, boundary-aware policy. Uniform random commands walk the robot into walls
and rarely touch a box, so the old random collector produced too little contact
data and too many out-of-bounds episodes.

## What it does

Each scenario seed deterministically picks one of four modes and places the robot,
one blue (movable) box, one red (resistant) box, and a goal:

| Mode | Default weight | Behavior |
| --- | --- | --- |
| push | 0.25 | Walks into the blue box and pushes it 0.5 to 1.0 m, then roams |
| resist | 0.15 | Walks into the red box, presses for 2 to 4 blocks (stall), turns away |
| detour | 0.25 | Goal sits behind the red box; walks around it |
| free | 0.35 | Roams with arcs, turns in place, and exact stops |

A wall guard looks 0.55 m ahead and turns toward the arena center. Each block has a 12% chance
of starting a random segment, and every command gets small noise, so the model sees actions
off the scripted manifold. The policy steers with privileged poses, which D-004
allows during collection only. Each episode's metadata records the policy id and mode.

## Commands

```bash
python -m go2wm.learning make-splits --seeds 1-400 --salt wave1 --out artifacts/splits-wave1.json
python scripts/collect_wave.py --backend mujoco --policy /tmp/robot_lab_policy.pt \
    --splits artifacts/splits-wave1.json --seeds 1-400 --out data/wave1
python -m go2wm.learning check-data --data data/wave1 --splits artifacts/splits-wave1.json
```

Test-split seeds are skipped unless you pass `--include-test`. Tune the mix with
`--mode-weights '{"push":0.2,"resist":0.1,"detour":0.25,"free":0.45}'` if the summary's
interaction fraction lands outside 0.20 to 0.30.

## Rehearsal on the fake backend (2026-09-20)

80 seeds, 40 blocks each: 72 episodes written (8 test seeds skipped), 0 out-of-bounds,
0 rejects, strict `check-data` passes. Interaction fraction 0.35 in the 3 x 3 m fake arena;
the 4 x 4 m MuJoCo arena has more free space, so measure it there before retuning.
This is software evidence only.
