# Direct-MjLab wave 120: G2/G3 evidence

Status: **G2 marginal; G3 failed; active fallback F3**.

This packet retains the small, reviewable evidence from the first sufficiently
long real LeWM run. It does not claim full G1, predictive control, dimOS
deployment, or test-set performance.

## Provenance

- Collection and plain LeWM source: `f3463e6a95e49ff3f626127acd659f7809846c1d`
- Optional auxiliary source: `343e7b4dbc22db7908b8e47ccd46a45053cc7375`
- LeWM source: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- Split: `splits-7ae472b8826f`; test seeds 377–400 remained uncached and unevaluated
- Controller policy SHA-256: `c1f66f9b85aa1ea4231c0025fe90d8d4636288657ccee6a490fde274edcaa9bc`
- Robot XML SHA-256: `077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912`
- Host: Apple M4, MPS, Python 3.10.19, torch 2.14.0
- Paid training compute: $0; RunPod remained stopped

## Dataset

The primary wave collected seeds 1–120 while automatically excluding test
assignments. Its training interaction fraction was 19.16%, just below the
predeclared 20–30% heuristic. A declared train-only supplement added ten push
and five resistant scenarios from new frozen-split seeds. The accepted dataset
then contained:

| Split | Episodes | Blocks | Interaction | Movable contacts | Resistant contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 92 | 3,680 | 24.16% | 566 | 336 |
| Validation | 19 | 760 | 26.71% | 135 | 68 |

Strict load: 111 episodes, zero rejects, zero leakage problems, zero coverage
warnings. The retained `montage.html` was inspected for free motion, turning,
stopping, movable pushes, brief contacts, resistant contacts, image coverage,
and requested/applied equality.

## Results

| Metric | Pixel baseline | Plain LeWM | Aux 0.1 diagnostic |
| --- | ---: | ---: | ---: |
| Real-frame robot readout, m | **0.100** | 1.281 | 1.302 |
| Real-frame object readout, m | **0.060** | 1.086 | 0.985 |
| Robot prediction at 0.5 s, m | **0.127** | 1.236 | 1.205 |
| Robot prediction at 3.0 s, m | **0.532** | 1.344 | 1.323 |
| Matched-vs-shuffled improvement | -0.22% | +0.02% | -0.99% |
| 64-candidate median latency | **32 ms** | 57 ms | 58 ms |

The plain run used 3,680 train clips, 760 validation clips, batch 32, ten
epochs, and 1,150 optimizer steps. It completed in 1,521.7 seconds. Epoch 9 was
best by validation latent prediction loss (`0.010526`), but copy-last was
`0.0025` and the shuffled-action gap was `-0.1%`. Its best weight SHA-256 is
`269500460733838bfb7c94e0c52b36dad46e1b3b9400ff9bd0f8c15c0f951423`.

The D-031 auxiliary diagnostic changed only one component: a training-only
state MLP with weight 0.1. It ran 345 steps. Validation auxiliary loss worsened
from 0.916 to 0.939 while train auxiliary loss fell; robot readout worsened and
the action gap stayed negative. It was rejected and not extended.

Both learned bundles reload deterministically and stay finite through six
blocks. Neither beats persistence at one step or reaches the 15% matched-action
criterion. Therefore G3 is failed and learned rollouts must not control the
robot.

## Retention boundary

This git packet intentionally excludes 775 MB of raw RGB episodes, 685 MB of
lossless caches, and large checkpoint files. Their local paths were
`/private/tmp/go2wm-data/20260920T081500Z-mjlab-wave120` and
`/private/tmp/go2wm-wave120-run` at creation time. Checkpoint hashes and all
small run/evaluation metadata are retained here. Copy the large artifacts to a
durable external volume before relying on rerun-free recovery.

## Next allowed experiment

Per G3 and P-010, the next model repair is a separately versioned short
multistep predictor fine-tune using train/validation only. Do not alter the
camera, split, action contract, or evaluation thresholds, and do not inspect
test seeds. Until that repair passes G3, the public deliverable is F3:
aligned Go2 interaction data, quantitative baselines, and clearly labeled
preliminary model failures.
