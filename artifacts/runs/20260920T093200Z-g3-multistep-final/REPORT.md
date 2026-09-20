# Final P-010 multistep diagnostic

Run status: **COMPLETE — G3 FAILED — FALLBACK F3 FROZEN**

This was the last repair allowed by the predeclared G3 plan. It used only the
frozen training and validation caches; reserved test seeds were not loaded.
The learned bundle is not approved to publish motion commands.

## Provenance

- go2wm source: clean `3adf69e153eff1247beb9c1a829da4ef4d4ac5fc`
- upstream LeWM: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- source plain checkpoint SHA-256:
  `269500460733838bfb7c94e0c52b36dad46e1b3b9400ff9bd0f8c15c0f951423`
- dataset: `mjlab-wave120-20260920T081500Z`
- split: `splits-7ae472b8826f` (92 train episodes, 19 validation episodes)
- objective: mean recursive latent MSE over six 0.5-second horizons
- trainable components: predictor, action encoder, predictor projector
- frozen components: image encoder and representation projector
- run: 300 updates, batch 32, fp32 MPS, 286.18 wall seconds
- selected checkpoint: `epoch_003.pt`, SHA-256
  `febbb90bec84653394533d3f9c2e1fc646a5cb62f79f9f2bd26302717cb541e4`

## Before/after validation controls

| Metric | Plain source | Multistep best | Verdict |
| --- | ---: | ---: | --- |
| Mean six-step latent MSE | 0.029653 | 0.023526 | improved 20.7% |
| 0.5 s latent MSE | 0.010740 | 0.010226 | improved 4.8% |
| 3.0 s latent MSE | 0.052081 | 0.038106 | improved 26.8% |
| Copy-last latent MSE | 0.008469 | 0.008428 | learned rollout still worse |
| Matched-vs-shuffled action gap | -0.079% | +0.095% | effectively action-blind |
| Nonfinite six-step rollouts | — | 0 | pass |

The lower recursive latent loss is real, but it does not satisfy G3: the model
still loses to latent persistence and the shuffled-action control is nearly
identical.

## Unchanged state-space evaluator

- G2 robot readout: 1.281 m versus 1.307 m constant mean — PASS, marginal.
- G2 object readout: 1.086 m versus 1.182 m constant mean — PASS.
- G2 split leakage and bundle reference replay — PASS.
- G3 robot one-step: 1.243 m versus 0.082 m persistence — FAIL.
- G3 object one-step: 1.168 m versus 0.000 m persistence — FAIL.
- G3 six-block robot action ablation: 1.336 m matched versus 1.332 m
  shuffled, -0.321% improvement — FAIL.
- G3 finite rollouts: zero nonfinite — PASS.
- evaluation bundle: `go2wm-b9a94a218341bf98` (retained for evidence only).

The evaluator exited 1 because the two G3 gates failed. That exit is the
expected, retained result; it is not an infrastructure failure.

## Freeze decision

Per G3 and the hard project cutoff, stop architecture/training experiments.
Fallback F3 is final: deliver the aligned Go2 dataset, collection and
instrumentation path, quantitative pooled-pixel baseline, complete negative
model results, montage, and metrics UI. Keep the existing deterministic
reference predictor available only for UI/transport rehearsal. Do not connect
this learned bundle to motion or claim predictive control.

Large raw data, caches, state files, checkpoints, and generated bundles remain
under `/private/tmp/go2wm-data` and `/private/tmp/go2wm-wave120-run`; they are
not in Git and must be copied to durable external storage if they need to
survive a reboot. The repository evidence packet retains configs, hashes,
metrics, and evaluation outputs needed to audit the conclusion.
