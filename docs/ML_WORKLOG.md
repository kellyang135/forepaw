# Person B (Learning & Control) worklog

Owner: Person B / ML. This file tracks the ML work items from
[TWO_PERSON_PLAN.md](TWO_PERSON_PLAN.md) §4. It is a status log, not
evidence: nothing below is VERIFIED until a real-path run artifact exists
(see [VERIFICATION.md](VERIFICATION.md)).

## Status by work item

| # | Work item | Code | Status |
| --- | --- | --- | --- |
| 1 | Data-contract consumer (strict loader, rejects) | `learning/dataset.py` | PROPOSED: unit-tested on fake data; awaiting SIM's G1 smoke packet |
| 1 | Episode-level splits frozen before fitting | `learning/splits.py` | PROPOSED: deterministic by `(salt, seed)`, write-once, hand-edit detection |
| 2 | Coverage diagnostics and leak check | `learning/diagnostics.py`, `splits.check_leakage` | PROPOSED |
| 3 | LeWM adaptation (2-D action, 3 frames, 0.5 s, 192-D) | `learning/lewm_adapter.py` | ASSUMPTION: written against upstream `8edfeb3`; no checkpoint loaded, no torch in this env |
| 4 | Linear readouts on real and predicted latents | `learning/readouts.py` | PROPOSED |
| 5 | Horizon metrics, free vs interaction, shuffled actions, baselines | `learning/prediction_eval.py` | PROPOSED |
| 6 | Planner (64 candidates, scoring) | `planning/` (pre-existing) | PROPOSED; weights still defaults, not validation-tuned |
| 7 | Surprise calibration | `learning/surprise_fit.py` + `runtime/surprise.py` | PROPOSED: calibration code path only |
| 8 | Bundle publication (ML → SIM handoff) | `learning/bundle_io.py` | PROPOSED: checksummed, content-addressed, reference replay on load |
| — | Linear latent baseline (pooled pixels + ridge dynamics) | `learning/baseline_model.py` | PROPOSED: the floor LeWM must beat |

## Commands

Requires NumPy: `uv sync --extra dev` (dev now includes the `ml` extra).

```bash
make ml-smoke                                   # whole ML pipeline on fake-sim data
uv run python -m go2wm.learning make-splits --seeds 1-400 --reserve-test <ids> --salt <salt> --out artifacts/splits.json
uv run python -m go2wm.learning check-data --data data/<wave> --splits artifacts/splits.json
uv run python -m go2wm.learning baseline --data data/<wave> --splits artifacts/splits.json --out runs/<run_id> --dataset-id <id>
uv run python -m go2wm.learning export-lewm --data data/<wave> --splits artifacts/splits.json --split train --out runs/<run_id>/train.npz
```

`baseline` writes `ml_report.json` (coverage, leak check, readout vs constant
mean, per-horizon errors for model / shuffled / encoded-real / persistence /
kinematic, split into free and interaction windows, surprise threshold, one
planning cycle, and G2/G3 checks) plus an immutable bundle under `bundles/`.

## Fake-backend smoke result (software-contract evidence only)

Run `runs/ml-smoke-20260919T185156Z` (Python 3.10.12, 60 fake episodes, bundle
`go2wm-95dcb74baa3c94ba`). Full suite at the same time: 90 passed (63
existing + 27 new), Ruff clean, line coverage 85%, `doctor`/`fake-smoke`/
`candidates`/`verify_protocol` all OK.

- G2 checks passed: real-frame readout beats the constant mean
  (robot 0.31 m vs 0.74 m median; objects 0.21 m vs 0.67 m). Split leak check
  passed. Bundle reload replays the reference packet.
- G3 checks **failed** for the linear pooled-pixel baseline: one-step robot
  error 0.27 m vs 0.02 m for persistence, and matched commands do not beat
  shuffled ones (-3%). The readout on the *true* next frame is already 0.31 m,
  so the bottleneck is the 8x8 pooled encoder, not the dynamics: at 32x32 a
  pooled cell is 0.4 m wide and cannot see a 0.2 m move. This is the expected
  behavior of a weak baseline and shows the gate catches it; it says nothing
  about LeWM.
- Coverage warnings fired: the fake scripted scenarios give 50–74% contact
  blocks against the 20–30% target, with resisted blocks dominating. Real
  collection should be checked with `check-data` after every wave.

## Decisions to raise with SIM (not yet accepted)

- **D-017 (proposed): NumPy is an ML-only dependency.** Core contracts,
  planner, and runtime stay dependency-free. `go2wm.learning` requires NumPy
  via the `ml` extra; its tests skip cleanly without it.
- **D-018 (proposed): LeWM action alignment.** LeWM predicts `emb[t+1]` from
  `emb[t-2..t]` and `act[t-2..t]` where `act[t]` is the command applied from
  frame `t`. With our 3-frame/2-command history, the first predicted latent
  therefore consumes the two history commands plus candidate command 0. The
  export writes `action[i]` = command applied from frame `i`, NaN on the last
  frame of each episode. SIM should confirm this matches the collector's
  "action at block k is applied from `t_k`" rule (it does in the fake path;
  `tests/test_learning_models.py::test_export_rows_pair_each_frame_with_its_outgoing_command`).
- **D-019 (proposed): action normalization belongs to the bundle.** Upstream
  z-scores the action column with training statistics; those numbers go in
  the LeWM encoder/predictor config and therefore change the bundle id.

## Fixes to shared files (flag to SIM)

- `src/go2wm/config.py` imported `tomllib`, which does not exist on Python
  3.10. LeWM's documented environment is Python 3.10, and CI tests 3.10. Now
  falls back to `tomli` (added as a `python_version < "3.11"` dependency).

## Next ML steps, in gate order

1. G1: run `check-data` on SIM's 20-block smoke packet; sign the packet only if
   it loads strictly and 10 inspected blocks match.
2. Freeze `artifacts/splits.json` with the 24 reserved evaluation scenario
   seeds forced to test, before any fitting.
3. G2: run `baseline` on wave 1 to get the floor; set up the LeWM env
   (Python 3.10, `stable-worldmodel[train,env]`, pinned upstream commit) on
   the CUDA machine and train from `export-lewm` output (a loader shim from
   the npz columns to `swm.data` still needs writing and checking).
4. Register a `lewm` component loader with `register_component_loader` and
   re-run the same evaluation/bundle path with the LeWM encoder/predictor.
5. G3: compare LeWM against the linear baseline, persistence, and kinematic
   references per horizon; run the shuffled-action ablation.
