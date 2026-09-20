# LeWM known-answer control: PASSED (2026-09-20)

Software evidence only (synthetic data, `rehearsal_only`). It shows the LeWM
training path can learn action-conditioned dynamics; it says nothing about Go2 data.

- Command: `LEWM_TRAINING.md` §3 exactly (`synthetic-control`, then `lewm_train`
  30 epochs, batch 16, 1,080 train / 180 val clips, 2,010 steps).
- Host: Apple M4, MPS, torch 2.14.0, le-wm `8edfeb3`. 25 min wall, 20-45 clips/s.
- Run dir (local, gitignored): `runs/control/lewm-mps/`, best `epoch_029.pt`.

| Criterion (§3) | Required | Epoch 30 |
| --- | --- | --- |
| val pred vs copy-last | well below | 0.0129 vs 0.1617 |
| shuffled-action gap | tens of percent | +95.1% |
| action path (AdaLN norm) | grows from 0 | 0.73 (ep 1) to 3.56 |
| embedding std | well above 0 | mean 0.86, min 0.13 |

Trajectory of the gap: +0.7, +1.4, +3.3, +4.1, +3.6, +8.0 % (epochs 1-6), then
+40.5 % at epoch 7, +80 % at epoch 10, +88-95 % from epoch 11 on. Action use
switches on after roughly 400 steps; the earlier 460-step CPU result (1-2 % gap)
was too short, not a pipeline defect. On real data, do not judge G3 from runs
shorter than about 1,000 steps.

Fix required to run on MPS: precise-BN recalibration ran the whole model in
train mode, so attention dropout reached MPS fused attention under `no_grad`
(`NotImplementedError: scaled_dot_product_attention for MPS does not support
dropout`). `recalibrate_batchnorm` now sets the model to eval and only the
BatchNorm layers to train. `tests/test_lewm_torch.py` passes (7/7) with the fix.
