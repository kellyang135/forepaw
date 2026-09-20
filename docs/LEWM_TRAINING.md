# LeWM training runbook (Person B)

How to go from collected go2wm episodes to a trained LeWM checkpoint, a G2/G3
report, and a published model bundle. Everything below also runs today on
fake-simulator data, so the whole path can be rehearsed before the real data
arrives. Fake-data numbers are software-contract evidence only.

```text
episodes (data/<wave>/)  ->  lewm-cache  ->  lewm_train  ->  lewm-eval  ->  bundle
      strict loader          lossless,       real LeWM,      same G2/G3      weights +
      + split manifest       split-pure      MPS/CUDA/CPU    code as the     readout +
                             .npy columns    checkpoints     baseline        surprise
```

## 1. What is and is not upstream LeWM

The model, loss, and optimizer come from `lucas-maes/le-wm` at commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The model classes are imported from
that checkout at runtime, not copied, and the code refuses any other commit
unless you pass an explicit override.

Checked by `tests/test_lewm_torch.py` against the upstream code itself:

- the model has the upstream `lewm.yaml` architecture (18.0M parameters);
- our loss equals upstream `train.py::lejepa_forward` on the same batch;
- our batched 64-candidate rollout equals upstream `JEPA.rollout`, which
  confirms the action alignment in decision D-022;
- batched and single-frame encoding agree.

Deliberate differences from upstream `train.py`, and why:

| Upstream | Here | Reason |
| --- | --- | --- |
| `frameskip: 5`, action dim from dataset | `frameskip = 1`, action dim 2 | our blocks are already 0.5 s `[forward, yaw_rate]` |
| random 90/10 split of *windows* | train/validation caches from the frozen *episode/seed* split | a random window split leaks episodes into validation |
| Lance/folder datasets (JPEG frames) | raw `uint8` `.npy` memmaps | JPEG would make training pixels differ from the raw RGB the controller sees |
| `bf16` on GPU | fp32 | bf16 is unreliable on Apple MPS |
| Lightning + stable-pretraining `Manager`, W&B | plain PyTorch loop, JSONL metrics | fewer moving parts on a laptop; every number is in `metrics.jsonl` |
| whole-object `torch.save` checkpoints | weights-only state dicts + SHA-256 + `run_config.json` | loading never unpickles code; bundles verify weights |
| BatchNorm running stats as trained | recomputed over 20 train batches before each validation/checkpoint ("precise BN") | lagging stats made eval-mode losses swing 0.5 -> 42 -> 0.2 on the rehearsal while training was smooth; planning runs in eval mode |

The optimizer is otherwise unchanged: AdamW lr 5e-5, weight decay 1e-3,
gradient clipping 1.0, linear warmup over 1% of steps, then cosine to 0,
`drop_last` training batches, SIGReg weight 0.09 (17 knots, 1024 projections).

## 2. One-time setup on the Mac

1. **Move the repo out of iCloud.** `~/Documents` on this Mac is synced with
   "Optimize Mac Storage"; source files were being evicted ("Resource deadlock
   avoided" errors). Datasets and checkpoints are far larger. Use e.g.
   `~/code/hackmit` and `~/code/le-wm`, or turn off Optimize Mac Storage.
2. **LeWM environment** (Python 3.10, as upstream requires):

   ```bash
   cd ~/code
   git clone https://github.com/lucas-maes/le-wm && cd le-wm
   git checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
   uv venv --python=3.10 && source .venv/bin/activate
   uv pip install "stable-worldmodel[train]"   # [env] needs `brew install swig`; not required
   uv pip install -e ~/code/hackmit            # makes `go2wm` importable here
   export GO2WM_LEWM_REPO=~/code/le-wm         # add to your shell profile
   ```

3. **Prove the install** (about 30 s on the Mac):

   ```bash
   cd ~/code/hackmit
   python -m pytest -q tests/test_lewm_torch.py tests/test_lewm_cache.py
   ```

   All tests must pass. `test_loss_matches_upstream_lejepa_forward` skips if
   upstream `train.py` cannot import (it needs `imageio`, from the `[env]`
   extra); `uv pip install imageio` makes it run.

## 3. Positive control: can this path learn action-conditioned dynamics?

Before any real data, prove on a known-answer problem that LeWM + this loop
learns to use commands. A white square moves by exactly 40 px x command, with a
fresh random command every block, so history alone cannot predict the next
frame.

```bash
python -m go2wm.learning synthetic-control --out-dir runs/control
caffeinate -i python -m go2wm.learning.lewm_train \
    --train-cache runs/control/cache-train --val-cache runs/control/cache-validation \
    --out runs/control/lewm --epochs 30 --batch-size 16
```

The default control has 1,080 training clips, so that is about 2,000 steps (30 epochs x 67), roughly 20-30 minutes on the M4 at the measured 28 clips/s. Pass: by the end,
`val pred` is well below `copy-last` **and** the shuffled-action `gap` is
clearly positive (tens of percent), and `action path` (the AdaLN weight norm)
has grown from 0. On the cloud CPU, 460 steps gave pred 0.24 vs copy-last 0.43
but a gap of only 1-2%: action use is learned slowly because upstream's
AdaLN-zero conditioning starts at zero. If the gap is still near zero after
those 2,000 steps, do not move on to real data; report it in the ledger.

## 4. Rehearsal on fake data (software check only)

`docs/GO2_CONTROLLER_RECOVERY.md` section 2 says Person B must not train the
visual world model on fake data or treat fake-backend results as robot
evidence. The rehearsal below exists only to prove the software path, the
device, and the timings. Caches built from fake scenes are marked
`rehearsal_only`; training prints a banner, the run config records it, and
`lewm-eval` labels the report and bundle "REHEARSAL ONLY". Never hand these
checkpoints or bundles to SIM. If the team reads that rule as forbidding even a
software rehearsal, skip this section and run `tests/test_lewm_torch.py` only.

```bash
cd ~/code/hackmit
R=runs/rehearsal-$(date -u +%Y%m%dT%H%M%SZ)

python -m go2wm.learning fake-data --out $R --episodes 60 --image-size 224
python -m go2wm.learning lewm-cache --data $R/data --splits $R/splits.json \
    --out-dir $R --dataset-id fake224-60

# preflight: the model must be able to overfit 2 batches (loss should fall steadily)
python -m go2wm.learning.lewm_train --train-cache $R/cache-train \
    --val-cache $R/cache-validation --out $R/overfit --overfit-batches 2 \
    --epochs 30 --batch-size 16 --val-max-batches 2

# a short real run
caffeinate -i python -m go2wm.learning.lewm_train --train-cache $R/cache-train \
    --val-cache $R/cache-validation --out $R/lewm --epochs 5 --batch-size 32

python -m go2wm.learning lewm-eval --run $R/lewm --data $R/data \
    --splits $R/splits.json --out $R/eval-lewm --dataset-id fake224-60 \
    --backend fake --device mps
python -m go2wm.learning baseline --data $R/data --splits $R/splits.json \
    --out $R/eval-baseline --dataset-id fake224-60 --backend fake
python -m go2wm.learning compare $R/eval-baseline/ml_report.json $R/eval-lewm/ml_report.json
```

The fake simulator draws 3-pixel dots at 224 x 224, so do not expect LeWM to
learn much from it. The rehearsal proves the pipeline, the device, and the
timings, not the model.

## 5. Real data (after G1 is signed)

1. `check-data` on the wave, and look at 10 blocks yourself.
2. `make-splits` once, with the 24 reserved evaluation seeds in `--reserve-test`.
   Never rebuild it after training starts.
3. `lewm-cache` builds `cache-train` and `cache-validation`. Test episodes are
   never cached for training.
4. Overfit preflight, then the main `lewm_train` run.
5. `lewm-eval` on the best checkpoint, and `compare` against the baseline.
6. Record the run in the ledger: run dir, `run_config.json`, `summary.json`,
   the eval `ml_report.json`, and the bundle id.

### Optional G2 repair: training-only state auxiliary

Use this only after the plain LeWM run has passed synchronization, visibility,
coverage, and non-collapse checks but its real-frame readouts remain inadequate.
It is the single-component repair allowed by D-031 and the build specification:

```bash
python -m go2wm.learning.lewm_train \
    --train-cache $R/cache-train --val-cache $R/cache-validation \
    --out $R/lewm-aux --epochs 10 --batch-size 32 \
    --aux-state-weight 0.1 --aux-hidden-dim 128
```

The auxiliary head sees aligned training labels only. Its normalization is fit
on the training cache, its loss is logged as `aux_state_loss`, and its weights
exist only in `state_last.pt` so a run can resume. Published runtime checkpoints
contain the RGB/action LeWM weights, not the head; poses and box labels remain
forbidden runtime inputs. Changing the auxiliary weight or width requires a new
run. Refit every readout and surprise threshold before evaluating the result.

## 6. Reading the training output

Every epoch prints one line and appends to `metrics.jsonl` (numbers below are illustrative):

```text
== epoch 3: train loss 0.2150 | val pred 0.0412 vs shuffled-action 0.0530 (gap +22.3%), copy-last 0.0900 | emb std mean 3.1e-01 min 8.0e-02 | action path 1.200 | 5.9 min  [best]
```

| Signal | Healthy | Red flag and first thing to check |
| --- | --- | --- |
| `val pred` | falls, then flattens | rises while train falls: overfitting; stop and keep `best` |
| `gap` (shuffled-action vs real) | clearly positive and growing | about 0%: the model ignores actions or has not learned them yet. Check training length (see the positive control), action variety in the data (`check-data` coverage), and the D-022 alignment |
| `copy-last` | clearly above `val pred` | `val pred` about equal: the predictor only copies the last latent |
| `action path` | grows from 0 | stays near 0: the AdaLN action conditioning is not training |
| `emb std` mean/min | stays well above 0 | shrinks toward 0: representation collapse; check SIGReg loss and data variety |
| `sigreg` | falls and stays bounded | explodes: lower lr, check for corrupted frames |
| `aux` (when enabled) | train and validation fall without a widening gap | train falls while validation rises: auxiliary overfit; do not select on test data |
| non-finite loss | never | the run stops itself with exit code 2 and logs the step |

Checkpoints: `checkpoints/epoch_NNN.pt` (weights-only), `checkpoints/index.json`
(SHA-256, val losses, `best`, `last`), `state_last.pt` (for `--resume`),
`run_config.json` (everything needed to rebuild the model), `summary.json`.
`--resume` refuses if the architecture, loss, action statistics, auxiliary
configuration, data, LeWM commit, or seed changed.

`best` is chosen on validation prediction loss only. G3 is decided by
`lewm-eval`, not by the training loss.

## 7. Time and speed on this Mac (M4, MPS)

Measured with the real model and our contract on random tensors: about 28
clips/s at batch 16-64, and about 34 ms for 64 candidates x 6 blocks of
planning (model only). So roughly 6 min per epoch per 10k training clips; 20
epochs of a 10k-clip dataset is about 2 hours. Real data adds data-loading time;
use `--num-workers 2` and keep the caches on the internal SSD.

## 8. Troubleshooting

- `LeWMSetupError: le-wm checkout is at ...`: `git -C $GO2WM_LEWM_REPO checkout 8edfeb3...`.
- `CacheError: expected 224x224`: the simulator camera changed; fix it there,
  never resize in the cache.
- `CacheError: ~zero variance`: the collection never varied one command field.
- MPS "operator not implemented": `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
  automatically; if it is still slow, report which op in the ledger.
- Out of memory: lower `--batch-size` to 16; throughput barely changes.
- Laptop slept: `--resume` with the same arguments.
- Positive control gap stays near 0: first try `--lr 3e-4` as a diagnostic (on
  the cloud CPU, 200 steps at 3e-4 gave a larger gap than 460 steps at 5e-5);
  upstream's 5e-5 was tuned for batch 128 and far more steps. Record which
  setting passes before using it on real data.
