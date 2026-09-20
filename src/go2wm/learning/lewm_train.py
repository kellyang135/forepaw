"""Train LeWM on go2wm training caches (Apple MPS, CUDA, or CPU).

Usage (inside the le-wm environment, with go2wm importable)::

    python -m go2wm.learning.lewm_train \\
        --lewm-repo ~/code/le-wm \\
        --train-cache runs/wave1/cache-train --val-cache runs/wave1/cache-validation \\
        --out runs/wave1/lewm-001 --epochs 20

The loop reproduces upstream ``train.py`` (same model, loss, AdamW settings,
gradient clipping, linear-warmup cosine schedule, drop_last training batches)
in plain PyTorch, with these deliberate differences:

* validation uses the frozen episode-level split, never a random window split;
* fp32 everywhere (bf16 autocast is unreliable on MPS);
* checkpoints are weights-only state dicts with SHA-256s, written atomically;
* every epoch also reports a shuffled-action validation loss and embedding
  spread, so action-blindness and representation collapse show up during
  training instead of after it.

Nothing here reads test data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import lewm_model as lm
from .lewm_cache import (
    TrainingCache,
    action_statistics,
    check_disjoint,
    read_training_cache,
)

RUN_SCHEMA = lm.CHECKPOINT_FORMAT


class ClipDataset(Dataset):
    """``num_steps``-frame clips that never cross an episode boundary.

    Pixels stay on disk: each worker opens its own read-only memmap, so
    DataLoader workers never pickle the frame array.
    """

    def __init__(
        self,
        cache: TrainingCache,
        num_steps: int,
        stats: lm.ActionStats,
        *,
        limit: int | None = None,
    ) -> None:
        self.pixels_path = cache.root / "pixels.npy"
        self.num_steps = num_steps
        self.starts = cache.clip_starts(num_steps)
        if limit is not None:
            self.starts = self.starts[:limit]
        if len(self.starts) == 0:
            raise ValueError(f"{cache.root}: no episode has {num_steps} frames")
        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        self.actions = np.nan_to_num((cache.action - mean) / std, nan=0.0).astype(np.float32)
        self._pixels: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_pixels"] = None
        return state

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._pixels is None:
            self._pixels = np.load(self.pixels_path, mmap_mode="r")
        start = int(self.starts[index])
        end = start + self.num_steps
        pixels = np.array(self._pixels[start:end], copy=True)
        return {
            "pixels": torch.from_numpy(pixels).permute(0, 3, 1, 2),
            "action": torch.from_numpy(self.actions[start:end]),
        }


def warmup_cosine(step: int, *, total_steps: int, warmup_steps: int) -> float:
    """Multiplier matching stable-pretraining ``LinearWarmupCosineAnnealingLR`` (eta_min=0)."""

    if step < warmup_steps:
        return step / warmup_steps
    span = max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(step - warmup_steps, span) / span))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    sigreg: torch.nn.Module,
    loader: DataLoader,
    arch: lm.LeWMArchitecture,
    loss_cfg: lm.LossConfig,
    device: torch.device,
    *,
    max_batches: int | None,
    seed: int,
) -> dict[str, float]:
    """Mean losses plus three references: shuffled actions, copy-last-latent, spread.

    ``pred_loss`` should fall clearly below ``copy_last_loss`` (the model predicts
    change, not just persistence) and below ``shuffled_pred_loss`` (it uses the
    commands).
    """

    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = {
        "pred_loss": 0.0,
        "sigreg_loss": 0.0,
        "loss": 0.0,
        "shuffled_pred_loss": 0.0,
        "copy_last_loss": 0.0,
    }
    embeddings: list[torch.Tensor] = []
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        pixels = lm.preprocess_pixels(batch["pixels"].to(device))
        actions = batch["action"].to(device)
        out = lm.lejepa_losses(model, sigreg, pixels, actions, arch, loss_cfg)
        for key in ("pred_loss", "sigreg_loss", "loss"):
            totals[key] += out[key].item()
        # "Nothing changes" reference in latent space: predict emb[t+1] = emb[t].
        totals["copy_last_loss"] += (
            (out["emb"][:, : arch.history_frames] - out["emb"][:, arch.num_preds :])
            .pow(2)
            .mean()
            .item()
        )
        if actions.shape[0] > 1:
            order = torch.randperm(actions.shape[0], generator=generator)
            if torch.equal(order, torch.arange(actions.shape[0])):
                order = torch.roll(order, 1)
            wrong_act = model.action_encoder(actions[order.to(device)])
            shuffled = model.predict(
                out["emb"][:, : arch.history_frames], wrong_act[:, : arch.history_frames]
            )
            totals["shuffled_pred_loss"] += float(
                (shuffled - out["emb"][:, arch.num_preds :]).pow(2).mean()
            )
        else:
            totals["shuffled_pred_loss"] += float("nan")
        embeddings.append(out["emb"][:, 0].float().cpu())
        batches += 1
    if batches == 0:
        raise ValueError("validation loader produced no batches")
    result = {key: value / batches for key, value in totals.items()}
    emb = torch.cat(embeddings)
    per_dim_std = emb.std(dim=0) if emb.shape[0] > 1 else torch.zeros(emb.shape[1])
    result["emb_std_mean"] = float(per_dim_std.mean())
    result["emb_std_min"] = float(per_dim_std.min())
    result["action_gap"] = (
        (result["shuffled_pred_loss"] - result["pred_loss"]) / result["shuffled_pred_loss"]
        if result["shuffled_pred_loss"] > 0
        else float("nan")
    )
    result["batches"] = batches
    return result


@torch.no_grad()
def recalibrate_batchnorm(
    model: torch.nn.Module,
    loader: DataLoader,
    stats_batches: int,
    device: torch.device,
) -> int:
    """Recompute BatchNorm running statistics for the current weights ("precise BN").

    LeWM's projector and predictor-projector use BatchNorm.  During training the
    running averages (momentum 0.1) lag the fast-changing weights, so eval-mode
    outputs, which planning uses, can be badly off-scale mid-training even when
    the training loss is smooth.  This pass only replaces running mean/var with
    an exact average over ``stats_batches`` training batches; weights, optimizer
    state, and train-mode forward passes are unchanged.  Returns batches used.
    """

    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not norms or stats_batches <= 0:
        return 0
    was_training = model.training
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over the batches below
    # Only BatchNorm layers collect statistics; dropout stays off so the
    # statistics match eval-mode planning (and MPS fused attention, which
    # does not support dropout under no_grad, can run).
    model.eval()
    for m in norms:
        m.train()
    used = 0
    for batch in loader:
        if used >= stats_batches:
            break
        pixels = lm.preprocess_pixels(batch["pixels"].to(device))
        actions = batch["action"].to(device)
        out = model.encode({"pixels": pixels, "action": actions})
        model.predict(out["emb"][:, :3], out["act_emb"][:, :3])
        used += 1
    for m, momentum in zip(norms, momenta, strict=True):
        m.momentum = momentum
    model.train(was_training)
    return used


def action_pathway_norm(model: torch.nn.Module) -> float:
    """Frobenius norm of the predictor's AdaLN modulation weights.

    Upstream initializes these to zero (AdaLN-zero), so commands have exactly no
    influence at step 0.  This number growing from 0 is the direct sign that the
    predictor is learning to condition on actions.
    """

    total = 0.0
    for module in model.predictor.modules():
        modulation = getattr(module, "adaLN_modulation", None)
        if modulation is not None:
            total += float(modulation[-1].weight.detach().float().pow(2).sum())
    return math.sqrt(total)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({k: _jsonable(v) for k, v in record.items()}, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _go2wm_revision() -> str:
    from .pipeline import git_revision

    return git_revision()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m go2wm.learning.lewm_train",
        description="Train LeWM on go2wm caches. Never reads test data.",
    )
    parser.add_argument("--lewm-repo", default=os.environ.get("GO2WM_LEWM_REPO"))
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--bn-recalibration-batches",
        type=int,
        default=20,
        help="precise-BN pass over N train batches before each validation/checkpoint (0 = off)",
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="preflight: repeatedly train on the first N train batches and expect loss to fall",
    )
    parser.add_argument("--resume", action="store_true", help="continue from out/state_last.pt")
    parser.add_argument("--allow-lewm-commit-mismatch", action="store_true")
    parser.add_argument("--verify-caches", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arch = lm.LeWMArchitecture()
    loss_cfg = lm.LossConfig()
    out: Path = args.out
    checkpoints = out / "checkpoints"
    metrics_path = out / "metrics.jsonl"

    src = lm.import_lewm(args.lewm_repo, allow_commit_mismatch=args.allow_lewm_commit_mismatch)
    train_cache = read_training_cache(args.train_cache, verify=args.verify_caches)
    val_cache = read_training_cache(args.val_cache, verify=args.verify_caches)
    if train_cache.split != "train" or val_cache.split != "validation":
        raise SystemExit(
            f"expected train + validation caches, got {train_cache.split} + {val_cache.split}"
        )
    check_disjoint([train_cache, val_cache])
    if train_cache.manifest["image_size"] != arch.image_size:
        raise SystemExit("cache image size does not match the model contract (224)")
    rehearsal = bool(
        train_cache.manifest.get("rehearsal_only") or val_cache.manifest.get("rehearsal_only")
    )
    if rehearsal:
        print(
            "*** REHEARSAL: fake-simulator data. This run tests the software path only. "
            "Do not hand its checkpoints or bundles to SIM or cite them as model evidence "
            "(docs/GO2_CONTROLLER_RECOVERY.md section 2). ***",
            flush=True,
        )
    mean, std = action_statistics(train_cache)
    stats = lm.ActionStats(mean, std)
    device = lm.pick_device(args.device)
    seed_everything(args.seed)

    train_ds = ClipDataset(train_cache, arch.num_steps, stats)
    val_ds = ClipDataset(val_cache, arch.num_steps, stats)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {"num_workers": args.num_workers}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.overfit_batches == 0,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    # Validation is small; worker processes would only add memory (each imports torch).
    # A fixed shuffle mixes episodes within each batch, so the shuffled-action check
    # swaps in commands from other episodes rather than from the neighbouring block.
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise SystemExit(
            f"only {len(train_ds)} training clips; need at least one full batch of "
            f"{args.batch_size} (lower --batch-size or collect more data)"
        )

    model = lm.build_model(arch, src).to(device)
    sigreg = lm.build_sigreg(loss_cfg, src).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(train_loader), args.overfit_batches or len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(args.warmup_fraction * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine(step, total_steps=total_steps, warmup_steps=warmup_steps),
    )

    run_config = {
        "format": RUN_SCHEMA,
        "run_id": out.name,
        "rehearsal_only": rehearsal,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, "-m", "go2wm.learning.lewm_train", *(argv or sys.argv[1:])],
        "architecture": arch.to_dict(),
        "loss": asdict(loss_cfg),
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "schedule": "linear warmup + cosine to 0 (per step)",
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "drop_last": True,
            "precision": "fp32",
        },
        "action_stats": {"mean": list(stats.mean), "std": list(stats.std), "source": "train cache"},
        "data": {
            "dataset_id": train_cache.manifest["dataset_id"],
            "split_id": train_cache.manifest["split_id"],
            "train_cache": str(args.train_cache),
            "val_cache": str(args.val_cache),
            "train_cache_files": train_cache.manifest["files"],
            "val_cache_files": val_cache.manifest["files"],
            "train_clips": len(train_ds),
            "val_clips": len(val_ds),
            "train_episodes": train_cache.manifest["episode_count"],
            "val_episodes": val_cache.manifest["episode_count"],
        },
        "lewm": {"repo": str(src.repo), "commit": src.commit},
        "go2wm_revision": _go2wm_revision(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "seed": args.seed,
        "overfit_batches": args.overfit_batches,
        "bn_recalibration_batches": args.bn_recalibration_batches,
        "parameters": lm.count_parameters(model),
    }

    start_epoch = 1
    global_step = 0
    index: dict[str, Any] = {
        "checkpoints": {},
        "best": None,
        "last": None,
        "best_val_pred_loss": None,
    }
    if args.resume:
        state_path = out / "state_last.pt"
        if not state_path.is_file():
            raise SystemExit(f"--resume given but {state_path} does not exist")
        previous = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        for key in ("architecture", "loss", "action_stats", "data", "lewm", "seed"):
            if key == "data":
                same = {
                    k: previous[key][k] for k in ("dataset_id", "split_id", "train_cache_files")
                } == {
                    k: run_config[key][k] for k in ("dataset_id", "split_id", "train_cache_files")
                }
            else:
                same = previous[key] == run_config[key]
            if not same:
                raise SystemExit(f"--resume refused: {key} differs from the original run")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        index = json.loads((checkpoints / "index.json").read_text(encoding="utf-8"))
        run_config = previous
        print(f"resuming after epoch {start_epoch - 1} (step {global_step})", flush=True)
    else:
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not empty; use a new --out or --resume")
        checkpoints.mkdir(parents=True, exist_ok=True)
        lm.write_json_atomic(out / "run_config.json", run_config)

    print(
        f"LeWM {run_config['parameters'] / 1e6:.1f}M params | device {device} | "
        f"{len(train_ds)} train clips, {len(val_ds)} val clips | "
        f"{steps_per_epoch} steps/epoch x {args.epochs} epochs (total {total_steps})",
        flush=True,
    )

    overfit_batches: list[dict[str, torch.Tensor]] | None = None
    if args.overfit_batches:
        overfit_batches = []
        for batch in train_loader:
            overfit_batches.append(batch)
            if len(overfit_batches) >= args.overfit_batches:
                break

    started = time.perf_counter()
    stop = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        seen = 0
        running: dict[str, float] = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
        batches = overfit_batches if overfit_batches is not None else train_loader
        for batch in batches:
            if global_step >= total_steps:
                stop = True
                break
            pixels = lm.preprocess_pixels(batch["pixels"].to(device, non_blocking=True))
            actions = batch["action"].to(device, non_blocking=True)
            out_losses = lm.lejepa_losses(model, sigreg, pixels, actions, arch, loss_cfg)
            loss = out_losses["loss"]
            if not torch.isfinite(loss):
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "nonfinite_loss",
                        "step": global_step,
                        "epoch": epoch,
                        "pred_loss": out_losses["pred_loss"].detach().item(),
                        "sigreg_loss": out_losses["sigreg_loss"].detach().item(),
                    },
                )
                print(f"non-finite loss at step {global_step}; stopping", flush=True)
                return 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            optimizer.step()
            scheduler.step()
            global_step += 1
            seen += pixels.shape[0]
            for key in running:
                running[key] += out_losses[key].detach().item()
            if global_step % args.log_every == 0 or global_step == 1:
                elapsed = time.perf_counter() - epoch_start
                record = {
                    "event": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss.detach().item(),
                    "pred_loss": out_losses["pred_loss"].detach().item(),
                    "sigreg_loss": out_losses["sigreg_loss"].detach().item(),
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "clips_per_s": seen / max(elapsed, 1e-9),
                }
                _append_jsonl(metrics_path, record)
                print(
                    f"epoch {epoch} step {global_step}/{total_steps} loss {record['loss']:.4f} "
                    f"(pred {record['pred_loss']:.4f}, sigreg {record['sigreg_loss']:.3f}) "
                    f"lr {record['lr']:.2e} {record['clips_per_s']:.1f} clips/s",
                    flush=True,
                )
        steps_this_epoch = max(1, round(seen / args.batch_size))
        train_means = {f"train_{k}": v / steps_this_epoch for k, v in running.items()}
        bn_batches = recalibrate_batchnorm(
            model,
            train_loader if overfit_batches is None else overfit_batches,
            args.bn_recalibration_batches,
            device,
        )
        val = validate(
            model,
            sigreg,
            val_loader,
            arch,
            loss_cfg,
            device,
            max_batches=args.val_max_batches,
            seed=args.seed + epoch,
        )
        name = f"epoch_{epoch:03d}.pt"
        sha = lm.save_weights(checkpoints / name, model)
        index["checkpoints"][name] = {
            "epoch": epoch,
            "step": global_step,
            "sha256": sha,
            "val_pred_loss": val["pred_loss"],
            "val_loss": val["loss"],
        }
        index["last"] = name
        if index["best_val_pred_loss"] is None or val["pred_loss"] < index["best_val_pred_loss"]:
            index["best"] = name
            index["best_val_pred_loss"] = val["pred_loss"]
        lm.write_json_atomic(checkpoints / "index.json", index)
        state_tmp = out / ".state_last.pt.tmp"
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            state_tmp,
        )
        os.replace(state_tmp, out / "state_last.pt")
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "event": "epoch",
            "epoch": epoch,
            "step": global_step,
            "epoch_seconds": epoch_seconds,
            "clips_per_s": seen / max(epoch_seconds, 1e-9),
            **train_means,
            **{f"val_{k}": v for k, v in val.items()},
            "checkpoint": name,
            "is_best": index["best"] == name,
            "bn_recalibration_batches": bn_batches,
            "action_pathway_norm": action_pathway_norm(model),
        }
        _append_jsonl(metrics_path, record)
        print(
            f"== epoch {epoch}: train loss {train_means['train_loss']:.4f} | val pred "
            f"{val['pred_loss']:.4f} vs shuffled-action {val['shuffled_pred_loss']:.4f} "
            f"(gap {val['action_gap']:+.1%}), copy-last {val['copy_last_loss']:.4f} | "
            f"emb std mean {val['emb_std_mean']:.2e} min {val['emb_std_min']:.2e} | "
            f"action path {record['action_pathway_norm']:.3f} | "
            f"{epoch_seconds / 60:.1f} min" + ("  [best]" if index["best"] == name else ""),
            flush=True,
        )
        if stop:
            break

    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": time.perf_counter() - started,
        "global_step": global_step,
        "best": index["best"],
        "best_val_pred_loss": index["best_val_pred_loss"],
        "last": index["last"],
        "stopped_early": stop,
    }
    lm.write_json_atomic(out / "summary.json", summary)
    run_config["best"] = {"checkpoint": index["best"], "val_pred_loss": index["best_val_pred_loss"]}
    lm.write_json_atomic(out / "run_config.json", run_config)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
