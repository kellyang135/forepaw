"""Adapter from a trained LeWM ``JEPA`` to this repository's model contracts.

Reviewed against lucas-maes/le-wm @ 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
(``jepa.py``, ``train.py``, ``utils.py``).  Upstream conventions preserved:

* ``action[t]`` is the command applied from frame ``t`` to ``t + 1``; training
  predicts ``emb[:, 1:4]`` from ``emb[:, :3]`` and ``act_emb[:, :3]``.  With
  three history frames the predictor therefore needs the two connecting
  history commands *plus the first candidate command*.
* Pixels are scaled to [0, 1] and ImageNet-normalized; frames must already be
  224 x 224 (resizing silently would hide a camera-contract break).
* Actions are z-scored with training statistics that travel with the weights.

The numpy helpers here have no torch dependency.  ``LeWMWorldModel`` imports
torch lazily through ``lewm_model``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from go2wm.contracts import RGBObservation
from go2wm.model import ActionBlock, Latent

LEWM_REVIEWED_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
WEIGHTS_FILENAME = "lewm_weights.pt"


def rgb_array(observation: RGBObservation, *, image_size: int = 224) -> np.ndarray:
    """RGB8 bytes -> uint8 ``(H, W, 3)``; refuses frames of the wrong size."""

    if (observation.width_px, observation.height_px) != (image_size, image_size):
        raise ValueError(
            f"expected {image_size}x{image_size} frames from the pinned camera, got "
            f"{observation.width_px}x{observation.height_px}"
        )
    return np.frombuffer(observation.rgb, dtype=np.uint8).reshape(image_size, image_size, 3)


def preprocess_rgb(observation: RGBObservation, *, image_size: int = 224) -> np.ndarray:
    """RGB8 bytes -> normalized float32 ``(3, H, W)``, identical to training preprocessing."""

    scaled = rgb_array(observation, image_size=image_size).astype(np.float32) / 255.0
    return ((scaled - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()


@dataclass(frozen=True, slots=True)
class ActionNormalizer:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def apply(self, actions: Sequence[ActionBlock]) -> np.ndarray:
        raw = np.asarray([[a.forward_mps, a.yaw_rate_rps] for a in actions], dtype=np.float32)
        return (raw - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )


def lewm_action_context(
    history_actions: Sequence[ActionBlock],
    candidate_actions: Sequence[ActionBlock],
    *,
    history_frames: int = 3,
) -> tuple[list[ActionBlock], list[ActionBlock]]:
    """Split commands into LeWM's ``act_0`` (one per history frame) and the future tail."""

    if len(history_actions) != history_frames - 1:
        raise ValueError(
            f"{history_frames} history frames need {history_frames - 1} connecting commands"
        )
    if not candidate_actions:
        raise ValueError("candidate must contain at least one command")
    return [*history_actions, candidate_actions[0]], list(candidate_actions[1:])


class LeWMWorldModel:
    """Implements ``ObservationEncoder`` and ``ActionConditionedPredictor``.

    Wraps a trained upstream ``jepa.JEPA`` in eval mode.  ``rollout_batch``
    predicts many candidates in one forward pass per step, which is what the
    planner uses for its 64 candidates.
    """

    def __init__(self, trained: Any, *, device: Any = "cpu") -> None:
        import torch

        from . import lewm_model

        self._torch = torch
        self._lm = lewm_model
        self.trained = trained
        self.device = torch.device(device)
        self.model = trained.model.to(self.device).eval()
        self.arch = trained.arch
        self.normalizer = ActionNormalizer(trained.action_stats.mean, trained.action_stats.std)

    @property
    def latent_dim(self) -> int:
        return int(self.arch.embed_dim)

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        *,
        checkpoint: str = "best",
        lewm_repo: str | Path | None = None,
        device: str = "cpu",
    ) -> LeWMWorldModel:
        from . import lewm_model

        src = lewm_model.import_lewm(lewm_repo) if lewm_repo else None
        trained = lewm_model.load_trained(run_dir, checkpoint=checkpoint, src=src, device=device)
        return cls(trained, device=device)

    @classmethod
    def from_bundle_config(
        cls, config: dict[str, Any], root: Path, context: dict[str, Any]
    ) -> LeWMWorldModel:
        """Rebuild from a published bundle: weights file inside the bundle, verified by hash."""

        from . import lewm_model

        cache = context.setdefault("_lewm_instances", {})
        key = config["weights_sha256"]
        if key in cache:
            return cache[key]
        src = lewm_model.import_lewm(
            context.get("lewm_repo"),
            expected_commit=config["lewm_commit"],
        )
        arch = lewm_model.LeWMArchitecture.from_dict(config["architecture"])
        model = lewm_model.build_model(arch, src)
        weights = root / config["weights_file"]
        lewm_model.load_weights(weights, model, expected_sha256=config["weights_sha256"])
        trained = lewm_model.TrainedLeWM(
            model=model,
            arch=arch,
            action_stats=lewm_model.ActionStats(
                tuple(config["action_mean"]), tuple(config["action_std"])
            ),
            weights_path=weights,
            weights_sha256=config["weights_sha256"],
            run_config={},
        )
        instance = cls(trained, device=context.get("device", "cpu"))
        cache[key] = instance
        return instance

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_file": WEIGHTS_FILENAME,
            "weights_sha256": self.trained.weights_sha256,
            "architecture": self.arch.to_dict(),
            "action_mean": list(self.normalizer.mean),
            "action_std": list(self.normalizer.std),
            "lewm_commit": self.trained.run_config.get("lewm", {}).get(
                "commit", LEWM_REVIEWED_COMMIT
            ),
            "source_run": self.trained.run_config.get("run_id", ""),
        }

    def encode(self, observation: RGBObservation) -> Latent:
        return tuple(self.encode_batch([observation])[0].tolist())

    def encode_batch(
        self, observations: Sequence[RGBObservation], batch_size: int = 64
    ) -> np.ndarray:
        torch = self._torch
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(observations), batch_size):
                chunk = observations[start : start + batch_size]
                raw = np.stack([rgb_array(o, image_size=self.arch.image_size) for o in chunk])
                pixels = torch.from_numpy(raw).permute(0, 3, 1, 2).to(self.device)
                info = {"pixels": self._lm.preprocess_pixels(pixels)[:, None]}
                emb = self.model.encode(info)["emb"][:, 0]
                outputs.append(emb.float().cpu().numpy())
        result = np.concatenate(outputs) if outputs else np.zeros((0, self.latent_dim))
        if not np.isfinite(result).all():
            raise FloatingPointError("LeWM encoder produced non-finite latents")
        return result

    def rollout(
        self,
        history_latents: tuple[Latent, ...],
        history_actions: tuple[ActionBlock, ...],
        candidate_actions: tuple[ActionBlock, ...],
    ) -> Sequence[Sequence[float]]:
        return self.rollout_batch(history_latents, history_actions, [candidate_actions])[0]

    def rollout_batch(
        self,
        history_latents: tuple[Latent, ...],
        history_actions: tuple[ActionBlock, ...],
        candidates: Sequence[Sequence[ActionBlock]],
    ) -> list[list[list[float]]]:
        """Mirror upstream ``JEPA.rollout`` but start from already encoded latents."""

        torch = self._torch
        count = len(candidates)
        horizons = {len(c) for c in candidates}
        if len(horizons) != 1:
            raise ValueError("all candidates in a batch must have the same horizon")
        steps = horizons.pop()
        hs = self.arch.history_frames
        if len(history_latents) != hs:
            raise ValueError(f"expected {hs} history latents, got {len(history_latents)}")
        contexts = [lewm_action_context(history_actions, c, history_frames=hs) for c in candidates]
        act0 = np.stack([self.normalizer.apply(ctx) for ctx, _ in contexts])
        future = (
            np.stack([self.normalizer.apply(tail) for _, tail in contexts])
            if steps > 1
            else np.zeros((count, 0, 2), dtype=np.float32)
        )
        with torch.no_grad():
            emb = torch.tensor(np.asarray(history_latents, dtype=np.float32), device=self.device)
            emb = emb[None].expand(count, -1, -1).clone()
            act = torch.from_numpy(act0).to(self.device)
            act_future = torch.from_numpy(future).to(self.device)
            predictions = []
            for t in range(steps):
                act_emb = self.model.action_encoder(act)
                pred = self.model.predict(emb[:, -hs:], act_emb[:, -hs:])[:, -1:]
                predictions.append(pred)
                emb = torch.cat([emb, pred], dim=1)
                if t < steps - 1:
                    act = torch.cat([act, act_future[:, t : t + 1]], dim=1)
            stacked = torch.cat(predictions, dim=1).float().cpu().numpy()
        if not np.isfinite(stacked).all():
            raise FloatingPointError("LeWM rollout produced non-finite latents")
        return stacked.tolist()
