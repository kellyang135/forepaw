"""Adapter from the upstream LeWM ``JEPA`` module to this repository's contracts.

Reviewed against lucas-maes/le-wm @ 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
(``jepa.py``, ``train.py``, ``utils.py``).  Upstream conventions this adapter
must preserve:

* ``action[t]`` is the command applied from frame ``t`` to ``t + 1``; training
  predicts ``emb[:, 1:4]`` from ``emb[:, :3]`` and ``act_emb[:, :3]``.  With
  three history frames the predictor therefore needs the two connecting
  history commands *plus the first candidate command*.
* Pixels go through ImageNet mean/std normalization and a resize to 224.
* Every non-pixel column, including ``action``, is z-scored with statistics
  computed from the training data.  Those statistics are part of the bundle.
* ``frameskip=1`` here, so the action encoder input dim is 2.

Nothing in this module is verified against a trained checkpoint yet.  PyTorch
is imported lazily so the dependency-light test suite does not need it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from go2wm.contracts import EpisodeRecord, RGBObservation
from go2wm.model import ActionBlock, Latent

from .readouts import TargetSpec
from .windows import episode_sequence

LEWM_REVIEWED_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgb(observation: RGBObservation, *, image_size: int = 224) -> np.ndarray:
    """RGB8 bytes -> normalized float32 ``(3, H, W)``; refuses silent resizing."""

    if (observation.width_px, observation.height_px) != (image_size, image_size):
        raise ValueError(
            f"expected {image_size}x{image_size} frames from the pinned camera, got "
            f"{observation.width_px}x{observation.height_px}"
        )
    image = np.frombuffer(observation.rgb, dtype=np.uint8).reshape(image_size, image_size, 3)
    scaled = image.astype(np.float32) / 255.0
    return ((scaled - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()


@dataclass(frozen=True, slots=True)
class ActionNormalizer:
    mean: tuple[float, float]
    std: tuple[float, float]

    @classmethod
    def fit(cls, actions: Iterable[ActionBlock]) -> ActionNormalizer:
        values = np.asarray([[a.forward_mps, a.yaw_rate_rps] for a in actions], dtype=np.float64)
        if len(values) < 2:
            raise ValueError("need at least two training actions")
        std = np.maximum(values.std(axis=0, ddof=1), 1e-6)
        mean = values.mean(axis=0)
        return cls((float(mean[0]), float(mean[1])), (float(std[0]), float(std[1])))

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


def export_sequences_npz(
    episodes: Iterable[EpisodeRecord],
    path: str | Path,
    *,
    spec: TargetSpec | None = None,
) -> dict[str, int]:
    """Flatten episodes into aligned columns for a LeWM-style training loader.

    Row ``i`` holds frame ``pixels[i]`` and ``action[i]``, the command applied
    from that frame to the next row of the same episode.  The final frame of
    every episode has a NaN action, matching upstream's boundary convention.
    ``state`` rows are privileged labels, stored only for readout probes.
    History warm-up frames have no labels, so their ``state`` rows are NaN.
    """

    pixels: list[np.ndarray] = []
    actions: list[list[float]] = []
    states: list[np.ndarray] = []
    episode_index: list[int] = []
    step_index: list[int] = []
    for number, episode in enumerate(episodes):
        observations, commands = episode_sequence(episode)
        spec = spec or TargetSpec(
            tuple(o.object_id for o in episode.reset_report.final_labels.objects)
        )
        labels_by_time = {}
        for block in episode.blocks:
            labels_by_time[block.transition.start_observation.observation_id] = (
                block.transition.start_labels
            )
            labels_by_time[block.transition.end_observation.observation_id] = (
                block.transition.end_labels
            )
        for step, observation in enumerate(observations):
            pixels.append(
                np.frombuffer(observation.rgb, dtype=np.uint8).reshape(
                    observation.height_px, observation.width_px, 3
                )
            )
            if step < len(commands):
                actions.append([commands[step].forward_mps, commands[step].yaw_rate_rps])
            else:
                actions.append([float("nan"), float("nan")])
            labels = labels_by_time.get(observation.observation_id)
            states.append(spec.encode(labels) if labels is not None else np.full(spec.dim, np.nan))
            episode_index.append(number)
            step_index.append(step)
    if not pixels:
        raise ValueError("no episodes to export")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    np.savez_compressed(
        target,
        pixels=np.stack(pixels),
        action=np.asarray(actions, dtype=np.float32),
        state=np.stack(states).astype(np.float32),
        state_names=np.asarray(spec.names if spec else ()),
        episode_index=np.asarray(episode_index, dtype=np.int32),
        step_index=np.asarray(step_index, dtype=np.int32),
    )
    return {"rows": len(pixels), "episodes": int(max(episode_index)) + 1}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LeWMWorldModel:
    """Implements both ``ObservationEncoder`` and ``ActionConditionedPredictor``.

    ``module`` is an upstream ``jepa.JEPA`` instance in eval mode.
    """

    def __init__(
        self,
        module: Any,
        *,
        action_normalizer: ActionNormalizer,
        checkpoint_path: str,
        checkpoint_sha256: str,
        device: str = "cpu",
        image_size: int = 224,
        history_frames: int = 3,
    ) -> None:
        import torch

        self._torch = torch
        self.module = module.to(device).eval()
        self.device = device
        self.action_normalizer = action_normalizer
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.image_size = image_size
        self.history_frames = history_frames

    @classmethod
    def from_checkpoint(cls, config: dict[str, Any], *, device: str = "cpu") -> LeWMWorldModel:
        import torch

        path = Path(config["checkpoint_path"])
        actual = sha256_file(path)
        if actual != config["checkpoint_sha256"]:
            raise ValueError(f"checkpoint checksum mismatch for {path}")
        module = torch.load(path, map_location=device, weights_only=False)
        return cls(
            module,
            action_normalizer=ActionNormalizer(
                tuple(config["action_mean"]), tuple(config["action_std"])
            ),
            checkpoint_path=str(path),
            checkpoint_sha256=actual,
            device=device,
            image_size=int(config.get("image_size", 224)),
            history_frames=int(config.get("history_frames", 3)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "action_mean": list(self.action_normalizer.mean),
            "action_std": list(self.action_normalizer.std),
            "image_size": self.image_size,
            "history_frames": self.history_frames,
            "upstream_commit": LEWM_REVIEWED_COMMIT,
        }

    def encode(self, observation: RGBObservation) -> Latent:
        torch = self._torch
        pixels = torch.from_numpy(preprocess_rgb(observation, image_size=self.image_size))
        info = {"pixels": pixels[None, None].to(self.device)}
        with torch.no_grad():
            emb = self.module.encode(info)["emb"][0, 0]
        return tuple(float(v) for v in emb.float().cpu().numpy())

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
        candidates: Sequence[tuple[ActionBlock, ...]],
    ) -> list[list[list[float]]]:
        """Mirror upstream ``JEPA.rollout`` but start from already encoded latents."""

        torch = self._torch
        count = len(candidates)
        horizons = {len(c) for c in candidates}
        if len(horizons) != 1:
            raise ValueError("all candidates in a batch must have the same horizon")
        steps = horizons.pop()
        contexts = [
            lewm_action_context(history_actions, c, history_frames=self.history_frames)
            for c in candidates
        ]
        act0 = np.stack([self.action_normalizer.apply(ctx) for ctx, _ in contexts])
        future = (
            np.stack([self.action_normalizer.apply(tail) for _, tail in contexts])
            if steps > 1
            else np.zeros((count, 0, 2), dtype=np.float32)
        )
        with torch.no_grad():
            emb = torch.tensor(np.asarray(history_latents, dtype=np.float32), device=self.device)
            emb = emb[None].expand(count, -1, -1).clone()
            act = torch.from_numpy(act0).to(self.device)
            act_future = torch.from_numpy(future).to(self.device)
            hs = self.history_frames
            predictions = []
            for t in range(steps):
                act_emb = self.module.action_encoder(act)
                pred = self.module.predict(emb[:, -hs:], act_emb[:, -hs:])[:, -1:]
                predictions.append(pred)
                emb = torch.cat([emb, pred], dim=1)
                if t < steps - 1:
                    act = torch.cat([act, act_future[:, t : t + 1]], dim=1)
            stacked = torch.cat(predictions, dim=1).float().cpu().numpy()
        if not np.isfinite(stacked).all():
            raise FloatingPointError("LeWM rollout produced non-finite latents")
        return stacked.tolist()
