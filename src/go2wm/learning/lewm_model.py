"""LeWM model construction, loss, preprocessing, and checkpoints for our contract.

Everything here mirrors the reviewed upstream code at ``LEWM_REVIEWED_COMMIT``:

* architecture: ``config/train/model/lewm.yaml`` (ViT-tiny/14 at 224, 192-D
  embeddings, 6-layer autoregressive predictor, BatchNorm MLP projectors);
* objective: ``train.py::lejepa_forward`` (next-embedding MSE plus 0.09 x
  SIGReg with 17 knots and 1024 projections);
* pixels: ImageNet mean/std after scaling uint8 to [0, 1]
  (``utils.get_img_preprocessor``);
* actions: z-scored with training statistics, NaN -> 0 at episode ends.

Only three things differ from upstream and all follow from the Go2 contract:
``frameskip=1`` (blocks are already 0.5 s), 2-D actions, and fp32 (bf16 is
unreliable on Apple MPS).  The upstream model classes are imported from a
pinned ``le-wm`` checkout rather than copied, so the weights are the real
LeWM architecture.

Importing this module imports PyTorch.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

LEWM_REVIEWED_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
CHECKPOINT_FORMAT = "go2wm.lewm-checkpoint.v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeWMSetupError(RuntimeError):
    """Raised when the LeWM source or a checkpoint does not match what was reviewed."""


@dataclass(frozen=True, slots=True)
class LeWMArchitecture:
    image_size: int = 224
    patch_size: int = 14
    encoder_size: str = "tiny"
    embed_dim: int = 192
    history_frames: int = 3
    num_preds: int = 1
    action_dim: int = 2
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    projector_hidden_dim: int = 2048

    @property
    def num_steps(self) -> int:
        """Frames per training clip: history plus predicted frames."""

        return self.history_frames + self.num_preds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LeWMArchitecture:
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise LeWMSetupError(f"unknown architecture fields {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class LossConfig:
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024


@dataclass(frozen=True, slots=True)
class ActionStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.std) or not self.mean:
            raise ValueError("action mean/std must be non-empty and the same length")
        if any(not math.isfinite(v) for v in (*self.mean, *self.std)):
            raise ValueError("action statistics must be finite")
        if any(s <= 0 for s in self.std):
            raise ValueError("action std must be positive")


@dataclass(slots=True)
class LeWMSource:
    """Classes imported from the pinned upstream checkout."""

    repo: Path
    commit: str
    JEPA: Any
    ARPredictor: Any
    Embedder: Any
    MLP: Any
    SIGReg: Any
    vit_hf: Any = field(repr=False)


def git_head(repo: Path) -> str:
    """Resolve HEAD without invoking git (works on detached and branch checkouts)."""

    git_dir = repo / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    ref_file = git_dir / ref
    if ref_file.is_file():
        return ref_file.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    raise LeWMSetupError(f"cannot resolve {ref} in {repo}")


def import_lewm(
    repo: str | Path | None = None,
    *,
    expected_commit: str = LEWM_REVIEWED_COMMIT,
    allow_commit_mismatch: bool = False,
) -> LeWMSource:
    """Import ``jepa.JEPA`` and ``module.*`` from a pinned le-wm checkout.

    ``repo`` defaults to ``$GO2WM_LEWM_REPO``.
    """

    raw = repo or os.environ.get("GO2WM_LEWM_REPO")
    if not raw:
        raise LeWMSetupError("pass --lewm-repo or set GO2WM_LEWM_REPO to the le-wm checkout")
    path = Path(raw).expanduser().resolve()
    if not (path / "jepa.py").is_file() or not (path / "module.py").is_file():
        raise LeWMSetupError(f"{path} does not look like a le-wm checkout (no jepa.py/module.py)")
    commit = git_head(path) if (path / ".git").exists() else "UNKNOWN"
    if commit != expected_commit and not allow_commit_mismatch:
        raise LeWMSetupError(
            f"le-wm checkout is at {commit}, reviewed commit is {expected_commit}. "
            f"Run `git -C {path} checkout {expected_commit}` or pass the explicit override."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    jepa = importlib.import_module("jepa")
    module = importlib.import_module("module")
    for name, mod in (("jepa", jepa), ("module", module)):
        origin = Path(getattr(mod, "__file__", "")).resolve()
        if origin.parent != path:
            raise LeWMSetupError(f"imported {name} from {origin}, not from {path}")
    from stable_pretraining.backbone.utils import vit_hf

    return LeWMSource(
        repo=path,
        commit=commit,
        JEPA=jepa.JEPA,
        ARPredictor=module.ARPredictor,
        Embedder=module.Embedder,
        MLP=module.MLP,
        SIGReg=module.SIGReg,
        vit_hf=vit_hf,
    )


def build_model(arch: LeWMArchitecture, src: LeWMSource) -> torch.nn.Module:
    """Instantiate exactly the upstream ``lewm.yaml`` model for our contract."""

    encoder = src.vit_hf(
        arch.encoder_size,
        patch_size=arch.patch_size,
        image_size=arch.image_size,
        pretrained=False,
        use_mask_token=False,
    )

    def mlp() -> torch.nn.Module:
        return src.MLP(
            input_dim=arch.embed_dim,
            hidden_dim=arch.projector_hidden_dim,
            output_dim=arch.embed_dim,
            norm_fn=torch.nn.BatchNorm1d,
        )

    return src.JEPA(
        encoder=encoder,
        predictor=src.ARPredictor(
            num_frames=arch.history_frames,
            input_dim=arch.embed_dim,
            hidden_dim=arch.embed_dim,
            output_dim=arch.embed_dim,
            depth=arch.predictor_depth,
            heads=arch.predictor_heads,
            mlp_dim=arch.predictor_mlp_dim,
            dim_head=arch.predictor_dim_head,
            dropout=arch.predictor_dropout,
            emb_dropout=arch.predictor_emb_dropout,
        ),
        action_encoder=src.Embedder(input_dim=arch.action_dim, emb_dim=arch.embed_dim),
        projector=mlp(),
        pred_proj=mlp(),
    )


def build_sigreg(loss: LossConfig, src: LeWMSource) -> torch.nn.Module:
    return src.SIGReg(knots=loss.sigreg_knots, num_proj=loss.sigreg_num_proj)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise LeWMSetupError("MPS requested but torch.backends.mps.is_available() is False")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise LeWMSetupError("CUDA requested but not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """uint8 ``(..., 3, H, W)`` -> ImageNet-normalized float32, on the input's device."""

    if pixels.dtype != torch.uint8:
        raise TypeError(f"expected uint8 pixels, got {pixels.dtype}")
    mean = torch.tensor(IMAGENET_MEAN, device=pixels.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=pixels.device).view(3, 1, 1)
    return (pixels.float() / 255.0 - mean) / std


def normalize_actions(actions: torch.Tensor, stats: ActionStats) -> torch.Tensor:
    """Z-score with training statistics, then NaN -> 0 exactly as upstream does."""

    mean = torch.tensor(stats.mean, dtype=torch.float32, device=actions.device)
    std = torch.tensor(stats.std, dtype=torch.float32, device=actions.device)
    return torch.nan_to_num((actions.float() - mean) / std, 0.0)


def lejepa_losses(
    model: torch.nn.Module,
    sigreg: torch.nn.Module,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    arch: LeWMArchitecture,
    loss: LossConfig,
) -> dict[str, torch.Tensor]:
    """Same computation as upstream ``lejepa_forward`` for one preprocessed batch.

    ``pixels``: float ``(B, T, 3, H, W)``; ``actions``: normalized ``(B, T, 2)``;
    ``T = history_frames + num_preds``.
    """

    output = model.encode({"pixels": pixels, "action": actions})
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, : arch.history_frames]
    ctx_act = act_emb[:, : arch.history_frames]
    tgt_emb = emb[:, arch.num_preds :]
    pred_emb = model.predict(ctx_emb, ctx_act)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    return {
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "loss": pred_loss + loss.sigreg_weight * sigreg_loss,
        "emb": emb,
        "act_emb": act_emb,
        "pred_emb": pred_emb,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_weights(path: Path, model: torch.nn.Module) -> str:
    """Write a CPU state_dict atomically and return its SHA-256."""

    tmp = path.with_name(f".{path.name}.tmp")
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    torch.save(state, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def load_weights(path: Path, model: torch.nn.Module, *, expected_sha256: str | None) -> None:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise LeWMSetupError(f"checksum mismatch for {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True, slots=True)
class TrainedLeWM:
    model: torch.nn.Module
    arch: LeWMArchitecture
    action_stats: ActionStats
    weights_path: Path
    weights_sha256: str
    run_config: dict[str, Any]


def load_trained(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    src: LeWMSource | None = None,
    device: str | torch.device = "cpu",
) -> TrainedLeWM:
    """Rebuild a trained model from ``run_config.json`` + a weights-only checkpoint.

    ``checkpoint`` is ``best``, ``last``, or an epoch number.  Loading never
    unpickles arbitrary objects (``weights_only=True``) and verifies the SHA-256
    recorded when the checkpoint was written.
    """

    root = Path(run_dir)
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if config.get("format") != CHECKPOINT_FORMAT:
        raise LeWMSetupError(f"{root} is not a go2wm LeWM run ({config.get('format')!r})")
    index = json.loads((root / "checkpoints" / "index.json").read_text(encoding="utf-8"))
    if checkpoint in ("best", "last"):
        name = index[checkpoint]
    else:
        name = f"epoch_{int(checkpoint):03d}.pt"
    entry = index["checkpoints"][name]
    arch = LeWMArchitecture.from_dict(config["architecture"])
    src = src or import_lewm(allow_commit_mismatch=config["lewm"]["commit"] != LEWM_REVIEWED_COMMIT)
    if src.commit != config["lewm"]["commit"]:
        raise LeWMSetupError(
            f"run was trained with le-wm {config['lewm']['commit']}, loaded {src.commit}"
        )
    model = build_model(arch, src)
    path = root / "checkpoints" / name
    load_weights(path, model, expected_sha256=entry["sha256"])
    model.to(device).eval()
    stats = ActionStats(tuple(config["action_stats"]["mean"]), tuple(config["action_stats"]["std"]))
    return TrainedLeWM(model, arch, stats, path, entry["sha256"], config)
