"""Publish and load immutable, checksummed model bundles (the ML -> SIM handoff).

Layout of one bundle directory::

    bundle.json          manifest, provenance, and SHA-256 of every other file
    encoder.json         encoder kind/config/weights
    predictor.json       predictor kind/weights
    readout.json         readout weights and normalization statistics
    surprise.json        frozen surprise calibration
    planner.json         candidate library and score weights
    reference/input.json   tiny history + candidate packet
    reference/output.json  expected predictions for that packet

The bundle id is derived from the content of every component, so changing any
member (including the surprise threshold or a planner weight) yields a new id.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from go2wm.contracts import RGBObservation
from go2wm.model import (
    ActionBlock,
    BundleCompatibilityError,
    BundleManifest,
    ComponentWorldModelBackend,
    ModelBundle,
    ModelInput,
)
from go2wm.planning import (
    CandidateSequence,
    PlannerSafetyPolicy,
    ScoreWeights,
    ScoringConfig,
)
from go2wm.runtime import SurpriseCalibration

from .baseline_model import LinearLatentPredictor, PooledPixelEncoder
from .readouts import LinearLatentReadout

BUNDLE_LAYOUT = "go2wm.bundle-dir.v1"
REFERENCE_TOLERANCE = 1e-6

ComponentLoader = Callable[[dict[str, Any]], Any]
ENCODER_LOADERS: dict[str, ComponentLoader] = {"pooled_pixels": PooledPixelEncoder.from_dict}
PREDICTOR_LOADERS: dict[str, ComponentLoader] = {"linear_latent": LinearLatentPredictor.from_dict}


def register_component_loader(role: str, kind: str, loader: ComponentLoader) -> None:
    """Let optional backends (for example LeWM/PyTorch) register how to load themselves."""

    registry = {"encoder": ENCODER_LOADERS, "predictor": PREDICTOR_LOADERS}[role]
    registry[kind] = loader


@dataclass(frozen=True, slots=True)
class Provenance:
    source_revision: str
    dataset_id: str
    split_id: str
    training_command: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    path: Path
    model_bundle: ModelBundle
    readout: LinearLatentReadout
    surprise: SurpriseCalibration
    candidates: tuple[CandidateSequence, ...]
    scoring: ScoringConfig
    safety: PlannerSafetyPolicy
    provenance: Mapping[str, Any]

    @property
    def bundle_id(self) -> str:
        return self.model_bundle.manifest.bundle_id


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _candidates_payload(candidates: Sequence[CandidateSequence]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": c.candidate_id,
            "family": c.family,
            "actions": [[a.forward_mps, a.yaw_rate_rps, a.duration_s] for a in c.actions],
        }
        for c in candidates
    ]


def _candidates_from_payload(raw: Sequence[Mapping[str, Any]]) -> tuple[CandidateSequence, ...]:
    return tuple(
        CandidateSequence(
            candidate_id=item["candidate_id"],
            family=item["family"],
            actions=tuple(ActionBlock(*values) for values in item["actions"]),
        )
        for item in raw
    )


def _observation_payload(observation: RGBObservation) -> dict[str, Any]:
    return {
        "observation_id": observation.observation_id,
        "episode_id": observation.episode_id,
        "sim_time_s": observation.sim_time_s,
        "camera_id": observation.camera_id,
        "width_px": observation.width_px,
        "height_px": observation.height_px,
        "rgb_base64": base64.b64encode(observation.rgb).decode("ascii"),
    }


def _observation_from_payload(raw: Mapping[str, Any]) -> RGBObservation:
    return RGBObservation(
        observation_id=raw["observation_id"],
        episode_id=raw["episode_id"],
        sim_time_s=raw["sim_time_s"],
        camera_id=raw["camera_id"],
        width_px=raw["width_px"],
        height_px=raw["height_px"],
        rgb=base64.b64decode(raw["rgb_base64"]),
    )


def _reference_input_payload(
    model_input: ModelInput, candidates: Sequence[CandidateSequence]
) -> dict[str, Any]:
    return {
        "observations": [_observation_payload(obs) for obs in model_input.observations],
        "command_history": [
            [a.forward_mps, a.yaw_rate_rps, a.duration_s] for a in model_input.command_history
        ],
        "candidate_ids": [c.candidate_id for c in candidates],
    }


def _rollouts_payload(rollouts: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": r.candidate_id,
            "states": [
                {
                    "step": s.step,
                    "robot": [s.robot.x_m, s.robot.y_m, s.robot.yaw_rad],
                    "objects": [[o.object_id, o.x_m, o.y_m] for o in s.objects],
                    "failure_risk": s.failure_risk,
                    "latent": list(s.latent or ()),
                }
                for s in r.states
            ],
        }
        for r in rollouts
    ]


def publish_bundle(
    output_root: str | Path,
    *,
    encoder: Any,
    encoder_kind: str,
    predictor: Any,
    predictor_kind: str,
    readout: LinearLatentReadout,
    surprise: SurpriseCalibration,
    candidates: Sequence[CandidateSequence],
    scoring: ScoringConfig,
    safety: PlannerSafetyPolicy,
    provenance: Provenance,
    reference_input: ModelInput,
    reference_candidate_count: int = 3,
    latent_dim: int = 192,
) -> LoadedBundle:
    """Write a bundle directory named by its content-derived id, then verify by reloading."""

    components: dict[str, Any] = {
        "encoder.json": {"kind": encoder_kind, "config": encoder.to_dict()},
        "predictor.json": {"kind": predictor_kind, "config": predictor.to_dict()},
        "readout.json": readout.to_dict(),
        "planner.json": {
            "candidates": _candidates_payload(candidates),
            "scoring": {
                "weights": asdict(scoring.weights),
                "minimum_progress_m": scoring.minimum_progress_m,
                "active_forward_threshold_mps": scoring.active_forward_threshold_mps,
            },
            "safety": asdict(safety),
        },
    }
    surprise_core = {
        "metric": surprise.metric,
        "threshold": surprise.threshold,
        "quantile": surprise.quantile,
        "consecutive_breaches": surprise.consecutive_breaches,
        "normal_sample_count": surprise.normal_sample_count,
        "latent_dim": surprise.latent_dim,
    }
    hashes = {name: _digest(_canonical(value)) for name, value in components.items()}
    surprise_version = "surprise-" + _digest(_canonical(surprise_core))[:12]
    hashes["surprise.core"] = _digest(_canonical(surprise_core))
    bundle_id = "go2wm-" + _digest(_canonical(hashes))[:16]
    surprise = replace(surprise, bundle_id=bundle_id, calibration_version=surprise_version)
    components["surprise.json"] = asdict(surprise)

    manifest = BundleManifest.create(
        bundle_id=bundle_id,
        encoder_version=f"{encoder_kind}:{hashes['encoder.json'][:12]}",
        predictor_version=f"{predictor_kind}:{hashes['predictor.json'][:12]}",
        readout_version=f"{readout.version}:{hashes['readout.json'][:12]}",
        normalization_version=f"readout-norm:{hashes['readout.json'][:12]}",
        surprise_calibration_version=surprise_version,
        latent_dim=latent_dim,
    )
    backend = ComponentWorldModelBackend(manifest, encoder, predictor, readout)
    model_bundle = ModelBundle(backend)
    surprise.validate_manifest(manifest)
    reference_candidates = tuple(candidates[:reference_candidate_count])
    reference_output = _rollouts_payload(
        model_bundle.predict_candidates(reference_input, reference_candidates)
    )
    components["reference/input.json"] = _reference_input_payload(
        reference_input, reference_candidates
    )
    components["reference/output.json"] = {
        "rollouts": reference_output,
        "tolerance": REFERENCE_TOLERANCE,
    }

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / bundle_id
    if target.exists():
        raise FileExistsError(f"bundle {bundle_id} already exists; bundles are immutable")
    staging = root / f".{bundle_id}.tmp-{uuid.uuid4().hex}"
    try:
        (staging / "reference").mkdir(parents=True)
        file_hashes: dict[str, str] = {}
        for name, value in components.items():
            payload = _canonical(value)
            (staging / name).write_bytes(payload)
            file_hashes[name] = _digest(payload)
        index = {
            "layout": BUNDLE_LAYOUT,
            "manifest": manifest.to_mapping(),
            "provenance": asdict(provenance),
            "files": file_hashes,
            "truthfulness_note": (
                "A bundle is an artifact, not evidence of task performance. Check its "
                "evaluation report and evidence ledger before making claims."
            ),
        }
        (staging / "bundle.json").write_bytes(_canonical(index))
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_bundle(target)


def load_bundle(path: str | Path, *, expected_bundle_id: str | None = None) -> LoadedBundle:
    """Verify every checksum, rebuild components, and replay the reference packet."""

    root = Path(path)
    index = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if index.get("layout") != BUNDLE_LAYOUT:
        raise BundleCompatibilityError([f"unsupported bundle layout {index.get('layout')!r}"])
    mismatches = []
    for name, expected in index["files"].items():
        file_path = root / name
        if not file_path.exists():
            mismatches.append(f"missing {name}")
        elif _digest(file_path.read_bytes()) != expected:
            mismatches.append(f"checksum mismatch for {name}")
    if mismatches:
        raise BundleCompatibilityError(mismatches)

    def read(name: str) -> Any:
        return json.loads((root / name).read_text(encoding="utf-8"))

    manifest = BundleManifest.from_mapping(index["manifest"])
    if expected_bundle_id is not None and manifest.bundle_id != expected_bundle_id:
        raise BundleCompatibilityError(
            [f"bundle_id {manifest.bundle_id!r} != expected {expected_bundle_id!r}"]
        )
    encoder_raw = read("encoder.json")
    predictor_raw = read("predictor.json")
    try:
        encoder = ENCODER_LOADERS[encoder_raw["kind"]](encoder_raw["config"])
        predictor = PREDICTOR_LOADERS[predictor_raw["kind"]](predictor_raw["config"])
    except KeyError as error:
        raise BundleCompatibilityError(
            [f"no loader registered for component kind {error}"]
        ) from None
    readout = LinearLatentReadout.from_dict(read("readout.json"))
    surprise = SurpriseCalibration(**read("surprise.json"))
    surprise.validate_manifest(manifest)
    planner = read("planner.json")
    candidates = _candidates_from_payload(planner["candidates"])
    scoring = ScoringConfig(
        weights=ScoreWeights(**planner["scoring"]["weights"]),
        minimum_progress_m=planner["scoring"]["minimum_progress_m"],
        active_forward_threshold_mps=planner["scoring"]["active_forward_threshold_mps"],
    )
    safety = PlannerSafetyPolicy(**planner["safety"])

    backend = ComponentWorldModelBackend(manifest, encoder, predictor, readout)
    model_bundle = ModelBundle(backend)

    reference_in = read("reference/input.json")
    reference_out = read("reference/output.json")
    model_input = ModelInput(
        observations=tuple(_observation_from_payload(o) for o in reference_in["observations"]),
        command_history=tuple(ActionBlock(*values) for values in reference_in["command_history"]),
    )
    by_id = {c.candidate_id: c for c in candidates}
    replay = _rollouts_payload(
        model_bundle.predict_candidates(
            model_input, tuple(by_id[cid] for cid in reference_in["candidate_ids"])
        )
    )
    _compare_reference(replay, reference_out["rollouts"], float(reference_out["tolerance"]))
    return LoadedBundle(
        path=root,
        model_bundle=model_bundle,
        readout=readout,
        surprise=surprise,
        candidates=candidates,
        scoring=scoring,
        safety=safety,
        provenance=index.get("provenance", {}),
    )


def _compare_reference(
    actual: Any, expected: Any, tolerance: float, where: str = "reference"
) -> None:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise BundleCompatibilityError([f"{where}: {actual!r} != {expected!r}"])
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise BundleCompatibilityError([f"{where}: length mismatch"])
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare_reference(a, e, tolerance, f"{where}[{i}]")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise BundleCompatibilityError([f"{where}: key mismatch"])
        for key in expected:
            _compare_reference(actual[key], expected[key], tolerance, f"{where}.{key}")
        return
    if actual != expected:
        raise BundleCompatibilityError([f"{where}: {actual!r} != {expected!r}"])
