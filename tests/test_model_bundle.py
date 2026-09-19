from __future__ import annotations

from dataclasses import replace

import pytest

from go2wm.model import (
    ActionBlock,
    BundleCompatibilityError,
    BundleManifest,
    CandidateRollout,
    ComponentWorldModelBackend,
    ModelBundle,
    ModelInput,
    PredictedState,
    RobotState,
    RuntimeRequirements,
)


class FakeBackend:
    def __init__(self, manifest: BundleManifest) -> None:
        self._manifest = manifest

    @property
    def manifest(self) -> BundleManifest:
        return self._manifest

    def predict_candidates(self, model_input, candidates):
        del model_input
        return [
            CandidateRollout(
                candidate_id=candidate.candidate_id,
                family=candidate.family,
                actions=candidate.actions,
                states=tuple(
                    PredictedState(
                        step=step,
                        robot=RobotState(float(step), 0.0, 0.0),
                        latent=(float(step), 0.0, 0.0),
                    )
                    for step in range(1, len(candidate.actions) + 1)
                ),
                bundle_id=self.manifest.bundle_id,
            )
            for candidate in candidates
        ]


def manifest() -> BundleManifest:
    return BundleManifest(
        bundle_id="test-bundle",
        encoder_version="encoder-a",
        predictor_version="predictor-a",
        readout_version="readout-a",
        normalization_version="norm-a",
        surprise_calibration_version="surprise-a",
        latent_dim=3,
        horizon_blocks=2,
    )


def requirements() -> RuntimeRequirements:
    return RuntimeRequirements(latent_dim=3, horizon_blocks=2)


def test_bundle_rejects_incompatible_manifest() -> None:
    bad = replace(manifest(), action_dim=3, block_duration_s=0.25)

    with pytest.raises(BundleCompatibilityError) as error:
        ModelBundle(FakeBackend(bad), requirements())

    assert "action_dim" in str(error.value)
    assert "block_duration_s" in str(error.value)
    assert len(error.value.mismatches) == 2


def test_model_input_contract_is_checked_before_backend() -> None:
    bundle = ModelBundle(FakeBackend(manifest()), requirements())
    malformed = ModelInput(observations=("only-one",), command_history=())

    with pytest.raises(ValueError, match="expected 3 observations"):
        bundle.predict_candidates(malformed, ())


def test_history_uses_commands_between_frames() -> None:
    action = ActionBlock(0.2, 0.0)
    model_input = ModelInput(
        observations=("old", "middle", "new"),
        command_history=(action, action),
    )

    model_input.validate(history_frames=3, block_duration_s=0.5)


def test_manifest_mapping_rejects_typo_instead_of_ignoring_it() -> None:
    values = manifest().to_mapping()
    values["predicter_version"] = "typo"

    with pytest.raises(ValueError, match="predicter_version"):
        BundleManifest.from_mapping(values)


def test_component_backend_rolls_out_and_reads_predicted_latents() -> None:
    class Encoder:
        def encode(self, observation):
            return (float(observation), 0.0, 0.0)

    class Predictor:
        def rollout(self, history_latents, history_actions, candidate_actions):
            assert len(history_latents) == 3
            assert len(history_actions) == 2
            return [
                (history_latents[-1][0] + index, action.forward_mps, action.yaw_rate_rps)
                for index, action in enumerate(candidate_actions, start=1)
            ]

    class Readout:
        def __init__(self) -> None:
            self.seen = []

        def decode(self, latent, *, step):
            self.seen.append(latent)
            return PredictedState(step, RobotState(latent[0], latent[1], latent[2]))

    action = ActionBlock(0.2, 0.1)

    class Candidate:
        candidate_id = "candidate"
        family = "test"
        actions = (action, action)

    readout = Readout()
    backend = ComponentWorldModelBackend(manifest(), Encoder(), Predictor(), readout)
    bundle = ModelBundle(backend, requirements())
    result = bundle.predict_candidates(
        ModelInput((0.0, 1.0, 2.0), (action, action)), (Candidate(),)
    )

    assert result[0].states[0].latent == (3.0, 0.2, 0.1)
    assert readout.seen == [(3.0, 0.2, 0.1), (4.0, 0.2, 0.1)]
