import math

import pytest

from go2wm.baselines import InteractionExample, MobilityLookup, MotionState, UnicycleModel


def test_unicycle_reference_integrates_forward_command() -> None:
    next_state = UnicycleModel().step(
        MotionState(0.0, 0.0, 0.0),
        commanded_forward_mps=0.4,
        commanded_yaw_rate_rps=0.0,
        duration_s=0.5,
    )
    assert next_state.x_m == pytest.approx(0.2)
    assert next_state.y_m == pytest.approx(0.0)


def test_unicycle_wraps_heading() -> None:
    next_state = UnicycleModel().step(
        MotionState(0.0, 0.0, math.pi - 0.1),
        commanded_forward_mps=0.0,
        commanded_yaw_rate_rps=1.0,
        duration_s=0.5,
    )
    assert -math.pi <= next_state.yaw_rad < math.pi


def test_mobility_lookup_is_fit_only_from_usable_interactions() -> None:
    lookup = MobilityLookup.fit(
        [
            InteractionExample("blue", 0.2, 0.16),
            InteractionExample("blue", 0.2, 0.12),
            InteractionExample("red", 0.2, 0.0),
            InteractionExample("red", 0.0, 0.0),
        ]
    )
    assert lookup.expected_displacement("blue", 0.1) == pytest.approx(0.07)
    assert lookup.expected_displacement("red", 0.1) == 0.0

