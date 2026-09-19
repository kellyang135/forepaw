from __future__ import annotations

from go2wm.planning import build_candidate_library, family_counts


def test_candidate_library_has_exact_interpretable_coverage() -> None:
    candidates = build_candidate_library()

    assert len(candidates) == 64
    assert family_counts(candidates) == {
        "approach_push": 12,
        "detour_left": 16,
        "detour_right": 16,
        "turn_adjust": 16,
        "stop": 4,
    }
    assert len({candidate.candidate_id for candidate in candidates}) == 64
    assert len({candidate.actions for candidate in candidates}) == 64


def test_every_candidate_matches_three_second_contract() -> None:
    for candidate in build_candidate_library():
        assert len(candidate.actions) == 6
        assert sum(action.duration_s for action in candidate.actions) == 3.0
        assert all(abs(action.forward_mps) <= 0.6 for action in candidate.actions)
        assert all(abs(action.yaw_rate_rps) <= 1.0 for action in candidate.actions)


def test_detours_are_symmetric_and_stop_has_zero_first_action() -> None:
    by_id = {candidate.candidate_id: candidate for candidate in build_candidate_library()}
    left = by_id["left_v0p30_turn0p60_n1_recover0p60"]
    right = by_id["right_v0p30_turn0p60_n1_recover0p60"]

    assert [action.forward_mps for action in left.actions] == [
        action.forward_mps for action in right.actions
    ]
    assert [action.yaw_rate_rps for action in left.actions] == [
        -action.yaw_rate_rps for action in right.actions
    ]
    assert by_id["stop_hold"].actions[0].forward_mps == 0.0
    assert by_id["stop_hold"].actions[0].yaw_rate_rps == 0.0

