import json

import pytest

from go2wm.integration import WorldModelSkillFacade


class FakeBackend:
    def __init__(self) -> None:
        self.stop_reason: str | None = None

    def imagine(self, candidate_commands: list[list[dict]]) -> dict:
        return {"status": "imagined", "candidate_count": len(candidate_commands)}

    def plan_to(self, x_m: float, y_m: float) -> dict:
        return {"status": "executed_one_block", "goal": [x_m, y_m]}

    def stop(self, reason: str) -> dict:
        self.stop_reason = reason
        return {"status": "stopped", "reason": reason}


def test_skill_facade_returns_json_safe_strings() -> None:
    backend = FakeBackend()
    facade = WorldModelSkillFacade(backend)
    response = facade.plan_to(1.0, -0.5)
    assert json.loads(response) == {
        "goal": [1.0, -0.5],
        "status": "executed_one_block",
    }


def test_imagine_rejects_wrong_action_duration() -> None:
    facade = WorldModelSkillFacade(FakeBackend())
    with pytest.raises(ValueError, match=r"0\.5-second"):
        facade.imagine(
            [[{"forward_velocity_mps": 0.2, "yaw_rate_rps": 0.0, "duration_s": 1.0}]]
        )


def test_stop_reason_is_explicit() -> None:
    backend = FakeBackend()
    facade = WorldModelSkillFacade(backend)
    assert json.loads(facade.stop())["status"] == "stopped"
    assert backend.stop_reason == "skill_requested"
