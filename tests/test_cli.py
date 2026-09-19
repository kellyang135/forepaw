import json

from go2wm.cli import main


def test_candidate_command_reports_exact_library(capsys) -> None:
    assert main(["candidates"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 64
    assert payload["first_action_only"] is True


def test_fake_smoke_is_explicitly_non_robot_evidence(capsys) -> None:
    assert main(["fake-smoke"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fake_smoke_passed"
    assert "not MuJoCo" in payload["evidence_scope"]
    assert payload["history_frames"] == 3
    assert payload["history_commands"] == 2

