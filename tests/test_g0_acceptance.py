from __future__ import annotations

from pathlib import Path

import pytest

from go2wm.sim.g0_acceptance import G0AcceptanceError, load_g0_acceptance

ROOT = Path(__file__).resolve().parents[1]


def test_repository_g0_acceptance_is_valid_and_discloses_pending_ack() -> None:
    acceptance = load_g0_acceptance(ROOT / "configs/g0_acceptance.toml")

    assert acceptance.status == "FROZEN_PENDING_COOWNER_ACK"
    assert not acceptance.coowner_acknowledged
    assert acceptance.max_forward_mps == 0.6
    assert acceptance.resistant_max_displacement_m == 0.05
    assert acceptance.required_successes_of_five == 4


def test_g0_acceptance_rejects_unfrozen_status(tmp_path: Path) -> None:
    source = (ROOT / "configs/g0_acceptance.toml").read_text(encoding="utf-8")
    path = tmp_path / "g0.toml"
    path.write_text(source.replace("FROZEN_PENDING_COOWNER_ACK", "PROVISIONAL"))

    with pytest.raises(G0AcceptanceError, match="not frozen"):
        load_g0_acceptance(path)


def test_g0_acceptance_rejects_ack_status_mismatch(tmp_path: Path) -> None:
    source = (ROOT / "configs/g0_acceptance.toml").read_text(encoding="utf-8")
    path = tmp_path / "g0.toml"
    path.write_text(source.replace("coowner_acknowledged = false", "coowner_acknowledged = true"))

    with pytest.raises(G0AcceptanceError, match="status must be"):
        load_g0_acceptance(path)
