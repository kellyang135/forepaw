from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import go2wm.sim.training_host as training_host
from go2wm.sim.training_host import TrainingHostFacts, assess_training_host


def _facts(**overrides: object) -> TrainingHostFacts:
    values = {
        "system": "Linux",
        "machine": "x86_64",
        "os_id": "ubuntu",
        "os_version": "22.04",
        "python_version": "3.11.9",
        "python_major_minor": (3, 11),
        "gpu_names": ("NVIDIA Test GPU",),
        "driver_versions": ("550.127.05",),
        "disk_free_gib": 100.0,
        "nvidia_smi_error": None,
    }
    values.update(overrides)
    return TrainingHostFacts(**values)  # type: ignore[arg-type]


def test_recommended_training_host_passes() -> None:
    assessment = assess_training_host(_facts())

    assert assessment.result == "PASS"
    assert assessment.errors == ()
    assert assessment.warnings == ()


def test_mac_without_nvidia_fails() -> None:
    assessment = assess_training_host(
        _facts(
            system="Darwin",
            machine="arm64",
            os_id=None,
            os_version=None,
            python_version="3.12.4",
            python_major_minor=(3, 12),
            gpu_names=(),
            driver_versions=(),
            disk_free_gib=19.0,
            nvidia_smi_error="nvidia-smi was not found",
        )
    )

    assert assessment.result == "FAIL"
    assert "training host must be Linux" in assessment.errors
    assert "training environment must use Python 3.11" in assessment.errors
    assert "nvidia-smi was not found" in assessment.errors
    assert len(assessment.warnings) == 2


def test_old_or_malformed_driver_fails() -> None:
    old = assess_training_host(_facts(driver_versions=("545.23",)))
    malformed = assess_training_host(_facts(driver_versions=("unknown",)))

    assert old.result == "FAIL"
    assert "below the pinned recommendation" in old.errors[0]
    assert malformed.result == "FAIL"
    assert "could not parse" in malformed.errors[0]


def test_supported_host_can_warn_about_os_and_disk() -> None:
    assessment = assess_training_host(
        _facts(os_version="24.04", disk_free_gib=40.0)
    )

    assert assessment.result == "PASS"
    assert len(assessment.warnings) == 2


def test_os_release_parser(tmp_path: Path) -> None:
    release = tmp_path / "os-release"
    release.write_text(
        '# comment\nID=ubuntu\nVERSION_ID="22.04"\nIGNORED_LINE\n', encoding="utf-8"
    )

    assert training_host._os_release(release) == {
        "ID": "ubuntu",
        "VERSION_ID": "22.04",
    }
    assert training_host._os_release(tmp_path / "missing") == {}


def test_nvidia_query_parses_multiple_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_host.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="NVIDIA A, 550.54.15\nNVIDIA B, 560.1\n",
        stderr="",
    )
    monkeypatch.setattr(training_host.subprocess, "run", lambda *args, **kwargs: completed)

    names, drivers, error = training_host._query_nvidia()

    assert names == ("NVIDIA A", "NVIDIA B")
    assert drivers == ("550.54.15", "560.1")
    assert error is None


def test_nvidia_query_reports_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_host.shutil, "which", lambda _: None)

    assert training_host._query_nvidia() == ((), (), "nvidia-smi was not found")


def test_training_host_report_serializes_tuples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_host, "collect_training_host_facts", lambda path=None: _facts())

    report = training_host.training_host_report()

    assert report["result"] == "PASS"
    assert report["facts"]["python_major_minor"] == [3, 11]
    assert report["facts"]["gpu_names"] == ["NVIDIA Test GPU"]
