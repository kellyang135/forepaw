"""Read-only preflight checks for the pinned Go2 controller training host."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingHostFacts:
    system: str
    machine: str
    os_id: str | None
    os_version: str | None
    python_version: str
    python_major_minor: tuple[int, int]
    gpu_names: tuple[str, ...]
    driver_versions: tuple[str, ...]
    disk_free_gib: float
    nvidia_smi_error: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingHostAssessment:
    result: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def _query_nvidia() -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return (), (), "nvidia-smi was not found"
    process = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if process.returncode != 0:
        error = process.stderr.strip() or process.stdout.strip() or "nvidia-smi failed"
        return (), (), error
    names: list[str] = []
    drivers: list[str] = []
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            names.append(parts[0])
            drivers.append(parts[1])
    if not names:
        return (), (), "nvidia-smi returned no GPUs"
    return tuple(names), tuple(drivers), None


def collect_training_host_facts(path: Path | None = None) -> TrainingHostFacts:
    path = path or Path.cwd()
    os_release = _os_release()
    gpu_names, drivers, nvidia_error = _query_nvidia()
    disk = shutil.disk_usage(path)
    return TrainingHostFacts(
        system=platform.system(),
        machine=platform.machine(),
        os_id=os_release.get("ID"),
        os_version=os_release.get("VERSION_ID"),
        python_version=platform.python_version(),
        python_major_minor=(sys.version_info.major, sys.version_info.minor),
        gpu_names=gpu_names,
        driver_versions=drivers,
        disk_free_gib=disk.free / (1024**3),
        nvidia_smi_error=nvidia_error,
    )


def _driver_major(version: str) -> int | None:
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None


def assess_training_host(facts: TrainingHostFacts) -> TrainingHostAssessment:
    errors: list[str] = []
    warnings: list[str] = []
    if facts.system != "Linux":
        errors.append("training host must be Linux")
    if facts.python_major_minor != (3, 11):
        errors.append("training environment must use Python 3.11")
    if not facts.gpu_names:
        errors.append(facts.nvidia_smi_error or "no NVIDIA GPU was detected")
    for version in facts.driver_versions:
        major = _driver_major(version)
        if major is None:
            errors.append(f"could not parse NVIDIA driver version {version!r}")
        elif major < 550:
            errors.append(f"NVIDIA driver {version} is below the pinned recommendation 550")
    if facts.os_id != "ubuntu" or facts.os_version != "22.04":
        warnings.append("upstream recommends Ubuntu 22.04; record and test any deviation")
    if facts.disk_free_gib < 50:
        warnings.append(
            "less than 50 GiB is free; confirm a separate checkpoint/log destination"
        )
    return TrainingHostAssessment(
        result="PASS" if not errors else "FAIL",
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def training_host_report(path: Path | None = None) -> dict[str, Any]:
    facts = collect_training_host_facts(path)
    assessment = assess_training_host(facts)
    payload = asdict(facts)
    payload["python_major_minor"] = list(facts.python_major_minor)
    payload["gpu_names"] = list(facts.gpu_names)
    payload["driver_versions"] = list(facts.driver_versions)
    return {
        "schema_version": "go2wm.go2-training-host-preflight.v1",
        "result": assessment.result,
        "facts": payload,
        "errors": list(assessment.errors),
        "warnings": list(assessment.warnings),
        "scope": "host suitability only; not dependency, training, or locomotion evidence",
    }
