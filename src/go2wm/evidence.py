"""Small, auditable evidence records for project gates and final evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

GateResult = Literal["PASS", "FAIL", "BLOCKED"]


@dataclass(frozen=True)
class EvidenceRecord:
    gate: str
    result: GateResult
    recorded_at_utc: str
    git_revision: str
    command: str
    metrics: dict[str, float | int | bool | str]
    artifacts: tuple[str, ...]
    notes: str

    @classmethod
    def create(
        cls,
        *,
        gate: str,
        result: GateResult,
        command: str,
        metrics: dict[str, float | int | bool | str],
        artifacts: tuple[str, ...],
        notes: str,
        root: Path,
    ) -> EvidenceRecord:
        if not gate.strip():
            raise ValueError("gate must not be empty")
        if result == "PASS" and not metrics:
            raise ValueError("a passing gate must include measured metrics")
        if result == "PASS" and not artifacts:
            raise ValueError("a passing gate must point to retained evidence")
        return cls(
            gate=gate,
            result=result,
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            git_revision=_git_revision(root),
            command=command,
            metrics=metrics,
            artifacts=artifacts,
            notes=notes,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def append_record(path: Path, record: EvidenceRecord) -> None:
    """Append one immutable JSONL record, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json())
        handle.write("\n")


def read_records(path: Path) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: dict[str, Any] = json.loads(line)
                raw["artifacts"] = tuple(raw["artifacts"])
                records.append(EvidenceRecord(**raw))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid evidence at {path}:{line_number}: {error}") from error
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "NO_COMMIT"

