from pathlib import Path

import pytest

from go2wm.evidence import EvidenceRecord, append_record, read_records


def test_passing_record_requires_metrics_and_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metrics"):
        EvidenceRecord.create(
            gate="sim-smoke",
            result="PASS",
            command="pytest",
            metrics={},
            artifacts=("video.mp4",),
            notes="",
            root=tmp_path,
        )


def test_evidence_jsonl_round_trip(tmp_path: Path) -> None:
    record = EvidenceRecord.create(
        gate="data-alignment",
        result="PASS",
        command="go2wm verify-dataset tiny.jsonl",
        metrics={"bad_blocks": 0, "duration_error_max_s": 0.001},
        artifacts=("tiny.jsonl", "alignment.png"),
        notes="Fake evidence used only by this unit test.",
        root=tmp_path,
    )
    destination = tmp_path / "evidence.jsonl"
    append_record(destination, record)
    assert read_records(destination) == [record]

