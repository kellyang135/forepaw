import pytest

from go2wm.evaluation import (
    TaskOutcome,
    action_conditioning_improvement,
    error_summary,
    percentile,
    ranking_success_fraction,
    summarize_surprise,
    summarize_tasks,
)


def test_percentile_and_error_summary_are_deterministic() -> None:
    assert percentile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5
    summary = error_summary([0.1, 0.2, 0.3])
    assert summary.count == 3
    assert summary.median == pytest.approx(0.2)
    assert summary.maximum == pytest.approx(0.3)


def test_action_and_ranking_metrics_keep_direction_clear() -> None:
    assert action_conditioning_improvement(matched_error=0.8, shuffled_error=1.0) == pytest.approx(
        0.2
    )
    assert ranking_success_fraction([True, False, True, True]) == pytest.approx(0.75)


def test_task_summary_reports_exclusions_without_counting_them() -> None:
    outcomes = [
        TaskOutcome("push-001", "push", 1, True, 4.0, False, False),
        TaskOutcome("push-002", "push", 1, False, 10.0, False, True),
        TaskOutcome(
            "push-003",
            "push",
            1,
            False,
            0.0,
            False,
            False,
            excluded=True,
            exclusion_reason="corrupt recording",
        ),
    ]
    summary = summarize_tasks(outcomes)
    assert summary.included_count == 2
    assert summary.excluded_count == 1
    assert summary.success_fraction == 0.5
    assert summary.median_success_time_s == 4.0


def test_surprise_summary_separates_false_stops_and_changed_trials() -> None:
    summary = summarize_surprise(
        normal_stop_flags=[False, False, True, False],
        changed_detection_delays=[1, 2, None],
    )
    assert summary.false_stop_fraction == 0.25
    assert summary.detection_fraction == pytest.approx(2 / 3)
    assert summary.median_detection_delay_blocks == 1.5

