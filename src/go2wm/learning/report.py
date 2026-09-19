"""Plain-text summaries of ``ml_report.json`` files for the terminal and the ledger."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def gate_table(report: dict[str, Any]) -> str:
    lines = [f"{report.get('model', '?')}  [{report.get('evidence_scope', '')}]"]
    for check in report.get("gate_checks", []):
        mark = "PASS" if check["passed"] else "FAIL"
        measured = ", ".join(f"{k}={_fmt(v)}" for k, v in check["measured"].items())
        lines.append(f"  {check['gate']} {mark}  {check['criterion']}  ({measured})")
    return "\n".join(lines)


def _row(report: dict[str, Any], group: str, step: str, family: str, method: str) -> Any:
    entry = report.get("prediction", {}).get("groups", {}).get(group, {}).get(step, {})
    return entry.get(family, {}).get(method) if entry.get("count") else None


def compare_reports(reports: Sequence[dict[str, Any]], names: Sequence[str]) -> str:
    """Median errors (metres) per horizon for each report, plus references from the first."""

    header = ["metric", *[r.get("model", n)[:28] for r, n in zip(reports, names, strict=True)]]
    rows: list[list[str]] = []

    def add(label: str, values: Sequence[Any]) -> None:
        rows.append([label, *[_fmt(v) for v in values]])

    add(
        "readout robot, true frames",
        [
            r.get("readout_validation_real_latents", {}).get("robot_position_median_m")
            for r in reports
        ],
    )
    add(
        "readout box, true frames",
        [
            r.get("readout_validation_real_latents", {}).get("object_position_median_m")
            for r in reports
        ],
    )
    add(
        "constant-mean robot",
        [
            r.get("readout_validation_constant_mean", {}).get("robot_position_median_m")
            for r in reports
        ],
    )
    for group in ("all", "free", "interaction"):
        for step in ("1", "2", "4", "6"):
            seconds = int(step) * 0.5
            add(
                f"{group} robot @{seconds:.1f}s model",
                [_row(r, group, step, "robot_position_median_m", "model") for r in reports],
            )
            add(
                f"{group} robot @{seconds:.1f}s shuffled",
                [_row(r, group, step, "robot_position_median_m", "shuffled") for r in reports],
            )
            add(
                f"{group} box @{seconds:.1f}s model",
                [_row(r, group, step, "object_position_median_m", "model") for r in reports],
            )
    first = reports[0]
    for step in ("1", "6"):
        for method in ("persistence", "kinematic"):
            add(
                f"reference {method} robot @{int(step) * 0.5:.1f}s",
                [_row(first, "all", step, "robot_position_median_m", method)] * len(reports),
            )
    add("plan latency median s", [r.get("latency", {}).get("median_s") for r in reports])
    add("all gate checks passed", [r.get("all_checks_passed") for r in reports])

    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))]
    out.append("  ".join("-" * w for w in widths))
    out.extend("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows)
    scopes = {r.get("evidence_scope", "") for r in reports}
    out.append("")
    out.append("scope: " + " | ".join(sorted(scopes)))
    return "\n".join(out)
