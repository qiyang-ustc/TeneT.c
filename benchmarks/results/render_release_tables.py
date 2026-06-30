#!/usr/bin/env python3
"""Render Markdown tables from committed TeneT.c benchmark TSVs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def render_comparison(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Completed GPU-Baseline Comparison",
        "",
        "| chi | TeneT.jl master GPU median (s) | TeneT.c GPU median (s) | speedup | master error | TeneT.c error | status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        master = "not measured" if row["master_seconds"] == "" else f"{float(row['master_seconds']):.6f}"
        speedup = "n/a" if row["speedup_master_over_tenetc"] == "" else f"{float(row['speedup_master_over_tenetc']):.2f}x"
        master_err = "n/a" if row["master_err"] == "" else f"{float(row['master_err']):.2e}"
        lines.append(
            f"| {row['chi']} | {master} | {float(row['tenetc_seconds']):.6f} | {speedup} | "
            f"{master_err} | {float(row['tenetc_err']):.2e} | {row['master_status']} |"
        )
    lines.append("")
    return lines


def render_native(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Native GPU Scaling",
        "",
        "| chi | TeneT.c GPU median (s) | p25 (s) | p75 (s) | TeneT.c error |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {chi} | {median:.6f} | {p25:.6f} | {p75:.6f} | {err:.2e} |".format(
                chi=row["chi"],
                median=float(row["tenetc_seconds"]),
                p25=float(row["tenetc_p25_seconds"]),
                p75=float(row["tenetc_p75_seconds"]),
                err=float(row["tenetc_err"]),
            )
        )
    lines.append("")
    return lines


def main() -> None:
    lines = ["# Generated Benchmark Tables", ""]
    lines += render_comparison(read_tsv(RESULTS / "tenetc_h100.tsv"))
    lines += render_native(read_tsv(RESULTS / "tenetc_native_h100.tsv"))
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
