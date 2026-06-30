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
    completed = [row for row in rows if row["master_seconds"]]
    all_comparable = bool(completed) and all(
        row.get("comparison_status") == "comparable" for row in completed
    )
    lines = [
        "## Completed Real GPU Comparison" if all_comparable else "## Completed GPU Timing Audit",
        "",
        "| chi | master eltype | TeneT.c eltype | TeneT.jl master GPU median (s) | TeneT.c GPU median (s) | ratio | master error | TeneT.c error | comparison status |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        master = "not measured" if row["master_seconds"] == "" else f"{float(row['master_seconds']):.6f}"
        ratio = "n/a" if row["raw_ratio_master_over_tenetc"] == "" else f"{float(row['raw_ratio_master_over_tenetc']):.2f}x"
        master_err = "n/a" if row["master_err"] == "" else f"{float(row['master_err']):.2e}"
        master_eltype = row.get("master_eltype", "") or "not measured"
        tenetc_eltype = row.get("tenetc_eltype", "") or "unknown"
        status = row.get("comparison_status", row.get("master_status", "unknown"))
        lines.append(
            f"| {row['chi']} | {master_eltype} | {tenetc_eltype} | {master} | "
            f"{float(row['tenetc_seconds']):.6f} | {ratio} | {master_err} | "
            f"{float(row['tenetc_err']):.2e} | {status} |"
        )
    lines.append("")
    return lines


def render_real_gpu_baseline(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Real GPU Baseline",
        "",
        "| chi | TeneT.jl branch | TeneT.jl eltype | TeneT.c eltype | TeneT.jl median (s) | TeneT.c median (s) | speedup | TeneT.jl VUMPS error | TeneT.jl abs delta f | TeneT.c error | run id |",
        "| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(
            "| {chi} | {branch} | {tenet_eltype} | {tenetc_eltype} | {tenet:.6f} | {tenetc:.6f} | {speedup:.2f}x | {tenet_err:.2e} | {ferr:.2e} | {tenetc_err:.2e} | {run_id} |".format(
                chi=row["chi"],
                branch=row["tenet_branch"],
                tenet_eltype=row["tenet_eltype"],
                tenetc_eltype=row["tenetc_eltype"],
                tenet=float(row["tenet_seconds"]),
                tenetc=float(row["tenetc_seconds"]),
                speedup=float(row["speedup_tenet_over_tenetc"]),
                tenet_err=float(row["tenet_err"]),
                ferr=float(row["tenet_free_energy_abs_error"]),
                tenetc_err=float(row["tenetc_err"]),
                run_id=row["run_id"],
            )
        )
    lines.append("")
    return lines


def render_native(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Native GPU Scaling",
        "",
        "| chi | backend | eltype | TeneT.c GPU median (s) | p25 (s) | p75 (s) | TeneT.c error |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {chi} | {backend} | {eltype} | {median:.6f} | {p25:.6f} | {p75:.6f} | {err:.2e} |".format(
                chi=row["chi"],
                backend=row.get("backend", "unknown"),
                eltype=row.get("tenetc_eltype", row.get("eltype", "unknown")),
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
    lines += render_real_gpu_baseline(read_tsv(RESULTS / "tenet_ipeps_h100.tsv"))
    lines += render_comparison(read_tsv(RESULTS / "tenetc_h100.tsv"))
    lines += render_native(read_tsv(RESULTS / "tenetc_native_h100.tsv"))
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
