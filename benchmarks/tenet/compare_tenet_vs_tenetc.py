#!/usr/bin/env python3
"""Compare warmed VUMPS-step timing summaries for one backend.

The release comparison is TeneT.jl iPEPS-unified versus TeneT.c/FastTeneT.
It only consumes the two warmed-step markers and never compares different
backends.
"""

from __future__ import annotations

import sys


def parse_line(line: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key] = value
    return data


def read_rows(path: str, marker: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if marker not in line:
                continue
            data = parse_line(line)
            rows[int(data["chi"])] = data
    if not rows:
        raise SystemExit(f"no {marker} rows found in {path}")
    return rows


if len(sys.argv) != 4:
    raise SystemExit(
        "usage: compare_tenet_vs_tenetc.py ipeps_vumps_step_summary.txt "
        "tenetc_vumps_step_summary.txt out.tsv"
    )

ipeps = read_rows(sys.argv[1], "TENET_IPEPS_VUMPS_STEP")
tenetc = read_rows(sys.argv[2], "TENETC_VUMPS_STEP")

with open(sys.argv[3], "w", encoding="utf-8") as out:
    out.write(
        "chi\tipeps_backend\ttenetc_backend\tipeps_device\ttenetc_device\t"
        "ipeps_eltype\ttenetc_eltype\tipeps_step\ttenetc_step\t"
        "ipeps_step_seconds\ttenetc_step_seconds\t"
        "ratio_tenetc_over_ipeps\tratio_ipeps_over_tenetc\t"
        "ipeps_status\ttenetc_status\tcomparison_status\n"
    )
    for chi in sorted(set(ipeps) | set(tenetc)):
        i = ipeps.get(chi)
        t = tenetc.get(chi)
        if i is None:
            tenetc_seconds = float(t["median_step_seconds"])
            out.write(
                "\t".join(
                    (
                        str(chi),
                        "",
                        t.get("backend", "unknown"),
                        "",
                        t.get("device", "unknown"),
                        "",
                        t.get("eltype", "Float64"),
                        "",
                        t.get("step", "unknown"),
                        "",
                        f"{tenetc_seconds:.9f}",
                        "",
                        "",
                        "missing",
                        "measured",
                        "missing_ipeps_row",
                    )
                )
                + "\n"
            )
            continue
        if t is None:
            ipeps_seconds = float(i["median_step_seconds"])
            out.write(
                "\t".join(
                    (
                        str(chi),
                        i.get("backend", "unknown"),
                        "",
                        i.get("device", "unknown"),
                        "",
                        i.get("eltype", "unknown"),
                        "",
                        i.get("step", "unknown"),
                        "",
                        f"{ipeps_seconds:.9f}",
                        "",
                        "",
                        "",
                        "measured",
                        "missing",
                        "missing_tenetc_row",
                    )
                )
                + "\n"
            )
            continue

        ipeps_backend = i.get("backend", "unknown")
        tenetc_backend = t.get("backend", "unknown")
        ipeps_device = i.get("device", "unknown")
        tenetc_device = t.get("device", "unknown")
        ipeps_eltype = i.get("eltype", "unknown")
        tenetc_eltype = t.get("eltype", "Float64")
        ipeps_step = i.get("step", "unknown")
        tenetc_step = t.get("step", "unknown")
        ipeps_seconds = float(i["median_step_seconds"])
        tenetc_seconds = float(t["median_step_seconds"])
        ratio = tenetc_seconds / ipeps_seconds
        comparison_status = (
            "comparable_warmed_step"
            if (
                ipeps_backend == tenetc_backend
                and ipeps_eltype == tenetc_eltype
                and ipeps_step == "vumps_step"
                and tenetc_step == "vumps_step_Hermitian"
            )
            else "invalid_mismatch"
        )
        out.write(
            "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.9f\tmeasured\tmeasured\t%s\n"
            % (
                chi,
                ipeps_backend,
                tenetc_backend,
                ipeps_device,
                tenetc_device,
                ipeps_eltype,
                tenetc_eltype,
                ipeps_step,
                tenetc_step,
                ipeps_seconds,
                tenetc_seconds,
                ratio,
                1.0 / ratio,
                comparison_status,
            )
        )
