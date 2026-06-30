#!/usr/bin/env python3
import sys


def parse_line(line):
    data = {}
    for part in line.strip().split():
        if "=" in part:
            key, value = part.split("=", 1)
            data[key] = value
    return data


def read_rows(path, label):
    rows = []
    with open(path) as handle:
        for line in handle:
            if label == "master" and "TENET_MASTER_2DISING" not in line:
                continue
            if label == "tenetc" and "TENETC_2DISING" not in line:
                continue
            data = parse_line(line)
            rows.append((int(data["chi"]), data))
    return rows


if len(sys.argv) != 4:
    raise SystemExit(
        "usage: compare_tenet_vs_tenetc.py master_summary.txt tenetc_summary.txt out.tsv"
    )

master = dict(read_rows(sys.argv[1], "master"))
tenetc = dict(read_rows(sys.argv[2], "tenetc"))

with open(sys.argv[3], "w") as out:
    out.write(
        "chi\tmaster_backend\ttenetc_backend\tmaster_device\ttenetc_device\t"
        "master_eltype\ttenetc_eltype\tmaster_seconds\ttenetc_seconds\t"
        "ratio_tenetc_over_master\traw_ratio_master_over_tenetc\tmaster_err\ttenetc_err\t"
        "master_status\ttenetc_status\tcomparison_status\n"
    )
    for chi in sorted(tenetc):
        t = tenetc[chi]
        tenetc_backend = t.get("backend", "unknown")
        tenetc_device = t.get("device", "unknown")
        tenetc_eltype = t.get("eltype", "unknown")
        tenetc_seconds = float(t["median_total_seconds"])
        if chi in master:
            m = master[chi]
            master_backend = m.get("backend", "unknown")
            master_device = m.get("device", "unknown")
            master_eltype = m.get("eltype", "unknown")
            master_seconds = float(m["median_total_seconds"])
            ratio = tenetc_seconds / master_seconds
            comparison_status = (
                "comparable"
                if master_backend == tenetc_backend and master_eltype == tenetc_eltype
                else "scalar_mismatch_audit_only"
            )
            out.write(
                "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%.9f\t%.9f\t%.9f\t%.9f\t%.9e\t%.9e\tmeasured\tmeasured\t%s\n"
                % (
                    chi,
                    master_backend,
                    tenetc_backend,
                    master_device,
                    tenetc_device,
                    master_eltype,
                    tenetc_eltype,
                    master_seconds,
                    tenetc_seconds,
                    ratio,
                    1.0 / ratio,
                    float(m["err"]),
                    float(t["err"]),
                    comparison_status,
                )
            )
        else:
            out.write(
                "%d\t\t%s\t\t%s\t\t%s\t\t%.9f\t\t\t\t%.9e\tnot measured\tmeasured\tnot measured\n"
                % (chi, tenetc_backend, tenetc_device, tenetc_eltype, tenetc_seconds, float(t["err"]))
            )
