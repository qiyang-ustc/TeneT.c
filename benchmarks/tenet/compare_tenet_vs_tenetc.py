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
        "chi\tmaster_seconds\ttenetc_seconds\tratio_tenetc_over_master\t"
        "speedup_master_over_tenetc\tmaster_err\ttenetc_err\n"
    )
    for chi in sorted(set(master) & set(tenetc)):
        m = master[chi]
        t = tenetc[chi]
        master_seconds = float(m["median_total_seconds"])
        tenetc_seconds = float(t["median_total_seconds"])
        ratio = tenetc_seconds / master_seconds
        out.write(
            "%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9e\t%.9e\n"
            % (
                chi,
                master_seconds,
                tenetc_seconds,
                ratio,
                1.0 / ratio,
                float(m["err"]),
                float(t["err"]),
            )
        )
