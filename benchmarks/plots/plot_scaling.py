#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, newline="") as handle:
        dialect = csv.Sniffer().sniff(handle.read(4096), delimiters=",\t")
        handle.seek(0)
        return list(csv.DictReader(handle, dialect=dialect))


if len(sys.argv) != 3:
    raise SystemExit("usage: plot_scaling.py input.csv-or-tsv output.png")

rows = load_rows(sys.argv[1])
xs = [int(r["chi"]) for r in rows]
series = []
for name in (
    "native_seconds_median",
    "krylov_seconds_median",
    "master_seconds",
    "tenetc_seconds",
):
    if name in rows[0]:
        series.append((name, [float(r[name]) for r in rows]))

plt.figure(figsize=(6.0, 4.0))
for name, ys in series:
    plt.plot(xs, ys, marker="o", label=name)
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("chi")
plt.ylabel("median seconds")
plt.title("Runtime scaling")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=200)
