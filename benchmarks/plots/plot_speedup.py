#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, newline="") as handle:
        dialect = csv.Sniffer().sniff(handle.read(4096), delimiters=",\t")
        handle.seek(0)
        return list(csv.DictReader(handle, dialect=dialect))


def speedup(row):
    if "native_over_krylov" in row and row["native_over_krylov"]:
        return 1.0 / float(row["native_over_krylov"])
    if "ratio_tenetc_over_master" in row and row["ratio_tenetc_over_master"]:
        return 1.0 / float(row["ratio_tenetc_over_master"])
    if "ratio_fasttenet_over_master" in row and row["ratio_fasttenet_over_master"]:
        return 1.0 / float(row["ratio_fasttenet_over_master"])
    raise KeyError("no speedup ratio column found")


if len(sys.argv) != 3:
    raise SystemExit("usage: plot_speedup.py input.csv-or-tsv output.png")

rows = load_rows(sys.argv[1])
xs = [int(r["chi"]) for r in rows]
ys = [speedup(r) for r in rows]

plt.figure(figsize=(6.0, 4.0))
plt.plot(xs, ys, marker="o")
plt.xscale("log", base=2)
plt.xlabel("chi")
plt.ylabel("speedup")
plt.title("Native backend speedup")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=200)
