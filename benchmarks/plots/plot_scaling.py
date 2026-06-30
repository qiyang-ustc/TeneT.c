#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
    }
)


def load_rows(path):
    with open(path, newline="") as handle:
        dialect = csv.Sniffer().sniff(handle.read(4096), delimiters=",\t")
        handle.seek(0)
        return list(csv.DictReader(handle, dialect=dialect))


if len(sys.argv) != 3:
    raise SystemExit("usage: plot_scaling.py input.csv-or-tsv output.svg")

rows = load_rows(sys.argv[1])
chis = [int(r["chi"]) for r in rows]
xpos = list(range(len(rows)))

if "tenetc_seconds" in rows[0]:
    ys = [float(r["tenetc_seconds"]) for r in rows]
    if "tenetc_p25_seconds" in rows[0] and "tenetc_p75_seconds" in rows[0]:
        lower = [y - float(r["tenetc_p25_seconds"]) for y, r in zip(ys, rows)]
        upper = [float(r["tenetc_p75_seconds"]) - y for y, r in zip(ys, rows)]
        yerr = [lower, upper]
    else:
        yerr = None
    label = "TeneT.c median"
else:
    raise KeyError("expected tenetc_seconds column")

fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
ax.errorbar(
    xpos,
    ys,
    yerr=yerr,
    marker="o",
    linewidth=2.0,
    markersize=5.5,
    capsize=4,
    color="#3568a7",
    label=label,
)
for i, y in enumerate(ys):
    ax.text(i, y + max(ys) * 0.035, f"{y:.2f}s", ha="center", va="bottom", fontsize=9)

ax.set_xticks(xpos)
ax.set_xticklabels([str(c) for c in chis])
ax.set_xlabel("bond dimension chi")
ax.set_ylabel("median wall time (s)")
ax.set_title("TeneT.c H100 GPU native runtime scaling")
ax.set_ylim(0, max(ys) * 1.22)
ax.grid(axis="y", alpha=0.28)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(
    0.01,
    0.98,
    "2D Ising; error bars show p25-p75 over repeats",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
    color="#555555",
)
fig.savefig(sys.argv[2])
