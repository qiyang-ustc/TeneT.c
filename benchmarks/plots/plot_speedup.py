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


def speedup(row):
    if "native_over_krylov" in row and row["native_over_krylov"]:
        return 1.0 / float(row["native_over_krylov"])
    if "ratio_tenetc_over_master" in row and row["ratio_tenetc_over_master"]:
        return 1.0 / float(row["ratio_tenetc_over_master"])
    if "ratio_fasttenet_over_master" in row and row["ratio_fasttenet_over_master"]:
        return 1.0 / float(row["ratio_fasttenet_over_master"])
    raise KeyError("no speedup ratio column found")


if len(sys.argv) != 3:
    raise SystemExit("usage: plot_speedup.py input.csv-or-tsv output.svg")

rows = load_rows(sys.argv[1])
chis = [int(r["chi"]) for r in rows]
xpos = list(range(len(rows)))
measured = []
for row in rows:
    try:
        measured.append(speedup(row))
    except KeyError:
        measured.append(None)

ymax = max([y for y in measured if y is not None] + [1.0])

fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
for i, (row, y) in enumerate(zip(rows, measured)):
    if y is None:
        ax.scatter(i, ymax * 0.045, marker="x", s=80, linewidths=2.0, color="#7d8790")
        ax.text(
            i,
            ymax * 0.13,
            "baseline\n timeout",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#555555",
        )
        continue
    ax.bar(i, y, width=0.62, color="#3568a7", edgecolor="#263238", linewidth=0.7)
    ax.text(i, y + ymax * 0.04, f"{y:.2f}x", ha="center", va="bottom", fontsize=9)

ax.axhline(1.0, color="#4f4f4f", linestyle="--", linewidth=1.0)
ax.set_xticks(xpos)
ax.set_xticklabels([str(c) for c in chis])
ax.set_xlabel("bond dimension chi")
ax.set_ylabel("speedup (TeneT.jl / TeneT.c)")
ax.set_title("TeneT.c H100 speedup vs TeneT.jl master")
ax.set_ylim(0, ymax * 1.28)
ax.grid(axis="y", alpha=0.28)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(
    0.01,
    0.98,
    "speedup is shown only where the master baseline completed",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
    color="#555555",
)
ax.text(
    len(rows) - 0.5,
    1.0 + ymax * 0.025,
    "parity",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color="#555555",
)
fig.savefig(sys.argv[2])
