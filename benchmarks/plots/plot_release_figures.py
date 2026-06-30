#!/usr/bin/env python3
"""Generate release README figures from committed TeneT.c artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
FIGURES = ROOT / "TeneTC" / "docs" / "figures"

BLUE = "#2f67a8"
GREEN = "#2f7d59"
RED = "#b45f4d"
GRAY = "#52575c"
LIGHT_GRID = "#b9c0c9"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
    }
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_completed_speedup(rows: list[dict[str, str]], path: Path, *, target: int) -> None:
    chis = [int(r["chi"]) for r in rows]
    xpos = list(range(len(rows)))
    measured = []
    for row in rows:
        ratio = row.get("ratio_tenetc_over_master", "")
        measured.append(None if ratio == "" else 1.0 / float(ratio))
    completed = sum(y is not None for y in measured)
    missing = len(measured) - completed
    ymax = max([y for y in measured if y is not None] + [1.0])

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for i, y in enumerate(measured):
        if y is None:
            ax.scatter(i, ymax * 0.045, marker="x", s=80, linewidths=2.0, color="#7d8790")
        else:
            ax.bar(i, y, width=0.62, color=BLUE, edgecolor="#263238", linewidth=0.7)
            ax.text(i, y + ymax * 0.04, f"{y:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.0)
    ax.text(0.01, 0.98, f"{completed}/{target} planned master baselines completed",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=GRAY)
    if missing:
        ax.text(0.99, 0.98, "x = baseline not measured",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=GRAY)
    ax.text(len(rows) - 0.45, 1.0 + ymax * 0.025, "parity",
            ha="right", va="bottom", fontsize=8.5, color=GRAY)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(c) for c in chis])
    ax.set_xlabel("bond dimension chi")
    ax.set_ylabel("speedup (TeneT.jl master / TeneT.c)")
    ax.set_title("Snellius H100, 2D Ising completed-baseline speedup\nwarmup=2 repeat=9")
    ax.set_ylim(0, ymax * 1.28)
    ax.grid(axis="y", alpha=0.28, color=LIGHT_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    finish(fig, path)


def plot_native_scaling(rows: list[dict[str, str]], path: Path, *, target: int) -> None:
    chis = [int(r["chi"]) for r in rows]
    xpos = list(range(len(rows)))
    ys = [float(r["tenetc_seconds"]) for r in rows]
    lower = [y - float(r["tenetc_p25_seconds"]) for y, r in zip(ys, rows)]
    upper = [float(r["tenetc_p75_seconds"]) - y for y, r in zip(ys, rows)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.errorbar(xpos, ys, yerr=[lower, upper], marker="o", linewidth=2.0,
                markersize=5.5, capsize=4, color=BLUE)
    for i, y in enumerate(ys):
        ax.text(i, y + max(ys) * 0.035, f"{y:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.text(0.01, 0.98, f"{len(rows)}/{target} planned native chi values measured; error bars p25-p75",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=GRAY)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(c) for c in chis])
    ax.set_xlabel("bond dimension chi")
    ax.set_ylabel("median wall time (s)")
    ax.set_title("Snellius H100, TeneT.c native runtime scaling, 2D Ising\nwarmup=2 repeat=9")
    ax.set_ylim(0, max(ys) * 1.22)
    ax.grid(axis="y", alpha=0.28, color=LIGHT_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    finish(fig, path)


def plot_error(rows: list[dict[str, str]], path: Path, *, target: int) -> None:
    chis = [int(r["chi"]) for r in rows]
    xpos = list(range(len(rows)))
    err = [float(r["tenetc_err"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(xpos, err, marker="o", linewidth=2.0, markersize=5.5, color=GREEN, label="TeneT.c")
    ax.text(0.99, 0.98, f"{len(rows)}/{target} planned native chi values measured",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=GRAY)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(c) for c in chis])
    ax.set_xlabel("bond dimension chi")
    ax.set_ylabel("reported 2D Ising error")
    ax.set_yscale("log")
    ax.set_title("Snellius H100, TeneT.c correctness trend, 2D Ising\nwarmup=2 repeat=9")
    ax.grid(axis="y", which="both", alpha=0.28, color=LIGHT_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    finish(fig, path)


def main() -> None:
    comparison = read_tsv(RESULTS / "tenetc_h100.tsv")
    native = read_tsv(RESULTS / "tenetc_native_h100.tsv")
    plot_completed_speedup(comparison, FIGURES / "tenetc_completed_speedup.svg", target=5)
    plot_native_scaling(native, FIGURES / "tenetc_native_scaling.svg", target=8)
    plot_error(native, FIGURES / "tenetc_error_vs_chi.svg", target=8)


if __name__ == "__main__":
    main()
