#!/usr/bin/env python3
"""Collect compact TeneT.c release benchmark artifacts."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def copy_required(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"missing input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h100-run-id", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--host-env", type=Path, required=True)
    parser.add_argument("--source-kind", default="public_main")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    copy_required(args.comparison, args.outdir / "tenetc_h100.tsv")
    copy_required(args.host_env, args.outdir / "tenetc_h100_host_env.txt")

    metadata = args.outdir / "metadata.toml"
    metadata.write_text(
        "\n".join(
            [
                f"generated_at_utc = {quote(datetime.now(timezone.utc).isoformat())}",
                f"source_kind = {quote(args.source_kind)}",
                "",
                "[tenetc_h100]",
                f"run_id = {quote(args.h100_run_id)}",
                'summary = "benchmarks/results/tenetc_h100.tsv"',
                'host_env = "benchmarks/results/tenetc_h100_host_env.txt"',
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
