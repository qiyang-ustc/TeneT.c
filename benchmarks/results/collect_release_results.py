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
    parser.add_argument("--comparison-run-id")
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--comparison-host-env", type=Path)
    parser.add_argument("--native-run-id")
    parser.add_argument("--native-summary", type=Path)
    parser.add_argument("--native-host-env", type=Path)
    parser.add_argument("--source-kind", default="public_main_partial")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    has_comparison = args.comparison_run_id or args.comparison or args.comparison_host_env
    has_native = args.native_run_id or args.native_summary or args.native_host_env
    if not has_comparison and not has_native:
        raise SystemExit("provide at least one comparison or native artifact")
    if has_comparison and not (
        args.comparison_run_id and args.comparison and args.comparison_host_env
    ):
        raise SystemExit(
            "comparison artifact requires --comparison-run-id, --comparison, and --comparison-host-env"
        )
    if has_native and not (args.native_run_id and args.native_summary and args.native_host_env):
        raise SystemExit(
            "native artifact requires --native-run-id, --native-summary, and --native-host-env"
        )

    if has_comparison:
        copy_required(args.comparison, args.outdir / "tenetc_h100.tsv")
        copy_required(
            args.comparison_host_env,
            args.outdir / "tenetc_master_h100_chi64_host_env.txt",
        )
    if has_native:
        copy_required(args.native_summary, args.outdir / "tenetc_native_h100.tsv")
        copy_required(args.native_host_env, args.outdir / "tenetc_native_h100_host_env.txt")

    lines = [
        f"generated_at_utc = {quote(datetime.now(timezone.utc).isoformat())}",
        f"source_kind = {quote(args.source_kind)}",
        "",
    ]
    if has_comparison:
        lines.extend(
            [
                "[tenetc_h100_comparison]",
                f"run_id = {quote(args.comparison_run_id)}",
                'summary = "benchmarks/results/tenetc_h100.tsv"',
                'host_env = "benchmarks/results/tenetc_master_h100_chi64_host_env.txt"',
            ]
        )
        lines.append("")
    if has_native:
        lines.extend(
            [
                "[tenetc_native_h100]",
                f"run_id = {quote(args.native_run_id)}",
                'summary = "benchmarks/results/tenetc_native_h100.tsv"',
                'host_env = "benchmarks/results/tenetc_native_h100_host_env.txt"',
            ]
        )
        lines.append("")

    metadata = args.outdir / "metadata.toml"
    metadata.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
