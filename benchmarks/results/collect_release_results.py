#!/usr/bin/env python3
"""Collect compact TeneT.c release benchmark artifacts."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_kv_line(line: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key] = value
    return data


def read_summary_rows(path: Path, marker: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if marker not in line:
                continue
            data = parse_kv_line(line)
            if "chi" not in data:
                raise SystemExit(f"missing chi in {path}: {line.strip()}")
            rows.append(data)
    if not rows:
        raise SystemExit(f"no {marker} rows found in {path}")
    rows.sort(key=lambda row: int(row["chi"]))
    return rows


def read_native_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            if not line.strip():
                continue
            values = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, values)))
    rows.sort(key=lambda row: int(row["chi"]))
    return rows


def copy_required(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"missing input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_native_tsv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write(
            "chi\tbackend\tdevice\ttenetc_seconds\ttenetc_p25_seconds\ttenetc_p75_seconds\t"
            "tenetc_eltype\ttenetc_err\ttenetc_status\n"
        )
        for row in rows:
            out.write(
                "{chi}\t{backend}\t{device}\t{median}\t{p25}\t{p75}\t{eltype}\t{err}\tmeasured\n".format(
                    chi=row["chi"],
                    backend=row.get("backend", "unknown"),
                    device=row.get("device", "unknown"),
                    median=f"{float(row['median_total_seconds']):.9f}",
                    p25=f"{float(row['p25_total_seconds']):.9f}",
                    p75=f"{float(row['p75_total_seconds']):.9f}",
                    eltype=row.get("eltype", "unknown"),
                    err=f"{float(row['err']):.9e}",
                )
            )


def write_comparison_tsv(
    native_rows: list[dict[str, str]],
    master_rows: list[dict[str, str]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    master_by_chi = {int(row["chi"]): row for row in master_rows}
    with path.open("w", encoding="utf-8") as out:
        out.write(
            "chi\tmaster_backend\ttenetc_backend\tmaster_device\ttenetc_device\t"
            "master_eltype\ttenetc_eltype\tmaster_seconds\ttenetc_seconds\t"
            "ratio_tenetc_over_master\traw_ratio_master_over_tenetc\tmaster_err\ttenetc_err\t"
            "master_status\ttenetc_status\tcomparison_status\n"
        )
        for native in native_rows:
            chi = int(native["chi"])
            tenetc_backend = native.get("backend", native.get("tenetc_backend", "unknown"))
            tenetc_device = native.get("device", native.get("tenetc_device", "unknown"))
            tenetc_eltype = native.get("eltype", native.get("tenetc_eltype", "unknown"))
            tenetc_seconds = float(
                native.get("tenetc_seconds", native.get("median_total_seconds", "nan"))
            )
            tenetc_err = float(native.get("tenetc_err", native.get("err", "nan")))
            if chi in master_by_chi:
                master = master_by_chi[chi]
                master_backend = master.get("backend", "unknown")
                master_device = master.get("device", "unknown")
                master_eltype = master.get("eltype", "unknown")
                master_seconds = float(master["median_total_seconds"])
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
                        float(master["err"]),
                        tenetc_err,
                        comparison_status,
                    )
                )
            else:
                out.write(
                    "%d\t\t%s\t\t%s\t\t%s\t\t%.9f\t\t\t\t%.9e\tnot measured\tmeasured\tnot measured\n"
                    % (chi, tenetc_backend, tenetc_device, tenetc_eltype, tenetc_seconds, tenetc_err)
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-run-id")
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--comparison-host-env", type=Path)
    parser.add_argument("--master-raw-summary", type=Path, action="append")
    parser.add_argument("--native-run-id")
    parser.add_argument("--native-summary", type=Path)
    parser.add_argument("--native-raw-summary", type=Path)
    parser.add_argument("--native-host-env", type=Path)
    parser.add_argument("--source-kind", default="public_main_partial")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    has_comparison = (
        args.comparison_run_id
        or args.comparison
        or args.comparison_host_env
        or args.master_raw_summary
    )
    has_native = (
        args.native_run_id or args.native_summary or args.native_raw_summary or args.native_host_env
    )
    if not has_comparison and not has_native:
        raise SystemExit("provide at least one comparison or native artifact")
    if has_comparison and not args.comparison_run_id:
        raise SystemExit("comparison artifact requires --comparison-run-id")
    if has_comparison and not args.comparison_host_env:
        raise SystemExit("comparison artifact requires --comparison-host-env")
    if has_comparison and not args.comparison and not args.master_raw_summary:
        raise SystemExit(
            "comparison artifact requires either --comparison or --master-raw-summary"
        )
    if has_native and not (
        args.native_run_id
        and (args.native_summary or args.native_raw_summary)
        and args.native_host_env
    ):
        raise SystemExit(
            "native artifact requires --native-run-id, --native-summary or "
            "--native-raw-summary, and --native-host-env"
        )
    if args.master_raw_summary and not has_native:
        raise SystemExit("--master-raw-summary requires native rows for chi coverage")

    native_rows: list[dict[str, str]] = []
    if args.native_raw_summary:
        native_rows = read_summary_rows(args.native_raw_summary, "TENETC_2DISING")
        write_native_tsv(native_rows, args.outdir / "tenetc_native_h100.tsv")
    elif args.native_summary:
        copy_required(args.native_summary, args.outdir / "tenetc_native_h100.tsv")
        native_rows = read_native_tsv(args.native_summary)

    if args.master_raw_summary:
        master_rows = []
        for summary in args.master_raw_summary:
            master_rows.extend(read_summary_rows(summary, "TENET_MASTER_2DISING"))
        master_rows.sort(key=lambda row: int(row["chi"]))
        write_comparison_tsv(native_rows, master_rows, args.outdir / "tenetc_h100.tsv")
    elif args.comparison:
        copy_required(args.comparison, args.outdir / "tenetc_h100.tsv")

    if has_comparison and args.comparison_host_env:
        copy_required(
            args.comparison_host_env,
            args.outdir / "tenetc_master_h100_host_env.txt",
        )
    if has_native:
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
                'host_env = "benchmarks/results/tenetc_master_h100_host_env.txt"',
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
