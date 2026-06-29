#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

jobctl run "$repo_root/benchmarks/jobfiles/snellius_tenetc_native_h100.jobfile.yaml" \
  --title "TeneT.c expanded native H100 sweep" \
  --tag tenetc --tag release --tag h100 --tag native

for chi in 32 48 64 96 128; do
  jobctl run "$repo_root/benchmarks/jobfiles/snellius_tenetc_master_h100_onechi.jobfile.yaml" \
    --title "TeneT.jl master H100 baseline chi=$chi" \
    --tag tenetc --tag release --tag h100 --tag master \
    --param chi="$chi"
done
