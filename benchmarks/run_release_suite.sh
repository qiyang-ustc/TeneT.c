#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

jobctl run "$repo_root/benchmarks/jobfiles/snellius_tenetc_native_h100.jobfile.yaml" \
  --title "TeneT.c expanded native H100 sweep" \
  --tag tenetc --tag release --tag h100 --tag native \
  --partition gpu_h100 --gres gpu:1 --mem 16G --time 01:30:00

for chi in 32 48 64 96 128; do
  jobctl run "$repo_root/benchmarks/jobfiles/snellius_tenet_ipeps_h100_onechi.jobfile.yaml" \
    --title "TeneT.jl iPEPS-unified real CUDA baseline chi=$chi" \
    --tag tenetc --tag release --tag h100 --tag tenet-ipeps \
    --partition gpu_h100 --gres gpu:1 --mem 16G --time 02:00:00 \
    --param chi="$chi"

  jobctl run "$repo_root/benchmarks/jobfiles/snellius_tenetc_master_h100_onechi.jobfile.yaml" \
    --title "TeneT.jl master ComplexF64 H100 audit chi=$chi" \
    --tag tenetc --tag release --tag h100 --tag master \
    --partition gpu_h100 --gres gpu:1 --mem 16G --time 02:00:00 \
    --param chi="$chi"
done
