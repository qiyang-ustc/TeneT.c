#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

jobctl run "$repo_root/benchmarks/jobfiles/oblix_tenet_vumps_cpu.jobfile.yaml" \
  --title "TeneT warmed VUMPS step CPU sweep on Oblix" \
  --tag tenetc --tag release --tag cpu --tag oblix --tag vumps-step \
  --backend slurm --server oblix --partition lerner --cpus 4 --mem 4G --time 02:00:00

jobctl run "$repo_root/benchmarks/jobfiles/snellius_tenet_vumps_h100.jobfile.yaml" \
  --title "TeneT warmed VUMPS step GPU sweep on Snellius H100" \
  --tag tenetc --tag release --tag h100 --tag snellius --tag vumps-step \
  --backend slurm --server snellius \
  --partition gpu_h100 --gres gpu:h100:1 --cpus 16 --mem 180G --time 01:30:00
