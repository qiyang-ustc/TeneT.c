# Reproducing TeneT.c Results

TeneT.c release benchmarks use a pinned TeneT.jl `master` checkout as the
reference baseline and the current TeneT.c implementation as the native backend.

Required metadata:

- TeneT.c commit and dirty status.
- TeneT.jl commit.
- TeneT master compatibility patch checksum.
- Julia version.
- CPU/GPU model.
- BLAS/CUDA/cuBLAS/cuTENSOR versions.
- Threads and environment variables.
- 2D Ising beta, `chi`, tolerance, VUMPS iteration limits, and seed.

Recommended commands from the repository root:

```sh
julia --project=TeneTC -e 'using Pkg; Pkg.instantiate()'
julia --project=TeneTC --startup-file=no benchmarks/tenet/run_tenetc.jl
python3 benchmarks/tenet/compare_tenet_vs_tenetc.py master_summary.txt tenetc_summary.txt comparison.tsv
python3 benchmarks/plots/plot_speedup.py comparison.tsv results/figures/tenetc_speedup.png
```

This branch contains local unregistered workspace dependency wiring for the
benchmark environment.
