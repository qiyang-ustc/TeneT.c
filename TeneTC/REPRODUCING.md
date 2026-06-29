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
TENETC_RUN_FASTTENET_GATE=1 julia --project=TeneTC -e 'using Pkg; Pkg.test()'
julia --project=TeneTC --startup-file=no benchmarks/tenet/run_tenetc.jl
python3 benchmarks/tenet/compare_tenet_vs_tenetc.py master_summary.txt tenetc_summary.txt comparison.tsv
python3 benchmarks/plots/plot_speedup.py benchmarks/results/tenetc_h100.tsv TeneTC/docs/figures/tenetc_speedup.svg
```

Use the JobFiles in `benchmarks/jobfiles/` for audited H100 runs. Commit only
compact summaries and host metadata from jobctl artifacts.
