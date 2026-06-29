# TeneT.c Benchmarks

These scripts compare a pinned TeneT.jl master checkout against `TeneTC`.

The TeneT.jl reference path may require the patch in
`patches/tenet_master_cuda_compat.patch` when running with modern CUDA.jl and
KrylovKit.jl versions. The patch only adapts package-ecosystem compatibility for
benchmarking; it is not part of TeneT.c and is not presented as a flaw in
TeneT.jl.

Default release sizes:

- TeneT.c H100 native: `TENET_BENCH_CHIS=32,48,64,96,128,192,256,384`
- TeneT.jl master H100 baseline: one `chi` per job for `32,48,64,96,128`

Run examples:

```sh
TENET_MASTER_REPO=/path/to/TeneT.jl julia --project=/path/to/TeneT.jl benchmarks/tenet/run_tenet_master.jl
julia --project=TeneTC -e 'using Pkg; Pkg.instantiate()'
julia --project=TeneTC benchmarks/tenet/run_tenetc.jl
python3 benchmarks/tenet/compare_tenet_vs_tenetc.py master_summary.txt tenetc_summary.txt comparison.tsv
```
