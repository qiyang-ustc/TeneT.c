# TeneT.c Benchmarks

These scripts compare GPU `TeneTC` against official TeneT.jl GPU baselines on
the 2D Ising benchmark. The fair baseline is TeneT.jl `iPEPS-unified` with
`TENET_IPEPS_MODE=general` and real `CuArray{Float64}` tensors. The pinned
TeneT.jl `master` runner is retained only as a `ComplexF64` timing audit.

The TeneT.jl master audit path may require the patch in
`patches/tenet_master_cuda_compat.patch` when running with modern CUDA.jl and
KrylovKit.jl versions. The patch only adapts package-ecosystem compatibility for
benchmarking; it is not part of TeneT.c and is not presented as a flaw in
TeneT.jl.

Default release sizes:

- GPU TeneT.c H100 native: `TENET_BENCH_CHIS=32,48,64,96,128,192,256,384`
- GPU TeneT.jl `iPEPS-unified` real H100 baseline: one `chi` per job for
  `32,48,64,96,128`
- GPU TeneT.jl master H100 audit: one `chi` per job for `32,48,64,96,128`

Run examples:

```sh
TENET_MASTER_REPO=/path/to/TeneT.jl julia --project=/path/to/TeneT.jl benchmarks/tenet/run_tenet_master.jl
TENET_IPEPS_MODE=general julia --project=/path/to/TeneT-iPEPS-unified benchmarks/tenet/run_tenet_ipeps_unified.jl
julia --project=TeneTC -e 'using Pkg; Pkg.instantiate()'
julia --project=TeneTC benchmarks/tenet/run_tenetc.jl
python3 benchmarks/tenet/compare_tenet_vs_tenetc.py master_summary.txt tenetc_summary.txt comparison.tsv
```
