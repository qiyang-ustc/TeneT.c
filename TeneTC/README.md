# TeneT.c

TeneT.c is a preview / benchmark artifact for native accelerated experiments on
selected TeneT-style tensor-network workloads. It is not a complete replacement
for TeneT.jl. The Julia module name is `TeneTC`.

## Acknowledgement

TeneT.jl is the scientific and API inspiration for this package. We are grateful
to Xingyu Zhang and the TeneT.jl contributors for the original implementation.
This package should be read as a specialized native backend and benchmark suite,
not as a replacement for TeneT.jl.

If this code is useful in your work, please cite and acknowledge the original
TeneT.jl work by Xingyu Zhang and contributors, and cite and acknowledge
KrylovKit.jl by Jutho Haegeman and contributors for the Krylov solver design.
Please do not cite TeneT.c or KrylovKit.c as the scientific source; these
repositories are engineering backends and benchmark artifacts.

The release benchmarks compare against a pinned TeneT.jl `master` commit. Any
CUDA compatibility patch used for the reference baseline is documented as an
ecosystem-version adapter, not as a criticism of the original project.

## Preview Scope

- 2D classical Ising boundary VUMPS.
- CPU `Float64` and CUDA `CuArray{Float64}` native fast path.
- TeneT.jl master comparison using a pinned commit and documented patch.
- Dependency on `KrylovKit.c` for generic Krylov backend ownership.

Large-cell, symmetry-sector, complex production tensors, and broad TeneT feature
coverage are intentionally out of scope for the first release.

## Basic Usage

```julia
using TeneTC

r = run_boundary(critical_beta(); chi=64, maxiter=20, maxiter_ad=0)
logz = log_partition_density(r)
```

CUDA uses the same array-type style as the implementation package:

```julia
using CUDA
using TeneTC

CUDA.allowscalar(false)
r = run_boundary(critical_beta(); chi=128, maxiter=20, maxiter_ad=0, arraytype=CuArray)
```

## Preliminary Performance

README figures are generated from compact summaries in `benchmarks/results/`,
not edited by hand. Current summaries are marked `pre_public_measured_subset` in
`benchmarks/results/metadata.toml`; a public-main rerun should replace them
before any formal release claim.

H100 run, Snellius `gpu_h100`; TeneT.jl master baseline from
`run-4ee25b9f6f9b`, source summary `benchmarks/results/tenetc_h100.tsv`:

| chi | TeneT.jl master median (s) | TeneT.c median (s) | speedup | master error | TeneT.c error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 40.187639 | 2.196951 | 18.29x | 1.31e-5 | 1.33e-5 |
| 128 | 346.016757 | 2.916907 | 118.62x | 6.32e-6 | 6.68e-6 |
| 256 | 426.729775 | 4.331296 | 98.52x | 3.65e-6 | 3.63e-6 |

![TeneT.c native speedup benchmark](docs/figures/tenetc_speedup.svg)

![TeneT.c runtime scaling benchmark](docs/figures/tenetc_scaling.svg)

Generate replacement figures from release artifacts:

```sh
python3 benchmarks/plots/plot_speedup.py benchmarks/results/tenetc_h100.tsv TeneTC/docs/figures/tenetc_speedup.svg
python3 benchmarks/plots/plot_scaling.py benchmarks/results/tenetc_h100.tsv TeneTC/docs/figures/tenetc_scaling.svg
```

## Benchmark Rules

Formal benchmark claims must use large workloads:

- CPU: `chi=32,64,128`, warmup 2, repeat 7.
- H100: `chi=64,128,256`, warmup 2, repeat 7.

Small `chi=8` and `chi=16` runs are smoke tests and correctness checks only.
