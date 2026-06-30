# TeneT.c

`TeneTC` exposes a native backend for selected TeneT-style 2D Ising boundary
VUMPS workloads. It is a benchmark-first engineering package, not a complete
replacement for TeneT.jl.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/qiyang-ustc/TeneT.c", subdir="TeneTC")
```

## Acknowledgement And Citation

TeneT.jl is the scientific and API inspiration for this package. We are
grateful to Xingyu Zhang and the TeneT.jl contributors for the original
implementation. Please cite and acknowledge TeneT.jl by Xingyu Zhang and
contributors when this backend is useful in scientific work. Please also cite
and acknowledge KrylovKit.jl by Jutho Haegeman and contributors for the Krylov
solver design. Please do not cite TeneT.c or KrylovKit.c as the scientific
source.

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

## What Is Measured

| Area | Coverage |
| :--- | :--- |
| Model | 2D classical Ising boundary VUMPS |
| Backend | CPU `Float64`, CUDA `CuArray{Float64}` native path |
| Baseline | TeneT.jl `master` pinned to `b9ac7919a96e930639935c9370ae568139bc8747` |
| Patch policy | `tenet_master_cuda_compat.patch` is an ecosystem-version adapter |
| Solver dependency | `KrylovKitC` native Krylov backend |

Large-cell, symmetry-sector, complex production tensors, and broad TeneT feature
coverage are intentionally out of scope for this first benchmark release.

## Correctness Gate

Run the release gate with:

```sh
TENETC_RUN_RELEASE_GATE=1 julia --project=TeneTC -e 'using Pkg; Pkg.test()'
```

The gate includes:

| Category | Cases |
| :--- | :--- |
| 2D Ising references | tensor construction, Onsager exact values |
| Smoke tests | CPU path and CUDA path when CUDA is available |
| Native parity | native full-step path vs reference path |
| Production defaults | real `Float64` tensors; no ComplexF64 production default |
| Benchmark artifacts | every README figure must have committed raw TSV/metadata |

## Baseline Policy

Speedup is shown only for `chi` values where the TeneT.jl master baseline
completed under the same benchmark settings. Timeout or not-measured rows are
visible and are not converted into speedup claims.

## Performance Evidence

Figures are generated from committed TSV artifacts under `benchmarks/results/`:

```sh
python3 benchmarks/plots/plot_release_figures.py
```

### Completed-baseline speedup

H100 public-main runs on Snellius `gpu_h100`; TeneT.c native run
`run-1723bfcdc707`, TeneT.jl master baseline runs
`run-24c4e94078f0`, `run-a0e0f4fa3a8f`, `run-5a9253bd14e9`,
`run-beb778b4ad87`, and `run-046dc6fb654f`.

| chi | TeneT.jl master median (s) | TeneT.c median (s) | speedup | master error | TeneT.c error | status |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 32 | 39.827650 | 1.404854 | 28.35x | 2.03e-05 | 3.36e-05 | measured |
| 48 | 38.011892 | 1.719294 | 22.11x | 1.51e-05 | 2.75e-05 | measured |
| 64 | 44.800633 | 2.109899 | 21.23x | 1.28e-05 | 1.33e-05 | measured |
| 96 | 68.559429 | 2.491735 | 27.51x | 7.55e-06 | 7.17e-06 | measured |
| 128 | 422.869580 | 2.830716 | 149.39x | 6.15e-06 | 6.68e-06 | measured |
| 192 | not measured | 3.331687 | n/a | n/a | 4.23e-06 | not measured |
| 256 | not measured | 4.234532 | n/a | n/a | 3.85e-06 | not measured |
| 384 | not measured | 7.143338 | n/a | n/a | 2.81e-06 | not measured |

![TeneT.c completed-baseline speedup](docs/figures/tenetc_completed_speedup.svg)

### Native scaling

The native-only scaling curve is kept separate so large `chi` measurements are
visible without implying a completed TeneT.jl baseline.

| chi | TeneT.c median (s) | p25 (s) | p75 (s) | TeneT.c error |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.404854 | 1.404048 | 1.405367 | 3.36e-05 |
| 48 | 1.719294 | 1.718513 | 1.719433 | 2.75e-05 |
| 64 | 2.109899 | 2.109182 | 2.113542 | 1.33e-05 |
| 96 | 2.491735 | 2.491423 | 2.492150 | 7.17e-06 |
| 128 | 2.830716 | 2.830250 | 2.831308 | 6.68e-06 |
| 192 | 3.331687 | 3.328526 | 3.333417 | 4.23e-06 |
| 256 | 4.234532 | 4.230963 | 4.237986 | 3.85e-06 |
| 384 | 7.143338 | 7.140215 | 7.143937 | 2.81e-06 |

![TeneT.c native runtime scaling](docs/figures/tenetc_native_scaling.svg)

![TeneT.c native correctness trend](docs/figures/tenetc_error_vs_chi.svg)

## Expanded Release Sweep

```sh
bash benchmarks/run_release_suite.sh
```

Planned matrix:

| Workload | chi values | warmup | repeats |
| :--- | :--- | ---: | ---: |
| TeneT.c H100 native | `32,48,64,96,128,192,256,384` | 2 | 9 |
| TeneT.jl master H100 baseline | `32,48,64,96,128` | 2 | 9 |

Larger master baselines may be attempted, but not-measured rows remain excluded
from speedup headlines until a completed baseline artifact exists.

## Limitations

- This is not full TeneT.jl feature coverage.
- Completed-baseline speedup is limited to `chi=32,48,64,96,128`; larger
  master baselines remain not measured.
- Native-only scaling is not a speedup claim.
- The CUDA compatibility patch is only for benchmarking the pinned TeneT.jl
  baseline on the current CUDA/Julia environment.
