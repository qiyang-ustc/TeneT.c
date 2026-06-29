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
completed under the same benchmark settings. Timeout rows are visible and are
not converted into speedup claims.

## Performance Evidence

Figures are generated from committed TSV artifacts under `benchmarks/results/`:

```sh
python3 benchmarks/plots/plot_release_figures.py
```

### Completed-baseline speedup

H100 public-main runs on Snellius `gpu_h100`; TeneT.c native run
`run-e51b2476d875`, TeneT.jl master `chi=64` baseline
`run-54ccea21ccc0`.

| chi | TeneT.jl master median (s) | TeneT.c median (s) | speedup | master error | TeneT.c error | status |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 64 | 40.995462 | 2.188001 | 18.74x | 1.53e-5 | 1.33e-5 | measured |
| 128 | not measured | 2.948597 | n/a | n/a | 6.68e-6 | master baseline timeout |
| 256 | not measured | 4.467242 | n/a | n/a | 3.67e-6 | master baseline timeout |

![TeneT.c completed-baseline speedup](docs/figures/tenetc_completed_speedup.svg)

### Native scaling

The native-only scaling curve is kept separate so large `chi` measurements are
visible without implying a completed TeneT.jl baseline.

| chi | TeneT.c median (s) | p25 (s) | p75 (s) | TeneT.c error |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.188001 | 2.187011 | 2.189780 | 1.33e-5 |
| 128 | 2.948597 | 2.946695 | 2.953534 | 6.68e-6 |
| 256 | 4.467242 | 4.464754 | 4.469334 | 3.67e-6 |

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

Larger master baselines may be attempted, but timeout rows remain `not
measured` and are not used for speedup headlines.

## Limitations

- This is not full TeneT.jl feature coverage.
- Current completed-baseline comparison is partial until the expanded master
  baseline jobs finish.
- Native-only scaling is not a speedup claim.
- The CUDA compatibility patch is only for benchmarking the pinned TeneT.jl
  baseline on the current CUDA/Julia environment.
