# TeneT.c

TeneT.c is a benchmark-first native backend for selected 2D Ising
TeneT-style tensor-network workloads. It depends on KrylovKit.c for the native
Krylov backend. This repository is not a complete replacement for TeneT.jl and
should not be cited as the scientific source.

```julia
using Pkg
Pkg.add(url="https://github.com/qiyang-ustc/TeneT.c", subdir="TeneTC")
```

## Acknowledgement

TeneT.jl is the scientific and API inspiration for this work. We are grateful to
Xingyu Zhang and the TeneT.jl contributors for the original implementation.
Please cite and acknowledge TeneT.jl by Xingyu Zhang and contributors when this
backend is useful in scientific work. Please also cite and acknowledge
KrylovKit.jl by Jutho Haegeman and contributors for the Krylov solver design.

Do not cite TeneT.c or KrylovKit.c as the scientific source; these repositories
are engineering backends and benchmark artifacts.

## What Is Measured

- 2D classical Ising boundary VUMPS.
- GPU TeneT.c native `CuArray{Float64}` path on H100.
- GPU TeneT.jl official `iPEPS-unified` branch real `CuArray{Float64}` path
  on H100 for the fair GPU-vs-GPU baseline.
- GPU TeneT.jl `master` timing using `CuArray{ComplexF64}` at a pinned commit
  with a documented CUDA compatibility patch, retained only as an audit row.
- Native runtime scaling at larger `chi` when the TeneT.jl baseline is not
  available.

`TeneTC` depends on `KrylovKitC` from
`https://github.com/qiyang-ustc/KrylovKit.c`.

## Correctness Before Speed

The release gate includes 2D Ising tensor construction, Onsager exact
references, CPU smoke tests, CUDA smoke tests when CUDA is available, native
path versus reference path parity, and checks that production defaults remain
real `Float64`.

```sh
TENETC_RUN_RELEASE_GATE=1 julia --project=TeneTC -e 'using Pkg; Pkg.test()'
```

Correctness artifacts report the VUMPS error returned by the solver and keep
the same `beta`, `tol`, `maxiter`, and seed across TeneT.jl and TeneT.c
comparisons.

## Baseline Policy

The fair GPU-vs-GPU reference baseline is the official TeneT.jl
`iPEPS-unified` branch at pinned commit
`a4bfff01bc898728b5b6af136f50e420aeeac5bc`, using real `Float64` CUDA arrays.

The pinned TeneT.jl `master` commit
`b9ac7919a96e930639935c9370ae568139bc8747` is also retained as a historical
timing audit with `tenet_master_cuda_compat.patch`. That patch is an
ecosystem-version adapter for current CUDA/Julia packages, not a criticism of
the original project. Because this master path uses `ComplexF64`, it is not
used for headline speedup claims against the real TeneT.c path.

Speedup is shown only for `chi` values where a real `Float64` TeneT.jl
baseline completed. Timeout, not-measured, scalar-mismatched, or audit-only
rows are reported separately and are not converted into speedup claims.

## Performance Evidence

All figures are generated from committed TSV artifacts:

```sh
python3 benchmarks/plots/plot_release_figures.py
```

Current public artifacts include 8 GPU TeneT.c H100 points and 5 completed GPU
TeneT.jl master H100 timing points. These master rows are retained as an audit
dataset, not as a headline speedup claim: TeneT.jl master uses `ComplexF64`,
while TeneT.c uses the real `Float64` native path. The fair real-vs-real
`iPEPS-unified` artifacts are collected separately before a speedup headline is
promoted.

The completed master/TeneT.c ratio is therefore a raw runtime ratio for a
non-equivalent scalar comparison. The real-vs-real `iPEPS-unified` baseline is
the required source for any headline GPU speedup.

![Completed GPU timing audit](TeneTC/docs/figures/tenetc_completed_speedup.svg)

![Native runtime scaling](TeneTC/docs/figures/tenetc_native_scaling.svg)

![Native correctness trend](TeneTC/docs/figures/tenetc_error_vs_chi.svg)

Detailed tables, run IDs, limitations, and reproduction commands are in
`TeneTC/README.md`.

## Expanded Release Sweep

```sh
bash benchmarks/run_release_suite.sh
```

Measured matrix:

- GPU TeneT.c H100 native: `chi=32,48,64,96,128,192,256,384`, warmup 2, repeat 9.
- GPU TeneT.jl `iPEPS-unified` real H100 baseline:
  `chi=32,48,64,96,128`, warmup 2, repeat 9.
- GPU TeneT.jl master H100 timing audit: completed `chi=32,48,64,96,128`,
  warmup 2, repeat 9; `chi=192,256,384` are not measured and do not appear as
  speedup.

No speedup claim is made for scalar-mismatched, missing, timed-out, or
smoke-test rows.
