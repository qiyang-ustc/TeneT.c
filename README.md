# TeneT.c

Preview / benchmark artifact for native accelerated experiments on selected
TeneT-style tensor-network workloads. This is not a complete replacement for
TeneT.jl. The Julia package lives in `TeneTC/`; install with:

```julia
using Pkg
Pkg.add(url="https://github.com/qiyang-ustc/TeneT.c", subdir="TeneTC")
```

## Acknowledgement

TeneT.jl is the scientific and API inspiration for this work. We are grateful to
Xingyu Zhang and the TeneT.jl contributors for the original implementation.
This repository should be read as a specialized native backend and benchmark
suite, not as a replacement for TeneT.jl. Any compatibility patch used for a
reference baseline is documented as an ecosystem-version adapter, not as a
criticism of the original project.

If this code is useful in your work, please cite and acknowledge the original
TeneT.jl work by Xingyu Zhang and contributors, and cite and acknowledge
KrylovKit.jl by Jutho Haegeman and contributors for the Krylov solver design.
Please do not cite TeneT.c or KrylovKit.c as the scientific source; these
repositories are engineering backends and benchmark artifacts.

## Layout

- `TeneTC/`: release-facing Julia wrapper package.
- `FastTeneT/`: 2D Ising boundary implementation and native integration layer.
- `TenetNative/`: native C++/CUDA implementation used by FastTeneT.
- `benchmarks/`: TeneT.jl master comparison, TeneT.c benchmark scripts,
  jobfiles, plotting scripts, and artifact conventions.

`TeneTC` depends on `KrylovKitC` from
`https://github.com/qiyang-ustc/KrylovKit.c`.

## Preliminary Performance Snapshot

The figures below are generated from compact summaries in `benchmarks/results/`.
Current summaries are marked `pre_public_measured_subset` in
`benchmarks/results/metadata.toml`; a public-main rerun should replace them
before any formal release claim. Small `chi=8` and `chi=16` runs are smoke tests
only.

![TeneT.c native speedup benchmark](TeneTC/docs/figures/tenetc_speedup.svg)

![TeneT.c runtime scaling benchmark](TeneTC/docs/figures/tenetc_scaling.svg)

See `TeneTC/README.md` for the full tables, run IDs, tolerances, and
reproduction commands.
