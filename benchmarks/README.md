# Release Benchmarks

This directory contains the release-facing benchmark suite for the two packages:

- `KrylovKit.c` (`KrylovKitC`): KrylovKit.jl vs native Krylov backend.
- `TeneT.c` (`TeneTC`): TeneT.jl master vs native TeneT backend.

Small jobs are smoke tests only. Headline release claims must use the large
defaults:

- TeneT.c H100 native: `chi=32,48,64,96,128,192,256,384`, warmup 2, repeat 9.
- TeneT.jl master H100 baseline: required `chi=32,48,64,96,128`, warmup 2,
  repeat 9; larger master baselines may be attempted but are not required for
  headline speedup.

Every run must preserve raw CSV/TSV data, host metadata, package commits, thread
counts, BLAS/CUDA versions, tolerance, Krylov dimension, max iteration count,
seed, residuals, and timing quantiles.

Checked-in summaries under `benchmarks/results/` are compact release artifacts.
Full jobctl run directories and Julia manifests are not committed.

Run the expanded release suite with:

```sh
bash benchmarks/run_release_suite.sh
```
