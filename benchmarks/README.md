# Release Benchmarks

This directory contains the release-facing benchmark suite for the two packages:

- `KrylovKit.c` (`KrylovKitC`): KrylovKit.jl vs native Krylov backend.
- `TeneT.c` (`TeneTC`): TeneT.jl master vs native TeneT backend.

Small jobs are smoke tests only. Headline release claims must use the large
defaults:

- CPU: `chi=32,64,128`, warmup 2, repeat 7.
- H100: `chi=64,128,256`, warmup 2, repeat 7.

Every run must preserve raw CSV/TSV data, host metadata, package commits, thread
counts, BLAS/CUDA versions, tolerance, Krylov dimension, max iteration count,
seed, residuals, and timing quantiles.

This branch keeps local unregistered Julia dependency wiring in the benchmark
projects. The release branch omits those `[sources]` entries.
