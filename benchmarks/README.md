# TeneT.c Benchmarks

The release benchmark suite measures one warmed VUMPS step. It does not measure
solve-to-convergence runtime.

Allowed comparisons:

- CPU on Oblix: TeneT.c/FastTeneT versus TeneT.jl iPEPS-unified, both `Float64`.
- GPU on Snellius H100: TeneT.c/FastTeneT versus TeneT.jl iPEPS-unified, both
  `CuArray{Float64}`.

Default sweep:

- `TENET_BENCH_CHIS=32,64,96,128,160,192,224,256`
- `TENET_BENCH_WARMUP=3`
- `TENET_BENCH_REPEATS=11`
- `TENET_BENCH_TOL=1e-10`
- `TENET_BENCH_MAXITER=20`
- `TENET_BENCH_MINITER=1`
- `TENET_IPEPS_METHOD=krylovkit`
- `TENET_IPEPS_MODE=general`

Run all official jobs through jobctl:

```bash
bash benchmarks/run_release_suite.sh
```
