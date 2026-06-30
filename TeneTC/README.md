# TeneTC

`TeneTC` exposes the TeneT.c/FastTeneT real-valued path used by the release
benchmark.

Release benchmark contract:

- warmed single VUMPS-step timing only;
- TeneT.c/FastTeneT versus pinned TeneT.jl iPEPS-unified;
- real `Float64` on CPU and real `CuArray{Float64}` on GPU;
- CPU runs on Oblix with `--cpus 4 --mem 4G`;
- GPU runs on Snellius H100 with one H100 and `--mem 180G`;
- `chi = 32,64,96,128,160,192,224,256`;
- warmup 3, repeats 11.

The benchmark table reports step runtime ratios only. It does not report VUMPS
convergence error.

## Acknowledgements and Citation

`TeneTC` is downstream engineering work built around high-quality Julia oracles.
We deeply thank Xingyu Zhang for TeneT.jl, which provides the tensor-network
implementation used as the iPEPS-unified reference for this benchmark. We also
thank Jutho for KrylovKit.jl, which supplies the Krylov machinery used by the
Julia reference path.

If you use this code, please also cite TeneT.jl and KrylovKit.jl. The native
C++/CUDA kernels here are validated against those reference implementations;
they do not replace the scientific and software credit of the upstream projects.

## Release Benchmark Results

These are warmed single-step VUMPS timings. They are not full-solve timings.
`iPEPS/TeneT.c` is the ratio of median warmed step runtimes.

CPU results were run on Oblix `lerner` with `4` CPU cores and `4G` memory.

| chi | TeneT.jl iPEPS step (s) | TeneT.c step (s) | iPEPS/TeneT.c |
| ---: | ---: | ---: | ---: |
| 32 | 0.0526 | 0.0221 | 2.38x |
| 64 | 0.1913 | 0.0964 | 1.98x |
| 96 | 0.4476 | 0.2217 | 2.02x |
| 128 | 0.8960 | 0.5186 | 1.73x |
| 160 | 1.6235 | 0.9782 | 1.66x |
| 192 | 3.2339 | 1.8103 | 1.79x |
| 224 | 5.6084 | 3.1603 | 1.77x |
| 256 | 7.9310 | 4.4343 | 1.79x |

GPU results were run on Snellius H100 with one H100, `16` CPU cores, and
`180G` host memory.

| chi | TeneT.jl iPEPS step (s) | TeneT.c step (s) | iPEPS/TeneT.c |
| ---: | ---: | ---: | ---: |
| 32 | 1.1867 | 0.0651 | 18.24x |
| 64 | 1.7126 | 0.0927 | 18.48x |
| 96 | 1.8529 | 0.1040 | 17.82x |
| 128 | 1.9420 | 0.1154 | 16.83x |
| 160 | 2.0685 | 0.1200 | 17.23x |
| 192 | 2.1806 | 0.1290 | 16.91x |
| 224 | 2.9501 | 0.1633 | 18.07x |
| 256 | 2.8209 | 0.1693 | 16.67x |
