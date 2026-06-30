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
