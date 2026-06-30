# TeneT.c

This release benchmark uses warmed single VUMPS-step timing only. It compares
TeneT.c/FastTeneT with the pinned TeneT.jl iPEPS-unified baseline on the same
backend and the same real `Float64` arithmetic.

Official CPU results are run on Oblix with `--cpus 4 --mem 4G`. Official GPU
results are run on Snellius H100 with one H100 allocation and `--mem 180G`.
CPU and GPU results are reported as separate tables.

The release sweep is `chi = 32,64,96,128,160,192,224,256`, with 3 warmup steps
and 11 timed repeats. VUMPS convergence error is not a reported benchmark
quantity for this suite.

Run definitions live under `benchmarks/jobfiles/`. Completed official artifacts
are recorded under `benchmarks/results/`.
