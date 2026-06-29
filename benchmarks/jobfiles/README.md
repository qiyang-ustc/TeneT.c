# Release JobFiles

These JobFiles are thin release entrypoints. They call the benchmark scripts in
`benchmarks/` and leave raw logs/results under the job run directory.

Use small `*_CHIS=8,16` overrides only for smoke tests. Release headline runs
must use the default large sizes.

Before running `snellius_tenetc_h100.jobfile.yaml`, prepare a pinned TeneT.jl
master checkout and apply `benchmarks/tenet/patches/tenet_master_cuda_compat.patch`
if needed by the CUDA/KrylovKit versions on the target host.

