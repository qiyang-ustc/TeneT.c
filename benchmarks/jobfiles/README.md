# Release JobFiles

These JobFiles are thin release entrypoints. They call the benchmark scripts in
`benchmarks/` and leave raw logs/results under the job run directory.

Use small `*_CHIS=8,16` overrides only for smoke tests. Formal benchmark claims
must use the default large sizes and committed compact artifacts.

Before running `snellius_tenetc_h100.jobfile.yaml`, prepare a pinned TeneT.jl
master checkout and apply `benchmarks/tenet/patches/tenet_master_cuda_compat.patch`
if needed by the CUDA/KrylovKit versions on the target host.

For long baselines, prefer `snellius_tenetc_master_h100_onechi.jobfile.yaml`
plus `snellius_tenetc_native_h100.jobfile.yaml` per `chi`. If the master
baseline times out for a size, report that size as not measured and keep the
native-only timing separate.
