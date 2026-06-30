# FastTeneT

FastTeneT is a real-valued, dense one-site tensor-network package used by the
TeneT.c preview benchmark artifact. It no longer depends on the sibling
`TeneT.jl` package; native kernel ownership for the fast dense path lives in
`TenetNative`.

Supported production scope:

- 2D classical Ising boundary VUMPS, one-site pattern `[1;;]`.
- Real `Float64` tensors on CPU and real `CuArray{Float64}` tensors on CUDA.
- Native dense Arnoldi fixed-point solves by default.
- Nearest-neighbor quantum TFIsing one-site Hamiltonian VUMPS, with exact
  thermodynamic reference helpers for validation.

Unsupported by design: larger unit cells, symmetry sectors, complex production
tensors, ROCArray, and the old TFIM row-transfer mapping.

## Local Tests

```sh
cd /path/to/TeneT.c
julia --project=FastTeneT --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Run the actual small VUMPS boundary smoke test:

```sh
FASTTENET_RUN_VUMPS_SMOKE=1 julia --project=FastTeneT --startup-file=no -e 'using Pkg; Pkg.test()'
```

Run the Onsager alignment test:

```sh
FASTTENET_RUN_ALIGNMENT=1 julia --project=FastTeneT --startup-file=no -e 'using Pkg; Pkg.test()'
```

## Examples

```julia
using FastTeneT

r = run_boundary(0.5; chi=4, maxiter=10, maxiter_ad=0)
logz = log_partition_density(r)
energy = energy_density(r)

tfi = run_tfising_vumps(1.0; chi=16)
e0 = tfising_energy_density(tfi)
exact = tfising_ground_state_energy_density_exact(1.0)
err = tfi.abs_energy_error
```

CUDA follows the same array-type style as the 2D Ising path:

```julia
using CUDA
using FastTeneT

CUDA.allowscalar(false)
r = run_boundary(critical_beta(); chi=4, maxiter=10, maxiter_ad=0, arraytype=CuArray)
tfi = run_tfising_vumps(1.0; chi=8, maxiter=25, arraytype=CuArray)
```

The 2D Ising boundary path uses the native Arnoldi fixed-point kernels owned by
`TenetNative/src/native`, with FastTeneT calling through the `TenetNative`
Julia API.
The TFIsing path uses a local dense real Hamiltonian VUMPS loop with FastTeneT's
native dense Arnoldi basis and projected solves. It does not call `MPSKit`,
`NNTF`, or KrylovKit.

## Standalone Native Driver

The CPU 2D Ising native kernel can also be run without starting a Julia process.
This builds the native dylib/shared library plus a small C++ executable that
constructs the one-site Ising tensor and calls the C ABI directly:

```sh
cd /path/to/TeneT.c
make -C TenetNative/src/native native-ising-cpu-driver \
  PREFIX=/path/to/TeneT.c/TenetNative/deps

TenetNative/deps/tenet_native_ising_cpu_driver \
  --chi 8 --maxiter 1 --miniter 1 --krylovdim 30 --seed 20260625 \
  --init native-canonical --init-relax 1
```

Use `--dump-state <path>` to write the C++ initial/final tensors for parity
checks against the Julia reference path. `--init-relax n` runs `n` native VUMPS
iterations before the timed repetitions; its Arnoldi tolerance defaults to
`1e-8` and can be overridden with `--init-relax-arnoldi-tol`.

`--krylovdim` (legacy alias: `--max-k`) is the Arnoldi basis dimension and
matches KrylovKit's `krylovdim` parameter. The native dominant Arnoldi kernels
use restarted thick Arnoldi with `TENET_NATIVE_ARNOLDI_RESTARTS=100` by default,
matching KrylovKit's default outer Arnoldi `maxiter`; standalone drivers also
accept `--arnoldi-restarts n` to set that environment variable explicitly.

## Native Benchmark

Use the release benchmark entrypoints under `benchmarks/tenet/` and the JobFiles
under `benchmarks/jobfiles/`. They record warmed single VUMPS-step timings for
TeneT.c and the pinned TeneT.jl iPEPS-unified baseline, with CPU results on
Oblix and GPU results on Snellius H100.
