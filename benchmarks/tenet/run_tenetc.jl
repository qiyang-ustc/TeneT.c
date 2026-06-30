using Dates
using CUDA
using FastTeneT
using Printf
using Random
using TeneTC

function env_value(name::String, default::String)
    value = get(ENV, name, "")
    return isempty(value) ? default : value
end

env_int(name::String, default::Int) = parse(Int, env_value(name, string(default)))
env_float(name::String, default::Float64) = parse(Float64, env_value(name, string(default)))
env_ints(name::String, default::String) =
    [parse(Int, strip(x)) for x in split(env_value(name, default), ",") if !isempty(strip(x))]

function median_value(xs)
    ys = sort(collect(Float64, xs))
    return ys[cld(length(ys), 2)]
end

function percentile_value(xs, q::Float64)
    ys = sort(collect(Float64, xs))
    idx = clamp(round(Int, 1 + (length(ys) - 1) * q), 1, length(ys))
    return ys[idx]
end

function backend_arraytype()
    backend = lowercase(env_value("TENET_BENCH_BACKEND", "cpu"))
    if backend == "cuda"
        CUDA.allowscalar(false)
        return backend, CUDA.CuArray
    elseif backend == "cpu"
        return backend, Array
    end
    error("TENET_BENCH_BACKEND must be cpu or cuda")
end

function sync_backend(backend)
    backend == "cuda" && CUDA.synchronize()
    return nothing
end

function backend_device(backend)
    backend == "cuda" || return "cpu"
    return replace(string(CUDA.name(CUDA.device())), ' ' => '_')
end

backend, arraytype = backend_arraytype()
device = backend_device(backend)
default_chis = "32,64,96,128,160,192,224,256"
chis = env_ints("TENET_BENCH_CHIS", default_chis)
beta = env_float("TENET_BENCH_BETA", critical_beta())
tol = env_float("TENET_BENCH_TOL", 1e-10)
maxiter = env_int("TENET_BENCH_MAXITER", 20)
miniter = env_int("TENET_BENCH_MINITER", 1)
warmup_steps = env_int("TENET_BENCH_WARMUP", 3)
repeats = env_int("TENET_BENCH_REPEATS", 11)
seed = env_int("TENET_BENCH_SEED", 42)
krylovdim = env_int("TENET_BENCH_KRYLOVDIM", 30)
arnoldi_tol = env_float("TENET_BENCH_ARNOLDI_TOL", 1e-12)
residual_tol = env_float("TENET_BENCH_RESIDUAL_TOL", backend == "cuda" ? 1e-10 : 1e-12)

if lowercase(env_value("TENET_BENCH_BUILD_NATIVE", "true")) in ("1", "true", "yes")
    build_native_arnoldi(target=(backend == "cuda" ? :cuda : :cpu))
end

for chi in chis
    Random.seed!(seed)
    GC.gc()
    network = ising_network(beta; arraytype)
    alg = vumps_algorithm(;
        tol,
        maxiter,
        miniter,
        maxiter_ad=0,
        miniter_ad=0,
        verbosity=0,
        ifupdown=false,
        native_arnoldi_krylovdim=krylovdim,
        native_arnoldi_tol=arnoldi_tol,
        native_arnoldi_check_residual=true,
        native_arnoldi_residual_tol=residual_tol,
    )

    setup_start = time_ns()
    rt = VUMPSRuntime(network, chi, alg)
    sync_backend(backend)
    setup_seconds = (time_ns() - setup_start) / 1.0e9

    for _ in 1:warmup_steps
        rt, _ = FastTeneT.vumps_step_Hermitian(rt, network, alg)
        sync_backend(backend)
    end

    step_seconds = Float64[]
    for _ in 1:repeats
        GC.gc()
        sync_backend(backend)
        t0 = time_ns()
        rt, _ = FastTeneT.vumps_step_Hermitian(rt, network, alg)
        sync_backend(backend)
        push!(step_seconds, (time_ns() - t0) / 1.0e9)
    end

    @printf(
        "TENETC_VUMPS_STEP backend=%s device=%s commit=current eltype=Float64 chi=%d beta=%.17g step=vumps_step_Hermitian warmed=true warmup_steps=%d repeats=%d setup_seconds=%.9f median_step_seconds=%.9f p25_step_seconds=%.9f p75_step_seconds=%.9f tol=%.3e maxiter=%d miniter=%d krylovdim=%d arnoldi_tol=%.3e residual_tol=%.3e seed=%d timestamp=%s\n",
        backend,
        device,
        chi,
        beta,
        warmup_steps,
        repeats,
        setup_seconds,
        median_value(step_seconds),
        percentile_value(step_seconds, 0.25),
        percentile_value(step_seconds, 0.75),
        tol,
        maxiter,
        miniter,
        krylovdim,
        arnoldi_tol,
        residual_tol,
        seed,
        string(now(UTC)),
    )
end
