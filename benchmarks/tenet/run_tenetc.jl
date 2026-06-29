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

backend, arraytype = backend_arraytype()
default_chis = backend == "cuda" ? "64,128,256" : "32,64,128"
chis = env_ints("TENET_BENCH_CHIS", default_chis)
beta = env_float("TENET_BENCH_BETA", critical_beta())
tol = env_float("TENET_BENCH_TOL", 1e-10)
maxiter = env_int("TENET_BENCH_MAXITER", 20)
miniter = env_int("TENET_BENCH_MINITER", 1)
warmup = env_int("TENET_BENCH_WARMUP", 2)
repeats = env_int("TENET_BENCH_REPEATS", 7)
seed = env_int("TENET_BENCH_SEED", 42)
krylovdim = env_int("TENET_BENCH_KRYLOVDIM", 30)
arnoldi_tol = env_float("TENET_BENCH_ARNOLDI_TOL", 1e-12)
residual_tol = env_float("TENET_BENCH_RESIDUAL_TOL", backend == "cuda" ? 1e-10 : 1e-12)

build_native_arnoldi(target=(backend == "cuda" ? :cuda : :cpu))

for chi in chis
    init_seconds = Float64[]
    iter_seconds = Float64[]
    total_seconds = Float64[]
    last_err = NaN
    for rep in 1:(warmup + repeats)
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
        t0 = time_ns()
        rt = VUMPSRuntime(network, chi, alg)
        sync_backend(backend)
        t1 = time_ns()
        _, err = FastTeneT.leading_boundary(rt, network, alg)
        sync_backend(backend)
        t2 = time_ns()
        if rep > warmup
            push!(init_seconds, (t1 - t0) / 1.0e9)
            push!(iter_seconds, (t2 - t1) / 1.0e9)
            push!(total_seconds, (t2 - t0) / 1.0e9)
        end
        last_err = Float64(real(err))
    end

    @printf(
        "TENETC_2DISING backend=%s commit=current eltype=Float64 chi=%d beta=%.17g tol=%.3e maxiter=%d miniter=%d krylovdim=%d arnoldi_tol=%.3e residual_tol=%.3e warmup=%d repeats=%d median_init_seconds=%.9f median_iter_seconds=%.9f median_total_seconds=%.9f p25_total_seconds=%.9f p75_total_seconds=%.9f err=%.9e timestamp=%s\n",
        backend,
        chi,
        beta,
        tol,
        maxiter,
        miniter,
        krylovdim,
        arnoldi_tol,
        residual_tol,
        warmup,
        repeats,
        median_value(init_seconds),
        median_value(iter_seconds),
        median_value(total_seconds),
        sort(total_seconds)[max(1, fld(length(total_seconds), 4))],
        sort(total_seconds)[min(length(total_seconds), cld(3 * length(total_seconds), 4))],
        last_err,
        string(now(UTC)),
    )
end
