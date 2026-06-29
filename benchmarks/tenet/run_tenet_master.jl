using Dates
using CUDA
using Printf
using Random
using TeneT

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

include(joinpath(dirname(dirname(pathof(TeneT))), "example", "exampletensors.jl"))

backend, arraytype = backend_arraytype()
default_chis = backend == "cuda" ? "64,128,256" : "32,64,128"
chis = env_ints("TENET_BENCH_CHIS", default_chis)
beta = env_float("TENET_BENCH_BETA", 0.44068679350977147)
tol = env_float("TENET_BENCH_TOL", 1e-10)
maxiter = env_int("TENET_BENCH_MAXITER", 20)
miniter = env_int("TENET_BENCH_MINITER", 1)
warmup = env_int("TENET_BENCH_WARMUP", 2)
repeats = env_int("TENET_BENCH_REPEATS", 7)
seed = env_int("TENET_BENCH_SEED", 42)
commit = env_value("TENET_MASTER_COMMIT", "unknown")
patch = env_value("TENET_MASTER_PATCH", "none")

for chi in chis
    model = Ising(1, 1, beta)
    M = arraytype(model_tensor(model, Val(:bulk)))
    sync_backend(backend)

    init_seconds = Float64[]
    iter_seconds = Float64[]
    total_seconds = Float64[]
    last_err = NaN
    for rep in 1:(warmup + repeats)
        Random.seed!(seed)
        GC.gc()
        t0 = time_ns()
        rt = TeneT.SquareVUMPSRuntime(M, Val(:random), chi; verbose=false)
        sync_backend(backend)
        t1 = time_ns()
        _, err = TeneT.vumps(rt; tol, maxiter, miniter, verbose=false, show_every=Inf)
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
        "TENET_MASTER_2DISING backend=%s branch=master commit=%s patch=%s eltype=%s chi=%d beta=%.17g tol=%.3e maxiter=%d miniter=%d warmup=%d repeats=%d median_init_seconds=%.9f median_iter_seconds=%.9f median_total_seconds=%.9f p25_total_seconds=%.9f p75_total_seconds=%.9f err=%.9e timestamp=%s\n",
        backend,
        commit,
        patch,
        string(eltype(M)),
        chi,
        beta,
        tol,
        maxiter,
        miniter,
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
