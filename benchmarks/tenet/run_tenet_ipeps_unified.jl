using CUDA
using Dates
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
chis = env_ints("TENET_BENCH_CHIS", "32,64,96,128,160,192,224,256")
beta = env_float("TENET_BENCH_BETA", log1p(sqrt(2.0)) / 2)
tol = env_float("TENET_BENCH_TOL", 1e-10)
maxiter = env_int("TENET_BENCH_MAXITER", 20)
miniter = env_int("TENET_BENCH_MINITER", 1)
warmup_steps = env_int("TENET_BENCH_WARMUP", 3)
repeats = env_int("TENET_BENCH_REPEATS", 11)
seed = env_int("TENET_BENCH_SEED", 42)
power_iter = env_int("TENET_IPEPS_POWER_ITER", 100)
forloop_iter = env_int("TENET_IPEPS_FORLOOP_ITER", 5)
method = lowercase(env_value("TENET_IPEPS_METHOD", "krylovkit"))
mode = lowercase(env_value("TENET_IPEPS_MODE", "general"))
commit = env_value("TENET_IPEPS_COMMIT", "unknown")

method == "krylovkit" || error("TENET_IPEPS_METHOD must be krylovkit for release benchmarks")
ifsimple_eig = false

for chi in chis
    Random.seed!(seed)
    backend == "cuda" && CUDA.seed!(seed)
    GC.gc()

    model = Ising(lattice=Square(), beta=beta)
    M, alg = if mode == "general"
        M = MPO(model, General; atype=arraytype)
        alg = VUMPS{General}(;
            verbosity=0,
            maxiter=maxiter,
            miniter=miniter,
            maxiter_ad=0,
            miniter_ad=0,
            tol=tol,
            ifsimple_eig=ifsimple_eig,
            power_iter=power_iter,
            ifupdown=false,
            forloop_iter=forloop_iter,
        )
        M, alg
    elseif mode == "c4v"
        M = MPO(model, C4v; atype=arraytype)
        alg = VUMPS{C4v}(;
            verbosity=0,
            maxiter=maxiter,
            miniter=miniter,
            maxiter_ad=0,
            miniter_ad=0,
            tol=tol,
            ifsimple_eig=ifsimple_eig,
            power_iter=power_iter,
            ifupdown=false,
            forloop_iter=forloop_iter,
        )
        M, alg
    else
        error("TENET_IPEPS_MODE must be general or c4v")
    end
    last_eltype = string(eltype(M.data[1]))

    sync_backend(backend)
    setup_start = time_ns()
    rt = init_env(M, chi, alg)
    sync_backend(backend)
    setup_seconds = (time_ns() - setup_start) / 1.0e9

    for _ in 1:warmup_steps
        rt, _ = TeneT.vumps_step(rt, M, alg)
        sync_backend(backend)
    end

    step_seconds = Float64[]
    for _ in 1:repeats
        GC.gc()
        sync_backend(backend)
        t0 = time_ns()
        rt, _ = TeneT.vumps_step(rt, M, alg)
        sync_backend(backend)
        push!(step_seconds, (time_ns() - t0) / 1.0e9)
    end

    @printf(
        "TENET_IPEPS_VUMPS_STEP branch=iPEPS-unified commit=%s method=%s mode=%s backend=%s device=%s eltype=%s chi=%d beta=%.17g step=vumps_step warmed=true warmup_steps=%d repeats=%d setup_seconds=%.9f median_step_seconds=%.9f p25_step_seconds=%.9f p75_step_seconds=%.9f tol=%.3e maxiter=%d miniter=%d power_iter=%d forloop_iter=%d seed=%d timestamp=%s\n",
        commit,
        method,
        mode,
        backend,
        device,
        last_eltype,
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
        power_iter,
        forloop_iter,
        seed,
        string(now(UTC)),
    )
end
