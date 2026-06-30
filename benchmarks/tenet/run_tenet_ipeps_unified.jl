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

function backend_arraytype()
    backend = lowercase(env_value("TENET_BENCH_BACKEND", "cuda"))
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
chis = env_ints("TENET_BENCH_CHIS", backend == "cuda" ? "32,48,64,96,128" : "16,32,64")
beta = env_float("TENET_BENCH_BETA", log1p(sqrt(2.0)) / 2)
tol = env_float("TENET_BENCH_TOL", 1e-10)
maxiter = env_int("TENET_BENCH_MAXITER", 20)
miniter = env_int("TENET_BENCH_MINITER", 1)
warmup = env_int("TENET_BENCH_WARMUP", 2)
repeats = env_int("TENET_BENCH_REPEATS", 7)
seed = env_int("TENET_BENCH_SEED", 42)
power_iter = env_int("TENET_IPEPS_POWER_ITER", 100)
forloop_iter = env_int("TENET_IPEPS_FORLOOP_ITER", 5)
method = lowercase(env_value("TENET_IPEPS_METHOD", "simple_eig"))
mode = lowercase(env_value("TENET_IPEPS_MODE", "general"))
commit = env_value("TENET_IPEPS_COMMIT", "unknown")
onsager_npts = env_int("TENET_BENCH_ONSAGER_NPTS", 2000)

ifsimple_eig =
    method == "simple_eig" ? true :
    method == "krylovkit" ? false :
    error("TENET_IPEPS_METHOD must be simple_eig or krylovkit")

model = Ising(lattice=Square(), beta=beta)
f_exact = exact_free_energy(model; npts=onsager_npts)

for chi in chis
    init_seconds = Float64[]
    iter_seconds = Float64[]
    total_seconds = Float64[]
    last_err = NaN
    last_free_energy_abs_error = NaN
    last_eltype = "unknown"

    for rep in 1:(warmup + repeats)
        Random.seed!(seed)
        backend == "cuda" && CUDA.seed!(seed)
        GC.gc()

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
        t0 = time_ns()
        rt = init_env(M, chi, alg)
        sync_backend(backend)
        t1 = time_ns()
        rt, err = leading_boundary(rt, M, alg)
        sync_backend(backend)
        t2 = time_ns()

        if rep > warmup
            push!(init_seconds, (t1 - t0) / 1.0e9)
            push!(iter_seconds, (t2 - t1) / 1.0e9)
            push!(total_seconds, (t2 - t0) / 1.0e9)
        end
        last_err = Float64(real(err))
        if rep == warmup + repeats
            fr = free_energy(rt, M, alg, model)
            sync_backend(backend)
            last_free_energy_abs_error = abs(fr.f - f_exact)
        end
    end

    @printf(
        "TENET_IPEPS_2DISING branch=iPEPS-unified commit=%s method=%s mode=%s backend=%s device=%s eltype=%s chi=%d beta=%.17g tol=%.3e maxiter=%d miniter=%d power_iter=%d forloop_iter=%d warmup=%d repeats=%d median_init_seconds=%.9f median_iter_seconds=%.9f median_total_seconds=%.9f p25_total_seconds=%.9f p75_total_seconds=%.9f err=%.9e free_energy_abs_error=%.9e onsager_npts=%d timestamp=%s\n",
        commit,
        method,
        mode,
        backend,
        device,
        last_eltype,
        chi,
        beta,
        tol,
        maxiter,
        miniter,
        power_iter,
        forloop_iter,
        warmup,
        repeats,
        median_value(init_seconds),
        median_value(iter_seconds),
        median_value(total_seconds),
        sort(total_seconds)[max(1, fld(length(total_seconds), 4))],
        sort(total_seconds)[min(length(total_seconds), cld(3 * length(total_seconds), 4))],
        last_err,
        last_free_energy_abs_error,
        onsager_npts,
        string(now(UTC)),
    )
end
