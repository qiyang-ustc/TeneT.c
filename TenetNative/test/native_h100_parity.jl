#!/usr/bin/env julia

const USAGE = """
Usage:
  julia --project=<FastTeneT env> TenetNative/test/native_h100_parity.jl [options]

Builds or uses the TenetNative CUDA C ABI, runs Float64 CuArray parity and
timing cases, and writes native_h100_parity.csv/native_h100_parity.md under an
output directory outside the repository.

Options:
  --repo PATH              Repository root. Defaults to the parent of TenetNative.
  --outdir PATH            Output directory for CSV/Markdown artifacts.
  --prefix PATH            Native build prefix. Defaults to OUTDIR/tenetnative_deps.
  --cuda-lib PATH          Existing libtenet_native_arnoldi_cuda library.
  --cpu-lib PATH           Existing libtenet_native_arnoldi CPU library.
  --build                  Build CPU and CUDA native libraries into --prefix.
  --no-build               Require --cuda-lib or TENET_NATIVE_ARNOLDI_CUDA_LIB.
  --no-cpu-comparison      Compare CUDA results to dense Julia residual checks only.
  --julia PATH             Julia executable for the CPU native Makefile.
  --nvcc PATH              nvcc executable for CUDA native build.
  --cuda-arch ARCH         nvcc architecture, e.g. sm_90 for H100.
  --seed INT               Deterministic input seed.
  --chi INT                Bond dimension for dense ABI fixtures.
  --phys INT               Physical dimension for dense ABI fixtures.
  --max-k INT              Arnoldi Krylov dimension; 0 means full dimension.
  --breakdown-tol FLOAT    Native Arnoldi breakdown tolerance.
  --atol FLOAT             Absolute parity tolerance.
  --rtol FLOAT             Relative parity tolerance.
  --warmup INT             Warmup calls per parity case.
  --repeats INT            Timed repeats per parity case.
  --skip-fasttenet-smoke   Skip the FastTeneT CuArray H100 smoke.
  --smoke-chi INT          FastTeneT smoke bond dimension.
  --smoke-repeats INT      Timed repeats for the FastTeneT smoke.
  --allow-repo-outdir      Permit generated outputs under the repo.
  --help                   Print this help.
"""

if any(arg -> arg == "--help" || arg == "-h", ARGS)
    print(USAGE)
    exit(0)
end

using Dates
using Libdl
using LinearAlgebra
using Printf
using Random
using Statistics

const CSV_COLUMNS = (
    "case",
    "operation",
    "status",
    "comparison",
    "max_abs_diff",
    "tolerance",
    "warmup",
    "repeats",
    "native_cuda_median_s",
    "comparison_median_s",
    "ratio",
    "device_name",
    "cuda_version",
    "cuda_lib_path",
    "cpu_lib_path",
    "detail",
)

mutable struct Options
    repo::String
    outdir::String
    prefix::String
    cuda_lib::Union{Nothing,String}
    cpu_lib::Union{Nothing,String}
    build::Bool
    cpu_comparison::Bool
    julia::String
    nvcc::String
    cuda_arch::String
    seed::Int
    chi::Int
    phys::Int
    max_k::Int
    breakdown_tol::Float64
    atol::Float64
    rtol::Float64
    warmup::Int
    repeats::Int
    fasttenet_smoke::Bool
    smoke_chi::Int
    smoke_repeats::Int
    allow_repo_outdir::Bool
end

struct BasisResult
    V::Matrix{Float64}
    H::Matrix{Float64}
    m::Int
    beta::Float64
    final_resnorm::Float64
end

struct DominantResult
    y::Array{Float64}
    lambda::Float64
end

timestamp_utc() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyymmddTHHMMSS") * "Z"
default_repo() = normpath(joinpath(@__DIR__, "..", ".."))
default_outdir() = joinpath(tempdir(), "tenet_native_h100_parity", "run_" * timestamp_utc())
fmt_float(x::Real) = @sprintf("%.9g", Float64(x))
fmt_diff(x::Real) = @sprintf("%.3e", Float64(x))
fmt_any(x) = x === nothing ? "" : string(x)

function parse_bool_build(argv)
    saw_build = any(==("--build"), argv)
    saw_no_build = any(==("--no-build"), argv)
    saw_build && saw_no_build && error("use only one of --build or --no-build")
    return saw_build ? true : saw_no_build ? false : nothing
end

function value_after(argv, i, name)
    i < length(argv) || error("$name requires a value")
    return argv[i + 1]
end

function env_path(name)
    value = get(ENV, name, "")
    return isempty(value) ? nothing : value
end

function parse_args(argv)
    repo = default_repo()
    outdir = get(ENV, "TENET_NATIVE_H100_PARITY_OUTDIR", default_outdir())
    prefix = nothing
    cuda_lib = env_path("TENET_NATIVE_ARNOLDI_CUDA_LIB")
    cpu_lib = env_path("TENET_NATIVE_ARNOLDI_LIB")
    build_flag = parse_bool_build(argv)
    cpu_comparison = true
    julia = get(ENV, "JULIA", joinpath(Sys.BINDIR, Sys.iswindows() ? "julia.exe" : "julia"))
    nvcc = get(ENV, "NVCC", "nvcc")
    cuda_arch = get(ENV, "TENET_NATIVE_CUDA_ARCH", "sm_90")
    seed = parse(Int, get(ENV, "TENET_NATIVE_H100_PARITY_SEED", "20260626"))
    chi = parse(Int, get(ENV, "TENET_NATIVE_H100_PARITY_CHI", "4"))
    phys = parse(Int, get(ENV, "TENET_NATIVE_H100_PARITY_PHYS", "2"))
    max_k = parse(Int, get(ENV, "TENET_NATIVE_H100_PARITY_MAX_K", "0"))
    breakdown_tol = parse(Float64, get(ENV, "TENET_NATIVE_H100_PARITY_BREAKDOWN_TOL", "1e-12"))
    atol = parse(Float64, get(ENV, "TENET_NATIVE_H100_PARITY_ATOL", "1e-10"))
    rtol = parse(Float64, get(ENV, "TENET_NATIVE_H100_PARITY_RTOL", "1e-10"))
    warmup = parse(Int, get(ENV, "TENET_NATIVE_H100_PARITY_WARMUP", "2"))
    repeats = parse(Int, get(ENV, "TENET_NATIVE_H100_PARITY_REPEATS", "5"))
    fasttenet_smoke = lowercase(get(ENV, "TENET_NATIVE_H100_FASTTENET_SMOKE", "1")) in ("1", "true", "yes", "on")
    smoke_chi = parse(Int, get(ENV, "TENET_NATIVE_H100_SMOKE_CHI", "4"))
    smoke_repeats = parse(Int, get(ENV, "TENET_NATIVE_H100_SMOKE_REPEATS", "1"))
    allow_repo_outdir = false

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--build" || arg == "--no-build"
        elseif startswith(arg, "--repo=")
            repo = split(arg, "=", limit=2)[2]
        elseif arg == "--repo"
            repo = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--outdir=")
            outdir = split(arg, "=", limit=2)[2]
        elseif arg == "--outdir"
            outdir = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--prefix=")
            prefix = split(arg, "=", limit=2)[2]
        elseif arg == "--prefix"
            prefix = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--cuda-lib=")
            cuda_lib = split(arg, "=", limit=2)[2]
        elseif arg == "--cuda-lib"
            cuda_lib = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--cpu-lib=")
            cpu_lib = split(arg, "=", limit=2)[2]
        elseif arg == "--cpu-lib"
            cpu_lib = value_after(argv, i, arg)
            i += 1
        elseif arg == "--no-cpu-comparison"
            cpu_comparison = false
        elseif startswith(arg, "--julia=")
            julia = split(arg, "=", limit=2)[2]
        elseif arg == "--julia"
            julia = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--nvcc=")
            nvcc = split(arg, "=", limit=2)[2]
        elseif arg == "--nvcc"
            nvcc = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--cuda-arch=")
            cuda_arch = split(arg, "=", limit=2)[2]
        elseif arg == "--cuda-arch"
            cuda_arch = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--seed=")
            seed = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--seed"
            seed = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--chi=")
            chi = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--chi"
            chi = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--phys=")
            phys = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--phys"
            phys = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--max-k=")
            max_k = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--max-k"
            max_k = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--breakdown-tol=")
            breakdown_tol = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--breakdown-tol"
            breakdown_tol = parse(Float64, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--atol=")
            atol = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--atol"
            atol = parse(Float64, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--rtol=")
            rtol = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--rtol"
            rtol = parse(Float64, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--warmup=")
            warmup = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--warmup"
            warmup = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--repeats=")
            repeats = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--repeats"
            repeats = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif arg == "--skip-fasttenet-smoke"
            fasttenet_smoke = false
        elseif startswith(arg, "--smoke-chi=")
            smoke_chi = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--smoke-chi"
            smoke_chi = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif startswith(arg, "--smoke-repeats=")
            smoke_repeats = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--smoke-repeats"
            smoke_repeats = parse(Int, value_after(argv, i, arg))
            i += 1
        elseif arg == "--allow-repo-outdir"
            allow_repo_outdir = true
        else
            error("unknown argument: $arg\n\n" * USAGE)
        end
        i += 1
    end

    repo = abspath(normpath(repo))
    outdir = abspath(normpath(outdir))
    prefix = prefix === nothing ? joinpath(outdir, "tenetnative_deps") : abspath(normpath(prefix))
    cuda_lib = cuda_lib === nothing ? nothing : abspath(normpath(cuda_lib))
    cpu_lib = cpu_lib === nothing ? nothing : abspath(normpath(cpu_lib))
    build = build_flag === nothing ? cuda_lib === nothing : build_flag

    chi > 0 || error("--chi must be positive")
    phys > 0 || error("--phys must be positive")
    max_k >= 0 || error("--max-k must be nonnegative")
    breakdown_tol >= 0 || error("--breakdown-tol must be nonnegative")
    atol >= 0 || error("--atol must be nonnegative")
    rtol >= 0 || error("--rtol must be nonnegative")
    warmup >= 0 || error("--warmup must be nonnegative")
    repeats > 0 || error("--repeats must be positive")
    smoke_chi > 0 || error("--smoke-chi must be positive")
    smoke_repeats > 0 || error("--smoke-repeats must be positive")

    return Options(repo, outdir, prefix, cuda_lib, cpu_lib, build, cpu_comparison,
                   julia, nvcc, cuda_arch, seed, chi, phys, max_k,
                   breakdown_tol, atol, rtol, warmup, repeats,
                   fasttenet_smoke, smoke_chi, smoke_repeats, allow_repo_outdir)
end

function is_subpath(path, root)
    rel = relpath(abspath(normpath(path)), abspath(normpath(root)))
    return rel == "." || !(rel == ".." || startswith(rel, ".." * Base.Filesystem.path_separator))
end

function guard_scratch_paths(opts::Options)
    opts.allow_repo_outdir && return nothing
    is_subpath(opts.outdir, opts.repo) &&
        error("outdir $(opts.outdir) is under repo $(opts.repo); use /tmp, /private/tmp, or a job run directory")
    is_subpath(opts.prefix, opts.repo) &&
        error("prefix $(opts.prefix) is under repo $(opts.repo); native build products must stay outside the repo")
    return nothing
end

function csv_escape(x)
    s = string(x)
    needs_quote = occursin(",", s) || occursin("\"", s) || occursin("\n", s)
    s = replace(s, "\"" => "\"\"")
    return needs_quote ? "\"" * s * "\"" : s
end

function md_escape(x)
    return replace(replace(string(x), "|" => "\\|"), "\n" => " ")
end

function row(meta; case_id, operation, status, comparison="", diff=nothing,
             tolerance=nothing, warmup="", repeats="", native_time=nothing,
             comparison_time=nothing, ratio=nothing, detail="")
    return Dict(
        "case" => string(case_id),
        "operation" => string(operation),
        "status" => string(status),
        "comparison" => string(comparison),
        "max_abs_diff" => diff === nothing ? "" : fmt_diff(diff),
        "tolerance" => tolerance === nothing ? "" : fmt_diff(tolerance),
        "warmup" => string(warmup),
        "repeats" => string(repeats),
        "native_cuda_median_s" => native_time === nothing ? "" : fmt_float(native_time),
        "comparison_median_s" => comparison_time === nothing ? "" : fmt_float(comparison_time),
        "ratio" => ratio === nothing ? "" : fmt_float(ratio),
        "device_name" => get(meta, "device_name", ""),
        "cuda_version" => get(meta, "cuda_version", ""),
        "cuda_lib_path" => get(meta, "cuda_lib_path", ""),
        "cpu_lib_path" => get(meta, "cpu_lib_path", ""),
        "detail" => string(detail),
    )
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, join(CSV_COLUMNS, ","))
        for r in rows
            println(io, join((csv_escape(get(r, col, "")) for col in CSV_COLUMNS), ","))
        end
    end
    return path
end

function write_markdown(path, rows, opts::Options, meta)
    nfail = count(r -> r["status"] == "fail", rows)
    nskip = count(r -> r["status"] == "skip", rows)
    open(path, "w") do io
        println(io, "# TenetNative H100 CUDA Parity")
        println(io)
        println(io, "- status: ", nfail == 0 ? "pass" : "fail")
        println(io, "- skipped: ", nskip)
        println(io, "- repo: `", opts.repo, "`")
        println(io, "- outdir: `", opts.outdir, "`")
        println(io, "- prefix: `", opts.prefix, "`")
        println(io, "- cuda_library: `", get(meta, "cuda_lib_path", ""), "`")
        println(io, "- cpu_library: `", get(meta, "cpu_lib_path", ""), "`")
        println(io, "- device: `", get(meta, "device_name", ""), "`")
        println(io, "- cuda_version: `", get(meta, "cuda_version", ""), "`")
        println(io, "- seed: `", opts.seed, "`")
        println(io, "- chi: `", opts.chi, "`")
        println(io, "- phys: `", opts.phys, "`")
        println(io, "- max_k: `", opts.max_k, "`")
        println(io, "- breakdown_tol: `", opts.breakdown_tol, "`")
        println(io, "- atol: `", opts.atol, "`")
        println(io, "- rtol: `", opts.rtol, "`")
        println(io)
        println(io, "Limitations:")
        println(io, "- The harness calls the TenetNative CUDA C ABI directly so it can validate the packaged shared library independently of FastTeneT integration.")
        println(io, "- Native CUDA timings include the Julia ccall, synchronization, and host materialization needed for parity checks.")
        println(io, "- Comparison timings use the TenetNative CPU ABI when available; otherwise numeric checks fall back to dense Julia residual/eigenvalue references.")
        println(io, "- The FastTeneT H100 smoke is skipped when FastTeneT dependencies cannot be loaded.")
        println(io)
        println(io, "| Case | Operation | Status | Comparison | max abs diff | tolerance | native CUDA median s | comparison median s | ratio | Detail |")
        println(io, "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in rows
            println(io, "| ", join((
                md_escape(r["case"]),
                md_escape(r["operation"]),
                md_escape(r["status"]),
                md_escape(r["comparison"]),
                md_escape(r["max_abs_diff"]),
                md_escape(r["tolerance"]),
                md_escape(r["native_cuda_median_s"]),
                md_escape(r["comparison_median_s"]),
                md_escape(r["ratio"]),
                md_escape(r["detail"]),
            ), " | "), " |")
        end
    end
    return path
end

function write_unavailable_artifacts(opts::Options, detail)
    mkpath(opts.outdir)
    meta = Dict(
        "device_name" => "",
        "cuda_version" => "",
        "cuda_lib_path" => fmt_any(opts.cuda_lib),
        "cpu_lib_path" => fmt_any(opts.cpu_lib),
    )
    rows = [row(meta; case_id="cuda_unavailable", operation="environment",
                status="fail", comparison="", detail)]
    csv_path = write_csv(joinpath(opts.outdir, "native_h100_parity.csv"), rows)
    md_path = write_markdown(joinpath(opts.outdir, "native_h100_parity.md"), rows, opts, meta)
    println("TENET_NATIVE_H100_PARITY artifacts csv=$csv_path markdown=$md_path")
    println("TENET_NATIVE_H100_PARITY_DONE status=fail rows=1 failures=1 outdir=$(opts.outdir)")
    return nothing
end

const OPTS = parse_args(ARGS)
guard_scratch_paths(OPTS)
mkpath(OPTS.outdir)

try
    @eval using CUDA
catch err
    write_unavailable_artifacts(OPTS, "CUDA.jl could not be loaded: " * sprint(showerror, err))
    exit(77)
end

cuda_functional = try
    CUDA.functional()
catch err
    write_unavailable_artifacts(OPTS, "CUDA.functional() threw: " * sprint(showerror, err))
    exit(77)
end

if !cuda_functional
    write_unavailable_artifacts(OPTS, "CUDA.functional() returned false")
    exit(77)
end

function add_load_path!(repo)
    for env in (joinpath(repo, "FastTeneT"), joinpath(repo, "TenetNative"))
        isfile(joinpath(env, "Project.toml")) || continue
        env_abs = abspath(normpath(env))
        env_abs in LOAD_PATH || pushfirst!(LOAD_PATH, env_abs)
    end
    return nothing
end

function build_native_cpu(opts::Options)
    native_dir = joinpath(opts.repo, "TenetNative", "src", "native")
    isdir(native_dir) || error("native source directory not found: $native_dir")
    mkpath(opts.prefix)
    run(`make -C $native_dir native-arnoldi-cpu PREFIX=$(opts.prefix) JULIA=$(opts.julia)`)
    libpath = joinpath(opts.prefix, "libtenet_native_arnoldi." * Libdl.dlext)
    isfile(libpath) || error("native CPU build did not produce $libpath")
    return libpath
end

function build_native_cuda(opts::Options)
    native_dir = joinpath(opts.repo, "TenetNative", "src", "native")
    cu_source = joinpath(native_dir, "tenet_native_arnoldi_cuda.cu")
    isfile(cu_source) || error("TenetNative CUDA source not found: $cu_source")
    mkpath(opts.prefix)
    libpath = joinpath(opts.prefix, "libtenet_native_arnoldi_cuda." * Libdl.dlext)
    blas_lib = Libdl.dlpath(Base.libblas_name)
    julia_libdir = dirname(blas_lib)
    cuda_libdir = dirname(CUDA.CUBLAS.libcublas)
    args = String[
        opts.nvcc,
        "-O3",
        "-std=c++17",
        "-Xcompiler=-fPIC",
        "-arch=$(opts.cuda_arch)",
        "-DTENET_NATIVE_USE_BLAS64",
        "--shared",
        "--cudart=shared",
        "-o", libpath,
        cu_source,
        "-L$(cuda_libdir)",
        "-lcublas",
        "-Xlinker", blas_lib,
        "-Xlinker", "-rpath",
        "-Xlinker", julia_libdir,
        "-Xlinker", "-rpath",
        "-Xlinker", cuda_libdir,
        "-Xlinker", "-rpath",
        "-Xlinker", opts.prefix,
    ]
    run(Cmd(args))
    isfile(libpath) || error("native CUDA build did not produce $libpath")
    return libpath
end

function resolve_libs(opts::Options)
    cuda_lib = opts.cuda_lib
    cpu_lib = opts.cpu_lib
    if opts.build
        cpu_lib = opts.cpu_comparison ? build_native_cpu(opts) : cpu_lib
        cuda_lib = build_native_cuda(opts)
    else
        cuda_lib === nothing && error("no CUDA native library configured; pass --build, --cuda-lib, or TENET_NATIVE_ARNOLDI_CUDA_LIB")
    end
    isfile(cuda_lib) || error("CUDA native library not found: $cuda_lib")
    if opts.cpu_comparison
        if cpu_lib !== nothing && !isfile(cpu_lib)
            error("CPU native library not found: $cpu_lib")
        end
    else
        cpu_lib = nothing
    end
    ENV["TENET_NATIVE_ARNOLDI_CUDA_LIB"] = cuda_lib
    cpu_lib !== nothing && (ENV["TENET_NATIVE_ARNOLDI_LIB"] = cpu_lib)
    return cuda_lib, cpu_lib
end

function safe_cuda_version()
    runtime = try
        string(CUDA.runtime_version())
    catch
        "unknown"
    end
    driver = try
        string(CUDA.driver_version())
    catch
        "unknown"
    end
    return "runtime=$runtime driver=$driver"
end

function inputs(seed::Integer, chi::Integer, phys::Integer)
    rng = MersenneTwister(seed)
    scale3 = inv(sqrt(Float64(chi * phys)))
    Aup = zeros(Float64, chi, phys, chi)
    for b in 1:phys
        S = scale3 .* randn(rng, chi, chi)
        Aup[:, b, :] .= 0.5 .* (S .+ S')
    end
    Adn = copy(Aup)
    W = copy(Aup)
    x0 = randn(rng, chi, chi)
    rho0 = randn(rng, chi, chi)
    rho = rho0 * rho0'
    rho ./= tr(rho)
    M = zeros(Float64, phys, phys, phys, phys)
    for d in 1:phys, g in 1:phys
        M[d, g, d, g] = 1.0
    end
    x3 = randn(rng, chi, phys, chi)
    return (; Aup, Adn, W, x0, rho, M, x3)
end

full_k(opts::Options, len::Integer) = opts.max_k == 0 ? Int(len) : min(opts.max_k, Int(len))

function two_layer_apply(Aup, Adn, X; transpose::Bool=false)
    chi, phys, _ = size(Aup)
    Y = zeros(Float64, chi, chi)
    for b in 1:phys
        A = Aup[:, b, :]
        B = Adn[:, b, :]
        if transpose
            Y .+= A * X * Base.transpose(B)
        else
            Y .+= Base.transpose(A) * X * B
        end
    end
    return Y
end

function projected_two_layer_apply(Aup, Adn, rho, X; transpose::Bool=false)
    Y = X .- two_layer_apply(Aup, Adn, X; transpose)
    projection = sum(rho .* X)
    for i in 1:size(X, 1)
        Y[i, i] += projection
    end
    return Y
end

rho_dot(rho, X) = sum(rho .* X)

function project_q(rho, X)
    return X .- tr(X) .* rho
end

function project_q_adj(rho, X)
    chi = size(X, 1)
    Y = copy(X)
    dot_rx = rho_dot(rho, X)
    for i in 1:chi
        Y[i, i] -= dot_rx
    end
    return Y
end

function qprojected_two_layer_apply(Aup, Adn, rho, X; transpose::Bool=false)
    if transpose
        return project_q(rho, two_layer_apply(Aup, Adn, project_q(rho, X);
                                             transpose=true))
    end
    return project_q_adj(rho, two_layer_apply(Aup, Adn, project_q_adj(rho, X);
                                             transpose=false))
end

function three_layer_apply(Aup, Adn, M, X; transpose::Bool=false)
    chi, phys, _ = size(Aup)
    Y = zeros(Float64, chi, phys, chi)
    if transpose
        for d in 1:phys, b in 1:phys
            accum = zeros(Float64, chi, chi)
            for e in 1:phys, g in 1:phys
                alpha = M[d, g, e, b]
                alpha == 0.0 && continue
                accum .+= alpha .* (X[:, e, :] * Base.transpose(Adn[:, g, :]))
            end
            Y[:, d, :] .+= Aup[:, b, :] * accum
        end
    else
        for e in 1:phys, b in 1:phys
            accum = zeros(Float64, chi, chi)
            for d in 1:phys, g in 1:phys
                alpha = M[d, g, e, b]
                alpha == 0.0 && continue
                accum .+= alpha .* (X[:, d, :] * Adn[:, g, :])
            end
            Y[:, e, :] .+= Base.transpose(Aup[:, b, :]) * accum
        end
    end
    return Y
end

function operator_matrix(apply, dims::Tuple)
    len = prod(dims)
    A = Matrix{Float64}(undef, len, len)
    basis = zeros(Float64, len)
    for j in 1:len
        basis[j] = 1.0
        A[:, j] = vec(apply(reshape(basis, dims)))
        basis[j] = 0.0
    end
    return A
end

function ritz_order(vals)
    order = collect(eachindex(vals))
    sort!(order; lt=(a, b) -> begin
        ma = abs(vals[a])
        mb = abs(vals[b])
        abs(ma - mb) > 1e-10 * max(1.0, ma, mb) && return ma > mb
        return real(vals[a]) > real(vals[b])
    end)
    return order
end

function target_eigenvalue(apply, dims::Tuple)
    F = eigen(operator_matrix(apply, dims))
    return F.values[first(ritz_order(F.values))]
end

function arnoldi_relation_residual(result::BasisResult, apply, dims::Tuple)
    m = result.m
    V = result.V
    H = result.H
    AV = Matrix{Float64}(undef, size(V, 1), m)
    for j in 1:m
        AV[:, j] = vec(apply(reshape(V[:, j], dims)))
    end
    VH = V[:, 1:(m + 1)] * H[1:(m + 1), 1:m]
    return norm(AV .- VH, Inf) / max(norm(AV, Inf), norm(VH, Inf), 1.0)
end

function arnoldi_orthogonality_residual(result::BasisResult)
    m = result.m
    V = result.V[:, 1:m]
    return norm(Base.transpose(V) * V - I, Inf)
end

function eigenpair_residual(result::DominantResult, apply, dims::Tuple)
    y = vec(result.y)
    fy = vec(apply(reshape(y, dims)))
    λ = result.lambda
    return norm(fy .- λ .* y) / max(norm(fy), abs(λ) * norm(y), norm(y), 1.0)
end

function status_message(handle::Ptr{Cvoid}, status::Integer)
    base = try
        fptr = Libdl.dlsym(handle, :tenet_native_status_string)
        unsafe_string(ccall(fptr, Cstring, (Cint,), Cint(status)))
    catch
        "status=$status"
    end
    last = try
        fptr = Libdl.dlsym(handle, :tenet_native_last_error)
        unsafe_string(ccall(fptr, Cstring, ()))
    catch
        ""
    end
    return isempty(last) || last == "success" ? base : "$base: $last"
end

function check_status(handle, status, context)
    status == 0 && return nothing
    error("$context failed: " * status_message(handle, status))
end

function basis_two_layer_cpu(handle, Aup, Adn, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * chi
    V = Matrix{Float64}(undef, len, k + 1)
    H = Matrix{Float64}(undef, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_two_layer_d_cpu)
    status = ccall(fptr, Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Int64, Float64, Cint, Ptr{Float64}, Int64, Ptr{Float64},
         Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(len), H, Int64(k + 1),
        beta, m, res)
    check_status(handle, status, "CPU two-layer Arnoldi")
    return BasisResult(V, H, Int(m[]), beta[], res[])
end

function basis_two_layer_cuda(handle, Aup, Adn, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * chi
    V = CuArray{Float64}(undef, len, k + 1)
    H = zeros(Float64, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_two_layer_d_cuda)
    status = ccall(fptr, Cint,
        (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
         Int64, Float64, Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64},
         Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(stride(V, 2)), H, Int64(stride(H, 2)),
        beta, m, res)
    check_status(handle, status, "CUDA two-layer Arnoldi")
    CUDA.synchronize()
    return BasisResult(Array(V), H, Int(m[]), beta[], res[])
end

function basis_projected_two_layer_cpu(handle, Aup, Adn, rho, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * chi
    V = Matrix{Float64}(undef, len, k + 1)
    H = Matrix{Float64}(undef, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_projected_two_layer_d_cpu)
    status = ccall(fptr, Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Int64,
         Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, rho, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(len), H, Int64(k + 1),
        beta, m, res)
    check_status(handle, status, "CPU projected two-layer Arnoldi")
    return BasisResult(V, H, Int(m[]), beta[], res[])
end

function basis_projected_two_layer_cuda(handle, Aup, Adn, rho, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * chi
    V = CuArray{Float64}(undef, len, k + 1)
    H = zeros(Float64, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_projected_two_layer_d_cuda)
    status = ccall(fptr, Cint,
        (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
         CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
         Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64}, Int64,
         Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, rho, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(stride(V, 2)), H, Int64(stride(H, 2)),
        beta, m, res)
    check_status(handle, status, "CUDA projected two-layer Arnoldi")
    CUDA.synchronize()
    return BasisResult(Array(V), H, Int(m[]), beta[], res[])
end

function basis_qprojected_two_layer_cpu(handle, Aup, Adn, rho, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * chi
    V = Matrix{Float64}(undef, len, k + 1)
    H = Matrix{Float64}(undef, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_qprojected_two_layer_d_cpu)
    status = ccall(fptr, Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Int64,
         Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, rho, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(len), H, Int64(k + 1),
        beta, m, res)
    check_status(handle, status, "CPU qprojected two-layer Arnoldi")
    return BasisResult(V, H, Int(m[]), beta[], res[])
end

function basis_qprojected_two_layer_cuda(handle, Aup, Adn, rho, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * chi
    V = CuArray{Float64}(undef, len, k + 1)
    H = zeros(Float64, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_qprojected_two_layer_d_cuda)
    status = ccall(fptr, Cint,
        (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
         CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
         Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64}, Int64,
         Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, rho, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(stride(V, 2)), H, Int64(stride(H, 2)),
        beta, m, res)
    check_status(handle, status, "CUDA qprojected two-layer Arnoldi")
    CUDA.synchronize()
    return BasisResult(Array(V), H, Int(m[]), beta[], res[])
end

function basis_three_layer_cpu(handle, Aup, Adn, M, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * phys * chi
    V = Matrix{Float64}(undef, len, k + 1)
    H = Matrix{Float64}(undef, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_three_layer_leg4_d_cpu)
    status = ccall(fptr, Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Int64,
         Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, M, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(len), H, Int64(k + 1),
        beta, m, res)
    check_status(handle, status, "CPU three-layer Arnoldi")
    return BasisResult(V, H, Int(m[]), beta[], res[])
end

function basis_three_layer_cuda(handle, Aup, Adn, M, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    len = chi * phys * chi
    V = CuArray{Float64}(undef, len, k + 1)
    H = zeros(Float64, k + 1, k)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_arnoldi_three_layer_leg4_d_cuda)
    status = ccall(fptr, Cint,
        (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
         CUDA.CuPtr{Float64}, Int64, Float64, Cint, CUDA.CuPtr{Float64}, Int64,
         Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, M, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), V, Int64(stride(V, 2)), H, Int64(stride(H, 2)),
        beta, m, res)
    check_status(handle, status, "CUDA three-layer Arnoldi")
    CUDA.synchronize()
    return BasisResult(Array(V), H, Int(m[]), beta[], res[])
end

function dominant_two_layer_cpu(handle, Aup, Adn, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    y = Matrix{Float64}(undef, chi, chi)
    λ = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_dominant_two_layer_d_cpu)
    status = ccall(fptr, Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Int64, Float64, Cint, Ptr{Float64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), y, λ)
    check_status(handle, status, "CPU dominant two-layer")
    return DominantResult(y, λ[])
end

function dominant_two_layer_cuda(handle, Aup, Adn, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    y = CuArray{Float64}(undef, chi, chi)
    λ = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_dominant_two_layer_d_cuda)
    status = ccall(fptr, Cint,
        (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
         CUDA.CuPtr{Float64}, Int64, Float64, Cint, CUDA.CuPtr{Float64},
         Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), y, λ)
    check_status(handle, status, "CUDA dominant two-layer")
    CUDA.synchronize()
    return DominantResult(Array(y), λ[])
end

function dominant_three_layer_cpu(handle, Aup, Adn, M, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    y = Array{Float64}(undef, chi, phys, chi)
    λ = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_dominant_three_layer_leg4_d_cpu)
    status = ccall(fptr, Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, M, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), y, λ)
    check_status(handle, status, "CPU dominant three-layer")
    return DominantResult(y, λ[])
end

function dominant_three_layer_cuda(handle, Aup, Adn, M, x0; k, tol, transpose=false)
    chi, phys, _ = size(Aup)
    y = CuArray{Float64}(undef, chi, phys, chi)
    λ = Ref{Float64}(0.0)
    fptr = Libdl.dlsym(handle, :tenet_native_dominant_three_layer_leg4_d_cuda)
    status = ccall(fptr, Cint,
        (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
         CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64, Cint,
         CUDA.CuPtr{Float64}, Ref{Float64}),
        Int64(chi), Int64(phys), Aup, Adn, M, x0, Int64(k), tol,
        Cint(transpose ? 1 : 0), y, λ)
    check_status(handle, status, "CUDA dominant three-layer")
    CUDA.synchronize()
    return DominantResult(Array(y), λ[])
end

function median_time(f; warmup::Integer, repeats::Integer, sync_cuda::Bool=false)
    last = nothing
    for _ in 1:warmup
        sync_cuda && CUDA.synchronize()
        last = f()
        sync_cuda && CUDA.synchronize()
    end
    times = Float64[]
    for _ in 1:repeats
        GC.gc(false)
        sync_cuda && CUDA.synchronize()
        t0 = time_ns()
        last = f()
        sync_cuda && CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return median(times), last
end

function tolerance(opts::Options, scale::Real)
    return max(opts.atol, opts.rtol * max(1.0, Float64(scale)))
end

function max_abs_or_zero(A)
    isempty(A) && return 0.0
    return maximum(abs, A)
end

function basis_diff_scale(cuda::BasisResult, cpu::BasisResult)
    if cuda.m != cpu.m
        return Inf, 1.0, "m mismatch cuda=$(cuda.m) cpu=$(cpu.m)"
    end
    m = cuda.m
    len = size(cpu.V, 1)
    compare_next_vector = m < len
    vcols = 1:(compare_next_vector ? m + 1 : m)
    v_diff = max_abs_or_zero(cuda.V[:, vcols] .- cpu.V[:, vcols])
    next_v_diff = compare_next_vector ? max_abs_or_zero(cuda.V[:, m + 1] .- cpu.V[:, m + 1]) : NaN
    h_diff = max_abs_or_zero(cuda.H[1:(m + 1), 1:m] .- cpu.H[1:(m + 1), 1:m])
    diffs = Float64[
        abs(cuda.beta - cpu.beta),
        abs(cuda.final_resnorm - cpu.final_resnorm),
        v_diff,
        h_diff,
    ]
    scale = max(1.0, max_abs_or_zero(cpu.V[:, 1:(m + 1)]),
                max_abs_or_zero(cpu.H[1:(m + 1), 1:m]), abs(cpu.beta))
    next_detail = compare_next_vector ? fmt_diff(next_v_diff) : "skipped_full_basis"
    detail = "m=$m beta_cuda=$(fmt_float(cuda.beta)) beta_cpu=$(fmt_float(cpu.beta)) " *
        "v_diff=$(fmt_diff(v_diff)) next_v_diff=$next_detail h_diff=$(fmt_diff(h_diff)) " *
        "resnorm_diff=$(fmt_diff(abs(cuda.final_resnorm - cpu.final_resnorm)))"
    return maximum(diffs), scale, detail
end

function basis_operator_diff_scale(cuda::BasisResult, cpu::BasisResult,
                                   apply, dims::Tuple)
    diff, scale, detail = basis_diff_scale(cuda, cpu)
    len = size(cpu.V, 1)
    if cuda.m == len && len <= 4096
        cuda_rel = arnoldi_relation_residual(cuda, apply, dims)
        cpu_rel = arnoldi_relation_residual(cpu, apply, dims)
        cuda_orth = arnoldi_orthogonality_residual(cuda)
        cpu_orth = arnoldi_orthogonality_residual(cpu)
        metric = max(abs(cuda.beta - cpu.beta),
                     abs(cuda.final_resnorm - cpu.final_resnorm),
                     cuda_rel, cpu_rel, cuda_orth, cpu_orth)
        detail *= " full_basis_metric=arnoldi_residual " *
            "cuda_rel=$(fmt_diff(cuda_rel)) cpu_rel=$(fmt_diff(cpu_rel)) " *
            "cuda_orth=$(fmt_diff(cuda_orth)) cpu_orth=$(fmt_diff(cpu_orth)) " *
            "elementwise_diff=$(fmt_diff(diff))"
        return metric, scale, detail
    end
    return diff, scale, detail
end

function dominant_diff_scale(cuda::DominantResult, cpu::DominantResult)
    yc = copy(cuda.y)
    yp = vec(yc)
    yref = vec(cpu.y)
    if dot(yp, yref) < 0
        yc .*= -1
        yp = vec(yc)
    end
    ydiff = max_abs_or_zero(yp .- yref)
    ldiff = abs(cuda.lambda - cpu.lambda)
    scale = max(1.0, abs(cpu.lambda), max_abs_or_zero(yref))
    detail = "lambda_cuda=$(fmt_float(cuda.lambda)) lambda_cpu=$(fmt_float(cpu.lambda))"
    return max(ydiff, ldiff), scale, detail
end

function dominant_operator_diff_scale(cuda::DominantResult, cpu::DominantResult,
                                      apply, dims::Tuple)
    yc = copy(cuda.y)
    yp = vec(yc)
    yref = vec(cpu.y)
    if dot(yp, yref) < 0
        yc .*= -1
        yp = vec(yc)
    end
    ydiff = max_abs_or_zero(yp .- yref)
    ldiff = abs(cuda.lambda - cpu.lambda)
    cuda_rel = eigenpair_residual(cuda, apply, dims)
    cpu_rel = eigenpair_residual(cpu, apply, dims)
    scale = max(1.0, abs(cpu.lambda), abs(cuda.lambda))
    detail = "lambda_cuda=$(fmt_float(cuda.lambda)) lambda_cpu=$(fmt_float(cpu.lambda)) " *
        "lambda_diff=$(fmt_diff(ldiff)) cuda_relres=$(fmt_diff(cuda_rel)) " *
        "cpu_relres=$(fmt_diff(cpu_rel)) y_max_abs_diff=$(fmt_diff(ydiff))"
    return max(ldiff, cuda_rel, cpu_rel), scale, detail
end

function basis_reference_metric(result::BasisResult, apply, dims::Tuple)
    rel = arnoldi_relation_residual(result, apply, dims)
    orth = arnoldi_orthogonality_residual(result)
    detail = "arnoldi_relation_relres=$(fmt_diff(rel)) orthogonality_inf=$(fmt_diff(orth)) m=$(result.m)"
    return max(rel, orth), 1.0, detail
end

function dominant_reference_metric(result::DominantResult, apply, dims::Tuple)
    ref = target_eigenvalue(apply, dims)
    rel = eigenpair_residual(result, apply, dims)
    ldiff = abs(result.lambda - real(ref))
    scale = max(1.0, abs(result.lambda), abs(ref))
    detail = "lambda_cuda=$(fmt_float(result.lambda)) lambda_ref=$(fmt_float(real(ref))) eigenpair_relres=$(fmt_diff(rel)) ref_imag=$(fmt_diff(abs(imag(ref))))"
    return max(ldiff, rel), scale, detail
end

function push_case!(rows, meta, opts::Options; case_id, operation, cuda_fun,
                    cpu_fun=nothing, ref_fun,
                    comparison_label="TenetNative CPU ABI")
    try
        native_time, cuda_result = median_time(cuda_fun; warmup=opts.warmup,
                                               repeats=opts.repeats, sync_cuda=true)
        comparison = ""
        comparison_time = nothing
        diff = nothing
        scale = 1.0
        detail = ""
        if cpu_fun !== nothing
            comparison = comparison_label
            comparison_time, cpu_result = median_time(cpu_fun; warmup=opts.warmup,
                                                      repeats=opts.repeats)
            diff, scale, detail = ref_fun(cuda_result, cpu_result)
        else
            comparison = "Julia dense reference"
            diff, scale, detail = ref_fun(cuda_result)
        end
        tol = tolerance(opts, scale)
        status = isfinite(diff) && diff <= tol ? "pass" : "fail"
        ratio = comparison_time === nothing || comparison_time == 0.0 ? nothing :
            native_time / comparison_time
        push!(rows, row(meta; case_id, operation, status, comparison, diff,
                        tolerance=tol, warmup=opts.warmup, repeats=opts.repeats,
                        native_time, comparison_time, ratio, detail))
    catch err
        push!(rows, row(meta; case_id, operation, status="fail",
                        comparison=cpu_fun === nothing ? "Julia dense reference" : comparison_label,
                        warmup=opts.warmup, repeats=opts.repeats,
                        detail=sprint(showerror, err)))
    end
    return nothing
end

function tenetnative_module(opts::Options)
    add_load_path!(opts.repo)
    return Base.require(Main, :TenetNative)
end

function wrapper_basis_result(result)
    final_resnorm = hasproperty(result, :final_resnorm) ?
        Float64(getproperty(result, :final_resnorm)) : NaN
    return BasisResult(Array(result.V), Matrix(result.H), Int(result.m),
                       Float64(result.beta), final_resnorm)
end

function wrapper_dominant_result(result)
    return DominantResult(Array(result.y), Float64(result.lambda))
end

function batch_output_diff_scale(cuda_result, ref::AbstractArray)
    Y = Array(cuda_result)
    diff = max_abs_or_zero(Y .- ref)
    scale = max(1.0, max_abs_or_zero(Y), max_abs_or_zero(ref))
    detail = "batch_max_abs_diff=$(fmt_diff(diff)) batch=$(size(ref, 3))"
    return diff, scale, detail
end

function batched_x0(data)
    chi = size(data.x0, 1)
    batch = 3
    X = Array{Float64}(undef, chi, chi, batch)
    X[:, :, 1] .= data.x0
    X[:, :, 2] .= 0.5 .* data.x0
    X[:, :, 3] .= data.x0 .+ 0.1 .* Matrix{Float64}(I, chi, chi)
    return X
end

function batched_w(data; batch::Integer=3)
    chi, phys, _ = size(data.W)
    W = Array{Float64}(undef, chi, phys, chi, batch)
    for b in 1:batch
        W[:, :, :, b] .= (1.0 + 0.01 * (b - 2)) .* data.W
    end
    return W
end

function batched_rho(data; batch::Integer=3)
    chi = size(data.rho, 1)
    R = Array{Float64}(undef, chi, chi, batch)
    for b in 1:batch
        R[:, :, b] .= data.rho
    end
    return R
end

function batch_reference(Aup, Adn, rho, X, apply; transpose::Bool=false)
    batch = size(X, 3)
    Y = similar(X)
    for b in 1:batch
        A = ndims(Aup) == 4 ? view(Aup, :, :, :, b) : Aup
        B = ndims(Adn) == 4 ? view(Adn, :, :, :, b) : Adn
        R = rho === nothing ? nothing :
            (ndims(rho) == 3 ? view(rho, :, :, b) : rho)
        Y[:, :, b] .= R === nothing ?
            apply(A, B, X[:, :, b]; transpose) :
            apply(A, B, R, X[:, :, b]; transpose)
    end
    return Y
end

function push_tenetnative_wrapper_cases!(rows, meta, opts::Options, data, ddata,
                                         cuda_handle, k2::Integer, k3::Integer,
                                         tol::Real)
    TN = try
        tenetnative_module(opts)
    catch err
        push!(rows, row(meta; case_id="tenetnative_julia_cuda_wrappers",
                        operation="TenetNative Julia CUDA wrapper load",
                        status="fail", comparison="direct CUDA C ABI",
                        detail=sprint(showerror, err)))
        return nothing
    end
    lib = get(meta, "cuda_lib_path", "")
    comparison_label = "direct CUDA C ABI"

    push_case!(rows, meta, opts;
        case_id="wrapper_two_layer_arnoldi",
        operation="TenetNative Julia CUDA wrapper: Arnoldi basis",
        cuda_fun=() -> wrapper_basis_result(Base.invokelatest(
            getproperty(TN, :tenet_native_arnoldi_two_layer_d_cuda),
            ddata.Aup, ddata.Adn, ddata.x0;
            max_k=k2, breakdown_tol=tol, lib)),
        cpu_fun=() -> basis_two_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn,
                                           ddata.x0; k=k2, tol),
        ref_fun=basis_diff_scale,
        comparison_label=comparison_label)

    push_case!(rows, meta, opts;
        case_id="wrapper_projected_two_layer_arnoldi",
        operation="TenetNative Julia CUDA wrapper: projected Arnoldi basis",
        cuda_fun=() -> wrapper_basis_result(Base.invokelatest(
            getproperty(TN, :tenet_native_arnoldi_projected_two_layer_d_cuda),
            ddata.W, ddata.W, ddata.rho, ddata.x0;
            max_k=k2, breakdown_tol=tol, lib)),
        cpu_fun=() -> basis_projected_two_layer_cuda(
            cuda_handle, ddata.W, ddata.W, ddata.rho, ddata.x0; k=k2, tol),
        ref_fun=basis_diff_scale,
        comparison_label=comparison_label)

    push_case!(rows, meta, opts;
        case_id="wrapper_qprojected_two_layer_arnoldi",
        operation="TenetNative Julia CUDA wrapper: qprojected Arnoldi basis",
        cuda_fun=() -> wrapper_basis_result(Base.invokelatest(
            getproperty(TN, :tenet_native_arnoldi_qprojected_two_layer_d_cuda),
            ddata.W, ddata.W, ddata.rho, ddata.x0;
            max_k=k2, breakdown_tol=tol, lib)),
        cpu_fun=() -> basis_qprojected_two_layer_cuda(
            cuda_handle, ddata.W, ddata.W, ddata.rho, ddata.x0; k=k2, tol),
        ref_fun=basis_diff_scale,
        comparison_label=comparison_label)

    Xbatch = batched_x0(data)
    dXbatch = CuArray(Xbatch)

    push_case!(rows, meta, opts;
        case_id="wrapper_batch_two_layer_apply_shared",
        operation="TenetNative Julia CUDA wrapper: batched two-layer apply (shared A)",
        cuda_fun=() -> Base.invokelatest(
            getproperty(TN, :tenet_native_two_layer_apply_batch_d_cuda),
            ddata.W, ddata.W, dXbatch; lib),
        cpu_fun=() -> batch_reference(data.W, data.W, nothing, Xbatch,
                                      two_layer_apply),
        ref_fun=batch_output_diff_scale,
        comparison_label="Julia dense batch reference")

    push_case!(rows, meta, opts;
        case_id="wrapper_batch_projected_apply_shared",
        operation="TenetNative Julia CUDA wrapper: batched projected apply (shared A,rho)",
        cuda_fun=() -> Base.invokelatest(
            getproperty(TN, :tenet_native_projected_two_layer_apply_batch_d_cuda),
            ddata.W, ddata.W, ddata.rho, dXbatch; lib),
        cpu_fun=() -> batch_reference(data.W, data.W, data.rho, Xbatch,
                                      projected_two_layer_apply),
        ref_fun=batch_output_diff_scale,
        comparison_label="Julia dense batch reference")

    Wbatch = batched_w(data)
    Rhobatch = batched_rho(data)
    dWbatch = CuArray(Wbatch)
    dRhobatch = CuArray(Rhobatch)

    push_case!(rows, meta, opts;
        case_id="wrapper_batch_qprojected_apply_batched_transpose",
        operation="TenetNative Julia CUDA wrapper: batched qprojected transpose apply (batched A,rho)",
        cuda_fun=() -> Base.invokelatest(
            getproperty(TN, :tenet_native_qprojected_two_layer_apply_batch_d_cuda),
            dWbatch, dWbatch, dRhobatch, dXbatch; transpose=true, lib),
        cpu_fun=() -> batch_reference(Wbatch, Wbatch, Rhobatch, Xbatch,
                                      qprojected_two_layer_apply;
                                      transpose=true),
        ref_fun=batch_output_diff_scale,
        comparison_label="Julia dense batch reference")

    push_case!(rows, meta, opts;
        case_id="wrapper_three_layer_leg4_arnoldi",
        operation="TenetNative Julia CUDA wrapper: three-layer leg4 Arnoldi basis",
        cuda_fun=() -> wrapper_basis_result(Base.invokelatest(
            getproperty(TN, :tenet_native_arnoldi_three_layer_leg4_d_cuda),
            ddata.Aup, ddata.Adn, ddata.M, ddata.x3;
            max_k=k3, breakdown_tol=tol, lib)),
        cpu_fun=() -> basis_three_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn,
                                             ddata.M, ddata.x3; k=k3, tol),
        ref_fun=basis_diff_scale,
        comparison_label=comparison_label)

    push_case!(rows, meta, opts;
        case_id="wrapper_dominant_two_layer",
        operation="TenetNative Julia CUDA wrapper: dominant eigenpair",
        cuda_fun=() -> wrapper_dominant_result(Base.invokelatest(
            getproperty(TN, :tenet_native_dominant_two_layer_d_cuda),
            ddata.Aup, ddata.Adn, ddata.x0;
            max_k=k2, breakdown_tol=tol, lib)),
        cpu_fun=() -> dominant_two_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn,
                                              ddata.x0; k=k2, tol),
        ref_fun=dominant_diff_scale,
        comparison_label=comparison_label)

    push_case!(rows, meta, opts;
        case_id="wrapper_dominant_three_layer",
        operation="TenetNative Julia CUDA wrapper: dominant three-layer eigenpair",
        cuda_fun=() -> wrapper_dominant_result(Base.invokelatest(
            getproperty(TN, :tenet_native_dominant_three_layer_leg4_d_cuda),
            ddata.Aup, ddata.Adn, ddata.M, ddata.x3;
            max_k=k3, breakdown_tol=tol, lib)),
        cpu_fun=() -> dominant_three_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn,
                                                ddata.M, ddata.x3; k=k3, tol),
        ref_fun=dominant_diff_scale,
        comparison_label=comparison_label)

    return nothing
end

function fasttenet_smoke_once(FT, opts::Options)
    beta = Base.invokelatest(getproperty(FT, :critical_beta))
    r = Base.invokelatest(
        getproperty(FT, :run_boundary),
        beta;
        chi=opts.smoke_chi,
        maxiter=10,
        miniter=1,
        maxiter_ad=0,
        verbosity=0,
        arraytype=CuArray,
    )
    CUDA.synchronize()
    logz = Base.invokelatest(getproperty(FT, :log_partition_density), r)
    logz_exact = Base.invokelatest(getproperty(FT, :log_partition_density_exact), beta)
    energy = Base.invokelatest(getproperty(FT, :energy_density), r)
    energy_exact = Base.invokelatest(getproperty(FT, :energy_density_exact), beta)
    logz_err = abs(logz - logz_exact)
    energy_err = abs(energy - energy_exact)
    logz_err < 1e-3 || error("FastTeneT 2DIsing logz error too large: $logz_err")
    energy_err < 2e-2 || error("FastTeneT 2DIsing energy error too large: $energy_err")

    tfi = Base.invokelatest(
        getproperty(FT, :run_tfising_vumps),
        1.0;
        chi=max(opts.smoke_chi, 4),
        maxiter=25,
        miniter=2,
        maxiter_ad=0,
        tol=1e-8,
        eig_maxiter=96,
        env_maxiter=96,
        verbosity=0,
        arraytype=CuArray,
    )
    CUDA.synchronize()
    tfi_energy = Base.invokelatest(getproperty(FT, :tfising_energy_density), tfi)
    tfi_exact = Base.invokelatest(getproperty(FT, :tfising_ground_state_energy_density_exact), 1.0)
    tfi_err = abs(tfi_energy - tfi_exact)
    tfi_err < 1e-4 || error("FastTeneT TFIsing energy error too large: $tfi_err")
    detail = "2DIsing logz_err=$(fmt_diff(logz_err)) energy_err=$(fmt_diff(energy_err)); TFIsing energy_err=$(fmt_diff(tfi_err))"
    return (; metric=max(logz_err, energy_err, tfi_err), detail)
end

function push_fasttenet_smoke!(rows, meta, opts::Options)
    if !opts.fasttenet_smoke
        push!(rows, row(meta; case_id="fasttenet_h100_smoke",
                        operation="FastTeneT CuArray full smoke", status="skip",
                        detail="disabled by --skip-fasttenet-smoke"))
        return nothing
    end

    FT = try
        add_load_path!(opts.repo)
        Base.require(Main, :FastTeneT)
    catch err
        push!(rows, row(meta; case_id="fasttenet_h100_smoke",
                        operation="FastTeneT CuArray full smoke", status="skip",
                        detail="FastTeneT dependencies unavailable: " * sprint(showerror, err)))
        return nothing
    end

    try
        native_time, result = median_time(() -> fasttenet_smoke_once(FT, opts);
                                          warmup=0, repeats=opts.smoke_repeats,
                                          sync_cuda=true)
        tol = 2e-2
        status = result.metric <= tol ? "pass" : "fail"
        push!(rows, row(meta; case_id="fasttenet_h100_smoke",
                        operation="FastTeneT CuArray full smoke", status,
                        comparison="exact references", diff=result.metric,
                        tolerance=tol, warmup=0, repeats=opts.smoke_repeats,
                        native_time, detail=result.detail))
    catch err
        push!(rows, row(meta; case_id="fasttenet_h100_smoke",
                        operation="FastTeneT CuArray full smoke", status="fail",
                        comparison="exact references", warmup=0,
                        repeats=opts.smoke_repeats, detail=sprint(showerror, err)))
    end
    return nothing
end

function all_ok(rows)
    return all(r -> r["status"] == "pass" || r["status"] == "skip", rows)
end

function main(opts::Options)
    CUDA.allowscalar(false)
    cuda_lib, cpu_lib = resolve_libs(opts)
    cuda_handle = Libdl.dlopen(cuda_lib)
    cpu_handle = cpu_lib === nothing ? nothing : Libdl.dlopen(cpu_lib)
    device_name = try
        CUDA.name(CUDA.device())
    catch
        "unknown"
    end
    meta = Dict(
        "device_name" => device_name,
        "cuda_version" => safe_cuda_version(),
        "cuda_lib_path" => cuda_lib,
        "cpu_lib_path" => cpu_lib === nothing ? "" : cpu_lib,
    )

    data = inputs(opts.seed, opts.chi, opts.phys)
    ddata = (;
        Aup=CuArray(data.Aup),
        Adn=CuArray(data.Adn),
        W=CuArray(data.W),
        x0=CuArray(data.x0),
        rho=CuArray(data.rho),
        M=CuArray(data.M),
        x3=CuArray(data.x3),
    )
    CUDA.synchronize()

    k2 = full_k(opts, opts.chi * opts.chi)
    k3 = full_k(opts, opts.chi * opts.phys * opts.chi)
    dims2 = (opts.chi, opts.chi)
    dims3 = (opts.chi, opts.phys, opts.chi)
    tol = opts.breakdown_tol
    rows = Vector{Dict{String,String}}()

    push_case!(rows, meta, opts;
        case_id="two_layer_arnoldi",
        operation="Arnoldi basis",
        cuda_fun=() -> basis_two_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn, ddata.x0; k=k2, tol),
        cpu_fun=cpu_handle === nothing ? nothing :
            () -> basis_two_layer_cpu(cpu_handle, data.Aup, data.Adn, data.x0; k=k2, tol),
        ref_fun=cpu_handle === nothing ?
            (r -> basis_reference_metric(r, X -> two_layer_apply(data.Aup, data.Adn, X), dims2)) :
            ((cuda, cpu) -> basis_operator_diff_scale(
                cuda, cpu, X -> two_layer_apply(data.Aup, data.Adn, X), dims2)))

    push_case!(rows, meta, opts;
        case_id="projected_two_layer_arnoldi",
        operation="Projected Arnoldi basis",
        cuda_fun=() -> basis_projected_two_layer_cuda(cuda_handle, ddata.W, ddata.W, ddata.rho, ddata.x0; k=k2, tol),
        cpu_fun=cpu_handle === nothing ? nothing :
            () -> basis_projected_two_layer_cpu(cpu_handle, data.W, data.W, data.rho, data.x0; k=k2, tol),
        ref_fun=cpu_handle === nothing ?
            (r -> basis_reference_metric(r, X -> projected_two_layer_apply(data.W, data.W, data.rho, X), dims2)) :
            ((cuda, cpu) -> basis_operator_diff_scale(
                cuda, cpu, X -> projected_two_layer_apply(data.W, data.W, data.rho, X), dims2)))

    push_case!(rows, meta, opts;
        case_id="qprojected_two_layer_arnoldi",
        operation="Q-projected Arnoldi basis",
        cuda_fun=() -> basis_qprojected_two_layer_cuda(cuda_handle, ddata.W, ddata.W, ddata.rho, ddata.x0; k=k2, tol),
        cpu_fun=cpu_handle === nothing ? nothing :
            () -> basis_qprojected_two_layer_cpu(cpu_handle, data.W, data.W, data.rho, data.x0; k=k2, tol),
        ref_fun=cpu_handle === nothing ?
            (r -> basis_reference_metric(r, X -> qprojected_two_layer_apply(data.W, data.W, data.rho, X), dims2)) :
            ((cuda, cpu) -> basis_operator_diff_scale(
                cuda, cpu, X -> qprojected_two_layer_apply(data.W, data.W, data.rho, X), dims2)))

    push_case!(rows, meta, opts;
        case_id="qprojected_two_layer_arnoldi_transpose",
        operation="Q-projected transpose Arnoldi basis",
        cuda_fun=() -> basis_qprojected_two_layer_cuda(cuda_handle, ddata.W, ddata.W, ddata.rho, ddata.x0; k=k2, tol, transpose=true),
        cpu_fun=cpu_handle === nothing ? nothing :
            () -> basis_qprojected_two_layer_cpu(cpu_handle, data.W, data.W, data.rho, data.x0; k=k2, tol, transpose=true),
        ref_fun=cpu_handle === nothing ?
            (r -> basis_reference_metric(r, X -> qprojected_two_layer_apply(data.W, data.W, data.rho, X; transpose=true), dims2)) :
            ((cuda, cpu) -> basis_operator_diff_scale(
                cuda, cpu, X -> qprojected_two_layer_apply(data.W, data.W, data.rho, X; transpose=true), dims2)))

    push_case!(rows, meta, opts;
        case_id="three_layer_leg4_arnoldi",
        operation="Three-layer leg4 Arnoldi basis",
        cuda_fun=() -> basis_three_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn, ddata.M, ddata.x3; k=k3, tol),
        cpu_fun=cpu_handle === nothing ? nothing :
            () -> basis_three_layer_cpu(cpu_handle, data.Aup, data.Adn, data.M, data.x3; k=k3, tol),
        ref_fun=cpu_handle === nothing ?
            (r -> basis_reference_metric(r, X -> three_layer_apply(data.Aup, data.Adn, data.M, X), dims3)) :
            ((cuda, cpu) -> basis_operator_diff_scale(
                cuda, cpu, X -> three_layer_apply(data.Aup, data.Adn, data.M, X), dims3)))

    push_case!(rows, meta, opts;
        case_id="dominant_two_layer",
        operation="Dominant eigenpair",
        cuda_fun=() -> dominant_two_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn, ddata.x0; k=k2, tol),
        cpu_fun=cpu_handle === nothing ? nothing :
            () -> dominant_two_layer_cpu(cpu_handle, data.Aup, data.Adn, data.x0; k=k2, tol),
        ref_fun=cpu_handle === nothing ?
            (r -> dominant_reference_metric(r, X -> two_layer_apply(data.Aup, data.Adn, X), dims2)) :
            ((cuda, cpu) -> dominant_operator_diff_scale(
                cuda, cpu, X -> two_layer_apply(data.Aup, data.Adn, X), dims2)))

    push_case!(rows, meta, opts;
        case_id="dominant_three_layer",
        operation="Dominant three-layer eigenpair",
        cuda_fun=() -> dominant_three_layer_cuda(cuda_handle, ddata.Aup, ddata.Adn, ddata.M, ddata.x3; k=k3, tol),
        cpu_fun=cpu_handle === nothing ? nothing :
            () -> dominant_three_layer_cpu(cpu_handle, data.Aup, data.Adn, data.M, data.x3; k=k3, tol),
        ref_fun=cpu_handle === nothing ?
            (r -> dominant_reference_metric(r, X -> three_layer_apply(data.Aup, data.Adn, data.M, X), dims3)) :
            ((cuda, cpu) -> dominant_operator_diff_scale(
                cuda, cpu, X -> three_layer_apply(data.Aup, data.Adn, data.M, X), dims3)))

    push_tenetnative_wrapper_cases!(rows, meta, opts, data, ddata, cuda_handle,
                                    k2, k3, tol)

    push_fasttenet_smoke!(rows, meta, opts)

    csv_path = write_csv(joinpath(opts.outdir, "native_h100_parity.csv"), rows)
    md_path = write_markdown(joinpath(opts.outdir, "native_h100_parity.md"), rows, opts, meta)
    println("TENET_NATIVE_H100_PARITY artifacts csv=$csv_path markdown=$md_path")

    if all_ok(rows)
        println("TENET_NATIVE_H100_PARITY_DONE status=pass rows=$(length(rows)) skipped=$(count(r -> r["status"] == "skip", rows)) outdir=$(opts.outdir)")
    else
        nfail = count(r -> r["status"] == "fail", rows)
        println("TENET_NATIVE_H100_PARITY_DONE status=fail rows=$(length(rows)) failures=$nfail outdir=$(opts.outdir)")
        error("native H100 parity failed with $nfail failing rows; see $csv_path and $md_path")
    end
    return nothing
end

try
    main(OPTS)
catch err
    println(stderr, "TENET_NATIVE_H100_PARITY_ERROR ", sprint(showerror, err))
    exit(1)
end
