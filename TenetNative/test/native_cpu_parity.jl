#!/usr/bin/env julia

using Dates
using Libdl
using LinearAlgebra
using Printf
using Random

const CSV_COLUMNS = (
    "case",
    "lhs",
    "rhs",
    "component",
    "lhs_m",
    "rhs_m",
    "max_abs_diff",
    "tolerance",
    "status",
    "detail",
)

mutable struct Options
    repo::String
    outdir::String
    prefix::String
    lib::Union{Nothing,String}
    build::Bool
    julia::String
    seed::Int
    chi::Int
    phys::Int
    max_k::Int
    atol::Float64
    rtol::Float64
    allow_repo_outdir::Bool
end

function usage()
    return """
    Usage:
      julia --project=<env> TenetNative/test/native_cpu_parity.jl [options]

    Options:
      --repo PATH           Repository root. Defaults to the parent of TenetNative.
      --outdir PATH         Output directory for CSV/Markdown artifacts.
      --prefix PATH         Native build prefix. Defaults to OUTDIR/tenetnative_deps.
      --lib PATH            Existing libtenet_native_arnoldi CPU library.
      --build               Build TenetNative CPU into --prefix before running.
      --no-build            Require --lib or TENET_NATIVE_ARNOLDI_LIB.
      --julia PATH          Julia executable for the native Makefile.
      --seed INT            Deterministic input seed.
      --chi INT             Bond dimension for dense ABI fixtures.
      --phys INT            Physical dimension for dense ABI fixtures.
      --max-k INT           Arnoldi Krylov dimension; 0 means full dimension.
      --atol FLOAT          Absolute tolerance for strict numeric parity.
      --rtol FLOAT          Relative tolerance for strict numeric parity.
      --allow-repo-outdir   Permit generated outputs under the repo.
      --help                Print this help.
    """
end

timestamp_utc() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyymmddTHHMMSS") * "Z"

function default_repo()
    return normpath(joinpath(@__DIR__, "..", ".."))
end

function default_outdir()
    return joinpath(tempdir(), "tenet_native_cpu_parity", "run_" * timestamp_utc())
end

function parse_bool_build(argv)
    saw_build = false
    saw_no_build = false
    for arg in argv
        saw_build |= arg == "--build"
        saw_no_build |= arg == "--no-build"
    end
    saw_build && saw_no_build && error("use only one of --build or --no-build")
    return saw_build ? true : saw_no_build ? false : nothing
end

function value_after(argv, i, name)
    i < length(argv) || error("$name requires a value")
    return argv[i + 1]
end

function parse_args(argv)
    repo = default_repo()
    outdir = get(ENV, "TENET_NATIVE_CPU_PARITY_OUTDIR", default_outdir())
    prefix = nothing
    lib_env = get(ENV, "TENET_NATIVE_ARNOLDI_LIB", "")
    lib = isempty(lib_env) ? nothing : lib_env
    build_flag = parse_bool_build(argv)
    julia = get(ENV, "JULIA", joinpath(Sys.BINDIR, Sys.iswindows() ? "julia.exe" : "julia"))
    seed = parse(Int, get(ENV, "TENET_NATIVE_CPU_PARITY_SEED", "20260626"))
    chi = parse(Int, get(ENV, "TENET_NATIVE_CPU_PARITY_CHI", "3"))
    phys = parse(Int, get(ENV, "TENET_NATIVE_CPU_PARITY_PHYS", "2"))
    max_k = parse(Int, get(ENV, "TENET_NATIVE_CPU_PARITY_MAX_K", "0"))
    atol = parse(Float64, get(ENV, "TENET_NATIVE_CPU_PARITY_ATOL", "1e-12"))
    rtol = parse(Float64, get(ENV, "TENET_NATIVE_CPU_PARITY_RTOL", "1e-12"))
    allow_repo_outdir = false

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            print(usage())
            exit(0)
        elseif arg == "--build" || arg == "--no-build"
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
        elseif startswith(arg, "--lib=")
            lib = split(arg, "=", limit=2)[2]
        elseif arg == "--lib"
            lib = value_after(argv, i, arg)
            i += 1
        elseif startswith(arg, "--julia=")
            julia = split(arg, "=", limit=2)[2]
        elseif arg == "--julia"
            julia = value_after(argv, i, arg)
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
        elseif arg == "--allow-repo-outdir"
            allow_repo_outdir = true
        else
            error("unknown argument: $arg\n\n" * usage())
        end
        i += 1
    end

    repo = abspath(normpath(repo))
    outdir = abspath(normpath(outdir))
    prefix = prefix === nothing ? joinpath(outdir, "tenetnative_deps") : abspath(normpath(prefix))
    build = build_flag === nothing ? lib === nothing : build_flag
    lib = lib === nothing ? nothing : abspath(normpath(lib))

    chi > 0 || error("--chi must be positive")
    phys > 0 || error("--phys must be positive")
    max_k >= 0 || error("--max-k must be nonnegative")
    atol >= 0 || error("--atol must be nonnegative")
    rtol >= 0 || error("--rtol must be nonnegative")

    return Options(repo, outdir, prefix, lib, build, julia, seed, chi, phys,
                   max_k, atol, rtol, allow_repo_outdir)
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

function build_native_cpu(opts::Options)
    native_dir = joinpath(opts.repo, "TenetNative", "src", "native")
    isdir(native_dir) || error("native source directory not found: $native_dir")
    mkpath(opts.prefix)
    run(`make -C $native_dir native-arnoldi-cpu PREFIX=$(opts.prefix) JULIA=$(opts.julia)`)
    libpath = joinpath(opts.prefix, "libtenet_native_arnoldi." * Libdl.dlext)
    isfile(libpath) || error("native build did not produce $libpath")
    return libpath
end

function resolve_native_lib(opts::Options)
    libpath = if opts.build
        build_native_cpu(opts)
    elseif opts.lib !== nothing
        opts.lib
    else
        error("no native library configured; pass --build, --lib, or TENET_NATIVE_ARNOLDI_LIB")
    end
    isfile(libpath) || error("native library not found: $libpath")
    ENV["TENET_NATIVE_ARNOLDI_LIB"] = libpath
    return libpath
end

function add_load_path!(repo)
    tenet = joinpath(repo, "TenetNative")
    isfile(joinpath(tenet, "Project.toml")) ||
        error("TenetNative Julia environment not found: $tenet")
    legacy_exactlr = isfile(joinpath(repo, "ARchivedExactLRuMPS", "Project.toml")) ?
        joinpath(repo, "ARchivedExactLRuMPS") :
        joinpath(repo, "ExactLRuMPS")
    envs = (legacy_exactlr, joinpath(repo, "FastTeneT"), tenet)
    for env in envs
        isfile(joinpath(env, "Project.toml")) || continue
        env_abs = abspath(normpath(env))
        env_abs in LOAD_PATH || pushfirst!(LOAD_PATH, env_abs)
    end
    return nothing
end

function load_module(name::Symbol)
    return Base.require(Main, name)
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

shape_string(A) = join(size(A), "x")

fmt_float(x::Real) = @sprintf("%.17g", Float64(x))
fmt_diff(x::Real) = @sprintf("%.3e", Float64(x))

function full_k(opts::Options, len::Integer)
    return opts.max_k == 0 ? Int(len) : min(opts.max_k, Int(len))
end

function two_layer_apply(Aup, Adn, X; transpose::Bool=false)
    chi, phys, chi2 = size(Aup)
    chi == chi2 || error("Aup must be chi x phys x chi")
    size(Adn) == size(Aup) || error("Adn size mismatch")
    size(X) == (chi, chi) || error("X size mismatch")
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

function three_layer_apply(Aup, Adn, M, X; transpose::Bool=false)
    chi, phys, chi2 = size(Aup)
    chi == chi2 || error("Aup must be chi x phys x chi")
    size(Adn) == size(Aup) || error("Adn size mismatch")
    size(M) == (phys, phys, phys, phys) || error("M size mismatch")
    size(X) == (chi, phys, chi) || error("X size mismatch")
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

function projected_two_layer_apply(Aup, Adn, rho, X; transpose::Bool=false)
    Y = X .- two_layer_apply(Aup, Adn, X; transpose)
    projection = rho_dot(rho, X)
    for i in 1:size(X, 1)
        Y[i, i] += projection
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

function ritz_order(vals; target::Symbol)
    order = collect(eachindex(vals))
    sort!(order; lt=(a, b) -> begin
        va = vals[a]
        vb = vals[b]
        if target === :smallest_real
            scale = max(1.0, abs(real(va)), abs(real(vb)))
            abs(real(va) - real(vb)) > 1e-10 * scale &&
                return real(va) < real(vb)
            return abs(imag(va)) < abs(imag(vb))
        end
        ma = abs(va)
        mb = abs(vb)
        abs(ma - mb) > 1e-10 * max(1.0, ma, mb) && return ma > mb
        return real(va) > real(vb)
    end)
    return order
end

function target_eigenvalue(apply, dims::Tuple; target::Symbol)
    F = eigen(operator_matrix(apply, dims))
    order = ritz_order(F.values; target)
    return F.values[first(order)]
end

function leading_eigenvalues(apply, dims::Tuple, nvalues::Integer; target::Symbol)
    F = eigen(operator_matrix(apply, dims))
    order = ritz_order(F.values; target)
    n = min(nvalues, length(order))
    return F.values[order[1:n]]
end

function row(case_id, lhs, rhs, component, lhs_m, rhs_m, diff, tol, status, detail="")
    return Dict(
        "case" => string(case_id),
        "lhs" => string(lhs),
        "rhs" => string(rhs),
        "component" => string(component),
        "lhs_m" => string(lhs_m),
        "rhs_m" => string(rhs_m),
        "max_abs_diff" => diff === nothing ? "" : fmt_diff(diff),
        "tolerance" => tol === nothing ? "" : fmt_diff(tol),
        "status" => string(status),
        "detail" => string(detail),
    )
end

function max_abs_diff(A, B)
    size(A) == size(B) || error("shape mismatch $(size(A)) != $(size(B))")
    isempty(A) && return 0.0
    return maximum(abs, A .- B)
end

function tolerance_for(opts::Options, lhs, rhs)
    isempty(lhs) && isempty(rhs) && return opts.atol
    scale = max(1.0, maximum(abs, lhs), maximum(abs, rhs))
    return max(opts.atol, opts.rtol * scale)
end

function tolerance_for_scalar(opts::Options, lhs::Real, rhs::Real)
    scale = max(1.0, abs(Float64(lhs)), abs(Float64(rhs)))
    return max(opts.atol, opts.rtol * scale)
end

function push_status!(rows, case_id, lhs, rhs, component, lhs_m, rhs_m, pass, detail)
    push!(rows, row(case_id, lhs, rhs, component, lhs_m, rhs_m, pass ? 0.0 : 1.0,
                    0.0, pass ? "pass" : "fail", detail))
end

function push_scalar_cmp!(rows, opts::Options, case_id, lhs_name, rhs_name,
                          component, lhs_m, rhs_m, lhs, rhs)
    diff = abs(Float64(lhs) - Float64(rhs))
    tol = tolerance_for_scalar(opts, lhs, rhs)
    status = isfinite(diff) && diff <= tol ? "pass" : "fail"
    detail = string(fmt_float(lhs), " vs ", fmt_float(rhs))
    push!(rows, row(case_id, lhs_name, rhs_name, component, lhs_m, rhs_m, diff,
                    tol, status, detail))
end

function push_array_cmp!(rows, opts::Options, case_id, lhs_name, rhs_name,
                         component, lhs_m, rhs_m, lhs, rhs)
    if size(lhs) != size(rhs)
        push!(rows, row(case_id, lhs_name, rhs_name, component, lhs_m, rhs_m,
                        nothing, nothing, "fail",
                        "shape $(shape_string(lhs)) vs $(shape_string(rhs))"))
        return nothing
    end
    diff = max_abs_diff(lhs, rhs)
    tol = tolerance_for(opts, lhs, rhs)
    status = isfinite(diff) && diff <= tol ? "pass" : "fail"
    detail = "shape=" * shape_string(lhs)
    push!(rows, row(case_id, lhs_name, rhs_name, component, lhs_m, rhs_m, diff,
                    tol, status, detail))
    return nothing
end

function as_basis(result)
    if result isa NamedTuple
        return (V=Matrix{Float64}(result.V), H=Matrix{Float64}(result.H),
                m=Int(result.m), beta=Float64(result.beta))
    elseif result isa Tuple && length(result) >= 4
        return (V=Matrix{Float64}(result[1]), H=Matrix{Float64}(result[2]),
                m=Int(result[3]), beta=Float64(result[4]))
    end
    error("unsupported basis result type: $(typeof(result))")
end

function compare_basis!(rows, opts::Options, case_id, lhs_name, lhs, rhs_name, rhs)
    push_status!(rows, case_id, lhs_name, rhs_name, "m", lhs.m, rhs.m,
                 lhs.m == rhs.m, "m=$(lhs.m) vs $(rhs.m)")
    push_status!(rows, case_id, lhs_name, rhs_name, "V_shape", lhs.m, rhs.m,
                 size(lhs.V) == size(rhs.V),
                 "$(shape_string(lhs.V)) vs $(shape_string(rhs.V))")
    push_status!(rows, case_id, lhs_name, rhs_name, "H_shape", lhs.m, rhs.m,
                 size(lhs.H) == size(rhs.H),
                 "$(shape_string(lhs.H)) vs $(shape_string(rhs.H))")
    push_scalar_cmp!(rows, opts, case_id, lhs_name, rhs_name, "beta",
                     lhs.m, rhs.m, lhs.beta, rhs.beta)

    common_m = min(lhs.m, rhs.m)
    vcols = min(size(lhs.V, 2), size(rhs.V, 2), common_m + 1)
    hrows = min(size(lhs.H, 1), size(rhs.H, 1), common_m + 1)
    hcols = min(size(lhs.H, 2), size(rhs.H, 2), common_m)
    push_array_cmp!(rows, opts, case_id, lhs_name, rhs_name, "V",
                    lhs.m, rhs.m, lhs.V[:, 1:vcols], rhs.V[:, 1:vcols])
    push_array_cmp!(rows, opts, case_id, lhs_name, rhs_name, "H",
                    lhs.m, rhs.m, lhs.H[1:hrows, 1:hcols],
                    rhs.H[1:hrows, 1:hcols])
    return nothing
end

function compare_vector_result!(rows, opts::Options, case_id, lhs_name, lhs,
                                rhs_name, rhs)
    push_scalar_cmp!(rows, opts, case_id, lhs_name, rhs_name, "lambda",
                     "", "", lhs.lambda, rhs.lambda)
    push_status!(rows, case_id, lhs_name, rhs_name, "y_shape", "", "",
                 size(lhs.y) == size(rhs.y),
                 "$(shape_string(lhs.y)) vs $(shape_string(rhs.y))")
    push_array_cmp!(rows, opts, case_id, lhs_name, rhs_name, "y", "", "",
                    lhs.y, rhs.y)
    return nothing
end

function push_metric!(rows, opts::Options, case_id, lhs_name, rhs_name,
                      component, lhs_m, rhs_m, value; scale::Real=1.0,
                      detail="")
    val = Float64(value)
    tol = max(opts.atol, opts.rtol * max(1.0, Float64(scale)))
    status = isfinite(val) && val <= tol ? "pass" : "fail"
    push!(rows, row(case_id, lhs_name, rhs_name, component, lhs_m, rhs_m,
                    val, tol, status, detail))
    return nothing
end

function arnoldi_relation_residual(result, apply, dims::Tuple)
    m = Int(result.m)
    m > 0 || error("empty Arnoldi basis")
    V = result.V
    H = result.H
    AV = Matrix{Float64}(undef, size(V, 1), m)
    for j in 1:m
        AV[:, j] = vec(apply(reshape(V[:, j], dims)))
    end
    VH = V[:, 1:(m + 1)] * H[1:(m + 1), 1:m]
    return norm(AV .- VH, Inf) / max(norm(AV, Inf), norm(VH, Inf), 1.0)
end

function arnoldi_orthogonality_residual(result)
    m = Int(result.m)
    m > 0 || error("empty Arnoldi basis")
    V = result.V[:, 1:m]
    return norm(Base.transpose(V) * V - I, Inf)
end

function eigenpair_residual(result, apply, dims::Tuple)
    y = vec(result.y)
    fy = vec(apply(reshape(y, dims)))
    λ = Float64(result.lambda)
    return norm(fy .- λ .* y) / max(norm(fy), abs(λ) * norm(y), norm(y), 1.0)
end

function compare_arnoldi_to_reference!(rows, opts::Options, case_id, result,
                                       apply, dims::Tuple, seed)
    push_metric!(rows, opts, case_id, "TenetNative", "JuliaReference",
                 "arnoldi_relation_relres", result.m, result.m,
                 arnoldi_relation_residual(result, apply, dims))
    push_metric!(rows, opts, case_id, "TenetNative", "JuliaReference",
                 "orthogonality_inf", result.m, result.m,
                 arnoldi_orthogonality_residual(result))
    push_scalar_cmp!(rows, opts, case_id, "TenetNative", "JuliaReference",
                     "beta", result.m, result.m, result.beta, 1.0)
    return nothing
end

function compare_eigenpair_to_reference!(rows, opts::Options, case_id, result,
                                         apply, dims::Tuple; target::Symbol)
    ref = target_eigenvalue(apply, dims; target)
    imag_ref = abs(imag(ref))
    push_metric!(rows, opts, case_id, "JuliaReference", "real-spectrum",
                 "target_imag_abs", "", "", imag_ref)
    push_scalar_cmp!(rows, opts, case_id, "TenetNative", "JuliaReference",
                     "lambda", "", "", Float64(result.lambda), real(ref))
    push_metric!(rows, opts, case_id, "TenetNative", "JuliaReference",
                 "eigenpair_relres", "", "",
                 eigenpair_residual(result, apply, dims))
    return nothing
end

function tenet_basis(TN, symbol::Symbol, args...; kwargs...)
    return as_basis(Base.invokelatest(getproperty(TN, symbol), args...; kwargs...))
end

function tenet_call(TN, symbol::Symbol, args...; kwargs...)
    return Base.invokelatest(getproperty(TN, symbol), args...; kwargs...)
end

function fast_basis(FT, symbol::Symbol, args...; kwargs...)
    return as_basis(Base.invokelatest(getproperty(FT, symbol), args...; kwargs...))
end

function exact_basis(EL, symbol::Symbol, args...; kwargs...)
    return as_basis(Base.invokelatest(getproperty(EL, symbol), args...; kwargs...))
end

function run_case!(f, rows, name)
    before = length(rows)
    try
        f()
    catch err
        push!(rows, row(name, "suite", "suite", "exception", "", "", nothing,
                        nothing, "fail", sprint(showerror, err)))
    end
    length(rows) > before || push!(rows, row(name, "suite", "suite", "empty",
                                             "", "", nothing, nothing, "fail",
                                             "case produced no rows"))
    return nothing
end

function csv_escape(x)
    s = string(x)
    needs_quote = occursin(",", s) || occursin("\"", s) || occursin("\n", s)
    s = replace(s, "\"" => "\"\"")
    return needs_quote ? "\"" * s * "\"" : s
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

function md_escape(x)
    return replace(replace(string(x), "|" => "\\|"), "\n" => " ")
end

function write_markdown(path, rows, opts::Options, libpath)
    nfail = count(r -> r["status"] != "pass", rows)
    open(path, "w") do io
        println(io, "# TenetNative CPU Native Parity")
        println(io)
        println(io, "- status: ", nfail == 0 ? "pass" : "fail")
        println(io, "- repo: `", opts.repo, "`")
        println(io, "- native_library: `", libpath, "`")
        println(io, "- seed: `", opts.seed, "`")
        println(io, "- chi: `", opts.chi, "`")
        println(io, "- phys: `", opts.phys, "`")
        println(io, "- max_k: `", opts.max_k, "`")
        println(io, "- atol: `", opts.atol, "`")
        println(io, "- rtol: `", opts.rtol, "`")
        println(io)
        println(io, "| Case | LHS | RHS | Component | m | max abs diff | tolerance | Status | Detail |")
        println(io, "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |")
        for r in rows
            m = isempty(r["lhs_m"]) && isempty(r["rhs_m"]) ? "" : r["lhs_m"] * "/" * r["rhs_m"]
            println(io, "| ", join((
                md_escape(r["case"]),
                md_escape(r["lhs"]),
                md_escape(r["rhs"]),
                md_escape(r["component"]),
                md_escape(m),
                md_escape(r["max_abs_diff"]),
                md_escape(r["tolerance"]),
                md_escape(r["status"]),
                md_escape(r["detail"]),
            ), " | "), " |")
        end
    end
    return path
end

function all_pass(rows)
    return all(r -> r["status"] == "pass", rows)
end

function main(argv)
    opts = parse_args(argv)
    guard_scratch_paths(opts)
    mkpath(opts.outdir)
    libpath = resolve_native_lib(opts)
    add_load_path!(opts.repo)

    TN = load_module(:TenetNative)

    data = inputs(opts.seed, opts.chi, opts.phys)
    k = full_k(opts, opts.chi * opts.chi)
    k3 = full_k(opts, opts.chi * opts.phys * opts.chi)
    tol = 1e-12
    dims2 = (opts.chi, opts.chi)
    dims3 = (opts.chi, opts.phys, opts.chi)
    two = X -> two_layer_apply(data.Aup, data.Adn, X)
    two_t = X -> two_layer_apply(data.Aup, data.Adn, X; transpose=true)
    projected = X -> projected_two_layer_apply(data.W, data.W, data.rho, X)
    projected_t = X -> projected_two_layer_apply(data.W, data.W, data.rho, X;
                                                 transpose=true)
    qprojected = X -> qprojected_two_layer_apply(data.W, data.W, data.rho, X)
    qprojected_t = X -> qprojected_two_layer_apply(data.W, data.W, data.rho, X;
                                                   transpose=true)
    three = X -> three_layer_apply(data.Aup, data.Adn, data.M, X)
    three_t = X -> three_layer_apply(data.Aup, data.Adn, data.M, X;
                                     transpose=true)

    rows = Vector{Dict{String,String}}()

    run_case!(rows, "two_layer_arnoldi") do
        result = tenet_basis(TN, :tenet_native_arnoldi_two_layer_d_cpu,
                             data.Aup, data.Adn, data.x0;
                             max_k=k, breakdown_tol=tol, lib=libpath)
        compare_arnoldi_to_reference!(rows, opts, "two_layer_arnoldi",
                                      result, two, dims2, data.x0)
    end

    run_case!(rows, "two_layer_arnoldi_transpose") do
        result = tenet_basis(TN, :tenet_native_arnoldi_two_layer_d_cpu,
                             data.Aup, data.Adn, data.x0;
                             max_k=k, breakdown_tol=tol, transpose=true,
                             lib=libpath)
        compare_arnoldi_to_reference!(rows, opts, "two_layer_arnoldi_transpose",
                                      result, two_t, dims2, data.x0)
    end

    run_case!(rows, "two_layer_ritz") do
        got = tenet_call(TN, :tenet_native_arnoldi_two_layer_ritz_d_cpu,
            data.Aup, data.Adn, data.x0;
            max_k=k, breakdown_tol=tol, nvalues=2, lib=libpath,
        )
        ref = leading_eigenvalues(two, dims2, length(got.values);
                                  target=:largest_magnitude)
        push_array_cmp!(rows, opts, "two_layer_ritz", "TenetNative",
                        "JuliaReference", "values", got.m, length(ref),
                        collect(got.values), collect(ref))
    end

    run_case!(rows, "dominant_two_layer") do
        result = tenet_call(TN, :tenet_native_dominant_two_layer_d_cpu,
            data.Aup, data.Adn, data.x0;
            max_k=k, breakdown_tol=tol, lib=libpath,
        )
        compare_eigenpair_to_reference!(rows, opts, "dominant_two_layer",
                                        result, two, dims2;
                                        target=:largest_magnitude)
    end

    run_case!(rows, "smallest_real_two_layer") do
        result = tenet_call(TN, :tenet_native_smallest_real_two_layer_d_cpu,
            data.Aup, data.Adn, data.x0;
            max_k=k, breakdown_tol=tol, lib=libpath,
        )
        compare_eigenpair_to_reference!(rows, opts, "smallest_real_two_layer",
                                        result, two, dims2;
                                        target=:smallest_real)
    end

    run_case!(rows, "projected_two_layer_arnoldi") do
        result = tenet_basis(TN, :tenet_native_arnoldi_projected_two_layer_d_cpu,
                             data.W, data.W, data.rho, data.x0;
                             max_k=k, breakdown_tol=tol, lib=libpath)
        compare_arnoldi_to_reference!(rows, opts, "projected_two_layer_arnoldi",
                                      result, projected, dims2, data.x0)
    end

    run_case!(rows, "projected_two_layer_arnoldi_transpose") do
        result = tenet_basis(TN, :tenet_native_arnoldi_projected_two_layer_d_cpu,
                             data.W, data.W, data.rho, data.x0;
                             max_k=k, breakdown_tol=tol, transpose=true,
                             lib=libpath)
        compare_arnoldi_to_reference!(rows, opts,
                                      "projected_two_layer_arnoldi_transpose",
                                      result, projected_t, dims2, data.x0)
    end

    run_case!(rows, "qprojected_two_layer_arnoldi") do
        result = tenet_basis(TN, :tenet_native_arnoldi_qprojected_two_layer_d_cpu,
                             data.W, data.W, data.rho, data.x0;
                             max_k=k, breakdown_tol=tol, lib=libpath)
        compare_arnoldi_to_reference!(rows, opts, "qprojected_two_layer_arnoldi",
                                      result, qprojected, dims2, data.x0)
    end

    run_case!(rows, "qprojected_two_layer_arnoldi_transpose") do
        result = tenet_basis(TN, :tenet_native_arnoldi_qprojected_two_layer_d_cpu,
                             data.W, data.W, data.rho, data.x0;
                             max_k=k, breakdown_tol=tol, transpose=true,
                             lib=libpath)
        compare_arnoldi_to_reference!(rows, opts,
                                      "qprojected_two_layer_arnoldi_transpose",
                                      result, qprojected_t, dims2, data.x0)
    end

    run_case!(rows, "three_layer_leg4_arnoldi") do
        result = tenet_basis(TN, :tenet_native_arnoldi_three_layer_leg4_d_cpu,
                             data.Aup, data.Adn, data.M, data.x3;
                             max_k=k3, breakdown_tol=tol, lib=libpath)
        compare_arnoldi_to_reference!(rows, opts, "three_layer_leg4_arnoldi",
                                      result, three, dims3, data.x3)
    end

    run_case!(rows, "three_layer_leg4_arnoldi_transpose") do
        result = tenet_basis(TN, :tenet_native_arnoldi_three_layer_leg4_d_cpu,
                             data.Aup, data.Adn, data.M, data.x3;
                             max_k=k3, breakdown_tol=tol, transpose=true,
                             lib=libpath)
        compare_arnoldi_to_reference!(rows, opts,
                                      "three_layer_leg4_arnoldi_transpose",
                                      result, three_t, dims3, data.x3)
    end

    run_case!(rows, "dominant_three_layer_leg4") do
        result = tenet_call(TN, :tenet_native_dominant_three_layer_leg4_d_cpu,
            data.Aup, data.Adn, data.M, data.x3;
            max_k=k3, breakdown_tol=tol, lib=libpath,
        )
        compare_eigenpair_to_reference!(rows, opts, "dominant_three_layer_leg4",
                                        result, three, dims3;
                                        target=:largest_magnitude)
    end

    run_case!(rows, "smallest_real_three_layer_leg4") do
        result = tenet_call(TN, :tenet_native_smallest_real_three_layer_leg4_d_cpu,
            data.Aup, data.Adn, data.M, data.x3;
            max_k=k3, breakdown_tol=tol, lib=libpath,
        )
        compare_eigenpair_to_reference!(rows, opts,
                                        "smallest_real_three_layer_leg4",
                                        result, three, dims3;
                                        target=:smallest_real)
    end

    csv_path = write_csv(joinpath(opts.outdir, "native_cpu_parity.csv"), rows)
    md_path = write_markdown(joinpath(opts.outdir, "native_cpu_parity.md"),
                             rows, opts, libpath)

    println("TENET_NATIVE_CPU_PARITY artifacts csv=$csv_path markdown=$md_path")
    if all_pass(rows)
        println("TENET_NATIVE_CPU_PARITY_DONE status=pass rows=$(length(rows)) outdir=$(opts.outdir)")
    else
        nfail = count(r -> r["status"] != "pass", rows)
        println("TENET_NATIVE_CPU_PARITY_DONE status=fail rows=$(length(rows)) failures=$nfail outdir=$(opts.outdir)")
        error("native CPU parity failed with $nfail failing rows; see $csv_path and $md_path")
    end
    return nothing
end

main(ARGS)
