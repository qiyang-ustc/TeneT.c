module FastTeneT

using Base.Threads: @spawn, @sync, threadid
using CUDA
using LinearAlgebra
using Libdl
using Parameters
using Printf
using Random
using TensorOperations
using Zygote
using cuTENSOR

import Base: +, -, *, getindex, Array
import CUDA: CuArray
import LinearAlgebra: norm
import VectorInterface: scale!!, scalartype, zerovector
import TenetNative

include("internal/defaults.jl")
include("internal/structarray/base.jl")
include("internal/structarray/initial.jl")
include("internal/structarray/buffer.jl")
include("internal/utilities.jl")
include("internal/initial_env.jl")
include("internal/contraction/basic.jl")
include("internal/contraction/forloop.jl")
include("internal/environment.jl")
include("internal/vumpsruntime.jl")
include("internal/native_arnoldi.jl")
include("internal/tfising_vumps.jl")

export BoundaryResult,
    SINGLE_UNITCELL_PATTERN,
    StructArray,
    TFIsingResult,
    TFIsingVUMPSEnv,
    TFIsingVUMPSState,
    VUMPS,
    VUMPSEnv,
    VUMPSRuntime,
    build_native_arnoldi,
    critical_beta,
    critical_tfim_field,
    energy_density,
    free_energy_density,
    ising_network,
    ising_tensor,
    log_partition_density,
    log_partition_density_exact,
    magnetization,
    magnetization_exact,
    native_eigsolve,
    native_krylov_capabilities,
    native_linsolve,
    energy_density_exact,
    run_boundary,
    run_tfising_vumps,
    tfising_energy_density,
    tfising_ground_state_energy_density_exact,
    tfising_mpo_tensor,
    tfising_network,
    tfim_ground_state_energy_density_exact,
    vumps_algorithm

const SINGLE_UNITCELL_PATTERN = [1;;]

struct BoundaryResult
    beta::Float64
    chi::Int
    network::StructArray
    alg::VUMPS
    runtime
    env::VUMPSEnv
    error
end

struct TFIsingResult
    field::Float64
    chi::Int
    network::StructArray
    energy_density::Float64
    exact_energy_density::Float64
    abs_energy_error::Float64
    error::Float64
    state
    env
    iterations::Int
    converged::Bool
end

critical_beta() = log1p(sqrt(2.0)) / 2
critical_tfim_field() = 1.0

function magnetization_exact(beta::Real)
    beta64 = Float64(beta)
    return beta64 > critical_beta() ? (1 - sinh(2 * beta64)^-4)^(1 / 8) : 0.0
end

function ising_tensor(beta::Real; kind::Symbol=:bulk)
    beta64 = Float64(beta)
    if kind === :bulk
        return _bulk_tensor(beta64)
    elseif kind === :energy
        return _energy_tensor(beta64)
    elseif kind === :mag || kind === :magnetization
        return _magnetization_tensor(beta64)
    end
    throw(ArgumentError("unsupported tensor kind $kind; expected :bulk, :energy, or :mag"))
end

function ising_network(beta::Real; kind::Symbol=:bulk, arraytype::Type=Array)
    return StructArray([arraytype(ising_tensor(beta; kind))], copy(SINGLE_UNITCELL_PATTERN))
end

function tfising_mpo_tensor(field::Real)
    h = Float64(field)
    h >= 0 || throw(DomainError(field, "transverse field must be nonnegative"))
    id = Float64[1 0; 0 1]
    x = Float64[0 1; 1 0]
    z = Float64[1 0; 0 -1]
    mpo = zeros(Float64, 3, 2, 3, 2)
    mpo[1, :, 1, :] .= id
    mpo[2, :, 1, :] .= z
    mpo[3, :, 1, :] .= -h .* x
    mpo[3, :, 2, :] .= -z
    mpo[3, :, 3, :] .= id
    return mpo
end

function tfising_network(field::Real; arraytype::Type=Array)
    return StructArray([arraytype(tfising_mpo_tensor(field))], copy(SINGLE_UNITCELL_PATTERN))
end

function _resolve_krylovdim(new_value, legacy_value, default::Integer;
                            new_name::AbstractString,
                            legacy_name::AbstractString)
    if new_value !== nothing && legacy_value !== nothing &&
       Int(new_value) != Int(legacy_value)
        throw(ArgumentError("$new_name and $legacy_name disagree"))
    end
    value = new_value === nothing ?
        (legacy_value === nothing ? default : legacy_value) :
        new_value
    return Int(value)
end

function vumps_algorithm(;
    tol::Real=1e-10,
    maxiter::Integer=100,
    miniter::Integer=1,
    maxiter_ad::Integer=0,
    miniter_ad::Integer=0,
    verbosity::Integer=0,
    ifupdown::Bool=false,
    ifdownfromup::Bool=true,
    ifparallelupdown::Bool=false,
    native_arnoldi_krylovdim=nothing,
    native_arnoldi_maxiter=nothing,
    native_arnoldi_tol::Real=1e-12,
    native_arnoldi_check_residual::Bool=true,
    native_arnoldi_residual_tol::Real=1e-8,
    kwargs...,
)
    haskey(kwargs, :ifsimple_eig) &&
        throw(ArgumentError("FastTeneT always sets VUMPS(ifsimple_eig=false, ...)"))
    haskey(kwargs, :eig_solver) &&
        throw(ArgumentError("FastTeneT always sets VUMPS(eig_solver=:native_arnoldi, ...)"))

    krylovdim = _resolve_krylovdim(
        native_arnoldi_krylovdim,
        native_arnoldi_maxiter,
        30;
        new_name="native_arnoldi_krylovdim",
        legacy_name="native_arnoldi_maxiter",
    )

    return VUMPS(;
        tol=Float64(tol),
        maxiter=Int(maxiter),
        miniter=Int(miniter),
        maxiter_ad=Int(maxiter_ad),
        miniter_ad=Int(miniter_ad),
        verbosity=Int(verbosity),
        ifupdown,
        ifdownfromup,
        ifparallelupdown,
        ifsimple_eig=false,
        eig_solver=:native_arnoldi,
        native_arnoldi_maxiter=krylovdim,
        native_arnoldi_tol=Float64(native_arnoldi_tol),
        native_arnoldi_check_residual,
        native_arnoldi_residual_tol=Float64(native_arnoldi_residual_tol),
        kwargs...,
    )
end

function run_boundary(beta::Real; chi::Integer=4, alg=nothing, arraytype::Type=Array, kwargs...)
    chi >= 1 || throw(ArgumentError("chi must be positive"))
    if alg === nothing
        alg = vumps_algorithm(; kwargs...)
    elseif !isempty(kwargs)
        throw(ArgumentError("pass VUMPS options either through alg or keyword arguments, not both"))
    else
        _check_algorithm(alg)
    end

    network = ising_network(beta; kind=:bulk, arraytype)
    runtime = VUMPSRuntime(network, Int(chi), alg)
    runtime, err = leading_boundary(runtime, network, alg)
    env = VUMPSEnv(runtime, network, alg)
    return BoundaryResult(Float64(beta), Int(chi), network, alg, runtime, env, err)
end

function log_partition_density(result::BoundaryResult)
    z = _partition_factor(result)
    return log(real(z))
end

function free_energy_density(result::BoundaryResult)
    iszero(result.beta) && throw(DomainError(result.beta, "free energy density is singular at beta=0"))
    return -log_partition_density(result) / result.beta
end

energy_density(result::BoundaryResult) = local_observable(result, :energy)
magnetization(result::BoundaryResult) = local_observable(result, :mag)

function log_partition_density_exact(beta::Real; panels::Integer=32768)
    beta64 = Float64(beta)
    beta64 > 0 || throw(DomainError(beta, "beta must be positive"))
    n = Int(panels)
    n > 0 && iseven(n) || throw(ArgumentError("panels must be a positive even integer"))

    x = cosh(2 * beta64) * coth(2 * beta64)
    integral = _simpson(theta -> acosh(x - cos(theta)), 0.0, Float64(pi), n)
    return 0.5 * log(2 * sinh(2 * beta64)) + integral / (2 * pi)
end

function energy_density_exact(beta::Real)
    beta64 = Float64(beta)
    beta64 > 0 || throw(DomainError(beta, "beta must be positive"))
    beta_c = critical_beta()
    if abs(beta64 - beta_c) <= 32 * eps(Float64) * max(1.0, abs(beta_c))
        return -sqrt(2.0)
    end

    t = tanh(2 * beta64)
    k = 2 * sinh(2 * beta64) / cosh(2 * beta64)^2
    return -coth(2 * beta64) * (1 + (2 / pi) * (2 * t^2 - 1) * _ellipk_mod(k))
end

function tfim_ground_state_energy_density_exact(field::Real; panels::Integer=32768)
    h = Float64(field)
    h >= 0 || throw(DomainError(field, "transverse field must be nonnegative"))
    n = Int(panels)
    n > 0 && iseven(n) || throw(ArgumentError("panels must be a positive even integer"))

    iszero(h) && return -1.0
    abs(h - critical_tfim_field()) <= 32 * eps(Float64) && return -4 / pi

    integral = _simpson(k -> sqrt(max(0.0, 1 + h^2 - 2 * h * cos(k))), 0.0, Float64(pi), n)
    return -integral / pi
end

tfising_ground_state_energy_density_exact(field::Real; panels::Integer=32768) =
    tfim_ground_state_energy_density_exact(field; panels)

function run_tfising_vumps(field::Real; chi::Integer=16, alg=nothing, arraytype::Type=Array,
                           seed::Integer=1234, eig_tol=nothing,
                           eig_krylovdim=nothing, eig_maxiter=nothing,
                           env_tol=nothing, env_krylovdim=nothing,
                           env_maxiter=nothing, kwargs...)
    chi >= 1 || throw(ArgumentError("chi must be positive"))
    if alg === nothing
        alg = vumps_algorithm(; kwargs...)
    elseif !isempty(kwargs)
        throw(ArgumentError("pass VUMPS options either through alg or keyword arguments, not both"))
    else
        alg isa VUMPS || throw(ArgumentError("alg must be a FastTeneT.VUMPS instance"))
    end
    eig_dim = _resolve_krylovdim(
        eig_krylovdim,
        eig_maxiter,
        alg.native_arnoldi_maxiter;
        new_name="eig_krylovdim",
        legacy_name="eig_maxiter",
    )
    env_dim = _resolve_krylovdim(
        env_krylovdim,
        env_maxiter,
        alg.native_arnoldi_maxiter;
        new_name="env_krylovdim",
        legacy_name="env_maxiter",
    )
    network = tfising_network(field; arraytype)
    state, env, energy, err, iterations, converged = _tfising_vumps(
        network[1, 1], Int(chi), alg;
        seed=Int(seed),
        eig_tol=eig_tol === nothing ? alg.tol : Float64(eig_tol),
        eig_krylovdim=eig_dim,
        env_tol=env_tol === nothing ? max(alg.tol * 0.1, 1e-13) : Float64(env_tol),
        env_krylovdim=env_dim,
    )
    exact = tfising_ground_state_energy_density_exact(field)
    return TFIsingResult(
        Float64(field),
        Int(chi),
        network,
        energy,
        exact,
        abs(energy - exact),
        err,
        state,
        env,
        iterations,
        converged,
    )
end

tfising_energy_density(result::TFIsingResult) = result.energy_density

function local_observable(result::BoundaryResult, kind::Symbol)
    tensor_kind = kind === :magnetization ? :mag : kind
    tensor_kind in (:energy, :mag) ||
        throw(ArgumentError("unsupported observable $kind; expected :energy or :mag"))

    obs_tensor = _array_like(ising_tensor(result.beta; kind=tensor_kind), result.network[1, 1])
    numerator = _cell_contract(result.env, obs_tensor)
    denominator = _cell_contract(result.env, result.network[1, 1])
    value = numerator / denominator
    tensor_kind === :mag && return abs(value)
    return real(value)
end

function _check_algorithm(alg::VUMPS)
    !alg.ifsimple_eig ||
        throw(ArgumentError("FastTeneT requires VUMPS(ifsimple_eig=false, ...)"))
    alg.eig_solver === :native_arnoldi ||
        throw(ArgumentError("FastTeneT requires VUMPS(eig_solver=:native_arnoldi, ...)"))
    return alg
end

_check_algorithm(_) = throw(ArgumentError("alg must be a FastTeneT.VUMPS instance"))

function _bulk_tensor(beta::Float64)
    ham = Float64[-1 1; 1 -1]
    w = exp.(-beta .* ham)
    wsq = sqrt(w)
    tensor = zeros(Float64, 2, 2, 2, 2)

    @inbounds for a in 1:2, b in 1:2, c in 1:2, d in 1:2, i in 1:2
        tensor[a, b, c, d] += wsq[i, a] * wsq[i, b] * wsq[i, c] * wsq[i, d]
    end
    return tensor
end

function _magnetization_tensor(beta::Float64)
    cbeta = sqrt(cosh(beta))
    sbeta = sqrt(sinh(beta))
    q = Float64[
        cbeta + sbeta cbeta - sbeta
        cbeta - sbeta cbeta + sbeta
    ] / sqrt(2)
    tensor = zeros(Float64, 2, 2, 2, 2)

    @inbounds for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        tensor[i, j, k, l] =
            q[1, i] * q[1, j] * q[1, k] * q[1, l] -
            q[2, i] * q[2, j] * q[2, k] * q[2, l]
    end
    return tensor
end

function _energy_tensor(beta::Float64)
    ham = Float64[-1 1; 1 -1]
    w = exp.(-beta .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = inv(wsq)
    tensor = zeros(Float64, 2, 2, 2, 2)

    @inbounds for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        total = 0.0
        for i in 1:2, m in 1:2
            total += wsqi[a, i] * we[i, m] * wsq[b, m] * wsq[c, m] * wsq[d, m]
            total += wsq[a, m] * wsqi[b, i] * we[i, m] * wsq[c, m] * wsq[d, m]
            total += wsq[a, m] * wsq[b, m] * wsqi[c, i] * we[i, m] * wsq[d, m]
            total += wsq[a, m] * wsq[b, m] * wsq[c, m] * wsqi[d, i] * we[i, m]
        end
        tensor[a, b, c, d] = total / 2
    end
    return tensor
end

function _partition_factor(result::BoundaryResult)
    env = result.env
    lambda_flo, _ = rightenv(env.ARu, env.ARu, result.network; ifobs=true, alg=result.alg, ifvalue=true)
    lambda_c, _ = rightCenv(env.ARu, env.ARu; ifobs=true, alg=result.alg, ifvalue=true)
    return _first_scalar(lambda_flo) / _first_scalar(lambda_c)
end

function _cell_contract(env::VUMPSEnv, tensor::AbstractArray)
    return _contract_scalar(
        Array(env.FLo[1, 1]),
        Array(env.ACu[1, 1]),
        Array(tensor),
        Array(env.ACd[1, 1]),
        Array(env.FRo[1, 1]),
    )
end

function _contract_scalar(flo, acu, tensor, acd, fro)
    total = zero(promote_type(eltype(flo), eltype(acu), eltype(tensor), eltype(acd), eltype(fro)))

    @inbounds for a in axes(flo, 1), d in axes(flo, 2), f in axes(flo, 3)
        for b in axes(acu, 2), c in axes(acu, 3), g in axes(tensor, 2), e in axes(tensor, 3), h in axes(acd, 3)
            total += flo[a, d, f] * acu[a, b, c] * tensor[d, g, e, b] * acd[f, g, h] * fro[c, e, h]
        end
    end
    return _scalar(total)
end

_scalar(x::Number) = x
_scalar(x::AbstractArray) = only(x)

_first_scalar(x::AbstractArray) = Array(x)[1]
_first_scalar(x::StructArray) = _scalar(x[1])

function _array_like(x::AbstractArray, template::AbstractArray)
    y = similar(template)
    copyto!(y, x)
    return y
end

function _simpson(f, a::Float64, b::Float64, n::Int)
    h = (b - a) / n
    total = f(a) + f(b)
    @inbounds for i in 1:n-1
        total += (isodd(i) ? 4.0 : 2.0) * f(a + i * h)
    end
    return total * h / 3
end

function _ellipk_mod(k::Float64)
    abs(k) <= 1 + 128 * eps(Float64) || throw(DomainError(k, "elliptic modulus must satisfy |k| <= 1"))
    k = min(abs(k), 1.0)
    k == 1.0 && return Inf

    a = 1.0
    b = sqrt(max(0.0, 1 - k^2))
    for _ in 1:80
        next_a = (a + b) / 2
        b = sqrt(a * b)
        a = next_a
        abs(a - b) <= eps(Float64) * max(1.0, abs(a)) && break
    end
    return pi / (2 * a)
end

end
