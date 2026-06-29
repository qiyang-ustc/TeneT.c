using LinearAlgebra
using Random
using Test
using TenetNative

function _restart_inputs(seed::Integer, chi::Integer, phys::Integer)
    rng = MersenneTwister(seed)
    scale3 = inv(sqrt(Float64(chi * phys)))
    Aup = zeros(Float64, chi, phys, chi)
    for b in 1:phys
        S = scale3 .* randn(rng, chi, chi)
        Aup[:, b, :] .= 0.5 .* (S .+ S')
    end
    Adn = copy(Aup)
    x0 = randn(rng, chi, chi)
    M = zeros(Float64, phys, phys, phys, phys)
    for d in 1:phys, g in 1:phys
        M[d, g, d, g] = 1.0
    end
    x3 = randn(rng, chi, phys, chi)
    return (; Aup, Adn, M, x0, x3)
end

function _two_layer_apply(Aup, Adn, X)
    chi, phys, _ = size(Aup)
    Y = zeros(Float64, chi, chi)
    for b in 1:phys
        A = Aup[:, b, :]
        B = Adn[:, b, :]
        Y .+= transpose(A) * X * B
    end
    return Y
end

function _three_layer_apply(Aup, Adn, M, X)
    chi, phys, _ = size(Aup)
    Y = zeros(Float64, chi, phys, chi)
    for e in 1:phys, b in 1:phys
        accum = zeros(Float64, chi, chi)
        for d in 1:phys, g in 1:phys
            alpha = M[d, g, e, b]
            alpha == 0.0 && continue
            accum .+= alpha .* (X[:, d, :] * Adn[:, g, :])
        end
        Y[:, e, :] .+= transpose(Aup[:, b, :]) * accum
    end
    return Y
end

function _eigenpair_relres(apply, y, lambda)
    yv = vec(y)
    fy = vec(apply(y))
    return norm(fy .- lambda .* yv) /
           max(norm(fy), abs(lambda) * norm(yv), norm(yv), 1.0)
end

@testset "TenetNative CPU wrapper smoke" begin
    prefix = mktempdir()
    lib = build_native_arnoldi(; target=:cpu, prefix)
    @test isfile(lib)
    @test native_arnoldi_library(; lib, target=:cpu, autobuild=false) == lib
    @test tenet_native_abi_version(; lib) == TENET_NATIVE_ABI_VERSION
    @test tenet_native_abi_version_string(; lib) == TENET_NATIVE_ABI_VERSION_STRING

    A = zeros(Float64, 2, 2, 2)
    A[:, 1, :] .= [0.7 0.2; 0.2 -0.3]
    A[:, 2, :] .= [0.1 -0.4; -0.4 0.5]
    x0 = Matrix{Float64}(I, 2, 2)
    raw = tenet_native_raw_two_layer_apply_d_cpu(A, A, x0; lib=lib)
    expected_raw = _two_layer_apply(A, A, x0)
    @test raw ≈ expected_raw atol=1e-12
    basis = tenet_native_arnoldi_two_layer_d_cpu(
        A, A, x0; max_k=4, breakdown_tol=1e-12, lib)
    dominant = tenet_native_dominant_two_layer_d_cpu(
        A, A, x0; max_k=4, breakdown_tol=1e-12, lib)

    @test basis.m >= 1
    @test size(basis.V, 1) == 4
    @test size(basis.H, 1) == 5
    @test basis.beta == 1.0
    @test isfinite(dominant.lambda)
    @test norm(dominant.y) ≈ 1.0 atol=1e-12

    data2 = _restart_inputs(20260626, 32, 2)
    dominant2 = tenet_native_dominant_two_layer_d_cpu(
        data2.Aup, data2.Adn, data2.x0; max_k=8, breakdown_tol=1e-12, lib)
    relres2 = _eigenpair_relres(
        X -> _two_layer_apply(data2.Aup, data2.Adn, X),
        dominant2.y, dominant2.lambda)
    @test relres2 <= 1e-10

    data3 = _restart_inputs(20260626, 16, 2)
    dominant3 = tenet_native_dominant_three_layer_leg4_d_cpu(
        data3.Aup, data3.Adn, data3.M, data3.x3;
        max_k=8, breakdown_tol=1e-12, lib)
    relres3 = _eigenpair_relres(
        X -> _three_layer_apply(data3.Aup, data3.Adn, data3.M, X),
        dominant3.y, dominant3.lambda)
    @test relres3 <= 1e-10
end

include("native_krylov_cpu.jl")
include("native_krylovkit_parity.jl")
