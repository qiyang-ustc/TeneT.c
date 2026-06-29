using Test
using FastTeneT
using LinearAlgebra
using Random

@testset "single-site tensor construction" begin
    beta = 0.5
    tensor = ising_tensor(beta)
    ham = Float64[-1 1; 1 -1]
    wsq = sqrt(exp.(-beta .* ham))

    @test size(tensor) == (2, 2, 2, 2)
    @test eltype(tensor) === Float64
    @test FastTeneT.SINGLE_UNITCELL_PATTERN == [1;;]
    @test tensor[1, 1, 1, 1] ≈ sum(wsq[i, 1]^4 for i in 1:2)
    @test tensor[1, 2, 1, 2] ≈ sum(wsq[i, 1] * wsq[i, 2] * wsq[i, 1] * wsq[i, 2] for i in 1:2)

    network = ising_network(beta)
    @test network isa FastTeneT.StructArray
    @test network.pattern == [1;;]
    @test length(network) == 1
    @test network[1, 1] ≈ tensor

    @test eltype(ising_tensor(beta; kind=:energy)) === Float64
    @test eltype(ising_tensor(beta; kind=:mag)) === Float64
    @test size(ising_tensor(beta; kind=:energy)) == (2, 2, 2, 2)
    @test size(ising_tensor(beta; kind=:mag)) == (2, 2, 2, 2)
    @test magnetization_exact(0.1) == 0.0
    @test magnetization_exact(1.0) > 0.0
    @test_throws ArgumentError ising_tensor(beta; kind=:plaquette)
end

@testset "Onsager exact references" begin
    beta = 0.5
    beta_c = critical_beta()

    @test log_partition_density_exact(beta) ≈ 1.0257928126949176 atol=1e-13
    @test energy_density_exact(beta) ≈ -1.745564575312554 atol=1e-13
    @test log_partition_density_exact(beta_c) ≈ 0.5 * log(2) + 2 * 0.915965594177219 / pi atol=1e-13
    @test energy_density_exact(beta_c) ≈ -sqrt(2) atol=1e-14
    @test_throws DomainError log_partition_density_exact(0.0)
    @test_throws DomainError energy_density_exact(0.0)
    @test_throws ArgumentError log_partition_density_exact(beta; panels=3)
end

@testset "nearest-neighbor TFIM exact reference" begin
    field_c = critical_tfim_field()

    @test field_c == 1.0
    @test tfim_ground_state_energy_density_exact(0.0) == -1.0
    @test tfim_ground_state_energy_density_exact(field_c) ≈ -4 / pi atol=1e-15
    @test_throws DomainError tfim_ground_state_energy_density_exact(-0.1)
    @test_throws ArgumentError tfim_ground_state_energy_density_exact(field_c; panels=3)

    for field in (field_c - 0.01, field_c + 0.01)
        reference = tfim_ground_state_energy_density_exact(field; panels=131072)
        @test abs(tfim_ground_state_energy_density_exact(field; panels=32768) - reference) < 1e-12
    end
end

@testset "nearest-neighbor TFIsing VUMPS" begin
    mpo = tfising_mpo_tensor(1.0)
    @test size(mpo) == (3, 2, 3, 2)
    @test eltype(mpo) === Float64
    @test_throws DomainError tfising_mpo_tensor(-0.1)

    network = tfising_network(1.0)
    @test network isa FastTeneT.StructArray
    @test network.pattern == [1;;]
    @test network[1, 1] ≈ mpo

    result = run_tfising_vumps(1.0; chi=4, maxiter=12, miniter=2, maxiter_ad=0,
                               tol=1e-7, eig_maxiter=64, env_maxiter=64)
    @test result.field == 1.0
    @test result.chi == 4
    @test isfinite(tfising_energy_density(result))
    @test result.exact_energy_density ≈ -4 / pi atol=1e-15
    @test result.abs_energy_error < 1e-3
    @test result.state isa FastTeneT.TFIsingVUMPSState
    @test result.env isa FastTeneT.TFIsingVUMPSEnv

    for field in (0.5, 2.0)
        r = run_tfising_vumps(field; chi=4, maxiter=12, miniter=2, maxiter_ad=0,
                              tol=1e-7, eig_maxiter=64, env_maxiter=64)
        @test abs(tfising_energy_density(r) - tfising_ground_state_energy_density_exact(field)) < 1e-5
    end
end

@testset "VUMPS main-env Krylov guard" begin
    alg = vumps_algorithm(maxiter=1, maxiter_ad=0, verbosity=0)
    @test alg isa FastTeneT.VUMPS
    @test alg.ifsimple_eig === false
    @test alg.eig_solver === :native_arnoldi
    @test_throws ArgumentError vumps_algorithm(ifsimple_eig=true)
    @test_throws ArgumentError vumps_algorithm(eig_solver=:krylovkit)
    @test vumps_algorithm(native_arnoldi_krylovdim=33).native_arnoldi_maxiter == 33
    @test vumps_algorithm(native_arnoldi_maxiter=34).native_arnoldi_maxiter == 34
    @test_throws ArgumentError vumps_algorithm(native_arnoldi_krylovdim=33,
                                               native_arnoldi_maxiter=34)
    @test_throws ArgumentError run_tfising_vumps(1.0; chi=2, maxiter=0,
                                                 eig_krylovdim=4, eig_maxiter=5)
    @test_throws ArgumentError run_tfising_vumps(1.0; chi=2, maxiter=0,
                                                 env_krylovdim=4, env_maxiter=5)

    source = replace(read(pathof(FastTeneT), String), r"\s+" => "")
    @test occursin("return VUMPS(;", read(pathof(FastTeneT), String))
    @test occursin("ifsimple_eig=false", source)
    @test occursin("eig_solver=:native_arnoldi", source)

    fasttenet_environment = read(joinpath(dirname(pathof(FastTeneT)), "internal", "environment.jl"), String)
    @test occursin("alg.eig_solver===:native_arnoldi", replace(fasttenet_environment, r"\s+" => ""))

    srcdir = dirname(pathof(FastTeneT))
    production_source = join(read.(filter(endswith(".jl"), readdir(srcdir; join=true)), String), "\n")
    production_source *= "\n" * join(read.(filter(endswith(".jl"), readdir(joinpath(srcdir, "internal"); join=true)), String), "\n")
    @test !occursin("KrylovKit", production_source)
    @test !occursin("GMRES", production_source)
    @test isdefined(FastTeneT, :native_eigsolve)
    @test isdefined(FastTeneT, :native_linsolve)
end

@testset "native Ising full-step parity" begin
    Random.seed!(20260625)
    network = ising_network(critical_beta())
    alg = vumps_algorithm(;
        maxiter=1,
        miniter=1,
        maxiter_ad=0,
        miniter_ad=0,
        verbosity=0,
        native_arnoldi_maxiter=64,
        native_arnoldi_tol=1e-12,
        native_arnoldi_check_residual=false,
    )
    runtime = VUMPSRuntime(network, 8, alg)

    old_disable = get(ENV, "FASTTENET_DISABLE_NATIVE_FULL_STEP", nothing)
    try
        ENV["FASTTENET_DISABLE_NATIVE_FULL_STEP"] = "1"
        ref_runtime, ref_err = FastTeneT.vumps_step_Hermitian(runtime, network, alg)
        delete!(ENV, "FASTTENET_DISABLE_NATIVE_FULL_STEP")
        got_runtime, got_err = FastTeneT.vumps_step_Hermitian(runtime, network, alg)

        rel(a, b) = norm(a .- b) / max(norm(a), norm(b), 1.0)
        signed_rel(a, b) = min(norm(a .- b), norm(a .+ b)) / max(norm(a), norm(b), 1.0)

        @test got_err ≈ ref_err atol=1e-12 rtol=1e-10
        @test signed_rel(got_runtime.AL[1, 1], ref_runtime.AL[1, 1]) < 1e-7
        @test signed_rel(got_runtime.AR[1, 1], ref_runtime.AR[1, 1]) < 1e-7
        @test rel(got_runtime.C[1, 1], ref_runtime.C[1, 1]) < 1e-12
        @test rel(got_runtime.FL[1, 1], ref_runtime.FL[1, 1]) < 1e-12
        @test rel(got_runtime.FR[1, 1], ref_runtime.FR[1, 1]) < 1e-12
    finally
        if old_disable === nothing
            delete!(ENV, "FASTTENET_DISABLE_NATIVE_FULL_STEP")
        else
            ENV["FASTTENET_DISABLE_NATIVE_FULL_STEP"] = old_disable
        end
    end
end

@testset "native checked residual tolerance gate" begin
    network = ising_network(critical_beta())
    alg = vumps_algorithm(;
        maxiter=1,
        miniter=1,
        maxiter_ad=0,
        miniter_ad=0,
        verbosity=0,
        native_arnoldi_maxiter=1,
        native_arnoldi_tol=0.0,
        native_arnoldi_check_residual=true,
        native_arnoldi_residual_tol=Inf,
    )
    runtime = VUMPSRuntime(network, 4, alg)
    stepped, err = FastTeneT._native_ising_vumps_step_cpu(runtime, network, alg)
    @test stepped isa FastTeneT.VUMPSRuntime
    @test isfinite(err)
end

@testset "optional VUMPS boundary smoke" begin
    if get(ENV, "FASTTENET_RUN_VUMPS_SMOKE", get(ENV, "TENETMINIMAL_RUN_VUMPS_SMOKE", "0")) == "1"
        result = run_boundary(0.3; chi=2, maxiter=1, miniter=1, maxiter_ad=0, verbosity=0)
        @test result.alg.ifsimple_eig === false
        @test result.alg.eig_solver === :native_arnoldi
        @test isfinite(log_partition_density(result))
        @test isfinite(free_energy_density(result))
        @test isfinite(energy_density(result))
        @test isfinite(magnetization(result))

        @test !isdefined(FastTeneT, :run_tfim_boundary)
    else
        @info "Skipping VUMPS boundary smoke; set FASTTENET_RUN_VUMPS_SMOKE=1 to run it"
        @test true
    end
end

@testset "optional Onsager alignment" begin
    if get(ENV, "FASTTENET_RUN_ALIGNMENT", get(ENV, "TENETMINIMAL_RUN_ALIGNMENT", "0")) == "1"
        for beta in (0.5, critical_beta() - 0.01, critical_beta() + 0.01)
            result = run_boundary(beta; chi=24, maxiter=100, miniter=1, maxiter_ad=0, verbosity=0)

            @test result.alg.ifsimple_eig === false
        @test result.alg.eig_solver === :native_arnoldi
            @test abs(log_partition_density(result) - log_partition_density_exact(beta)) < 1e-8
            @test abs(energy_density(result) - energy_density_exact(beta)) < 1e-8
        end
    else
        @info "Skipping Onsager alignment; set FASTTENET_RUN_ALIGNMENT=1 to run it"
        @test true
    end
end
