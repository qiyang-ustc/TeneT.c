using LinearAlgebra
using Test
using TenetNative

function _relerr(a, b)
    return norm(a .- b) / max(norm(a), norm(b), 1.0)
end

function _eig_relres(A, λ, v)
    Av = A * vec(v)
    vv = vec(v)
    return norm(Av .- λ .* vv) / max(norm(Av), abs(λ) * norm(vv), norm(vv), 1.0)
end

function _select_ref(A, which::Symbol)
    F = eigen(A)
    order = if which === :LM
        sortperm(abs.(F.values); rev=true)
    elseif which === :SM
        sortperm(abs.(F.values); rev=false)
    elseif which === :LR
        sortperm(real.(F.values); rev=true)
    elseif which === :SR
        sortperm(real.(F.values); rev=false)
    elseif which === :LI
        sortperm(imag.(F.values); rev=true)
    elseif which === :SI
        sortperm(imag.(F.values); rev=false)
    else
        error("unsupported reference selector $which")
    end
    return F.values[first(order)]
end

function _select_refs(A, which::Symbol, howmany::Integer)
    F = eigen(A)
    order = if which === :LM
        sortperm(abs.(F.values); rev=true)
    elseif which === :SM
        sortperm(abs.(F.values); rev=false)
    elseif which === :LR
        sortperm(real.(F.values); rev=true)
    elseif which === :SR
        sortperm(real.(F.values); rev=false)
    elseif which === :LI
        sortperm(imag.(F.values); rev=true)
    elseif which === :SI
        sortperm(imag.(F.values); rev=false)
    else
        error("unsupported reference selector $which")
    end
    return F.values[order[1:howmany]]
end

@testset "TenetNative generic CPU Krylov v1" begin
    prefix = mktempdir()
    lib = build_native_arnoldi(; target=:cpu, prefix)
    caps = native_krylov_capabilities(; lib)
    @test caps.legacy_abi_version == TENET_NATIVE_ABI_VERSION
    @test caps.abi_version == TENET_NATIVE_KRYLOV_ABI_VERSION
    @test caps.generic_cpu_callback
    @test caps.generic_cpu_dense
    @test caps.generic_cpu_cg_callback
    @test caps.generic_cpu_cg_dense
    @test caps.generic_cpu_bicgstab_callback
    @test caps.generic_cpu_bicgstab_dense
    @test !caps.generic_gpu_callback

    Aherm = [3.0 0.2 0.0;
             0.2 2.0 0.1;
             0.0 0.1 1.0]
    x0 = [1.0, 0.4, -0.2]
    vals, vecs, info = native_eigsolve(Aherm, x0, 1, :LM;
                                       krylovdim=3, tol=1e-13, lib,
                                       issymmetric=true, ishermitian=true)
    λref = _select_ref(Aherm, :LM)
    @test abs(vals[1] - λref) <= 1e-11 * max(1.0, abs(λref))
    @test vecs[1] isa Vector{Float64}
    @test _eig_relres(Aherm, vals[1], vecs[1]) <= 1e-11
    @test _relerr(info.residual[1], Aherm * vecs[1] .- vals[1] .* vecs[1]) <= 1e-9
    @test info.converged >= 1
    @test info.scalar === :float64
    @test info.path === :generic_cpu_dense

    @test_throws ArgumentError native_eigsolve(Aherm, x0, 1, :SM;
                                               krylovdim=3, tol=1e-13, lib)

    vals, vecs, info = native_eigsolve(Aherm, x0, 2, :LM;
                                       krylovdim=3, tol=1e-13, lib)
    λrefs = _select_refs(Aherm, :LM, 2)
    @test length(vals) == 2
    @test info.converged == 2
    @test maximum(abs.(sort(real.(vals)) .- sort(real.(λrefs)))) <= 1e-11
    @test maximum(_eig_relres(Aherm, vals[i], vecs[i]) for i in eachindex(vals)) <= 1e-11

    @test_throws ArgumentError native_eigsolve(Aherm, x0, 2, :LM;
                                               krylovdim=2, tol=1e-13, lib)

    Athick = Matrix(Diagonal([5.0, 4.0, 3.0, 2.0, 1.0]))
    Athick[1, 2] = 0.1
    Athick[2, 3] = 0.2
    Athick[3, 4] = 0.3
    Athick[4, 5] = 0.4
    xthick = [1.0, 0.7, -0.2, 0.3, 0.5]
    vals, vecs, info = native_eigsolve(Athick, xthick, 2, :LM;
                                       krylovdim=3, maxiter=80,
                                       tol=1e-12, algorithm=:krylovschur, lib)
    @test length(vals) == 2
    @test info.algorithm === :krylovschur
    @test info.requested_algorithm === :krylovschur
    @test info.thick_restart
    @test info.schur_keep >= 2
    @test info.converged == 2
    @test maximum(abs.(sort(real.(vals)) .- [4.0, 5.0])) <= 1e-10
    @test maximum(_eig_relres(Athick, vals[i], vecs[i]) for i in eachindex(vals)) <= 1e-10

    Anon = [1.0 4.0 0.0;
            0.0 2.0 0.5;
            0.0 0.0 3.0]
    vals, vecs, info = native_eigsolve(x -> Anon * x, x0, 1, :LR;
                                       krylovdim=3, tol=1e-13, lib)
    @test abs(vals[1] - 3.0) <= 1e-11
    @test _eig_relres(Anon, vals[1], vecs[1]) <= 1e-10
    @test _relerr(info.residual[1], Anon * vecs[1] .- vals[1] .* vecs[1]) <= 1e-9
    @test info.numops >= 1
    @test info.path === :generic_cpu_callback

    Zherm = ComplexF64[2.0 0.2 + 0.3im 0.0;
                       0.2 - 0.3im 3.0 0.4im;
                       0.0 -0.4im 1.0]
    z0 = ComplexF64[1.0 + 0.2im, -0.4 + 0.1im, 0.3 - 0.7im]
    vals, vecs, info = native_eigsolve(Zherm, z0, 1, :LM;
                                       krylovdim=3, tol=1e-13, lib,
                                       ishermitian=true)
    λref = _select_ref(Zherm, :LM)
    @test abs(vals[1] - λref) <= 1e-10 * max(1.0, abs(λref))
    @test vecs[1] isa Vector{ComplexF64}
    @test _eig_relres(Zherm, vals[1], vecs[1]) <= 1e-10
    @test info.scalar === :complexf64
    @test info.path === :generic_cpu_dense

    Znormal = Diagonal(ComplexF64[0.3 + 0.4im, 2.0 - 0.7im, -1.0 + 0.1im])
    vals, vecs, info = native_eigsolve(Matrix(Znormal), z0, 1, :LM;
                                       krylovdim=3, tol=1e-13, lib)
    @test abs(vals[1] - (2.0 - 0.7im)) <= 1e-10
    @test _eig_relres(Matrix(Znormal), vals[1], vecs[1]) <= 1e-10

    Abreak = Diagonal([5.0, 3.0, 1.0])
    vals, vecs, info = native_eigsolve(Matrix(Abreak), [1.0, 0.0, 0.0], 1, :LM;
                                       krylovdim=3, tol=1e-13, lib)
    @test vals[1] ≈ 5.0 atol=1e-13
    @test _eig_relres(Matrix(Abreak), vals[1], vecs[1]) <= 1e-13
    @test info.converged == 1
    @test info.numops == 1

    Arepeated = Diagonal([2.0, 2.0, 1.0])
    vals, vecs, info = native_eigsolve(Matrix(Arepeated), [1.0, -2.0, 0.3], 1, :LM;
                                       krylovdim=3, tol=1e-13, lib)
    @test abs(vals[1] - 2.0) <= 1e-12
    @test _eig_relres(Matrix(Arepeated), vals[1], vecs[1]) <= 1e-11

    Acluster = Diagonal([1.0, 1.0 + 1e-9, 3.0])
    vals, vecs, info = native_eigsolve(Matrix(Acluster), [0.7, -0.2, 1.0], 1, :SR;
                                       krylovdim=3, tol=1e-13, lib)
    @test abs(vals[1] - 1.0) <= 1e-8
    @test _eig_relres(Matrix(Acluster), vals[1], vecs[1]) <= 1e-9

    Ajordan = [2.0 1.0 0.0;
               0.0 2.0 0.0;
               0.0 0.0 1.0]
    vals, vecs, info = native_eigsolve(Ajordan, [1.0, 0.5, -0.7], 1, :SR;
                                       krylovdim=3, tol=1e-13, lib)
    @test abs(vals[1] - 1.0) <= 1e-11
    @test _eig_relres(Ajordan, vals[1], vecs[1]) <= 1e-10

    Apair = [0.0 -2.0 0.0;
             2.0  0.0 0.0;
             0.0  0.0 0.5]
    @test_throws ArgumentError native_eigsolve(Apair, [1.0, 0.3, 0.7], 1, :LI;
                                               krylovdim=3, tol=1e-13, lib)
    vals, vecs, info = native_eigsolve(Apair, [1.0, 0.3, 0.7], 1, :LI;
                                       krylovdim=3, tol=1e-13,
                                       scalar=:complexf64, lib)
    λref = _select_ref(Apair, :LI)
    @test abs(vals[1] - λref) <= 1e-11
    @test vecs[1] isa Vector{ComplexF64}
    @test _eig_relres(Apair, vals[1], vecs[1]) <= 1e-10

    vals, vecs, info = native_eigsolve(Znormal, [1.0, 0.3, 0.7], 1, :LI;
                                       krylovdim=3, tol=1e-13, lib)
    @test info.scalar === :complexf64
    @test vecs[1] isa Vector{ComplexF64}
    @test _eig_relres(Matrix(Znormal), vals[1], vecs[1]) <= 1e-10

    vals, vecs, info = native_eigsolve(Aherm, x0, 1, :LM;
                                       krylovdim=1, maxiter=1, tol=1e-16, lib)
    @test info.converged == 0
    @test info.numiter == 1
    @test info.normres[1] > 1e-16

    b = [1.0, -2.0, 0.5]
    x, linfo = native_linsolve(Aherm, b; krylovdim=3, maxiter=4,
                               tol=1e-13, lib)
    xref = Aherm \ b
    @test _relerr(x, xref) <= 1e-10
    @test norm(linfo.residual) <= 1e-11
    @test linfo.converged == 1
    @test linfo.status === :converged
    @test linfo.reason === :converged
    @test linfo.algorithm === :gmres
    @test linfo.tol == 1e-13
    @test linfo.tol_source === :tol
    @test linfo.scalar === :float64
    @test linfo.path === :generic_cpu_dense

    xscaled, sinfo_scaled = native_linsolve(Aherm, b; krylovdim=3,
                                            maxiter=4, atol=1e-8,
                                            rtol=1e-3, lib)
    @test _relerr(xscaled, xref) <= 1e-10
    @test sinfo_scaled.converged == 1
    @test sinfo_scaled.status === :converged
    @test sinfo_scaled.tol ≈ max(1e-8, 1e-3 * norm(b))
    @test sinfo_scaled.atol == 1e-8
    @test sinfo_scaled.rtol == 1e-3
    @test sinfo_scaled.tol_source === :atol_rtol

    x, linfo = native_linsolve(Aherm, b, xref; krylovdim=3, maxiter=4,
                               tol=1e-13, lib)
    @test _relerr(x, xref) <= 1e-12
    @test linfo.converged == 1
    @test linfo.numiter == 0
    @test linfo.normres <= 1e-12

    x, linfo = native_linsolve(Aherm, b; algorithm=:cg, maxiter=6,
                               tol=1e-13, lib)
    @test _relerr(x, xref) <= 1e-10
    @test linfo.normres <= 1e-11
    @test linfo.converged == 1
    @test linfo.path === :generic_cpu_dense

    xshort, short_info = native_linsolve(Aherm, b; krylovdim=1, maxiter=1,
                                         tol=1e-16, lib)
    @test short_info.converged == 0
    @test short_info.status === :not_converged
    @test short_info.reason === :maxiter
    @test short_info.numiter == 1
    @test short_info.normres > 1e-16

    x, linfo = native_linsolve(v -> Aherm * v, b; algorithm=:cg, maxiter=6,
                               tol=1e-13, lib)
    @test _relerr(x, xref) <= 1e-10
    @test linfo.normres <= 1e-11
    @test linfo.converged == 1
    @test linfo.path === :generic_cpu_callback

    shifted = I - 0.2 .* Anon
    x, linfo = native_linsolve(Anon, b, nothing, 1.0, -0.2;
                               krylovdim=3, maxiter=4, tol=1e-13, lib)
    @test _relerr(x, shifted \ b) <= 1e-10
    @test linfo.normres <= 1e-11

    x, linfo = native_linsolve(Anon, b; algorithm=:bicgstab,
                               maxiter=20, tol=1e-13, lib)
    @test _relerr(x, Anon \ b) <= 1e-10
    @test linfo.normres <= 1e-11
    @test linfo.converged == 1
    @test linfo.path === :generic_cpu_dense

    x, linfo = native_linsolve(v -> Anon * v, b, nothing, 1.0, -0.2;
                               algorithm=:bicgstab, maxiter=20,
                               tol=1e-13, lib)
    @test _relerr(x, shifted \ b) <= 1e-10
    @test linfo.normres <= 1e-11
    @test linfo.converged == 1
    @test linfo.path === :generic_cpu_callback

    bz = ComplexF64[1.0 + 0.2im, -0.5 + 0.1im, 0.7 - 0.3im]
    xz, zinfo = native_linsolve(Zherm, bz; krylovdim=3, maxiter=4,
                                tol=1e-13, lib)
    @test _relerr(xz, Zherm \ bz) <= 1e-9
    @test zinfo.normres <= 1e-10
    @test zinfo.scalar === :complexf64

    xz_promoted, zprom_info = native_linsolve(Zherm, b; krylovdim=3,
                                              maxiter=4, tol=1e-13, lib)
    @test xz_promoted isa Vector{ComplexF64}
    @test zprom_info.scalar === :complexf64
    @test _relerr(xz_promoted, Zherm \ ComplexF64.(b)) <= 1e-9

    xz, zinfo = native_linsolve(Zherm, bz; algorithm=:cg, maxiter=8,
                                tol=1e-13, lib)
    @test _relerr(xz, Zherm \ bz) <= 1e-9
    @test zinfo.normres <= 1e-10
    @test zinfo.converged == 1
    @test zinfo.scalar === :complexf64
    @test_throws ArgumentError native_linsolve(
        Zherm, bz, nothing, 0.0 + 0.0im, 1.0 + 0.0im;
        algorithm=:cg, maxiter=8, tol=1e-13, lib)

    Znon = ComplexF64[
        2.0 + 0.4im  0.3 - 0.2im  0.1;
        0.0          1.5 - 0.3im  0.2 + 0.1im;
        0.0          0.0          2.5 + 0.2im
    ]
    zshift = (0.5 + 0.1im) .* I + (0.8 - 0.2im) .* Znon
    xz, zinfo = native_linsolve(Znon, bz, nothing, 0.5 + 0.1im, 0.8 - 0.2im;
                                algorithm=:bicgstab, maxiter=20,
                                tol=1e-12, lib)
    @test _relerr(xz, zshift \ bz) <= 1e-9
    @test zinfo.normres <= 1e-10
    @test zinfo.converged == 1
    @test zinfo.scalar === :complexf64

    xshift, sinfo = native_linsolve(Aherm, b, nothing, 0.5 + 0.1im, 1.0;
                                    krylovdim=3, maxiter=4, tol=1e-13, lib)
    @test xshift isa Vector{ComplexF64}
    @test sinfo.scalar === :complexf64
    @test _relerr(xshift, ((0.5 + 0.1im) * I + Aherm) \ ComplexF64.(b)) <= 1e-9

    Hill = [1.0          0.5          1/3          0.25;
            0.5          1/3          0.25         0.2;
            1/3          0.25         0.2          1/6;
            0.25         0.2          1/6          1/7]
    bill = [1.0, -0.25, 0.5, 0.75]
    xill, iinfo = native_linsolve(Hill, bill; krylovdim=4, maxiter=8,
                                  tol=1e-13, lib)
    @test _relerr(xill, Hill \ bill) <= 1e-8
    @test iinfo.converged == 1
    @test iinfo.normres <= 1e-9

    zero_sol, zero_info = native_linsolve(Aherm, zeros(3); krylovdim=3,
                                          maxiter=2, tol=1e-13, lib)
    @test norm(zero_sol) <= 1e-14
    @test zero_info.converged == 1

    singular_sol, singular_info = native_linsolve(zeros(3, 3), b;
                                                  krylovdim=2, maxiter=3,
                                                  tol=1e-13, lib)
    @test norm(singular_sol) <= 1e-14
    @test singular_info.converged == 0
    @test singular_info.status === :not_converged
    @test singular_info.reason === :breakdown_or_stagnation
    @test singular_info.normres ≈ norm(b)

    @test_throws ArgumentError native_eigsolve(Aherm, x0; backend=:cuda, lib)
end
