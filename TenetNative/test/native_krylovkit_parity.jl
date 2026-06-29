using LinearAlgebra
using Test
using TenetNative
using KrylovKit

function _kk_select(vals, which::Symbol)
    if which === :LM
        return vals[argmax(abs.(vals))]
    elseif which === :SM
        return vals[argmin(abs.(vals))]
    elseif which === :LR
        return vals[argmax(real.(vals))]
    elseif which === :SR
        return vals[argmin(real.(vals))]
    elseif which === :LI
        return vals[argmax(imag.(vals))]
    elseif which === :SI
        return vals[argmin(imag.(vals))]
    end
    error("unsupported selector $which")
end

function _kk_select_many(vals, which::Symbol, howmany::Integer)
    order = if which === :LM
        sortperm(abs.(vals); rev=true)
    elseif which === :SM
        sortperm(abs.(vals); rev=false)
    elseif which === :LR
        sortperm(real.(vals); rev=true)
    elseif which === :SR
        sortperm(real.(vals); rev=false)
    elseif which === :LI
        sortperm(imag.(vals); rev=true)
    elseif which === :SI
        sortperm(imag.(vals); rev=false)
    else
        error("unsupported selector $which")
    end
    return vals[order[1:howmany]]
end

function _kk_relres(A, x, b)
    r = b .- A * x
    return norm(r) / max(norm(b), norm(A * x), 1.0)
end

@testset "TenetNative vs KrylovKit CPU parity" begin
        prefix = mktempdir()
        lib = build_native_arnoldi(; target=:cpu, prefix)

        A = [0.2 2.0 0.0 0.0;
             -1.0 0.1 0.0 0.0;
              0.0 0.0 3.0 0.4;
              0.0 0.0 0.0 -2.0]
        x0 = [1.0, -0.3, 0.2, 0.7]
        for which in (:LM, :LR, :SR)
            vals_native, vecs_native, info_native = native_eigsolve(
                x -> A * x,
                x0,
                1,
                which;
                krylovdim=4,
                tol=1e-13,
                lib,
            )
            vals_kk, _vecs_kk, _info_kk = KrylovKit.eigsolve(
                x -> A * x,
                x0,
                1,
                which;
                krylovdim=4,
                tol=1e-13,
                maxiter=20,
            )
            λ_native = vals_native[1]
            λ_kk = _kk_select(vals_kk, which)
            @test abs(λ_native - λ_kk) <= 5e-11 * max(1.0, abs(λ_kk))
            @test norm(info_native.residual[1]) <= 1e-10
            @test norm(A * vec(vecs_native[1]) .- λ_native .* vec(vecs_native[1])) <= 1e-10
        end

        vals_native, vecs_native, info_native = native_eigsolve(
            x -> A * x,
            x0,
            2,
            :LM;
            krylovdim=4,
            tol=1e-13,
            lib,
        )
        vals_kk, _vecs_kk, _info_kk = KrylovKit.eigsolve(
            x -> A * x,
            x0,
            2,
            :LM;
            krylovdim=4,
            tol=1e-13,
            maxiter=20,
        )
        @test length(vals_native) == 2
        @test info_native.converged == 2
        vals_kk_selected = _kk_select_many(vals_kk, :LM, 2)
        @test maximum(abs.(sort(real.(vals_native)) .- sort(real.(vals_kk_selected)))) <= 5e-11
        @test maximum(norm(A * vec(vecs_native[i]) .- vals_native[i] .* vec(vecs_native[i]))
                      for i in eachindex(vals_native)) <= 1e-10

        x0c = ComplexF64.(x0)
        for which in (:LI, :SI)
            vals_native, vecs_native, info_native = native_eigsolve(
                x -> A * x,
                x0c,
                1,
                which;
                krylovdim=4,
                tol=1e-13,
                lib,
            )
            vals_kk, _vecs_kk, _info_kk = KrylovKit.eigsolve(
                x -> A * x,
                x0c,
                1,
                which;
                krylovdim=4,
                tol=1e-13,
                maxiter=20,
            )
            λ_native = vals_native[1]
            λ_kk = _kk_select(vals_kk, which)
            @test abs(λ_native - λ_kk) <= 5e-11 * max(1.0, abs(λ_kk))
            @test norm(info_native.residual[1]) <= 1e-10
            @test norm(A * vec(vecs_native[1]) .- λ_native .* vec(vecs_native[1])) <= 1e-10
        end

        Z = ComplexF64[
            2.0 + 0.3im  0.2 - 0.4im  0.0;
            0.2 + 0.4im  3.0 - 0.2im  0.1im;
            0.0          -0.1im        -1.0 + 0.7im
        ]
        z0 = ComplexF64[0.7 + 0.2im, -0.4 + 0.6im, 1.0 - 0.1im]
        vals_native, vecs_native, info_native = native_eigsolve(
            Z,
            z0,
            1,
            :LM;
            krylovdim=3,
            tol=1e-13,
            lib,
        )
        vals_kk, _vecs_kk, _info_kk = KrylovKit.eigsolve(
            x -> Z * x,
            z0,
            1,
            :LM;
            krylovdim=3,
            tol=1e-13,
            maxiter=20,
        )
        λ_kk = _kk_select(vals_kk, :LM)
        @test abs(vals_native[1] - λ_kk) <= 5e-10 * max(1.0, abs(λ_kk))
        @test vecs_native[1] isa Vector{ComplexF64}
        @test info_native.normres[1] <= 1e-10

        B = [4.0 0.5 -0.2 0.0;
             0.1 3.0 0.4 0.0;
             0.0 -0.3 2.0 0.2;
             0.1 0.0 0.0 1.5]
        b = [1.0, -2.0, 0.5, 0.3]
        x_native, info_native = native_linsolve(
            B,
            b;
            krylovdim=4,
            maxiter=4,
            tol=1e-13,
            lib,
        )
        x_kk, _info_kk = KrylovKit.linsolve(
            x -> B * x,
            b;
            krylovdim=4,
            maxiter=20,
            tol=1e-13,
        )
        @test norm(x_native - x_kk) / max(norm(x_kk), 1.0) <= 1e-10
        @test _kk_relres(B, x_native, b) <= 1e-11
        @test info_native.converged == 1

        x_native_default, info_native_default = native_linsolve(
            B,
            b;
            krylovdim=4,
            maxiter=4,
            lib,
        )
        x_kk_default, _info_kk_default = KrylovKit.linsolve(
            x -> B * x,
            b;
            krylovdim=4,
            maxiter=20,
        )
        @test info_native_default.tol ≈ max(1e-12, 1e-12 * norm(b))
        @test info_native_default.tol_source === :atol_rtol
        @test norm(x_native_default - x_kk_default) / max(norm(x_kk_default), 1.0) <= 1e-10

        x_native, info_native = native_linsolve(
            B,
            b;
            algorithm=:bicgstab,
            maxiter=20,
            tol=1e-13,
            lib,
        )
        x_kk, _info_kk = KrylovKit.linsolve(
            x -> B * x,
            b,
            zeros(4),
            KrylovKit.BiCGStab(; maxiter=20, tol=1e-13),
        )
        @test norm(x_native - x_kk) / max(norm(x_kk), 1.0) <= 1e-10
        @test _kk_relres(B, x_native, b) <= 1e-11
        @test info_native.converged == 1

        H = [4.0 0.3 0.0 0.1;
             0.3 3.0 0.2 0.0;
             0.0 0.2 2.5 0.4;
             0.1 0.0 0.4 2.0]
        x_native, info_native = native_linsolve(
            H,
            b;
            algorithm=:cg,
            maxiter=12,
            tol=1e-13,
            lib,
        )
        x_kk, _info_kk = KrylovKit.linsolve(
            x -> H * x,
            b,
            zero(b),
            KrylovKit.CG(; maxiter=20, tol=1e-13),
        )
        @test norm(x_native - x_kk) / max(norm(x_kk), 1.0) <= 1e-10
        @test _kk_relres(H, x_native, b) <= 1e-11
        @test info_native.converged == 1

        a0 = 0.7
        a1 = -0.25
        shifted = a0 .* I + a1 .* B
        x_native, info_native = native_linsolve(
            B,
            b,
            zeros(4),
            a0,
            a1;
            krylovdim=4,
            maxiter=4,
            tol=1e-13,
            lib,
        )
        x_kk, _info_kk = KrylovKit.linsolve(
            x -> B * x,
            b,
            zeros(4),
            a0,
            a1;
            krylovdim=4,
            maxiter=20,
            tol=1e-13,
        )
        @test norm(x_native - x_kk) / max(norm(x_kk), 1.0) <= 1e-10
        @test _kk_relres(shifted, x_native, b) <= 1e-11
        @test info_native.converged == 1

        bstab_a0 = 1.0
        bstab_a1 = 0.1
        shifted_bicgstab = bstab_a0 .* I + bstab_a1 .* B
        x_native, info_native = native_linsolve(
            B,
            b,
            zeros(4),
            bstab_a0,
            bstab_a1;
            algorithm=:bicgstab,
            maxiter=20,
            tol=1e-13,
            lib,
        )
        x_kk, _info_kk = KrylovKit.linsolve(
            x -> B * x,
            b,
            zeros(4),
            KrylovKit.BiCGStab(; maxiter=20, tol=1e-13),
            bstab_a0,
            bstab_a1,
        )
        @test norm(x_native - x_kk) / max(norm(x_kk), 1.0) <= 1e-10
        @test _kk_relres(shifted_bicgstab, x_native, b) <= 1e-11
        @test info_native.converged == 1

        cg_a0 = 0.5
        cg_a1 = 0.75
        shifted_spd = cg_a0 .* I + cg_a1 .* H
        x_native, info_native = native_linsolve(
            H,
            b,
            zeros(4),
            cg_a0,
            cg_a1;
            algorithm=:cg,
            maxiter=12,
            tol=1e-13,
            lib,
        )
        x_kk, _info_kk = KrylovKit.linsolve(
            x -> H * x,
            b,
            zeros(4),
            KrylovKit.CG(; maxiter=20, tol=1e-13),
            cg_a0,
            cg_a1,
        )
        @test norm(x_native - x_kk) / max(norm(x_kk), 1.0) <= 1e-10
        @test _kk_relres(shifted_spd, x_native, b) <= 1e-11
        @test info_native.converged == 1

        C = ComplexF64[
            3.0 + 0.1im  0.2 - 0.3im  0.0;
            -0.1 + 0.2im 2.5 - 0.4im  0.5;
            0.3im        0.1 + 0.2im  1.7 + 0.2im
        ]
        cb = ComplexF64[1.0 - 0.2im, -0.3 + 0.7im, 0.4 + 0.5im]
        cx_native, cinfo_native = native_linsolve(
            C,
            cb;
            krylovdim=3,
            maxiter=4,
            tol=1e-13,
            lib,
        )
        cx_kk, _cinfo_kk = KrylovKit.linsolve(
            x -> C * x,
            cb;
            krylovdim=3,
            maxiter=20,
            tol=1e-13,
        )
        @test norm(cx_native - cx_kk) / max(norm(cx_kk), 1.0) <= 1e-9
        @test _kk_relres(C, cx_native, cb) <= 1e-10
        @test cinfo_native.converged == 1

        cx_native, cinfo_native = native_linsolve(
            C,
            cb;
            algorithm=:bicgstab,
            maxiter=20,
            tol=1e-12,
            lib,
        )
        cx_kk, _cinfo_kk = KrylovKit.linsolve(
            x -> C * x,
            cb,
            zero(cb),
            KrylovKit.BiCGStab(; maxiter=20, tol=1e-12),
        )
        @test norm(cx_native - cx_kk) / max(norm(cx_kk), 1.0) <= 1e-9
        @test _kk_relres(C, cx_native, cb) <= 1e-10
        @test cinfo_native.converged == 1

        HP = ComplexF64[
            4.0          0.2 + 0.3im  0.1;
            0.2 - 0.3im  3.0          0.4im;
            0.1          -0.4im       2.5
        ]
        cx_native, cinfo_native = native_linsolve(
            HP,
            cb;
            algorithm=:cg,
            maxiter=12,
            tol=1e-13,
            lib,
        )
        cx_kk, _cinfo_kk = KrylovKit.linsolve(
            x -> HP * x,
            cb,
            zero(cb),
            KrylovKit.CG(; maxiter=20, tol=1e-13),
        )
        @test norm(cx_native - cx_kk) / max(norm(cx_kk), 1.0) <= 1e-9
        @test _kk_relres(HP, cx_native, cb) <= 1e-10
        @test cinfo_native.converged == 1
end
