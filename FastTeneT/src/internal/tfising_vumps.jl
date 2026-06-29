struct TFIsingVUMPSState
    AL
    AR
    C
    AC
end

struct TFIsingVUMPSEnv
    left
    right
end

const _TFISING_MPO_NONZERO = ((1, 1), (2, 1), (3, 1), (3, 2), (3, 3))

function _tfising_vumps(W::AbstractArray, χ::Int, alg::VUMPS;
                        seed::Int,
                        eig_tol::Float64,
                        eig_krylovdim::Int,
                        env_tol::Float64,
                        env_krylovdim::Int)
    ndims(W) == 4 || throw(DimensionMismatch("TFIsing MPO tensor must be rank-4"))
    size(W) == (3, 2, 3, 2) ||
        throw(DimensionMismatch("TFIsing MPO tensor must have size (3,2,3,2), got $(size(W))"))
    eltype(W) === Float64 || throw(ArgumentError("TFIsing VUMPS supports Float64 tensors only"))

    state = _tfising_initial_state(W, χ; seed)
    env = _tfising_environments(state, W; env_tol, env_krylovdim, verbosity=alg.verbosity)
    energy = _tfising_energy_density(state, W, env)
    err = Inf
    converged = false
    iterations = 0

    for iter in 1:alg.maxiter
        iterations = iter
        state′, err = _tfising_vumps_step(state, W, env; eig_tol, eig_krylovdim,
                                          verbosity=alg.verbosity)
        state = state′
        env = _tfising_environments(state, W; env_tol, env_krylovdim, verbosity=alg.verbosity)
        energy = _tfising_energy_density(state, W, env)
        if err < alg.tol && iter >= alg.miniter
            converged = true
            break
        end
    end

    return state, env, Float64(real(energy)), Float64(real(err)), iterations, converged
end

function _tfising_initial_state(W::AbstractArray, χ::Int; seed::Int)
    rng = MersenneTwister(seed)
    A = randn(rng, Float64, χ, 2, χ)
    A ./= sqrt(2χ)
    A = _arraytype(W)(A)
    SA = StructArray([A], copy(SINGLE_UNITCELL_PATTERN))
    AL, L, _ = left_canonical(SA; tol=1e-12, maxiter=100)
    R, AR, _ = right_canonical(AL; tol=1e-12, maxiter=100)
    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)
    return TFIsingVUMPSState(AL[1, 1], AR[1, 1], C[1, 1], AC[1, 1])
end

function _tfising_vumps_step(state::TFIsingVUMPSState, W::AbstractArray;
                             eig_tol::Float64, eig_krylovdim::Int, verbosity::Int=0)
    return _tfising_vumps_step(
        state, W, _tfising_environments(state, W; env_tol=eig_tol, env_krylovdim=eig_krylovdim,
                                         verbosity);
        eig_tol, eig_krylovdim, verbosity,
    )
end

function _tfising_vumps_step(state::TFIsingVUMPSState, W::AbstractArray,
                             env::TFIsingVUMPSEnv;
                             eig_tol::Float64, eig_krylovdim::Int, verbosity::Int=0)
    _, AC = _tfising_native_ac_eig(state.AC, W, env; tol=eig_tol,
                                   krylovdim=eig_krylovdim, verbosity)
    _, C = _tfising_native_c_eig(state.C, env; tol=eig_tol,
                                 krylovdim=eig_krylovdim, verbosity)
    C ./= norm(C)

    ACsa = StructArray([AC], copy(SINGLE_UNITCELL_PATTERN))
    Csa = StructArray([C], copy(SINGLE_UNITCELL_PATTERN))
    ALsa, ARsa, errL, errR = ACCtoALAR(ACsa, Csa)
    AL = ALsa[1, 1]
    AR = ARsa[1, 1]
    ACnext = ALCtoAC(ALsa, Csa)[1, 1]
    err = max(errL, errR)
    return TFIsingVUMPSState(AL, AR, C, ACnext), Float64(real(err))
end

function _tfising_environments(state::TFIsingVUMPSState, W::AbstractArray;
                               env_tol::Float64, env_krylovdim::Int, verbosity::Int=0)
    AL, AR, C = state.AL, state.AR, state.C
    Iχ = _tfising_identity_like(AL, size(AL, 1))
    L = Vector{typeof(Iχ)}(undef, 3)
    R = Vector{typeof(Iχ)}(undef, 3)

    L[3] = Iχ
    L[2] = _tfising_left_apply(Iχ, AL, _tfising_wslice(W, 3, 2))
    rawL1 = _tfising_left_apply(L[2], AL, _tfising_wslice(W, 2, 1)) +
            _tfising_left_apply(Iχ, AL, _tfising_wslice(W, 3, 1))

    ρR = _tfising_right_density(C)
    eL = _tfising_inner(rawL1, ρR)
    rhsL = rawL1 .- eL .* Iχ
    L[1] = _tfising_native_projected_solve(
        AL, _tfising_wslice(W, 1, 1), ρR, rhsL, zero(rhsL);
        side=:left, tol=env_tol, krylovdim=env_krylovdim, verbosity,
    )

    R[1] = Iχ
    R[2] = _tfising_right_apply(Iχ, AR, _tfising_wslice(W, 2, 1))
    rawR3 = _tfising_right_apply(R[2], AR, _tfising_wslice(W, 3, 2)) +
            _tfising_right_apply(Iχ, AR, _tfising_wslice(W, 3, 1))

    ρL = _tfising_left_density(C)
    eR = _tfising_inner(rawR3, ρL)
    rhsR = rawR3 .- eR .* Iχ
    R[3] = _tfising_native_projected_solve(
        AR, _tfising_wslice(W, 3, 3), ρL, rhsR, zero(rhsR);
        side=:right, tol=env_tol, krylovdim=env_krylovdim, verbosity,
    )

    return TFIsingVUMPSEnv(L, R)
end

function _tfising_energy_density(state::TFIsingVUMPSState, W::AbstractArray,
                                 env::TFIsingVUMPSEnv)
    AL = state.AL
    raw = _tfising_left_apply(env.left[2], AL, _tfising_wslice(W, 2, 1)) +
          _tfising_left_apply(env.left[3], AL, _tfising_wslice(W, 3, 1))
    return _tfising_inner(raw, _tfising_right_density(state.C))
end

function _tfising_effective_ac(AC::AbstractArray, W::AbstractArray, env::TFIsingVUMPSEnv)
    out = zero(AC)
    @inbounds for (left, right) in _TFISING_MPO_NONZERO
        out .+= _tfising_ac_apply(env.left[left], AC, _tfising_wslice(W, left, right),
                                  env.right[right])
    end
    return out
end

function _tfising_effective_c(C::AbstractMatrix, env::TFIsingVUMPSEnv)
    out = zero(C)
    @inbounds for level in 1:3
        out .+= _tfising_c_apply(env.left[level], C, env.right[level])
    end
    return out
end

function _tfising_left_apply(L::AbstractMatrix, A::AbstractArray, O::AbstractMatrix)
    @tensor out[c, e] := L[a, d] * A[d, g, e] * O[g, b] * A[a, b, c]
    return out
end

function _tfising_right_apply(R::AbstractMatrix, A::AbstractArray, O::AbstractMatrix)
    @tensor out[a, d] := A[a, b, c] * R[c, e] * O[g, b] * A[d, g, e]
    return out
end

function _tfising_ac_apply(L::AbstractMatrix, AC::AbstractArray, O::AbstractMatrix,
                           R::AbstractMatrix)
    @tensor out[f, g, h] := L[a, f] * AC[a, b, c] * O[g, b] * R[c, h]
    return out
end

function _tfising_c_apply(L::AbstractMatrix, C::AbstractMatrix, R::AbstractMatrix)
    @tensor out[f, h] := L[a, f] * C[a, c] * R[c, h]
    return out
end

function _tfising_native_alg(krylovdim::Int, tol::Float64)
    return VUMPS(;
        ifsimple_eig=false,
        eig_solver=:native_arnoldi,
        native_arnoldi_maxiter=krylovdim,
        native_arnoldi_tol=tol,
        native_arnoldi_check_residual=false,
    )
end

function _tfising_backend_vector(coeff::AbstractVector, ref::AbstractArray)
    return _arraytype(ref)(collect(Float64, coeff))
end

function _tfising_select_smallest_real(H::AbstractMatrix, V::AbstractMatrix,
                                       beta::Float64, m::Int,
                                       shape::Tuple, ref::AbstractArray)
    m > 0 || throw(ArgumentError("TFIsing native Arnoldi produced an empty basis"))
    F = eigen(H[1:m, 1:m])
    vals = F.values
    vecs = F.vectors
    idx = argmin(real.(vals))
    λ = vals[idx]
    coeff = vecs[:, idx]
    scale = max(norm(real.(coeff)), 1.0)
    if norm(imag.(coeff)) > 1e-8 * scale || abs(imag(λ)) > 1e-8 * max(abs(real(λ)), 1.0)
        throw(ArgumentError("TFIsing VUMPS selected a complex Ritz pair λ=$λ"))
    end
    c = _tfising_backend_vector(beta .* real.(coeff), ref)
    y = (@view V[:, 1:m]) * c
    y = reshape(y, shape)
    y ./= norm(y)
    return Float64(real(λ)), y
end

function _tfising_stack_leg3(mats::Vector)
    χ = size(mats[1], 1)
    n = length(mats)
    out = _arraytype(mats[1])(zeros(Float64, χ, n, χ))
    @inbounds for p in 1:n
        out[:, p, :] .= mats[p]
    end
    return out
end

function _tfising_pad_ac(AC::AbstractArray, phys::Int=3)
    χ = size(AC, 1)
    out = _arraytype(AC)(zeros(Float64, χ, phys, χ))
    out[:, 1:size(AC, 2), :] .= AC
    return out
end

function _tfising_pad_mpo(W::AbstractArray, phys::Int=3)
    out = _arraytype(W)(zeros(Float64, phys, phys, phys, phys))
    out[1:size(W, 1), 1:size(W, 2), 1:size(W, 3), 1:size(W, 4)] .= W
    return out
end

function _tfising_unpad_ac(AC::AbstractArray, phys::Int=2)
    return copy(@view AC[:, 1:phys, :])
end

function _tfising_smallest_real_two_layer_cpu(Aup::AbstractArray,
                                              Adn::AbstractArray,
                                              x0::AbstractMatrix;
                                              transpose::Bool=false,
                                              alg::VUMPS)
    backend = _native_backend(Aup, Adn, x0)
    backend === :cpu || return nothing
    A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
    X = _native_array(x0, (chi, chi), "x0", backend)
    len = chi * chi
    kmax = _native_maxdim(alg, len)
    f = _tenet_native_required_function(:tenet_native_smallest_real_two_layer_d_cpu)
    result = _tenet_native_invoke(
        f,
        A, B, X;
        max_k=kmax,
        breakdown_tol=alg.native_arnoldi_tol,
        transpose,
        lib=native_arnoldi_library(; target=:cpu),
    )
    return result.lambda, result.y
end

function _tfising_smallest_real_three_layer_cpu(Aup::AbstractArray,
                                                Adn::AbstractArray,
                                                M::AbstractArray,
                                                x0::AbstractArray;
                                                transpose::Bool=false,
                                                alg::VUMPS)
    backend = _native_backend(Aup, Adn, M, x0)
    backend === :cpu || return nothing
    A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
    Mc = _native_array(M, (phys, phys, phys, phys), "M", backend)
    X = _native_array(x0, (chi, phys, chi), "x0", backend)
    len = chi * phys * chi
    kmax = _native_maxdim(alg, len)
    f = _tenet_native_required_function(:tenet_native_smallest_real_three_layer_leg4_d_cpu)
    result = _tenet_native_invoke(
        f,
        A, B, Mc, X;
        max_k=kmax,
        breakdown_tol=alg.native_arnoldi_tol,
        transpose,
        lib=native_arnoldi_library(; target=:cpu),
    )
    return result.lambda, result.y
end

function _tfising_native_ac_eig(AC::AbstractArray, W::AbstractArray,
                                env::TFIsingVUMPSEnv;
                                tol::Float64, krylovdim::Int,
                                verbosity::Int=0)
    alg = _tfising_native_alg(krylovdim, tol)
    FL = _tfising_stack_leg3(env.left)
    FR = _tfising_stack_leg3(env.right)
    M = _tfising_pad_mpo(W)
    AC0 = _tfising_pad_ac(AC, size(M, 1))
    Mp = permutedims(M, (4, 3, 2, 1))
    native = _tfising_smallest_real_three_layer_cpu(FL, FR, Mp, AC0; alg)
    if native === nothing
        V, H, m, beta = _native_arnoldi_three_layer_leg4_basis(FL, FR, Mp, AC0; alg)
        λ, ACp = _tfising_select_smallest_real(H, V, beta, m, size(AC0), AC0)
    else
        λ, ACp = native
    end
    ACphys = _tfising_unpad_ac(ACp, size(AC, 2))
    verbosity >= 1 && begin
        res = norm(_tfising_effective_ac(ACphys, W, env) .- λ .* ACphys)
        res > max(100tol, 1e-12) &&
            @warn "TFIsing native AC Arnoldi residual is high" residual=res tol=tol krylovdim=krylovdim
    end
    return λ, ACphys
end

function _tfising_native_c_eig(C::AbstractMatrix, env::TFIsingVUMPSEnv;
                               tol::Float64, krylovdim::Int,
                               verbosity::Int=0)
    alg = _tfising_native_alg(krylovdim, tol)
    FL = _tfising_stack_leg3(env.left)
    FR = _tfising_stack_leg3(env.right)
    native = _tfising_smallest_real_two_layer_cpu(FL, FR, C; alg)
    if native === nothing
        V, H, m, beta = _native_arnoldi_two_layer_basis(FL, FR, C; alg)
        λ, Cp = _tfising_select_smallest_real(H, V, beta, m, size(C), C)
    else
        λ, Cp = native
    end
    verbosity >= 1 && begin
        res = norm(_tfising_effective_c(Cp, env) .- λ .* Cp)
        res > max(100tol, 1e-12) &&
            @warn "TFIsing native C Arnoldi residual is high" residual=res tol=tol krylovdim=krylovdim
    end
    return λ, Cp
end

function _tfising_apply_physical(A::AbstractArray, O::AbstractMatrix)
    @tensor out[d, b, e] := A[d, g, e] * O[g, b]
    return out
end

function _tfising_projected_apply(X::AbstractMatrix, A::AbstractArray,
                                  O::AbstractMatrix, rho::AbstractMatrix,
                                  side::Symbol)
    Iχ = _tfising_identity_like(X, size(X, 1))
    if side === :left
        return X .- _tfising_left_apply(X, A, O) .+ _tfising_inner(X, rho) .* Iχ
    elseif side === :right
        return X .- _tfising_right_apply(X, A, O) .+ _tfising_inner(X, rho) .* Iχ
    end
    throw(ArgumentError("unsupported TFIsing projected solve side $side"))
end

function _tfising_native_projected_solve(A::AbstractArray, O::AbstractMatrix,
                                         rho::AbstractMatrix, rhs::AbstractMatrix,
                                         x0::AbstractMatrix;
                                         side::Symbol, tol::Float64,
                                         krylovdim::Int, verbosity::Int=0)
    AO = _tfising_apply_physical(A, O)
    r0 = rhs .- _tfising_projected_apply(x0, A, O, rho, side)
    β = Float64(norm(r0))
    β <= tol * max(Float64(norm(rhs)), 1.0) && return x0
    alg = _tfising_native_alg(krylovdim, tol)
    V, H, m, _ = _native_arnoldi_projected_two_layer_basis(
        A, AO, rho, r0; transpose=(side === :right), alg,
    )
    m > 0 || return x0
    e1 = zeros(Float64, m + 1)
    e1[1] = β
    y = H[1:(m + 1), 1:m] \ e1
    yb = _tfising_backend_vector(y, rhs)
    xvec = vec(x0) .+ (@view V[:, 1:m]) * yb
    x = reshape(xvec, size(rhs))
    res = Float64(norm(rhs .- _tfising_projected_apply(x, A, O, rho, side)))
    verbosity >= 1 && res > max(100tol * max(Float64(norm(rhs)), 1.0), 1e-12) &&
        @warn "TFIsing native Arnoldi linear residual is high" residual=res tol=tol krylovdim=krylovdim
    return x
end

function _tfising_wslice(W::AbstractArray, left::Int, right::Int)
    return @view W[left, :, right, :]
end

function _tfising_right_density(C::AbstractMatrix)
    ρ = C * C'
    ρ ./= _tfising_trace(ρ)
    return ρ
end

function _tfising_left_density(C::AbstractMatrix)
    ρ = C' * C
    ρ ./= _tfising_trace(ρ)
    return ρ
end

function _tfising_identity_like(A::AbstractArray, n::Int)
    return _arraytype(A)(Matrix{Float64}(I, n, n))
end

_tfising_trace(A::AbstractMatrix) = sum(diag(A))
_tfising_inner(A::AbstractMatrix, B::AbstractMatrix) = real(dot(A, B))
