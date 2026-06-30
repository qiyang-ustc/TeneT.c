# Native dense Arnoldi fixed-point solver for the real single-site dense FastTeneT
# contractions. This layer stays narrow by design; TenetNative owns the native
# C++/CUDA source and build logic.

_native_arnoldi_default_prefix() = normpath(joinpath(dirname(pathof(TenetNative)), "..", "deps"))

function _native_arnoldi_libname(target::Symbol=:cpu)
    if target === :cpu
        return "libtenet_native_arnoldi." * Libdl.dlext
    elseif target === :cuda
        return "libtenet_native_arnoldi_cuda." * Libdl.dlext
    end
    throw(ArgumentError("unsupported native Arnoldi target $target; expected :cpu or :cuda"))
end

const _TENET_NATIVE_UNAVAILABLE = gensym(:TenetNativeUnavailable)

function _tenet_native_function(name::Symbol)
    return isdefined(TenetNative, name) ? getproperty(TenetNative, name) : nothing
end

function _tenet_native_required_function(name::Symbol)
    f = _tenet_native_function(name)
    f !== nothing && return f
    error("TenetNative.$name is required for FastTeneT Krylov wrappers; upgrade TenetNative or call the legacy native Arnoldi APIs directly")
end

_tenet_native_invoke(f, args...; kwargs...) =
    Base.invokelatest(f, args...; kwargs...)

function native_krylov_capabilities(; kwargs...)
    f = _tenet_native_required_function(:native_krylov_capabilities)
    return _tenet_native_invoke(f; kwargs...)
end

function native_eigsolve(args...; kwargs...)
    f = _tenet_native_required_function(:native_eigsolve)
    return _tenet_native_invoke(f, args...; kwargs...)
end

function native_linsolve(args...; kwargs...)
    f = _tenet_native_required_function(:native_linsolve)
    return _tenet_native_invoke(f, args...; kwargs...)
end

function _tenet_native_cuda_unavailable(err)
    err isa MethodError && return true
    err isa UndefVarError && return true
    if err isa ArgumentError
        msg = sprint(showerror, err)
        return occursin("TenetNative CUDA wrappers require CUDA.jl", msg) ||
               occursin("not exported by the selected native library", msg)
    end
    return false
end

function _tenet_native_cuda_invoke(name::Symbol, args...; kwargs...)
    f = _tenet_native_function(name)
    f === nothing && return _TENET_NATIVE_UNAVAILABLE
    try
        return _tenet_native_invoke(f, args...; kwargs...)
    catch err
        _tenet_native_cuda_unavailable(err) || rethrow()
        return _TENET_NATIVE_UNAVAILABLE
    end
end

function _tenet_native_build_native_arnoldi(target::Symbol,
                                            prefix::AbstractString)
    f = _tenet_native_function(:build_native_arnoldi)
    f === nothing && return _TENET_NATIVE_UNAVAILABLE
    return String(_tenet_native_invoke(f; target, prefix))
end

function _tenet_native_arnoldi_library(; lib=nothing, target::Symbol=:cpu,
                                       autobuild::Bool=true)
    f = _tenet_native_function(:native_arnoldi_library)
    f === nothing && return _TENET_NATIVE_UNAVAILABLE
    try
        return String(_tenet_native_invoke(f; lib, target, autobuild))
    catch err
        (err isa ErrorException || err isa ArgumentError) &&
            return _TENET_NATIVE_UNAVAILABLE
        rethrow()
    end
end

function build_native_arnoldi(; target::Symbol=:cpu,
                              prefix::AbstractString=_native_arnoldi_default_prefix())
    target in (:cpu, :cuda) ||
        throw(ArgumentError("unsupported native Arnoldi target $target; expected :cpu or :cuda"))
    libpath = _tenet_native_build_native_arnoldi(target, prefix)
    libpath !== _TENET_NATIVE_UNAVAILABLE && return libpath
    error("FastTeneT uses TenetNative for native Arnoldi builds; load TenetNative in the current environment")
end

function native_arnoldi_library(; lib=nothing, target::Symbol=:cpu, autobuild::Bool=true)
    lib !== nothing && return String(lib)
    target in (:cpu, :cuda) ||
        throw(ArgumentError("unsupported native Arnoldi target $target; expected :cpu or :cuda"))
    envkey = target === :cpu ? "TENET_NATIVE_ARNOLDI_LIB" : "TENET_NATIVE_ARNOLDI_CUDA_LIB"
    if haskey(ENV, envkey) && !isempty(ENV[envkey])
        return ENV[envkey]
    end
    libpath = _tenet_native_arnoldi_library(; lib, target, autobuild)
    libpath !== _TENET_NATIVE_UNAVAILABLE && return libpath
    autobuild && return build_native_arnoldi(; target)
    path = joinpath(_native_arnoldi_default_prefix(), _native_arnoldi_libname(target))
    error("native Arnoldi $target library not found at $path; build it with build_native_arnoldi(target=:$target) or set $envkey")
end

const _native_arnoldi_handles = Dict{String, Ptr{Cvoid}}()

function _native_arnoldi_handle(libpath::AbstractString)
    return get!(_native_arnoldi_handles, String(libpath)) do
        Libdl.dlopen(String(libpath))
    end
end

function _native_with_backend_device(f::Function, backend::Symbol, A::AbstractArray)
    backend === :cuda || return f()
    old = CUDA.device()
    target = CUDA.device(A)
    CUDA.device!(target)
    try
        CUDA.synchronize()
        return f()
    finally
        CUDA.device!(old)
    end
end

function _native_to_backend(Y::AbstractArray, backend::Symbol, ref::AbstractArray)
    backend === :cuda || return Y
    Y isa CuArray && return Y
    return _native_with_backend_device(backend, ref) do
        CuArray(Y)
    end
end

function _native_status_message(handle::Ptr{Cvoid}, status::Integer)
    handle == C_NULL && return "status=$status"
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
    isempty(last) || last == "success" ? base : "$base: $last"
end

_native_backend(A::Array) = :cpu
_native_backend(A::CuArray) = :cuda
_native_backend(A::AbstractArray) =
    throw(ArgumentError("native Arnoldi supports Array and CUDA.CuArray only, got $(typeof(A))"))

function _native_backend(args::AbstractArray...)
    backend = _native_backend(args[1])
    for A in args[2:end]
        other = _native_backend(A)
        other === backend ||
            throw(ArgumentError("native Arnoldi inputs must all use one backend; got $backend and $other"))
    end
    return backend
end

function _native_is_dense_column_major(A::AbstractArray)
    expected = 1
    for dim in 1:ndims(A)
        stride(A, dim) == expected || return false
        expected *= size(A, dim)
    end
    return true
end

function _native_array(A::AbstractArray, dims::Tuple, name::AbstractString, backend::Symbol)
    if backend === :cpu
        A isa Array || throw(ArgumentError("native Arnoldi CPU backend requires Array for $name, got $(typeof(A))"))
    elseif backend === :cuda
        A isa CuArray || throw(ArgumentError("native Arnoldi CUDA backend requires CUDA.CuArray for $name, got $(typeof(A))"))
    else
        throw(ArgumentError("unsupported native Arnoldi backend $backend"))
    end
    eltype(A) === Float64 || throw(ArgumentError("native Arnoldi supports Float64 only for $name, got $(eltype(A))"))
    size(A) == dims || throw(DimensionMismatch("$name must have size $dims, got $(size(A))"))
    return _native_is_dense_column_major(A) ? A : copy(A)
end

function _native_leg3_pair(Aup::AbstractArray, Adn::AbstractArray, backend::Symbol)
    ndims(Aup) == 3 || throw(DimensionMismatch("Aup must be rank-3, got rank $(ndims(Aup))"))
    size(Aup, 1) == size(Aup, 3) ||
        throw(DimensionMismatch("Aup must have size chi x phys x chi, got $(size(Aup))"))
    chi = size(Aup, 1)
    phys = size(Aup, 2)
    A = _native_array(Aup, (chi, phys, chi), "Aup", backend)
    B = _native_array(Adn, (chi, phys, chi), "Adn", backend)
    return A, B, chi, phys
end

function _native_maxdim(alg::VUMPS, len::Integer)
    k = alg.native_arnoldi_maxiter <= 0 ? len : min(len, alg.native_arnoldi_maxiter)
    return Int(k)
end

function _native_check_fixed_point(λ::Real, y::AbstractArray, fy::AbstractArray, alg::VUMPS, context::AbstractString)
    alg.native_arnoldi_check_residual || return nothing
    residual = fy .- λ .* y
    denom = max(norm(fy), abs(λ) * norm(y), norm(y), 1.0)
    relres = norm(residual) / denom
    if !isfinite(relres) || relres > alg.native_arnoldi_residual_tol
        throw(ArgumentError("native Arnoldi $context residual $relres exceeds tolerance $(alg.native_arnoldi_residual_tol); increase native_arnoldi_krylovdim/native_arnoldi_maxiter for this case"))
    end
    return nothing
end

function _native_select_ritz(H::AbstractMatrix, V::AbstractMatrix, beta::Real, m::Integer)
    m > 0 || throw(ArgumentError("native Arnoldi produced an empty basis"))
    Hm = @view H[1:m, 1:m]
    F = eigen(Matrix(Hm))
    vals = F.values
    mags = abs.(vals)
    maxmag = maximum(mags)
    candidates = findall(i -> abs(mags[i] - maxmag) <= 1e-10 * max(1.0, maxmag), eachindex(vals))
    idx = candidates[argmax(real.(vals[candidates]))]
    λ = vals[idx]
    coeff = F.vectors[:, idx]
    if V isa CuArray
        scale = max(1.0, norm(real.(coeff)))
        if norm(imag.(coeff)) > 1e-8 * scale || abs(imag(λ)) > 1e-8 * max(1.0, abs(real(λ)))
            throw(ArgumentError("native Arnoldi dominant Ritz pair is not real enough: λ=$λ, coeff_imag_norm=$(norm(imag.(coeff)))"))
        end
        c = CuArray(beta .* real.(coeff))
        y = V[:, 1:m] * c
        yn = norm(y)
        yn > 0 || throw(ArgumentError("native Arnoldi selected a zero Ritz vector"))
        y ./= yn
        return real(λ), y
    end
    y = beta .* (V[:, 1:m] * coeff)
    yn = norm(y)
    yn > 0 || throw(ArgumentError("native Arnoldi selected a zero Ritz vector"))
    y ./= yn
    scale = max(1.0, norm(real.(y)))
    if norm(imag.(y)) > 1e-8 * scale || abs(imag(λ)) > 1e-8 * max(1.0, abs(real(λ)))
        throw(ArgumentError("native Arnoldi dominant Ritz pair is not real enough: λ=$λ, imag_norm=$(norm(imag.(y)))"))
    end
    yr = real.(y)
    pivot = argmax(abs.(yr))
    yr[pivot] < 0 && (yr .*= -1)
    return real(λ), yr
end

function _native_arnoldi_two_layer_basis(Aup::AbstractArray, Adn::AbstractArray,
                                         x0::AbstractMatrix;
                                         transpose::Bool=false,
                                         alg::VUMPS,
                                         lib=nothing)
    backend = _native_backend(Aup, Adn, x0)
    if backend === :cpu
        A, B, chi, _phys = _native_leg3_pair(Aup, Adn, backend)
        X = _native_array(x0, (chi, chi), "x0", backend)
        libpath = native_arnoldi_library(; lib, target=:cpu)
        f = _tenet_native_required_function(:tenet_native_arnoldi_two_layer_d_cpu)
        result = _tenet_native_invoke(
            f,
            A, B, X;
            max_k=_native_maxdim(alg, chi * chi),
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        return result.V, result.H, result.m, result.beta
    end
    A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
    X = _native_array(x0, (chi, chi), "x0", backend)
    len = chi * chi
    kmax = _native_maxdim(alg, len)
    libpath = native_arnoldi_library(; lib, target=backend)
    if backend === :cuda
        result = _tenet_native_cuda_invoke(
            :tenet_native_arnoldi_two_layer_d_cuda,
            A, B, X;
            max_k=kmax,
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        result !== _TENET_NATIVE_UNAVAILABLE &&
            return result.V, result.H, result.m, result.beta
    end
    V = backend === :cuda ? CuArray{Float64}(undef, len, kmax + 1) : Matrix{Float64}(undef, len, kmax + 1)
    H = zeros(Float64, kmax + 1, kmax)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    handle = _native_arnoldi_handle(libpath)
    msg = Ref{String}("")
    symbol = backend === :cpu ? :tenet_native_arnoldi_two_layer_d_cpu : :tenet_native_arnoldi_two_layer_d_cuda
    fptr = Libdl.dlsym(handle, symbol)
    status = _native_with_backend_device(backend, A) do
        if backend === :cpu
            ccall(fptr, Cint,
                  (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Int64, Float64, Cint, Ptr{Float64}, Int64, Ptr{Float64},
                   Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol), Cint(transpose ? 1 : 0),
                  V, Int64(len), H, Int64(kmax + 1), beta, m, res)
        else
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   Int64, Float64, Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64},
                   Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol), Cint(transpose ? 1 : 0),
                  V, Int64(stride(V, 2)), H, Int64(stride(H, 2)), beta, m, res)
        end
    end
    status == 0 || (msg[] = _native_status_message(handle, status))
    status == 0 || error("native two-layer Arnoldi failed: " * msg[])
    return V, H, Int(m[]), beta[]
end

function _native_arnoldi_two_layer_ritz_values(Aup::AbstractArray,
                                               Adn::AbstractArray,
                                               x0::AbstractMatrix;
                                               nvalues::Integer=2,
                                               transpose::Bool=false,
                                               alg::VUMPS,
                                               lib=nothing)
    backend = _native_backend(Aup, Adn, x0)
    backend === :cpu ||
        throw(ArgumentError("native restarted Ritz values currently require CPU Array inputs"))
    nvalues > 0 || throw(ArgumentError("nvalues must be positive"))
    A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
    X = _native_array(x0, (chi, chi), "x0", backend)
    len = chi * chi
    kmax = _native_maxdim(alg, len)
    libpath = native_arnoldi_library(; lib, target=:cpu)
    f = _tenet_native_required_function(:tenet_native_arnoldi_two_layer_ritz_d_cpu)
    result = _tenet_native_invoke(
        f,
        A, B, X;
        nvalues,
        max_k=kmax,
        breakdown_tol=alg.native_arnoldi_tol,
        transpose,
        lib=libpath,
    )
    return result.values, result.m
end

function _native_arnoldi_projected_two_layer_basis(Aup::AbstractArray,
                                                   Adn::AbstractArray,
                                                   rho::AbstractMatrix,
                                                   x0::AbstractMatrix;
                                                   transpose::Bool=false,
                                                   alg::VUMPS,
                                                   lib=nothing)
    backend = _native_backend(Aup, Adn, rho, x0)
    if backend === :cpu
        A, B, chi, _phys = _native_leg3_pair(Aup, Adn, backend)
        Rho = _native_array(rho, (chi, chi), "rho", backend)
        X = _native_array(x0, (chi, chi), "x0", backend)
        libpath = native_arnoldi_library(; lib, target=:cpu)
        f = _tenet_native_required_function(:tenet_native_arnoldi_projected_two_layer_d_cpu)
        result = _tenet_native_invoke(
            f,
            A, B, Rho, X;
            max_k=_native_maxdim(alg, chi * chi),
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        return result.V, result.H, result.m, result.beta
    end
    A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
    Rho = _native_array(rho, (chi, chi), "rho", backend)
    X = _native_array(x0, (chi, chi), "x0", backend)
    len = chi * chi
    kmax = _native_maxdim(alg, len)
    libpath = native_arnoldi_library(; lib, target=backend)
    if backend === :cuda
        result = _tenet_native_cuda_invoke(
            :tenet_native_arnoldi_projected_two_layer_d_cuda,
            A, B, Rho, X;
            max_k=kmax,
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        result !== _TENET_NATIVE_UNAVAILABLE &&
            return result.V, result.H, result.m, result.beta
    end
    V = backend === :cuda ? CuArray{Float64}(undef, len, kmax + 1) : Matrix{Float64}(undef, len, kmax + 1)
    H = zeros(Float64, kmax + 1, kmax)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    handle = _native_arnoldi_handle(libpath)
    msg = Ref{String}("")
    symbol = backend === :cpu ? :tenet_native_arnoldi_projected_two_layer_d_cpu : :tenet_native_arnoldi_projected_two_layer_d_cuda
    fptr = Libdl.dlsym(handle, symbol)
    status = _native_with_backend_device(backend, A) do
        if backend === :cpu
            ccall(fptr, Cint,
                  (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Int64,
                   Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, Rho, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol), Cint(transpose ? 1 : 0),
                  V, Int64(len), H, Int64(kmax + 1), beta, m, res)
        else
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
                   Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64}, Int64,
                   Ref{Float64}, Ref{Int64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, Rho, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol), Cint(transpose ? 1 : 0),
                  V, Int64(stride(V, 2)), H, Int64(stride(H, 2)), beta, m, res)
        end
    end
    status == 0 || (msg[] = _native_status_message(handle, status))
    status == 0 || error("native projected two-layer Arnoldi failed: " * msg[])
    return V, H, Int(m[]), beta[]
end

function _native_arnoldi_two_layer(Aup::AbstractArray, Adn::AbstractArray,
                                   x0::AbstractMatrix;
                                   transpose::Bool=false,
                                   alg::VUMPS,
                                   lib=nothing)
    backend = _native_backend(Aup, Adn, x0)
    if backend === :cpu
        A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
        X = _native_array(x0, (chi, chi), "x0", backend)
        len = chi * chi
        kmax = _native_maxdim(alg, len)
        libpath = native_arnoldi_library(; lib, target=:cpu)
        f = _tenet_native_required_function(:tenet_native_dominant_two_layer_d_cpu)
        result = _tenet_native_invoke(
            f,
            A, B, X;
            max_k=kmax,
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        return result.lambda, result.y
    end
    if backend === :cuda
        A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
        X = _native_array(x0, (chi, chi), "x0", backend)
        len = chi * chi
        kmax = _native_maxdim(alg, len)
        libpath = native_arnoldi_library(; lib, target=:cuda)
        result = _tenet_native_cuda_invoke(
            :tenet_native_dominant_two_layer_d_cuda,
            A, B, X;
            max_k=kmax,
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        result !== _TENET_NATIVE_UNAVAILABLE && return result.lambda, result.y
        Y = CuArray{Float64}(undef, chi, chi)
        λ = Ref{Float64}(0.0)
        handle = _native_arnoldi_handle(libpath)
        fptr = Libdl.dlsym(handle, :tenet_native_dominant_two_layer_d_cuda)
        status = _native_with_backend_device(backend, A) do
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, Int64, Float64, Cint,
                   CUDA.CuPtr{Float64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol),
                  Cint(transpose ? 1 : 0), Y, λ)
        end
        status == 0 || error("native CUDA dominant two-layer failed: " * _native_status_message(handle, status))
        return λ[], Y
    end
    V, H, m, beta = _native_arnoldi_two_layer_basis(Aup, Adn, x0;
                                                    transpose, alg, lib)
    λ, y = _native_with_backend_device(backend, x0) do
        _native_select_ritz(H, V, beta, m)
    end
    Y = reshape(y, size(x0))
    return λ, _native_to_backend(Y, backend, x0)
end

function _native_arnoldi_three_layer_leg4_basis(Aup::AbstractArray,
                                                Adn::AbstractArray,
                                                M::AbstractArray,
                                                x0::AbstractArray;
                                                transpose::Bool=false,
                                                alg::VUMPS,
                                                lib=nothing)
    backend = _native_backend(Aup, Adn, M, x0)
    if backend === :cpu
        A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
        ndims(M) == 4 || throw(DimensionMismatch("M must be rank-4, got rank $(ndims(M))"))
        Mc = _native_array(M, (phys, phys, phys, phys), "M", backend)
        X = _native_array(x0, (chi, phys, chi), "x0", backend)
        libpath = native_arnoldi_library(; lib, target=:cpu)
        f = _tenet_native_required_function(:tenet_native_arnoldi_three_layer_leg4_d_cpu)
        result = _tenet_native_invoke(
            f,
            A, B, Mc, X;
            max_k=_native_maxdim(alg, chi * phys * chi),
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        return result.V, result.H, result.m, result.beta
    end
    A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
    ndims(M) == 4 || throw(DimensionMismatch("M must be rank-4, got rank $(ndims(M))"))
    Mc = _native_array(M, (phys, phys, phys, phys), "M", backend)
    X = _native_array(x0, (chi, phys, chi), "x0", backend)
    len = chi * phys * chi
    kmax = _native_maxdim(alg, len)
    libpath = native_arnoldi_library(; lib, target=backend)
    if backend === :cuda
        result = _tenet_native_cuda_invoke(
            :tenet_native_arnoldi_three_layer_leg4_d_cuda,
            A, B, Mc, X;
            max_k=kmax,
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        result !== _TENET_NATIVE_UNAVAILABLE &&
            return result.V, result.H, result.m, result.beta
    end
    V = backend === :cuda ? CuArray{Float64}(undef, len, kmax + 1) : Matrix{Float64}(undef, len, kmax + 1)
    H = zeros(Float64, kmax + 1, kmax)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    res = Ref{Float64}(0.0)
    handle = _native_arnoldi_handle(libpath)
    msg = Ref{String}("")
    symbol = backend === :cpu ? :tenet_native_arnoldi_three_layer_leg4_d_cpu : :tenet_native_arnoldi_three_layer_leg4_d_cuda
    fptr = Libdl.dlsym(handle, symbol)
    status = _native_with_backend_device(backend, A) do
        if backend === :cpu
            ccall(fptr, Cint,
                  (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Int64,
                   Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, Mc, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol), Cint(transpose ? 1 : 0),
                  V, Int64(len), H, Int64(kmax + 1), beta, m, res)
        else
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, Int64, Float64, Cint, CUDA.CuPtr{Float64}, Int64,
                   Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, Mc, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol), Cint(transpose ? 1 : 0),
                  V, Int64(stride(V, 2)), H, Int64(stride(H, 2)), beta, m, res)
        end
    end
    status == 0 || (msg[] = _native_status_message(handle, status))
    status == 0 || error("native three-layer Arnoldi failed: " * msg[])
    return V, H, Int(m[]), beta[]
end

function _native_arnoldi_three_layer_leg4(Aup::AbstractArray, Adn::AbstractArray,
                                          M::AbstractArray, x0::AbstractArray;
                                          transpose::Bool=false,
                                          alg::VUMPS,
                                          lib=nothing)
    backend = _native_backend(Aup, Adn, M, x0)
    if backend === :cpu
        A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
        Mc = _native_array(M, (phys, phys, phys, phys), "M", backend)
        X = _native_array(x0, (chi, phys, chi), "x0", backend)
        len = chi * phys * chi
        kmax = _native_maxdim(alg, len)
        libpath = native_arnoldi_library(; lib, target=:cpu)
        f = _tenet_native_required_function(:tenet_native_dominant_three_layer_leg4_d_cpu)
        result = _tenet_native_invoke(
            f,
            A, B, Mc, X;
            max_k=kmax,
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        return result.lambda, result.y
    end
    if backend === :cuda
        A, B, chi, phys = _native_leg3_pair(Aup, Adn, backend)
        Mc = _native_array(M, (phys, phys, phys, phys), "M", backend)
        X = _native_array(x0, (chi, phys, chi), "x0", backend)
        len = chi * phys * chi
        kmax = _native_maxdim(alg, len)
        libpath = native_arnoldi_library(; lib, target=:cuda)
        result = _tenet_native_cuda_invoke(
            :tenet_native_dominant_three_layer_leg4_d_cuda,
            A, B, Mc, X;
            max_k=kmax,
            breakdown_tol=alg.native_arnoldi_tol,
            transpose,
            lib=libpath,
        )
        result !== _TENET_NATIVE_UNAVAILABLE && return result.lambda, result.y
        Y = CuArray{Float64}(undef, chi, phys, chi)
        λ = Ref{Float64}(0.0)
        handle = _native_arnoldi_handle(libpath)
        fptr = Libdl.dlsym(handle, :tenet_native_dominant_three_layer_leg4_d_cuda)
        status = _native_with_backend_device(backend, A) do
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64,
                   Float64, Cint, CUDA.CuPtr{Float64}, Ref{Float64}),
                  Int64(chi), Int64(phys), A, B, Mc, X, Int64(kmax),
                  Float64(alg.native_arnoldi_tol),
                  Cint(transpose ? 1 : 0), Y, λ)
        end
        status == 0 || error("native CUDA dominant three-layer failed: " * _native_status_message(handle, status))
        return λ[], Y
    end
    V, H, m, beta = _native_arnoldi_three_layer_leg4_basis(Aup, Adn, M, x0;
                                                           transpose, alg, lib)
    λ, y = _native_with_backend_device(backend, x0) do
        _native_select_ritz(H, V, beta, m)
    end
    Y = reshape(y, size(x0))
    return λ, _native_to_backend(Y, backend, x0)
end

_native_full_step_disabled() =
    lowercase(get(ENV, "FASTTENET_DISABLE_NATIVE_FULL_STEP", "")) in ("1", "true", "yes", "on")

function _native_single_site_cpu_ising_inputs(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    _native_full_step_disabled() && return nothing
    alg.ifsimple_eig && return nothing
    alg.eig_solver === :native_arnoldi || return nothing
    alg.ifcheckpoint && return nothing
    alg.forloop_iter == 1 || return nothing
    size(M) == (1, 1) || return nothing
    size(rt.AL) == (1, 1) || return nothing
    size(rt.AR) == (1, 1) || return nothing
    size(rt.C) == (1, 1) || return nothing
    size(rt.FL) == (1, 1) || return nothing
    size(rt.FR) == (1, 1) || return nothing

    AL0 = rt.AL[1, 1]
    AR0 = rt.AR[1, 1]
    C0 = rt.C[1, 1]
    FL0 = rt.FL[1, 1]
    FR0 = rt.FR[1, 1]
    M0 = M[1, 1]
    (AL0 isa Array && AR0 isa Array && C0 isa Array && FL0 isa Array &&
     FR0 isa Array && M0 isa Array) || return nothing
    eltype(AL0) === Float64 || return nothing
    eltype(AR0) === Float64 || return nothing
    eltype(C0) === Float64 || return nothing
    eltype(FL0) === Float64 || return nothing
    eltype(FR0) === Float64 || return nothing
    eltype(M0) === Float64 || return nothing
    ndims(AL0) == 3 || return nothing
    ndims(AR0) == 3 || return nothing
    ndims(C0) == 2 || return nothing
    ndims(FL0) == 3 || return nothing
    ndims(FR0) == 3 || return nothing
    ndims(M0) == 4 || return nothing
    chi = size(AL0, 1)
    phys = size(AL0, 2)
    size(AL0) == (chi, phys, chi) || return nothing
    size(AR0) == (chi, phys, chi) || return nothing
    size(C0) == (chi, chi) || return nothing
    size(FL0) == (chi, phys, chi) || return nothing
    size(FR0) == (chi, phys, chi) || return nothing
    size(M0) == (phys, phys, phys, phys) || return nothing

    AL = copy(_native_array(AL0, (chi, phys, chi), "AL", :cpu))
    AR = copy(_native_array(AR0, (chi, phys, chi), "AR", :cpu))
    C = copy(_native_array(C0, (chi, chi), "C", :cpu))
    FL = copy(_native_array(FL0, (chi, phys, chi), "FL", :cpu))
    FR = copy(_native_array(FR0, (chi, phys, chi), "FR", :cpu))
    Mc = _native_array(M0, (phys, phys, phys, phys), "M", :cpu)
    return chi, phys, Mc, AL, AR, C, FL, FR
end

function _native_single_site_cuda_ising_inputs(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    _native_full_step_disabled() && return nothing
    alg.ifsimple_eig && return nothing
    alg.eig_solver === :native_arnoldi || return nothing
    alg.ifcheckpoint && return nothing
    alg.forloop_iter == 1 || return nothing
    size(M) == (1, 1) || return nothing
    size(rt.AL) == (1, 1) || return nothing
    size(rt.AR) == (1, 1) || return nothing
    size(rt.C) == (1, 1) || return nothing
    size(rt.FL) == (1, 1) || return nothing
    size(rt.FR) == (1, 1) || return nothing

    AL0 = rt.AL[1, 1]
    AR0 = rt.AR[1, 1]
    C0 = rt.C[1, 1]
    FL0 = rt.FL[1, 1]
    FR0 = rt.FR[1, 1]
    M0 = M[1, 1]
    (AL0 isa CuArray && AR0 isa CuArray && C0 isa CuArray &&
     FL0 isa CuArray && FR0 isa CuArray && M0 isa CuArray) || return nothing
    eltype(AL0) === Float64 || return nothing
    eltype(AR0) === Float64 || return nothing
    eltype(C0) === Float64 || return nothing
    eltype(FL0) === Float64 || return nothing
    eltype(FR0) === Float64 || return nothing
    eltype(M0) === Float64 || return nothing
    ndims(AL0) == 3 || return nothing
    ndims(AR0) == 3 || return nothing
    ndims(C0) == 2 || return nothing
    ndims(FL0) == 3 || return nothing
    ndims(FR0) == 3 || return nothing
    ndims(M0) == 4 || return nothing
    chi = size(AL0, 1)
    phys = size(AL0, 2)
    size(AL0) == (chi, phys, chi) || return nothing
    size(AR0) == (chi, phys, chi) || return nothing
    size(C0) == (chi, chi) || return nothing
    size(FL0) == (chi, phys, chi) || return nothing
    size(FR0) == (chi, phys, chi) || return nothing
    size(M0) == (phys, phys, phys, phys) || return nothing

    AL = copy(_native_array(AL0, (chi, phys, chi), "AL", :cuda))
    AR = copy(_native_array(AR0, (chi, phys, chi), "AR", :cuda))
    C = copy(_native_array(C0, (chi, chi), "C", :cuda))
    FL = copy(_native_array(FL0, (chi, phys, chi), "FL", :cuda))
    FR = copy(_native_array(FR0, (chi, phys, chi), "FR", :cuda))
    Mc = _native_array(M0, (phys, phys, phys, phys), "M", :cuda)
    return chi, phys, Mc, AL, AR, C, FL, FR
end

function _native_wrap_ising_runtime(rt::VUMPSRuntime, AL, AR, C, FL, FR)
    return VUMPSRuntime(
        StructArray([AL], copy(rt.AL.pattern)),
        StructArray([AR], copy(rt.AR.pattern)),
        StructArray([C], copy(rt.C.pattern)),
        StructArray([FL], copy(rt.FL.pattern)),
        StructArray([FR], copy(rt.FR.pattern)),
    )
end

function _native_acc_to_alar_cpu(AC::StructArray, C::StructArray)
    lowercase(get(ENV, "FASTTENET_DISABLE_NATIVE_ACC_TO_ALAR", "")) in ("1", "true", "yes", "on") &&
        return nothing
    size(AC) == (1, 1) || return nothing
    size(C) == (1, 1) || return nothing
    AC0 = AC[1, 1]
    C0 = C[1, 1]
    AC0 isa Array || return nothing
    C0 isa Array || return nothing
    eltype(AC0) === Float64 || return nothing
    eltype(C0) === Float64 || return nothing
    ndims(AC0) == 3 || return nothing
    ndims(C0) == 2 || return nothing
    chi = size(AC0, 1)
    phys = size(AC0, 2)
    size(AC0) == (chi, phys, chi) || return nothing
    size(C0) == (chi, chi) || return nothing
    ACc = _native_array(AC0, (chi, phys, chi), "AC", :cpu)
    Cc = _native_array(C0, (chi, chi), "C", :cpu)
    f = _tenet_native_required_function(:tenet_native_acc_to_alar_d_cpu)
    result = _tenet_native_invoke(
        f,
        ACc, Cc; lib=native_arnoldi_library(; target=:cpu)
    )
    return StructArray([result.AL], copy(AC.pattern)),
           StructArray([result.AR], copy(AC.pattern)), result.err, 0.0
end

function _native_ising_vumps_step_cpu(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    inputs = _native_single_site_cpu_ising_inputs(rt, M, alg)
    inputs === nothing && return nothing
    chi, phys, Mc, AL, AR, C, FL, FR = inputs
    libpath = native_arnoldi_library(; target=:cpu)
    max_k = _native_maxdim(alg, chi * phys * chi)
    result = if alg.native_arnoldi_check_residual
        f = _tenet_native_required_function(:tenet_native_ising_vumps_step_checked_d_cpu)
        _tenet_native_invoke(
            f,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            residual_tol=alg.native_arnoldi_residual_tol,
            lib=libpath,
        )
    else
        f = _tenet_native_required_function(:tenet_native_ising_vumps_step_d_cpu)
        _tenet_native_invoke(
            f,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            lib=libpath,
        )
    end
    return _native_wrap_ising_runtime(rt, result.AL, result.AR, result.C, result.FL, result.FR), result.err
end

function _native_ising_vumps_step_cuda(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    inputs = _native_single_site_cuda_ising_inputs(rt, M, alg)
    inputs === nothing && return nothing
    chi, phys, Mc, AL, AR, C, FL, FR = inputs
    libpath = native_arnoldi_library(; target=:cuda)
    max_k = _native_maxdim(alg, chi * phys * chi)
    tenet_result = if alg.native_arnoldi_check_residual
        _tenet_native_cuda_invoke(
            :tenet_native_ising_vumps_step_checked_d_cuda,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            residual_tol=alg.native_arnoldi_residual_tol,
            lib=libpath,
        )
    else
        _tenet_native_cuda_invoke(
            :tenet_native_ising_vumps_step_d_cuda,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            lib=libpath,
        )
    end
    if tenet_result !== _TENET_NATIVE_UNAVAILABLE
        CUDA.synchronize()
        return _native_wrap_ising_runtime(
            rt,
            tenet_result.AL,
            tenet_result.AR,
            tenet_result.C,
            tenet_result.FL,
            tenet_result.FR,
        ), tenet_result.err
    end

    handle = _native_arnoldi_handle(libpath)
    err = Ref{Float64}(0.0)
    status = if alg.native_arnoldi_check_residual
        fptr = Libdl.dlsym(handle, :tenet_native_ising_vumps_step_checked_d_cuda)
        _native_with_backend_device(:cuda, AL) do
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, Int64, Float64, Float64, Ref{Float64}),
                  Int64(chi), Int64(phys), Mc, AL, AR, C, FL, FR, Int64(max_k),
                  Float64(alg.native_arnoldi_tol),
                  Float64(alg.native_arnoldi_residual_tol), err)
        end
    else
        fptr = Libdl.dlsym(handle, :tenet_native_ising_vumps_step_d_cuda)
        _native_with_backend_device(:cuda, AL) do
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, Int64, Float64, Ref{Float64}),
                  Int64(chi), Int64(phys), Mc, AL, AR, C, FL, FR, Int64(max_k),
                  Float64(alg.native_arnoldi_tol), err)
        end
    end
    status == 0 || error("native CUDA Ising VUMPS step failed: " * _native_status_message(handle, status))
    CUDA.synchronize()
    return _native_wrap_ising_runtime(rt, AL, AR, C, FL, FR), err[]
end

function _native_ising_vumps_run_cpu(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    alg.maxiter_ad == 0 || return nothing
    alg.verbosity <= 1 || return nothing
    inputs = _native_single_site_cpu_ising_inputs(rt, M, alg)
    inputs === nothing && return nothing
    chi, phys, Mc, AL, AR, C, FL, FR = inputs
    libpath = native_arnoldi_library(; target=:cpu)
    max_k = _native_maxdim(alg, chi * phys * chi)
    result = if alg.native_arnoldi_check_residual
        f = _tenet_native_required_function(:tenet_native_ising_vumps_run_checked_d_cpu)
        _tenet_native_invoke(
            f,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            tol=alg.tol,
            miniter=alg.miniter,
            maxiter=alg.maxiter,
            residual_tol=alg.native_arnoldi_residual_tol,
            lib=libpath,
        )
    else
        f = _tenet_native_required_function(:tenet_native_ising_vumps_run_d_cpu)
        _tenet_native_invoke(
            f,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            tol=alg.tol,
            miniter=alg.miniter,
            maxiter=alg.maxiter,
            lib=libpath,
        )
    end
    return _native_wrap_ising_runtime(rt, result.AL, result.AR, result.C, result.FL, result.FR), result.err
end

function _native_ising_vumps_run_cuda(rt::VUMPSRuntime, M::StructArray, alg::VUMPS)
    alg.maxiter_ad == 0 || return nothing
    alg.verbosity <= 1 || return nothing
    inputs = _native_single_site_cuda_ising_inputs(rt, M, alg)
    inputs === nothing && return nothing
    chi, phys, Mc, AL, AR, C, FL, FR = inputs
    libpath = native_arnoldi_library(; target=:cuda)
    max_k = _native_maxdim(alg, chi * phys * chi)
    tenet_result = if alg.native_arnoldi_check_residual
        _tenet_native_cuda_invoke(
            :tenet_native_ising_vumps_run_checked_d_cuda,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            tol=alg.tol,
            miniter=alg.miniter,
            maxiter=alg.maxiter,
            residual_tol=alg.native_arnoldi_residual_tol,
            lib=libpath,
        )
    else
        _tenet_native_cuda_invoke(
            :tenet_native_ising_vumps_run_d_cuda,
            Mc, AL, AR, C, FL, FR;
            max_k,
            breakdown_tol=alg.native_arnoldi_tol,
            tol=alg.tol,
            miniter=alg.miniter,
            maxiter=alg.maxiter,
            lib=libpath,
        )
    end
    if tenet_result !== _TENET_NATIVE_UNAVAILABLE
        CUDA.synchronize()
        return _native_wrap_ising_runtime(
            rt,
            tenet_result.AL,
            tenet_result.AR,
            tenet_result.C,
            tenet_result.FL,
            tenet_result.FR,
        ), tenet_result.err
    end
    handle = _native_arnoldi_handle(libpath)
    err = Ref{Float64}(0.0)
    iterations = Ref{Int64}(0)
    converged = Ref{Cint}(0)
    status = if alg.native_arnoldi_check_residual
        fptr = Libdl.dlsym(handle, :tenet_native_ising_vumps_run_checked_d_cuda)
        _native_with_backend_device(:cuda, AL) do
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, Int64, Float64, Float64, Int64, Int64,
                   Float64, Ref{Float64}, Ref{Int64}, Ref{Cint}),
                  Int64(chi), Int64(phys), Mc, AL, AR, C, FL, FR, Int64(max_k),
                  Float64(alg.native_arnoldi_tol), Float64(alg.tol),
                  Int64(alg.miniter), Int64(alg.maxiter),
                  Float64(alg.native_arnoldi_residual_tol), err, iterations,
                  converged)
        end
    else
        fptr = Libdl.dlsym(handle, :tenet_native_ising_vumps_run_d_cuda)
        _native_with_backend_device(:cuda, AL) do
            ccall(fptr, Cint,
                  (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                   CUDA.CuPtr{Float64}, Int64, Float64, Float64, Int64, Int64,
                   Ref{Float64}, Ref{Int64}, Ref{Cint}),
                  Int64(chi), Int64(phys), Mc, AL, AR, C, FL, FR, Int64(max_k),
                  Float64(alg.native_arnoldi_tol), Float64(alg.tol),
                  Int64(alg.miniter), Int64(alg.maxiter), err, iterations,
                  converged)
        end
    end
    status == 0 || error("native CUDA Ising VUMPS run failed: " * _native_status_message(handle, status))
    CUDA.synchronize()
    return _native_wrap_ising_runtime(rt, AL, AR, C, FL, FR), err[]
end

function _native_FLmap_eig(FL, ALu, ALd, M; alg::VUMPS)
    λ, y = _native_arnoldi_three_layer_leg4(ALu, ALd, M, FL; alg)
    _native_check_fixed_point(λ, y, FLmap_forloop(y, ALu, ALd, M; forloop_iter=alg.forloop_iter), alg, "FLmap")
    return λ, y
end

function _native_FRmap_eig(FR, ARu, ARd, M; alg::VUMPS)
    λ, y = _native_arnoldi_three_layer_leg4(ARu, ARd, M, FR; transpose=true, alg)
    _native_check_fixed_point(λ, y, FRmap_forloop(y, ARu, ARd, M; forloop_iter=alg.forloop_iter), alg, "FRmap")
    return λ, y
end

function _native_Lmap_eig(L, ALu, ALd; alg::VUMPS)
    λ, y = _native_arnoldi_two_layer(ALu, ALd, L; alg)
    _native_check_fixed_point(λ, y, Lmap(y, ALu, ALd), alg, "Lmap")
    return λ, y
end

function _native_Rmap_eig(R, ARu, ARd; alg::VUMPS)
    λ, y = _native_arnoldi_two_layer(ARu, ARd, R; transpose=true, alg)
    _native_check_fixed_point(λ, y, Rmap(y, ARu, ARd), alg, "Rmap")
    return λ, y
end

function _native_ACmap_eig(AC, FL, FR, M; alg::VUMPS)
    Mp = permutedims(M, (4, 3, 2, 1))
    λ, y = _native_arnoldi_three_layer_leg4(FL, FR, Mp, AC; alg)
    _native_check_fixed_point(λ, y, ACmap_forloop(y, FL, FR, M; forloop_iter=alg.forloop_iter), alg, "ACmap")
    return λ, y
end

function _native_Cmap_eig(C, FL, FR; alg::VUMPS)
    λ, y = _native_arnoldi_two_layer(FL, FR, C; alg)
    _native_check_fixed_point(λ, y, Cmap(y, FL, FR), alg, "Cmap")
    return λ, y
end

function _native_require_single_site(axis_len::Integer, solver::Symbol, context::AbstractString)
    if solver === :native_arnoldi && axis_len != 1
        throw(ArgumentError("native Arnoldi currently supports single-site dense maps only in $context; got length=$axis_len"))
    end
    return nothing
end
