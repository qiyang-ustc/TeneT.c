module TenetNative

using Libdl
using LinearAlgebra

export build_native_arnoldi,
       native_arnoldi_library,
       TENET_NATIVE_ABI_VERSION,
       TENET_NATIVE_ABI_VERSION_STRING,
       TENET_NATIVE_KRYLOV_ABI_VERSION,
       TENET_NATIVE_KRYLOV_ABI_VERSION_STRING,
       tenet_native_abi_version,
       tenet_native_abi_version_string,
       tenet_native_status_string,
       tenet_native_last_error,
       tenet_native_raw_two_layer_apply_d_cpu,
       tenet_native_raw_transfer_op_d_cpu,
       tenet_native_raw_rowmajor_transfer_d_cpu,
       tenet_native_raw_rowmajor_transfer_adj_d_cpu,
       tenet_native_raw_rowmajor_transfer_op_d_cpu,
       tenet_native_arnoldi_two_layer_d_cpu,
       tenet_native_arnoldi_two_layer_ritz_d_cpu,
       tenet_native_arnoldi_projected_two_layer_d_cpu,
       tenet_native_arnoldi_qprojected_two_layer_d_cpu,
       tenet_native_arnoldi_three_layer_leg4_d_cpu,
       tenet_native_arnoldi_two_layer_d_cuda,
       tenet_native_arnoldi_two_layer_ritz_d_cuda,
       tenet_native_arnoldi_projected_two_layer_d_cuda,
       tenet_native_arnoldi_qprojected_two_layer_d_cuda,
       tenet_native_raw_two_layer_apply_batch_d_cuda,
       tenet_native_two_layer_apply_batch_d_cuda,
       tenet_native_projected_two_layer_apply_batch_d_cuda,
       tenet_native_qprojected_two_layer_apply_batch_d_cuda,
       tenet_native_arnoldi_three_layer_leg4_d_cuda,
       tenet_native_dominant_two_layer_d_cpu,
       tenet_native_dominant_two_layer_d_cuda,
       tenet_native_smallest_real_two_layer_d_cpu,
       tenet_native_dominant_three_layer_leg4_d_cpu,
       tenet_native_dominant_three_layer_leg4_d_cuda,
       tenet_native_smallest_real_three_layer_leg4_d_cpu,
       tenet_native_ising_vumps_step_d_cpu,
       tenet_native_ising_vumps_step_d_cuda,
       tenet_native_ising_vumps_step_checked_d_cpu,
       tenet_native_ising_vumps_step_checked_d_cuda,
       tenet_native_ising_vumps_run_d_cpu,
       tenet_native_ising_vumps_run_d_cuda,
       tenet_native_ising_vumps_run_checked_d_cpu,
       tenet_native_ising_vumps_run_checked_d_cuda,
       tenet_native_acc_to_alar_d_cpu,
       native_eigsolve,
       native_linsolve,
       native_krylov_capabilities

const _LIB_BASENAME = "libtenet_native_arnoldi"
const TENET_NATIVE_ABI_VERSION = 3
const TENET_NATIVE_ABI_VERSION_STRING = "tenet_native_arnoldi_abi_v3"
const TENET_NATIVE_KRYLOV_ABI_VERSION = 4
const TENET_NATIVE_KRYLOV_ABI_VERSION_STRING = "tenet_native_krylov_abi_v4"
const _NATIVE_KRYLOV_DEFAULT_TOL = 1.0e-12
const _SUPPORTED_ABI_VERSIONS = (3,)
const _SUPPORTED_ABI_VERSION_STRINGS = (
    "tenet_native_arnoldi_abi_v3",
)
const _handles = Dict{String, Ptr{Cvoid}}()
const _abi_checked = Set{Tuple{String, Symbol}}()
const _cuda_module_ref = Ref{Any}(nothing)
const _cuda_wrappers_defined = Ref(false)

struct _NativeComplex64
    re::Float64
    im::Float64
end

_native_complex64(z::Complex) = _NativeComplex64(Float64(real(z)), Float64(imag(z)))
_native_complex64(x::Real) = _NativeComplex64(Float64(x), 0.0)

_native_dir() = joinpath(@__DIR__, "native")
_default_prefix() = normpath(joinpath(@__DIR__, "..", "deps"))
function _native_libname(target::Symbol=:cpu)
    if target === :cpu
        return _LIB_BASENAME * "." * Libdl.dlext
    elseif target === :cuda
        return _LIB_BASENAME * "_cuda." * Libdl.dlext
    end
    throw(ArgumentError("unsupported TenetNative Arnoldi target $target; expected :cpu or :cuda"))
end

function build_native_arnoldi(; target::Symbol=:cpu,
                              prefix::AbstractString=_default_prefix())
    target in (:cpu, :cuda) ||
        throw(ArgumentError("unsupported TenetNative Arnoldi target $target; expected :cpu or :cuda"))
    mkpath(prefix)
    make_target = target === :cpu ? "native-arnoldi-cpu" : "native-arnoldi-cuda"
    julia_exe = joinpath(Sys.BINDIR, Sys.iswindows() ? "julia.exe" : "julia")
    _forget_library!(joinpath(prefix, _native_libname(target)))
    run(`make -C $(_native_dir()) $make_target PREFIX=$prefix JULIA=$julia_exe`)
    return joinpath(prefix, _native_libname(target))
end

function _forget_library!(libpath::AbstractString)
    path = String(libpath)
    if haskey(_handles, path)
        handle = _handles[path]
        delete!(_handles, path)
        try
            Libdl.dlclose(handle)
        catch
        end
    end
    for key in collect(_abi_checked)
        first(key) == path || continue
        delete!(_abi_checked, key)
    end
    return nothing
end

function _path_has_current_abi(path::AbstractString, target::Symbol)
    handle = Libdl.dlopen(String(path))
    try
        _verify_abi(handle, String(path); target)
        return true
    finally
        try
            Libdl.dlclose(handle)
        catch
        end
    end
end

function native_arnoldi_library(; lib=nothing, target::Symbol=:cpu,
                                autobuild::Bool=true)
    target in (:cpu, :cuda) ||
        throw(ArgumentError("unsupported TenetNative Arnoldi target $target; expected :cpu or :cuda"))
    lib !== nothing && return String(lib)
    envkey = target === :cpu ? "TENET_NATIVE_ARNOLDI_LIB" : "TENET_NATIVE_ARNOLDI_CUDA_LIB"
    if haskey(ENV, envkey) && !isempty(ENV[envkey])
        return ENV[envkey]
    end
    path = joinpath(_default_prefix(), _native_libname(target))
    if isfile(path)
        try
            _path_has_current_abi(path, target) && return path
        catch err
            autobuild || rethrow()
            _forget_library!(path)
            # On macOS, reloading a rebuilt dylib at the same path inside the
            # current process can keep returning the old image. Rebuild into a
            # unique prefix and pin the current process to that fresh path.
            prefix = mktempdir()
            fresh = build_native_arnoldi(; target, prefix)
            ENV[envkey] = fresh
            return fresh
        end
    end
    autobuild && return build_native_arnoldi(; target)
    if target === :cpu
        error("native Arnoldi CPU library not found at $path; build it with build_native_arnoldi() or set TENET_NATIVE_ARNOLDI_LIB")
    end
    error("native Arnoldi CUDA library not found at $path; build it with build_native_arnoldi(target=:cuda) or set TENET_NATIVE_ARNOLDI_CUDA_LIB")
end

const _ABI_SYMBOLS_CPU = (
    :tenet_native_abi_version,
    :tenet_native_abi_version_string,
    :tenet_native_raw_two_layer_apply_d_cpu,
    :tenet_native_raw_transfer_op_d_cpu,
    :tenet_native_raw_rowmajor_transfer_d_cpu,
    :tenet_native_raw_rowmajor_transfer_adj_d_cpu,
    :tenet_native_raw_rowmajor_transfer_op_d_cpu,
)

const _ABI_SYMBOLS_CPU_V4_OPTIONAL = (
    :tenet_native_krylov_arnoldi_d_cpu,
    :tenet_native_krylov_arnoldi_z_cpu,
    :tenet_native_krylov_arnoldi_prefilled_d_cpu,
    :tenet_native_krylov_arnoldi_prefilled_z_cpu,
    :tenet_native_krylov_gmres_d_cpu,
    :tenet_native_krylov_gmres_z_cpu,
    :tenet_native_krylov_cg_d_cpu,
    :tenet_native_krylov_cg_z_cpu,
    :tenet_native_krylov_bicgstab_d_cpu,
    :tenet_native_krylov_bicgstab_z_cpu,
    :tenet_native_krylov_arnoldi_dense_d_cpu,
    :tenet_native_krylov_arnoldi_dense_z_cpu,
    :tenet_native_krylov_arnoldi_prefilled_dense_d_cpu,
    :tenet_native_krylov_arnoldi_prefilled_dense_z_cpu,
    :tenet_native_krylov_gmres_dense_d_cpu,
    :tenet_native_krylov_gmres_dense_z_cpu,
    :tenet_native_krylov_cg_dense_d_cpu,
    :tenet_native_krylov_cg_dense_z_cpu,
    :tenet_native_krylov_bicgstab_dense_d_cpu,
    :tenet_native_krylov_bicgstab_dense_z_cpu,
)

const _ABI_SYMBOLS_CUDA = (
    :tenet_native_abi_version,
    :tenet_native_abi_version_string,
    :tenet_native_raw_two_layer_apply_batch_d_cuda,
)

function _check_abi_symbol(handle::Ptr{Cvoid}, name::Symbol, libpath::AbstractString,
                           target::Symbol)
    try
        Libdl.dlsym(handle, name)
    catch
        throw(ArgumentError("TenetNative $target library at $libpath does not export required symbol $name"))
    end
    return nothing
end

function _verify_abi(handle::Ptr{Cvoid}, libpath::AbstractString; target::Symbol=:cpu)
    key = (String(libpath), target)
    if key in _abi_checked
        return
    end
    required = target === :cuda ? _ABI_SYMBOLS_CUDA : _ABI_SYMBOLS_CPU
    for name in required
        _check_abi_symbol(handle, name, String(libpath), target)
    end
    version_ptr = Libdl.dlsym(handle, :tenet_native_abi_version)
    version = ccall(version_ptr, Cint, ())
    if !(version in _SUPPORTED_ABI_VERSIONS)
        throw(ArgumentError("TenetNative $target ABI version mismatch at $libpath: expected one of $(_SUPPORTED_ABI_VERSIONS), got $version"))
    end
    version_string_ptr = Libdl.dlsym(handle, :tenet_native_abi_version_string)
    version_string = unsafe_string(ccall(version_string_ptr, Cstring, ()))
    if !(version_string in _SUPPORTED_ABI_VERSION_STRINGS)
        throw(ArgumentError("TenetNative $target ABI version string mismatch at $libpath: expected one of $(_SUPPORTED_ABI_VERSION_STRINGS), got $version_string"))
    end
    push!(_abi_checked, key)
end

function _handle(lib; target::Symbol=:cpu)
    libpath = native_arnoldi_library(; lib, target)
    return get!(_handles, libpath) do
        handle = Libdl.dlopen(libpath)
        _verify_abi(handle, libpath; target)
        handle
    end
end

_symbol(lib, name::Symbol; target::Symbol=:cpu) = Libdl.dlsym(_handle(lib; target), name)

function _has_symbol(lib, name::Symbol; target::Symbol=:cpu)
    handle = _handle(lib; target)
    try
        Libdl.dlsym(handle, name)
        return true
    catch
        return false
    end
end

function _required_optional_symbol(lib, name::Symbol; target::Symbol=:cpu)
    try
        return _symbol(lib, name; target)
    catch
        throw(ArgumentError("TenetNative $target library does not export required v4 Krylov symbol $name; rebuild TenetNative or unset the native library env var if it points at a legacy v3 library"))
    end
end

function native_krylov_capabilities(; lib=nothing, target::Symbol=:cpu)
    target in (:cpu, :cuda) ||
        throw(ArgumentError("unsupported TenetNative target $target; expected :cpu or :cuda"))
    version = _native_krylov_abi_version(; lib, target)
    legacy_version = tenet_native_abi_version(; lib, target)
    fixed_native = target === :cpu ||
        _has_symbol(lib, :tenet_native_raw_two_layer_apply_batch_d_cuda; target)
    generic_cpu = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_arnoldi_d_cpu,
             :tenet_native_krylov_arnoldi_z_cpu,
             :tenet_native_krylov_gmres_d_cpu,
             :tenet_native_krylov_gmres_z_cpu))
    generic_cpu_prefilled = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_arnoldi_prefilled_d_cpu,
             :tenet_native_krylov_arnoldi_prefilled_z_cpu))
    dense_cpu = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_arnoldi_dense_d_cpu,
             :tenet_native_krylov_arnoldi_dense_z_cpu,
             :tenet_native_krylov_gmres_dense_d_cpu,
             :tenet_native_krylov_gmres_dense_z_cpu))
    dense_cpu_prefilled = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_arnoldi_prefilled_dense_d_cpu,
             :tenet_native_krylov_arnoldi_prefilled_dense_z_cpu))
    cg_callback = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_cg_d_cpu,
             :tenet_native_krylov_cg_z_cpu))
    cg_dense = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_cg_dense_d_cpu,
             :tenet_native_krylov_cg_dense_z_cpu))
    bicgstab_callback = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_bicgstab_d_cpu,
             :tenet_native_krylov_bicgstab_z_cpu))
    bicgstab_dense = target === :cpu &&
        all(name -> _has_symbol(lib, name; target=:cpu),
            (:tenet_native_krylov_bicgstab_dense_d_cpu,
             :tenet_native_krylov_bicgstab_dense_z_cpu))
    return (;
        abi_version=version,
        legacy_abi_version=legacy_version,
        target,
        fixed_native,
        generic_cpu_callback=generic_cpu,
        generic_cpu_prefilled_callback=generic_cpu_prefilled,
        generic_cpu_dense=dense_cpu,
        generic_cpu_prefilled_dense=dense_cpu_prefilled,
        generic_cpu_cg_callback=cg_callback,
        generic_cpu_cg_dense=cg_dense,
        generic_cpu_bicgstab_callback=bicgstab_callback,
        generic_cpu_bicgstab_dense=bicgstab_dense,
        generic_gpu_callback=false,
    )
end

function tenet_native_status_string(status::Integer; lib=nothing, target::Symbol=:cpu)
    fptr = _symbol(lib, :tenet_native_status_string; target)
    return unsafe_string(ccall(fptr, Cstring, (Cint,), Cint(status)))
end

function tenet_native_abi_version(; lib=nothing, target::Symbol=:cpu)
    fptr = _symbol(lib, :tenet_native_abi_version; target)
    return Int(ccall(fptr, Cint, ()))
end

function tenet_native_abi_version_string(; lib=nothing, target::Symbol=:cpu)
    fptr = _symbol(lib, :tenet_native_abi_version_string; target)
    return unsafe_string(ccall(fptr, Cstring, ()))
end

function _native_krylov_abi_version(; lib=nothing, target::Symbol=:cpu)
    if _has_symbol(lib, :tenet_native_krylov_abi_version; target)
        fptr = _symbol(lib, :tenet_native_krylov_abi_version; target)
        return Int(ccall(fptr, Cint, ()))
    end
    return tenet_native_abi_version(; lib, target)
end

function _native_krylov_abi_version_string(; lib=nothing, target::Symbol=:cpu)
    if _has_symbol(lib, :tenet_native_krylov_abi_version_string; target)
        fptr = _symbol(lib, :tenet_native_krylov_abi_version_string; target)
        return unsafe_string(ccall(fptr, Cstring, ()))
    end
    return tenet_native_abi_version_string(; lib, target)
end

function tenet_native_last_error(; lib=nothing, target::Symbol=:cpu)
    fptr = _symbol(lib, :tenet_native_last_error; target)
    return unsafe_string(ccall(fptr, Cstring, ()))
end

function _status_message(status::Integer; lib=nothing, target::Symbol=:cpu)
    base = tenet_native_status_string(status; lib, target)
    last = tenet_native_last_error(; lib, target)
    return isempty(last) || last == "success" ? base : "$base: $last"
end

function _check_status(status::Integer, context::AbstractString; lib=nothing,
                       target::Symbol=:cpu)
    status == 0 && return nothing
    error("$context failed: " * _status_message(status; lib, target))
end

function _check_leg3_pair(Aup::Array{Float64,3}, Adn::Array{Float64,3})
    chi, phys, chi2 = size(Aup)
    chi == chi2 || throw(DimensionMismatch("Aup must have size chi x phys x chi, got $(size(Aup))"))
    size(Adn) == (chi, phys, chi) ||
        throw(DimensionMismatch("Adn must have size $(size(Aup)), got $(size(Adn))"))
    return Int64(chi), Int64(phys)
end

function _check_matrix(X::Array{Float64,2}, dims::Tuple{Int,Int}, name::AbstractString)
    size(X) == dims || throw(DimensionMismatch("$name must have size $dims, got $(size(X))"))
    return X
end

function _rowmajor_flat_matrix(X::Matrix{Float64})
    D, D2 = size(X)
    D == D2 || throw(DimensionMismatch("row-major matrix input must be square, got $D x $D2"))
    flat = Vector{Float64}(undef, D * D)
    for i in 1:D, j in 1:D
        flat[(i - 1) * D + j] = X[i, j]
    end
    return flat
end

function _rowmajor_matrix_from_flat(flat::AbstractVector{Float64}, D::Integer)
    D > 0 || throw(ArgumentError("matrix dimension must be positive, got $D"))
    length(flat) == D * D ||
        throw(ArgumentError("row-major matrix buffer must have length $(D * D), got $(length(flat))"))
    out = Matrix{Float64}(undef, D, D)
    for i in 1:D, j in 1:D
        out[i, j] = flat[(i - 1) * D + j]
    end
    return out
end

function _rowmajor_flat_tensor(W::Array{Float64,3})
    d, D, D2 = size(W)
    D == D2 || throw(DimensionMismatch("row-major tensor input must have equal trailing dimensions, got $D x $D2"))
    flat = Vector{Float64}(undef, d * D * D)
    for s in 1:d, i in 1:D, j in 1:D
        flat[(s - 1) * D * D + (i - 1) * D + j] = W[s, i, j]
    end
    return flat
end

function _check_leg3(X::Array{Float64,3}, dims::Tuple{Int,Int,Int}, name::AbstractString)
    size(X) == dims || throw(DimensionMismatch("$name must have size $dims, got $(size(X))"))
    return X
end

function _check_leg4(M::Array{Float64,4}, phys::Int)
    dims = (phys, phys, phys, phys)
    size(M) == dims || throw(DimensionMismatch("M must have size $dims, got $(size(M))"))
    return M
end

function _resolved_max_k(max_k, len::Integer)
    k = max_k === nothing ? Int64(len) : Int64(max_k)
    k <= 0 && return Int64(len)
    1 <= k <= len || throw(ArgumentError("max_k must be in 1:$len, got $max_k"))
    return k
end

function _nonnegative_float(x, name::AbstractString)
    y = Float64(x)
    y >= 0.0 || throw(ArgumentError("$name must be nonnegative, got $x"))
    return y
end

_transpose_flag(transpose::Bool) = Cint(transpose ? 1 : 0)

function _two_layer_inputs(Aup, Adn, x0)
    chi64, phys64 = _check_leg3_pair(Aup, Adn)
    chi = Int(chi64)
    _check_matrix(x0, (chi, chi), "x0")
    return chi64, phys64, Int64(chi * chi)
end

function _projected_two_layer_inputs(Aup, Adn, rho, x0)
    chi64, phys64, len = _two_layer_inputs(Aup, Adn, x0)
    chi = Int(chi64)
    _check_matrix(rho, (chi, chi), "rho")
    return chi64, phys64, len
end

function _three_layer_inputs(Aup, Adn, M, x0)
    chi64, phys64 = _check_leg3_pair(Aup, Adn)
    chi = Int(chi64)
    phys = Int(phys64)
    _check_leg4(M, phys)
    _check_leg3(x0, (chi, phys, chi), "x0")
    return chi64, phys64, Int64(chi * phys * chi)
end

function tenet_native_raw_two_layer_apply_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    x0::Matrix{Float64};
    transpose::Bool=false,
    lib=nothing,
)
    chi, phys, _ = _two_layer_inputs(Aup, Adn, x0)
    y = Matrix{Float64}(undef, Int(chi), Int(chi))
    fptr = _symbol(lib, :tenet_native_raw_two_layer_apply_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint,
         Ptr{Float64}),
        chi,
        phys,
        Aup,
        Adn,
        x0,
        _transpose_flag(transpose),
        y,
    )
    _check_status(status, "tenet_native_raw_two_layer_apply_d_cpu"; lib)
    return y
end

function tenet_native_raw_transfer_op_d_cpu(
    W::Array{Float64,3},
    O::Matrix{Float64},
    x::Matrix{Float64};
    lib=nothing,
)
    d, D, D2 = size(W)
    D == D2 || throw(DimensionMismatch("W must have size d x D x D, got $(size(W))"))
    size(O) == (d, d) ||
        throw(DimensionMismatch("O must be size d x d, got $(size(O))"))
    _check_matrix(x, (D, D), "x")
    y = Matrix{Float64}(undef, D, D)
    fptr = _symbol(lib, :tenet_native_raw_transfer_op_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        Int64(d),
        Int64(D),
        W,
        O,
        x,
        y,
    )
    _check_status(status, "tenet_native_raw_transfer_op_d_cpu"; lib)
    return y
end

function tenet_native_raw_rowmajor_transfer_d_cpu(
    W::Array{Float64,3},
    x::Matrix{Float64};
    lib=nothing,
)
    d, D, D2 = size(W)
    D == D2 || throw(DimensionMismatch("W must have size d x D x D, got $(size(W))"))
    _check_matrix(x, (D, D), "x")
    W_rm = _rowmajor_flat_tensor(W)
    x_rm = _rowmajor_flat_matrix(x)
    y_rm = Vector{Float64}(undef, D * D)
    fptr = _symbol(lib, :tenet_native_raw_rowmajor_transfer_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        Int64(d),
        Int64(D),
        W_rm,
        x_rm,
        y_rm,
    )
    _check_status(status, "tenet_native_raw_rowmajor_transfer_d_cpu"; lib)
    return _rowmajor_matrix_from_flat(y_rm, D)
end

function tenet_native_raw_rowmajor_transfer_adj_d_cpu(
    W::Array{Float64,3},
    x::Matrix{Float64};
    lib=nothing,
)
    d, D, D2 = size(W)
    D == D2 || throw(DimensionMismatch("W must have size d x D x D, got $(size(W))"))
    _check_matrix(x, (D, D), "x")
    W_rm = _rowmajor_flat_tensor(W)
    x_rm = _rowmajor_flat_matrix(x)
    y_rm = Vector{Float64}(undef, D * D)
    fptr = _symbol(lib, :tenet_native_raw_rowmajor_transfer_adj_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        Int64(d),
        Int64(D),
        W_rm,
        x_rm,
        y_rm,
    )
    _check_status(status, "tenet_native_raw_rowmajor_transfer_adj_d_cpu"; lib)
    return _rowmajor_matrix_from_flat(y_rm, D)
end

function tenet_native_raw_rowmajor_transfer_op_d_cpu(
    W::Array{Float64,3},
    O::Matrix{Float64},
    x::Matrix{Float64};
    lib=nothing,
)
    d, D, D2 = size(W)
    D == D2 || throw(DimensionMismatch("W must have size d x D x D, got $(size(W))"))
    size(O) == (d, d) ||
        throw(DimensionMismatch("O must be size d x d, got $(size(O))"))
    _check_matrix(x, (D, D), "x")
    W_rm = _rowmajor_flat_tensor(W)
    O_rm = _rowmajor_flat_matrix(O)
    x_rm = _rowmajor_flat_matrix(x)
    y_rm = Vector{Float64}(undef, D * D)
    fptr = _symbol(lib, :tenet_native_raw_rowmajor_transfer_op_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        Int64(d),
        Int64(D),
        W_rm,
        O_rm,
        x_rm,
        y_rm,
    )
    _check_status(status, "tenet_native_raw_rowmajor_transfer_op_d_cpu"; lib)
    return _rowmajor_matrix_from_flat(y_rm, D)
end

function _vumps_inputs(M, AL, AR, C, FL, FR)
    chi64, phys64 = _check_leg3_pair(AL, AR)
    chi = Int(chi64)
    phys = Int(phys64)
    _check_leg4(M, phys)
    _check_matrix(C, (chi, chi), "C")
    _check_leg3(FL, (chi, phys, chi), "FL")
    _check_leg3(FR, (chi, phys, chi), "FR")
    return chi64, phys64, Int64(max(chi * chi, chi * phys * chi))
end

function _basis_result(V, H, beta, m, final_resnorm)
    return (V=V, H=H, m=Int(m[]), beta=beta[], final_resnorm=final_resnorm[])
end

function _cuda_binding(owner::Module)
    isdefined(owner, :CUDA) || return nothing
    mod = getproperty(owner, :CUDA)
    mod isa Module && nameof(mod) === :CUDA || return nothing
    _cuda_module_ref[] = mod
    return mod
end

function _cuda_module()
    mod = _cuda_module_ref[]
    mod isa Module && return mod
    for owner in (parentmodule(@__MODULE__), Main)
        found = _cuda_binding(owner)
        found === nothing || return found
    end
    if isdefined(Base, :loaded_modules)
        try
            loaded = getproperty(Base, :loaded_modules)
            modules = loaded isa Function ? values(loaded()) : values(loaded)
            for candidate in modules
                candidate isa Module && nameof(candidate) === :CUDA || continue
                _cuda_module_ref[] = candidate
                return candidate
            end
        catch err
            err isa InterruptException && rethrow()
        end
    end
    for owner in (parentmodule(@__MODULE__), @__MODULE__, Main)
        try
            loaded = Base.require(owner, :CUDA)
            loaded isa Module || continue
            _cuda_module_ref[] = loaded
            return loaded
        catch err
            err isa InterruptException && rethrow()
            err isa ArgumentError || rethrow()
        end
    end
    throw(ArgumentError("TenetNative CUDA wrappers require CUDA.jl to be loaded or available in the active environment"))
end

function _ensure_cuda_wrappers!()
    _cuda_wrappers_defined[] && return _cuda_module()
    cuda = _cuda_module()
    if !isdefined(@__MODULE__, :CUDA)
        @eval const CUDA = $cuda
    end
    @eval begin
        function _cuda_with_device(f::Function, A)
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

        function _check_cuda_array(A, dims::Tuple, name::AbstractString)
            A isa CUDA.CuArray ||
                throw(ArgumentError("$name must be a CUDA.CuArray, got $(typeof(A))"))
            eltype(A) === Float64 ||
                throw(ArgumentError("$name must have eltype Float64, got $(eltype(A))"))
            size(A) == dims ||
                throw(DimensionMismatch("$name must have size $dims, got $(size(A))"))
            expected = 1
            for dim in 1:ndims(A)
                stride(A, dim) == expected || return copy(A)
                expected *= size(A, dim)
            end
            return A
        end

        function _check_cuda_leg3_pair(Aup, Adn)
            Aup isa CUDA.CuArray ||
                throw(ArgumentError("Aup must be a CUDA.CuArray, got $(typeof(Aup))"))
            ndims(Aup) == 3 ||
                throw(DimensionMismatch("Aup must be rank-3, got rank $(ndims(Aup))"))
            chi, phys, chi2 = size(Aup)
            chi == chi2 ||
                throw(DimensionMismatch("Aup must have size chi x phys x chi, got $(size(Aup))"))
            A = _check_cuda_array(Aup, (chi, phys, chi), "Aup")
            B = _check_cuda_array(Adn, (chi, phys, chi), "Adn")
            return Int64(chi), Int64(phys), A, B
        end

        function _cuda_two_layer_inputs(Aup, Adn, x0)
            chi64, phys64, A, B = _check_cuda_leg3_pair(Aup, Adn)
            chi = Int(chi64)
            X = _check_cuda_array(x0, (chi, chi), "x0")
            return chi64, phys64, Int64(chi * chi), A, B, X
        end

        function _cuda_projected_two_layer_inputs(Aup, Adn, rho, x0)
            chi64, phys64, len, A, B, X = _cuda_two_layer_inputs(Aup, Adn, x0)
            chi = Int(chi64)
            Rho = _check_cuda_array(rho, (chi, chi), "rho")
            return chi64, phys64, len, A, B, Rho, X
        end

        function _check_cuda_leg3_or_batch(A, chi::Integer, phys::Integer,
                                           batch::Integer,
                                           name::AbstractString)
            A isa CUDA.CuArray ||
                throw(ArgumentError("$name must be a CUDA.CuArray, got $(typeof(A))"))
            if ndims(A) == 3
                Ac = _check_cuda_array(A, (chi, phys, chi), name)
                return Ac, Int64(0)
            elseif ndims(A) == 4
                Ac = _check_cuda_array(A, (chi, phys, chi, batch), name)
                return Ac, Int64(stride(Ac, 4))
            else
                throw(DimensionMismatch("$name must be rank-3 shared or rank-4 batched, got rank $(ndims(A))"))
            end
        end

        function _check_cuda_rho_or_batch(rho, chi::Integer, batch::Integer)
            rho isa CUDA.CuArray ||
                throw(ArgumentError("rho must be a CUDA.CuArray, got $(typeof(rho))"))
            if ndims(rho) == 2
                Rho = _check_cuda_array(rho, (chi, chi), "rho")
                return Rho, Int64(0)
            elseif ndims(rho) == 3
                Rho = _check_cuda_array(rho, (chi, chi, batch), "rho")
                return Rho, Int64(stride(Rho, 3))
            else
                throw(DimensionMismatch("rho must be rank-2 shared or rank-3 batched, got rank $(ndims(rho))"))
            end
        end

        function _cuda_two_layer_batch_inputs(Aup, Adn, X)
            Aup isa CUDA.CuArray ||
                throw(ArgumentError("Aup must be a CUDA.CuArray, got $(typeof(Aup))"))
            ndims(Aup) in (3, 4) ||
                throw(DimensionMismatch("Aup must be rank-3 shared or rank-4 batched, got rank $(ndims(Aup))"))
            chi, phys, chi2 = size(Aup, 1), size(Aup, 2), size(Aup, 3)
            chi == chi2 ||
                throw(DimensionMismatch("Aup must have size chi x phys x chi[, batch], got $(size(Aup))"))
            X isa CUDA.CuArray ||
                throw(ArgumentError("X must be a CUDA.CuArray, got $(typeof(X))"))
            ndims(X) == 3 ||
                throw(DimensionMismatch("X must have size chi x chi x batch, got rank $(ndims(X))"))
            size(X, 1) == chi && size(X, 2) == chi ||
                throw(DimensionMismatch("X must have leading size ($chi, $chi), got $(size(X))"))
            batch = size(X, 3)
            ndims(Aup) == 4 && size(Aup, 4) == batch ||
                ndims(Aup) == 3 ||
                throw(DimensionMismatch("Aup batch $(size(Aup, 4)) != X batch $batch"))
            Xc = _check_cuda_array(X, (chi, chi, batch), "X")
            A, stride_A = _check_cuda_leg3_or_batch(Aup, chi, phys, batch, "Aup")
            B, stride_B = _check_cuda_leg3_or_batch(Adn, chi, phys, batch, "Adn")
            return Int64(batch), Int64(chi), Int64(phys), Int64(chi * chi),
                   A, stride_A, B, stride_B, Xc, Int64(stride(Xc, 3))
        end

        function _cuda_projected_two_layer_batch_inputs(Aup, Adn, rho, X)
            batch, chi64, phys64, len, A, stride_A, B, stride_B, Xc, stride_X =
                _cuda_two_layer_batch_inputs(Aup, Adn, X)
            Rho, stride_Rho = _check_cuda_rho_or_batch(rho, Int(chi64), Int(batch))
            return batch, chi64, phys64, len, A, stride_A, B, stride_B,
                   Rho, stride_Rho, Xc, stride_X
        end

        function _cuda_three_layer_inputs(Aup, Adn, M, x0)
            chi64, phys64, A, B = _check_cuda_leg3_pair(Aup, Adn)
            chi = Int(chi64)
            phys = Int(phys64)
            Mc = _check_cuda_array(M, (phys, phys, phys, phys), "M")
            X = _check_cuda_array(x0, (chi, phys, chi), "x0")
            return chi64, phys64, Int64(chi * phys * chi), A, B, Mc, X
        end

        function _cuda_vumps_inputs(M, AL, AR, C, FL, FR)
            chi64, phys64, ALc, ARc = _check_cuda_leg3_pair(AL, AR)
            chi = Int(chi64)
            phys = Int(phys64)
            Mc = _check_cuda_array(M, (phys, phys, phys, phys), "M")
            Cc = _check_cuda_array(C, (chi, chi), "C")
            FLc = _check_cuda_array(FL, (chi, phys, chi), "FL")
            FRc = _check_cuda_array(FR, (chi, phys, chi), "FR")
            return chi64, phys64, Int64(max(chi * chi, chi * phys * chi)),
                   Mc, ALc, ARc, Cc, FLc, FRc
        end

        function _cuda_two_layer_basis_impl(symbol::Symbol,
                                            context::AbstractString,
                                            Aup, Adn, x0;
                                            max_k=nothing,
                                            breakdown_tol::Real=1e-12,
                                            transpose::Bool=false,
                                            lib=nothing)
            chi, phys, len, A, B, X = _cuda_two_layer_inputs(Aup, Adn, x0)
            k = _resolved_max_k(max_k, len)
            V = CUDA.CuArray{Float64}(undef, Int(len), Int(k) + 1)
            H = zeros(Float64, Int(k) + 1, Int(k))
            beta = Ref{Float64}(0.0)
            m = Ref{Int64}(0)
            final_resnorm = Ref{Float64}(0.0)
            fptr = _symbol(lib, symbol; target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, Int64, Float64, Cint,
                     CUDA.CuPtr{Float64}, Int64, Ptr{Float64}, Int64,
                     Ref{Float64}, Ref{Int64}, Ref{Float64}),
                    chi, phys, A, B, X, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _transpose_flag(transpose), V, Int64(stride(V, 2)), H,
                    Int64(stride(H, 2)), beta, m, final_resnorm,
                )
            end
            _check_status(status, context; lib, target=:cuda)
            return _basis_result(V, H, beta, m, final_resnorm)
        end

        function _tenet_native_arnoldi_two_layer_d_cuda_impl(Aup, Adn, x0; kwargs...)
            return _cuda_two_layer_basis_impl(
                :tenet_native_arnoldi_two_layer_d_cuda,
                "tenet_native_arnoldi_two_layer_d_cuda",
                Aup, Adn, x0;
                kwargs...,
            )
        end

        # The header declares this symbol, but current CUDA libraries may not
        # export it. Probe before calling so older libraries fail clearly.
        function _tenet_native_arnoldi_two_layer_ritz_d_cuda_impl(
            Aup, Adn, x0;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            transpose::Bool=false,
            nvalues::Integer=2,
            lib=nothing,
        )
            chi, phys, len, A, B, X = _cuda_two_layer_inputs(Aup, Adn, x0)
            k = _resolved_max_k(max_k, len)
            nvalues64 = Int64(nvalues)
            nvalues64 > 0 || throw(ArgumentError("nvalues must be positive, got $nvalues"))
            fptr = try
                _symbol(lib, :tenet_native_arnoldi_two_layer_ritz_d_cuda; target=:cuda)
            catch err
                err isa InterruptException && rethrow()
                throw(ArgumentError("tenet_native_arnoldi_two_layer_ritz_d_cuda is declared in the CUDA ABI but not exported by the selected native library"))
            end
            lambda_real = Vector{Float64}(undef, Int(nvalues64))
            lambda_imag = Vector{Float64}(undef, Int(nvalues64))
            m = Ref{Int64}(0)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, Int64, Float64, Cint, Int64,
                     Ptr{Float64}, Ptr{Float64}, Ref{Int64}),
                    chi, phys, A, B, X, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _transpose_flag(transpose), nvalues64, lambda_real,
                    lambda_imag, m,
                )
            end
            _check_status(status, "tenet_native_arnoldi_two_layer_ritz_d_cuda";
                          lib, target=:cuda)
            return (values=complex.(lambda_real, lambda_imag), m=Int(m[]))
        end

        function _tenet_native_arnoldi_projected_two_layer_d_cuda_impl(
            Aup, Adn, rho, x0;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            transpose::Bool=false,
            lib=nothing,
        )
            chi, phys, len, A, B, Rho, X =
                _cuda_projected_two_layer_inputs(Aup, Adn, rho, x0)
            k = _resolved_max_k(max_k, len)
            V = CUDA.CuArray{Float64}(undef, Int(len), Int(k) + 1)
            H = zeros(Float64, Int(k) + 1, Int(k))
            beta = Ref{Float64}(0.0)
            m = Ref{Int64}(0)
            final_resnorm = Ref{Float64}(0.0)
            fptr = _symbol(lib, :tenet_native_arnoldi_projected_two_layer_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64,
                     Float64, Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64},
                     Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
                    chi, phys, A, B, Rho, X, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _transpose_flag(transpose), V, Int64(stride(V, 2)), H,
                    Int64(stride(H, 2)), beta, m, final_resnorm,
                )
            end
            _check_status(status, "tenet_native_arnoldi_projected_two_layer_d_cuda";
                          lib, target=:cuda)
            return _basis_result(V, H, beta, m, final_resnorm)
        end

        function _tenet_native_arnoldi_qprojected_two_layer_d_cuda_impl(
            Aup, Adn, rho, x0;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            transpose::Bool=false,
            lib=nothing,
        )
            chi, phys, len, A, B, Rho, X =
                _cuda_projected_two_layer_inputs(Aup, Adn, rho, x0)
            k = _resolved_max_k(max_k, len)
            V = CUDA.CuArray{Float64}(undef, Int(len), Int(k) + 1)
            H = zeros(Float64, Int(k) + 1, Int(k))
            beta = Ref{Float64}(0.0)
            m = Ref{Int64}(0)
            final_resnorm = Ref{Float64}(0.0)
            fptr = _symbol(lib, :tenet_native_arnoldi_qprojected_two_layer_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64,
                     Float64, Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64},
                     Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
                    chi, phys, A, B, Rho, X, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _transpose_flag(transpose), V, Int64(stride(V, 2)), H,
                    Int64(stride(H, 2)), beta, m, final_resnorm,
                )
            end
            _check_status(status, "tenet_native_arnoldi_qprojected_two_layer_d_cuda";
                          lib, target=:cuda)
            return _basis_result(V, H, beta, m, final_resnorm)
        end

        function _tenet_native_two_layer_apply_batch_d_cuda_impl(
            Aup, Adn, X; transpose::Bool=false, lib=nothing)
            batch, chi, phys, _len, A, stride_A, B, stride_B, Xc, stride_X =
                _cuda_two_layer_batch_inputs(Aup, Adn, X)
            Y = CUDA.CuArray{Float64}(undef, Int(chi), Int(chi), Int(batch))
            fptr = _symbol(lib, :tenet_native_two_layer_apply_batch_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, Int64, CUDA.CuPtr{Float64}, Int64,
                     CUDA.CuPtr{Float64}, Int64, CUDA.CuPtr{Float64}, Int64,
                     Cint, CUDA.CuPtr{Float64}, Int64),
                    batch, chi, phys, A, stride_A, B, stride_B, Xc, stride_X,
                    _transpose_flag(transpose), Y, Int64(stride(Y, 3)),
                )
            end
            _check_status(status, "tenet_native_two_layer_apply_batch_d_cuda";
                          lib, target=:cuda)
            return Y
        end

        function _tenet_native_raw_two_layer_apply_batch_d_cuda_impl(
            Aup, Adn, X; transpose::Bool=false, lib=nothing)
            batch, chi, phys, _len, A, stride_A, B, stride_B, Xc, stride_X =
                _cuda_two_layer_batch_inputs(Aup, Adn, X)
            Y = CUDA.CuArray{Float64}(undef, Int(chi), Int(chi), Int(batch))
            fptr = _symbol(lib, :tenet_native_raw_two_layer_apply_batch_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, Int64, CUDA.CuPtr{Float64}, Int64,
                     CUDA.CuPtr{Float64}, Int64, CUDA.CuPtr{Float64}, Int64,
                     Cint, CUDA.CuPtr{Float64}, Int64),
                    batch, chi, phys, A, stride_A, B, stride_B, Xc, stride_X,
                    _transpose_flag(transpose), Y, Int64(stride(Y, 3)),
                )
            end
            _check_status(status, "tenet_native_raw_two_layer_apply_batch_d_cuda";
                          lib, target=:cuda)
            return Y
        end

        function _tenet_native_projected_two_layer_apply_batch_d_cuda_impl(
            Aup, Adn, rho, X; transpose::Bool=false, lib=nothing)
            batch, chi, phys, _len, A, stride_A, B, stride_B, Rho, stride_Rho,
                Xc, stride_X =
                _cuda_projected_two_layer_batch_inputs(Aup, Adn, rho, X)
            Y = CUDA.CuArray{Float64}(undef, Int(chi), Int(chi), Int(batch))
            fptr = _symbol(lib,
                           :tenet_native_projected_two_layer_apply_batch_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, Int64, CUDA.CuPtr{Float64}, Int64,
                     CUDA.CuPtr{Float64}, Int64, CUDA.CuPtr{Float64}, Int64,
                     CUDA.CuPtr{Float64}, Int64, Cint, CUDA.CuPtr{Float64},
                     Int64),
                    batch, chi, phys, A, stride_A, B, stride_B, Rho,
                    stride_Rho, Xc, stride_X, _transpose_flag(transpose), Y,
                    Int64(stride(Y, 3)),
                )
            end
            _check_status(status,
                          "tenet_native_projected_two_layer_apply_batch_d_cuda";
                          lib, target=:cuda)
            return Y
        end

        function _tenet_native_qprojected_two_layer_apply_batch_d_cuda_impl(
            Aup, Adn, rho, X; transpose::Bool=false, lib=nothing)
            batch, chi, phys, _len, A, stride_A, B, stride_B, Rho, stride_Rho,
                Xc, stride_X =
                _cuda_projected_two_layer_batch_inputs(Aup, Adn, rho, X)
            Y = CUDA.CuArray{Float64}(undef, Int(chi), Int(chi), Int(batch))
            fptr = _symbol(
                lib, :tenet_native_qprojected_two_layer_apply_batch_d_cuda;
                target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, Int64, CUDA.CuPtr{Float64}, Int64,
                     CUDA.CuPtr{Float64}, Int64, CUDA.CuPtr{Float64}, Int64,
                     CUDA.CuPtr{Float64}, Int64, Cint, CUDA.CuPtr{Float64},
                     Int64),
                    batch, chi, phys, A, stride_A, B, stride_B, Rho,
                    stride_Rho, Xc, stride_X, _transpose_flag(transpose), Y,
                    Int64(stride(Y, 3)),
                )
            end
            _check_status(status,
                          "tenet_native_qprojected_two_layer_apply_batch_d_cuda";
                          lib, target=:cuda)
            return Y
        end

        function _tenet_native_arnoldi_three_layer_leg4_d_cuda_impl(
            Aup, Adn, M, x0;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            transpose::Bool=false,
            lib=nothing,
        )
            chi, phys, len, A, B, Mc, X = _cuda_three_layer_inputs(Aup, Adn, M, x0)
            k = _resolved_max_k(max_k, len)
            V = CUDA.CuArray{Float64}(undef, Int(len), Int(k) + 1)
            H = zeros(Float64, Int(k) + 1, Int(k))
            beta = Ref{Float64}(0.0)
            m = Ref{Int64}(0)
            final_resnorm = Ref{Float64}(0.0)
            fptr = _symbol(lib, :tenet_native_arnoldi_three_layer_leg4_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
                     Cint, CUDA.CuPtr{Float64}, Int64, Ptr{Float64}, Int64,
                     Ref{Float64}, Ref{Int64}, Ref{Float64}),
                    chi, phys, A, B, Mc, X, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _transpose_flag(transpose), V, Int64(stride(V, 2)), H,
                    Int64(stride(H, 2)), beta, m, final_resnorm,
                )
            end
            _check_status(status, "tenet_native_arnoldi_three_layer_leg4_d_cuda";
                          lib, target=:cuda)
            return _basis_result(V, H, beta, m, final_resnorm)
        end

        function _cuda_two_layer_vector(symbol::Symbol,
                                        context::AbstractString,
                                        Aup, Adn, x0;
                                        max_k=nothing,
                                        breakdown_tol::Real=1e-12,
                                        transpose::Bool=false,
                                        lib=nothing)
            chi, phys, len, A, B, X = _cuda_two_layer_inputs(Aup, Adn, x0)
            k = _resolved_max_k(max_k, len)
            y = CUDA.CuArray{Float64}(undef, Int(chi), Int(chi))
            lambda = Ref{Float64}(0.0)
            fptr = _symbol(lib, symbol; target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, Int64, Float64, Cint,
                     CUDA.CuPtr{Float64}, Ref{Float64}),
                    chi, phys, A, B, X, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _transpose_flag(transpose), y, lambda,
                )
            end
            _check_status(status, context; lib, target=:cuda)
            return (lambda=lambda[], y=y)
        end

        function _tenet_native_dominant_two_layer_d_cuda_impl(Aup, Adn, x0; kwargs...)
            return _cuda_two_layer_vector(
                :tenet_native_dominant_two_layer_d_cuda,
                "tenet_native_dominant_two_layer_d_cuda",
                Aup, Adn, x0;
                kwargs...,
            )
        end

        function _cuda_three_layer_vector(symbol::Symbol,
                                          context::AbstractString,
                                          Aup, Adn, M, x0;
                                          max_k=nothing,
                                          breakdown_tol::Real=1e-12,
                                          transpose::Bool=false,
                                          lib=nothing)
            chi, phys, len, A, B, Mc, X = _cuda_three_layer_inputs(Aup, Adn, M, x0)
            k = _resolved_max_k(max_k, len)
            y = CUDA.CuArray{Float64}(undef, Int(chi), Int(phys), Int(chi))
            lambda = Ref{Float64}(0.0)
            fptr = _symbol(lib, symbol; target=:cuda)
            status = _cuda_with_device(A) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
                     Cint, CUDA.CuPtr{Float64}, Ref{Float64}),
                    chi, phys, A, B, Mc, X, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _transpose_flag(transpose), y, lambda,
                )
            end
            _check_status(status, context; lib, target=:cuda)
            return (lambda=lambda[], y=y)
        end

        function _tenet_native_dominant_three_layer_leg4_d_cuda_impl(
            Aup, Adn, M, x0;
            kwargs...,
        )
            return _cuda_three_layer_vector(
                :tenet_native_dominant_three_layer_leg4_d_cuda,
                "tenet_native_dominant_three_layer_leg4_d_cuda",
                Aup, Adn, M, x0;
                kwargs...,
            )
        end

        function _tenet_native_ising_vumps_step_d_cuda_impl(
            M, AL, AR, C, FL, FR;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            lib=nothing,
        )
            chi, phys, len, Mc, ALc, ARc, Cc, FLc, FRc =
                _cuda_vumps_inputs(M, AL, AR, C, FL, FR)
            k = _resolved_max_k(max_k, len)
            err = Ref{Float64}(0.0)
            fptr = _symbol(lib, :tenet_native_ising_vumps_step_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(ALc) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
                     Ref{Float64}),
                    chi, phys, Mc, ALc, ARc, Cc, FLc, FRc, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"), err,
                )
            end
            _check_status(status, "tenet_native_ising_vumps_step_d_cuda";
                          lib, target=:cuda)
            return (AL=ALc, AR=ARc, C=Cc, FL=FLc, FR=FRc, err=err[])
        end

        function _tenet_native_ising_vumps_step_checked_d_cuda_impl(
            M, AL, AR, C, FL, FR;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            residual_tol::Real=1e-8,
            lib=nothing,
        )
            chi, phys, len, Mc, ALc, ARc, Cc, FLc, FRc =
                _cuda_vumps_inputs(M, AL, AR, C, FL, FR)
            k = _resolved_max_k(max_k, len)
            err = Ref{Float64}(0.0)
            fptr = _symbol(lib, :tenet_native_ising_vumps_step_checked_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(ALc) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
                     Float64, Ref{Float64}),
                    chi, phys, Mc, ALc, ARc, Cc, FLc, FRc, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _nonnegative_float(residual_tol, "residual_tol"), err,
                )
            end
            _check_status(status, "tenet_native_ising_vumps_step_checked_d_cuda";
                          lib, target=:cuda)
            return (AL=ALc, AR=ARc, C=Cc, FL=FLc, FR=FRc, err=err[])
        end

        function _tenet_native_ising_vumps_run_d_cuda_impl(
            M, AL, AR, C, FL, FR;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            tol::Real=1e-10,
            miniter::Integer=1,
            maxiter::Integer=100,
            lib=nothing,
        )
            chi, phys, len, Mc, ALc, ARc, Cc, FLc, FRc =
                _cuda_vumps_inputs(M, AL, AR, C, FL, FR)
            k = _resolved_max_k(max_k, len)
            err = Ref{Float64}(0.0)
            iterations = Ref{Int64}(0)
            converged = Ref{Cint}(0)
            fptr = _symbol(lib, :tenet_native_ising_vumps_run_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(ALc) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
                     Float64, Int64, Int64, Ref{Float64}, Ref{Int64},
                     Ref{Cint}),
                    chi, phys, Mc, ALc, ARc, Cc, FLc, FRc, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _nonnegative_float(tol, "tol"), Int64(miniter),
                    Int64(maxiter), err, iterations, converged,
                )
            end
            _check_status(status, "tenet_native_ising_vumps_run_d_cuda";
                          lib, target=:cuda)
            return (AL=ALc, AR=ARc, C=Cc, FL=FLc, FR=FRc, err=err[],
                    iterations=Int(iterations[]), converged=converged[] != 0)
        end

        function _tenet_native_ising_vumps_run_checked_d_cuda_impl(
            M, AL, AR, C, FL, FR;
            max_k=nothing,
            breakdown_tol::Real=1e-12,
            tol::Real=1e-10,
            miniter::Integer=1,
            maxiter::Integer=100,
            residual_tol::Real=1e-8,
            lib=nothing,
        )
            chi, phys, len, Mc, ALc, ARc, Cc, FLc, FRc =
                _cuda_vumps_inputs(M, AL, AR, C, FL, FR)
            k = _resolved_max_k(max_k, len)
            err = Ref{Float64}(0.0)
            iterations = Ref{Int64}(0)
            converged = Ref{Cint}(0)
            fptr = _symbol(lib, :tenet_native_ising_vumps_run_checked_d_cuda;
                           target=:cuda)
            status = _cuda_with_device(ALc) do
                ccall(
                    fptr,
                    Cint,
                    (Int64, Int64, CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64},
                     CUDA.CuPtr{Float64}, CUDA.CuPtr{Float64}, Int64, Float64,
                     Float64, Int64, Int64, Float64, Ref{Float64},
                     Ref{Int64}, Ref{Cint}),
                    chi, phys, Mc, ALc, ARc, Cc, FLc, FRc, k,
                    _nonnegative_float(breakdown_tol, "breakdown_tol"),
                    _nonnegative_float(tol, "tol"), Int64(miniter),
                    Int64(maxiter), _nonnegative_float(residual_tol, "residual_tol"),
                    err, iterations, converged,
                )
            end
            _check_status(status, "tenet_native_ising_vumps_run_checked_d_cuda";
                          lib, target=:cuda)
            return (AL=ALc, AR=ARc, C=Cc, FL=FLc, FR=FRc, err=err[],
                    iterations=Int(iterations[]), converged=converged[] != 0)
        end
    end
    _cuda_wrappers_defined[] = true
    return cuda
end

function _cuda_invoke(impl::Symbol, args...; kwargs...)
    _ensure_cuda_wrappers!()
    f = getproperty(@__MODULE__, impl)
    return Base.invokelatest(f, args...; kwargs...)
end

function tenet_native_arnoldi_two_layer_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_arnoldi_two_layer_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_arnoldi_two_layer_ritz_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_arnoldi_two_layer_ritz_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_arnoldi_projected_two_layer_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_arnoldi_projected_two_layer_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_arnoldi_qprojected_two_layer_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_arnoldi_qprojected_two_layer_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_two_layer_apply_batch_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_two_layer_apply_batch_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_raw_two_layer_apply_batch_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_raw_two_layer_apply_batch_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_projected_two_layer_apply_batch_d_cuda(args...; kwargs...)
    return _cuda_invoke(
        :_tenet_native_projected_two_layer_apply_batch_d_cuda_impl,
        args...; kwargs...)
end

function tenet_native_qprojected_two_layer_apply_batch_d_cuda(args...; kwargs...)
    return _cuda_invoke(
        :_tenet_native_qprojected_two_layer_apply_batch_d_cuda_impl,
        args...; kwargs...)
end

function tenet_native_arnoldi_three_layer_leg4_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_arnoldi_three_layer_leg4_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_dominant_two_layer_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_dominant_two_layer_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_dominant_three_layer_leg4_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_dominant_three_layer_leg4_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_ising_vumps_step_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_ising_vumps_step_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_ising_vumps_step_checked_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_ising_vumps_step_checked_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_ising_vumps_run_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_ising_vumps_run_d_cuda_impl,
                        args...; kwargs...)
end

function tenet_native_ising_vumps_run_checked_d_cuda(args...; kwargs...)
    return _cuda_invoke(:_tenet_native_ising_vumps_run_checked_d_cuda_impl,
                        args...; kwargs...)
end

mutable struct _NativeKrylovCallbackContext
    f::Any
    shape::Tuple
    failure::Any
end

function _native_krylov_call_shape(x)
    return size(x)
end

function _native_krylov_flat(x::Array{T}) where {T<:Union{Float64,ComplexF64}}
    return vec(x)
end

function _native_krylov_backend(A_or_f, xs...; backend::Symbol)
    backend in (:auto, :cpu, :cuda) ||
        throw(ArgumentError("backend must be :auto, :cpu, or :cuda, got $backend"))
    if backend === :cuda
        throw(ArgumentError("TenetNative generic GPU callback API is unavailable in CPU-first v1; existing fixed CUDA native MPS paths remain available"))
    end
    for x in xs
        x isa Array && continue
        throw(ArgumentError("TenetNative generic Krylov v1 supports CPU Array inputs only, got $(typeof(x))"))
    end
    return :cpu
end

function _native_krylov_scalar(x::Array{T}; scalar::Symbol=:auto,
                               promote_complex::Bool=false) where {T}
    scalar in (:auto, :float64, :complexf64) ||
        throw(ArgumentError("scalar must be :auto, :float64, or :complexf64, got $scalar"))
    if scalar === :auto
        promote_complex && T === Float64 && return ComplexF64
        T === Float64 && return Float64
        T === ComplexF64 && return ComplexF64
        throw(ArgumentError("TenetNative generic Krylov supports Float64 and ComplexF64, got $T"))
    elseif scalar === :float64
        T === Float64 || throw(ArgumentError("scalar=:float64 requires Float64 input, got $T"))
        return Float64
    else
        (T === Float64 || T === ComplexF64) ||
            throw(ArgumentError("scalar=:complexf64 requires Float64 or ComplexF64 input, got $T"))
        return ComplexF64
    end
end

function _native_krylov_operator(A_or_f, shape::Tuple, ::Type{T}) where {T}
    n = prod(shape)
    if A_or_f isa AbstractMatrix
        size(A_or_f) == (n, n) ||
            throw(DimensionMismatch("matrix operator must have size ($n, $n), got $(size(A_or_f))"))
        A = Matrix{T}(A_or_f)
        return x -> reshape(A * vec(x), shape)
    elseif A_or_f isa Function || applicable(A_or_f, reshape(zeros(T, n), shape))
        return A_or_f
    end
    throw(ArgumentError("A_or_f must be an AbstractMatrix or callable object, got $(typeof(A_or_f))"))
end

function _copy_callback_output!(yvec, yobj, n::Int, ::Type{T}) where {T}
    length(yobj) == n ||
        throw(DimensionMismatch("matvec returned length $(length(yobj)), expected $n"))
    copyto!(yvec, vec(yobj))
    return nothing
end

function _matvec_d_cpu_callback(n::Int64, xptr::Ptr{Float64},
                                yptr::Ptr{Float64}, ctxptr::Ptr{Cvoid})::Cint
    ctx = unsafe_pointer_to_objref(ctxptr)::_NativeKrylovCallbackContext
    try
        xvec = unsafe_wrap(Array, xptr, (Int(n),); own=false)
        yvec = unsafe_wrap(Array, yptr, (Int(n),); own=false)
        xobj = reshape(xvec, ctx.shape)
        yobj = ctx.f(xobj)
        _copy_callback_output!(yvec, yobj, Int(n), Float64)
        return Cint(0)
    catch err
        ctx.failure = (err, catch_backtrace())
        return Cint(1)
    end
end

function _matvec_z_cpu_callback(n::Int64, xptr::Ptr{ComplexF64},
                                yptr::Ptr{ComplexF64}, ctxptr::Ptr{Cvoid})::Cint
    ctx = unsafe_pointer_to_objref(ctxptr)::_NativeKrylovCallbackContext
    try
        xvec = unsafe_wrap(Array, xptr, (Int(n),); own=false)
        yvec = unsafe_wrap(Array, yptr, (Int(n),); own=false)
        xobj = reshape(xvec, ctx.shape)
        yobj = ctx.f(xobj)
        _copy_callback_output!(yvec, yobj, Int(n), ComplexF64)
        return Cint(0)
    catch err
        ctx.failure = (err, catch_backtrace())
        return Cint(1)
    end
end

_matvec_d_cpu_c() = @cfunction(
    _matvec_d_cpu_callback, Cint,
    (Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Cvoid}))
_matvec_z_cpu_c() = @cfunction(
    _matvec_z_cpu_callback, Cint,
    (Int64, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Cvoid}))

function _throw_callback_failure(ctx::_NativeKrylovCallbackContext)
    ctx.failure === nothing && return nothing
    err, bt = ctx.failure
    throw(err)
end

function _native_dense_matrix(A_or_f, shape::Tuple, ::Type{T}) where {T}
    A_or_f isa AbstractMatrix || return nothing
    n = prod(shape)
    size(A_or_f) == (n, n) ||
        throw(DimensionMismatch("matrix operator must have size ($n, $n), got $(size(A_or_f))"))
    return Matrix{T}(A_or_f)
end

function _native_arnoldi_dense_cpu(A::Matrix{T}, x0::Array{T};
                                   krylovdim::Integer=30,
                                   tol::Real=1e-12,
                                   lib=nothing) where {T<:Union{Float64,ComplexF64}}
    n = length(x0)
    size(A) == (n, n) ||
        throw(DimensionMismatch("matrix operator must have size ($n, $n), got $(size(A))"))
    k = Int64(min(max(Int(krylovdim), 1), n))
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    final_resnorm = Ref{Float64}(0.0)
    numops = Ref{Int64}(0)
    x0v = Vector{T}(vec(x0))
    if T === Float64
        V = Matrix{Float64}(undef, n, Int(k) + 1)
        H = Matrix{Float64}(undef, Int(k) + 1, Int(k))
        fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_dense_d_cpu)
        status = GC.@preserve A x0v V H begin
            ccall(fptr, Cint,
                  (Int64, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Float64,
                   Ptr{Float64}, Int64, Ptr{Float64}, Int64, Ref{Float64},
                   Ref{Int64}, Ref{Float64}, Ref{Int64}),
                  Int64(n), A, Int64(size(A, 1)), x0v, k,
                  _nonnegative_float(tol, "tol"), V, Int64(size(V, 1)), H,
                  Int64(size(H, 1)), beta, m, final_resnorm, numops)
        end
        _check_status(status, "tenet_native_krylov_arnoldi_dense_d_cpu"; lib)
        return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
                numops=Int(numops[]), path=:generic_cpu_dense)
    end
    V = Matrix{ComplexF64}(undef, n, Int(k) + 1)
    H = Matrix{ComplexF64}(undef, Int(k) + 1, Int(k))
    fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_dense_z_cpu)
    status = GC.@preserve A x0v V H begin
        ccall(fptr, Cint,
              (Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Float64,
               Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Ref{Float64},
               Ref{Int64}, Ref{Float64}, Ref{Int64}),
              Int64(n), A, Int64(size(A, 1)), x0v, k,
              _nonnegative_float(tol, "tol"), V, Int64(size(V, 1)), H,
              Int64(size(H, 1)), beta, m, final_resnorm, numops)
    end
    _check_status(status, "tenet_native_krylov_arnoldi_dense_z_cpu"; lib)
    return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
            numops=Int(numops[]), path=:generic_cpu_dense)
end

function _native_arnoldi_cpu(A_or_f, x0::Array{T};
                             krylovdim::Integer=30,
                             tol::Real=1e-12,
                             lib=nothing) where {T<:Union{Float64,ComplexF64}}
    n = length(x0)
    k = Int64(min(max(Int(krylovdim), 1), n))
    dense_A = _native_dense_matrix(A_or_f, size(x0), T)
    dense_A === nothing || return _native_arnoldi_dense_cpu(
        dense_A, x0; krylovdim, tol, lib)
    op = _native_krylov_operator(A_or_f, size(x0), T)
    ctx = _NativeKrylovCallbackContext(op, size(x0), nothing)
    ctxptr = pointer_from_objref(ctx)
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    final_resnorm = Ref{Float64}(0.0)
    numops = Ref{Int64}(0)
    x0v = Vector{T}(vec(x0))
    if T === Float64
        V = Matrix{Float64}(undef, n, Int(k) + 1)
        H = Matrix{Float64}(undef, Int(k) + 1, Int(k))
        fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_d_cpu)
        status = GC.@preserve ctx x0v V H begin
            ccall(fptr, Cint,
                  (Int64, Ptr{Float64}, Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
                   Ptr{Float64}, Int64, Ptr{Float64}, Int64, Ref{Float64},
                   Ref{Int64}, Ref{Float64}, Ref{Int64}),
                  Int64(n), x0v, k, _nonnegative_float(tol, "tol"),
                  _matvec_d_cpu_c(), ctxptr, V, Int64(size(V, 1)), H,
                  Int64(size(H, 1)), beta, m, final_resnorm, numops)
        end
        _throw_callback_failure(ctx)
        _check_status(status, "tenet_native_krylov_arnoldi_d_cpu"; lib)
        return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
                numops=Int(numops[]), path=:generic_cpu_callback)
    end
    V = Matrix{ComplexF64}(undef, n, Int(k) + 1)
    H = Matrix{ComplexF64}(undef, Int(k) + 1, Int(k))
    fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_z_cpu)
    status = GC.@preserve ctx x0v V H begin
        ccall(fptr, Cint,
              (Int64, Ptr{ComplexF64}, Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
               Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Ref{Float64},
               Ref{Int64}, Ref{Float64}, Ref{Int64}),
              Int64(n), x0v, k, _nonnegative_float(tol, "tol"),
              _matvec_z_cpu_c(), ctxptr, V, Int64(size(V, 1)), H,
              Int64(size(H, 1)), beta, m, final_resnorm, numops)
    end
    _throw_callback_failure(ctx)
    _check_status(status, "tenet_native_krylov_arnoldi_z_cpu"; lib)
    return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
            numops=Int(numops[]), path=:generic_cpu_callback)
end

function _native_arnoldi_prefilled_dense_cpu(A::Matrix{T},
                                             initial_V::Matrix{T},
                                             initial_H::Matrix{T},
                                             completed_cols::Integer;
                                             tol::Real=1e-12,
                                             lib=nothing) where {T<:Union{Float64,ComplexF64}}
    n = size(initial_V, 1)
    k = size(initial_H, 2)
    size(A) == (n, n) ||
        throw(DimensionMismatch("matrix operator must have size ($n, $n), got $(size(A))"))
    size(initial_H, 1) == k + 1 ||
        throw(DimensionMismatch("initial_H must have size ($(k + 1), $k), got $(size(initial_H))"))
    completed = Int64(completed_cols)
    initial_cols = Int64(size(initial_V, 2))
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    final_resnorm = Ref{Float64}(0.0)
    numops = Ref{Int64}(0)
    if T === Float64
        V = Matrix{Float64}(undef, n, k + 1)
        H = Matrix{Float64}(undef, k + 1, k)
        fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_prefilled_dense_d_cpu)
        status = GC.@preserve A initial_V initial_H V H begin
            ccall(fptr, Cint,
                  (Int64, Ptr{Float64}, Int64, Ptr{Float64}, Int64, Int64,
                   Ptr{Float64}, Int64, Int64, Int64, Float64, Ptr{Float64},
                   Int64, Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64},
                   Ref{Float64}, Ref{Int64}),
                  Int64(n), A, Int64(size(A, 1)), initial_V,
                  Int64(size(initial_V, 1)), initial_cols, initial_H,
                  Int64(size(initial_H, 1)), completed, Int64(k),
                  _nonnegative_float(tol, "tol"), V, Int64(size(V, 1)), H,
                  Int64(size(H, 1)), beta, m, final_resnorm, numops)
        end
        _check_status(status, "tenet_native_krylov_arnoldi_prefilled_dense_d_cpu"; lib)
        return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
                numops=Int(numops[]), path=:generic_cpu_dense)
    end
    V = Matrix{ComplexF64}(undef, n, k + 1)
    H = Matrix{ComplexF64}(undef, k + 1, k)
    fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_prefilled_dense_z_cpu)
    status = GC.@preserve A initial_V initial_H V H begin
        ccall(fptr, Cint,
              (Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Int64,
               Ptr{ComplexF64}, Int64, Int64, Int64, Float64,
               Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Ref{Float64},
               Ref{Int64}, Ref{Float64}, Ref{Int64}),
              Int64(n), A, Int64(size(A, 1)), initial_V,
              Int64(size(initial_V, 1)), initial_cols, initial_H,
              Int64(size(initial_H, 1)), completed, Int64(k),
              _nonnegative_float(tol, "tol"), V, Int64(size(V, 1)), H,
              Int64(size(H, 1)), beta, m, final_resnorm, numops)
    end
    _check_status(status, "tenet_native_krylov_arnoldi_prefilled_dense_z_cpu"; lib)
    return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
            numops=Int(numops[]), path=:generic_cpu_dense)
end

function _native_arnoldi_prefilled_cpu(A_or_f, initial_V::Matrix{T},
                                       initial_H::Matrix{T},
                                       completed_cols::Integer,
                                       shape::Tuple;
                                       tol::Real=1e-12,
                                       lib=nothing) where {T<:Union{Float64,ComplexF64}}
    n = size(initial_V, 1)
    dense_A = _native_dense_matrix(A_or_f, shape, T)
    dense_A === nothing || return _native_arnoldi_prefilled_dense_cpu(
        dense_A, initial_V, initial_H, completed_cols; tol, lib)
    op = _native_krylov_operator(A_or_f, shape, T)
    ctx = _NativeKrylovCallbackContext(op, shape, nothing)
    ctxptr = pointer_from_objref(ctx)
    k = size(initial_H, 2)
    completed = Int64(completed_cols)
    initial_cols = Int64(size(initial_V, 2))
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    final_resnorm = Ref{Float64}(0.0)
    numops = Ref{Int64}(0)
    if T === Float64
        V = Matrix{Float64}(undef, n, k + 1)
        H = Matrix{Float64}(undef, k + 1, k)
        fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_prefilled_d_cpu)
        status = GC.@preserve ctx initial_V initial_H V H begin
            ccall(fptr, Cint,
                  (Int64, Ptr{Float64}, Int64, Int64, Ptr{Float64}, Int64,
                   Int64, Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
                   Ptr{Float64}, Int64, Ptr{Float64}, Int64, Ref{Float64},
                   Ref{Int64}, Ref{Float64}, Ref{Int64}),
                  Int64(n), initial_V, Int64(size(initial_V, 1)), initial_cols,
                  initial_H, Int64(size(initial_H, 1)), completed, Int64(k),
                  _nonnegative_float(tol, "tol"), _matvec_d_cpu_c(), ctxptr,
                  V, Int64(size(V, 1)), H, Int64(size(H, 1)), beta, m,
                  final_resnorm, numops)
        end
        _throw_callback_failure(ctx)
        _check_status(status, "tenet_native_krylov_arnoldi_prefilled_d_cpu"; lib)
        return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
                numops=Int(numops[]), path=:generic_cpu_callback)
    end
    V = Matrix{ComplexF64}(undef, n, k + 1)
    H = Matrix{ComplexF64}(undef, k + 1, k)
    fptr = _required_optional_symbol(lib, :tenet_native_krylov_arnoldi_prefilled_z_cpu)
    status = GC.@preserve ctx initial_V initial_H V H begin
        ccall(fptr, Cint,
              (Int64, Ptr{ComplexF64}, Int64, Int64, Ptr{ComplexF64}, Int64,
               Int64, Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
               Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Ref{Float64},
               Ref{Int64}, Ref{Float64}, Ref{Int64}),
              Int64(n), initial_V, Int64(size(initial_V, 1)), initial_cols,
              initial_H, Int64(size(initial_H, 1)), completed, Int64(k),
              _nonnegative_float(tol, "tol"), _matvec_z_cpu_c(), ctxptr, V,
              Int64(size(V, 1)), H, Int64(size(H, 1)), beta, m,
              final_resnorm, numops)
    end
    _throw_callback_failure(ctx)
    _check_status(status, "tenet_native_krylov_arnoldi_prefilled_z_cpu"; lib)
    return (; V, H, beta=beta[], m=Int(m[]), final_resnorm=final_resnorm[],
            numops=Int(numops[]), path=:generic_cpu_callback)
end

function _native_eig_order(vals, which)
    if which === :LM
        return sortperm(abs.(vals); rev=true)
    elseif which === :LR
        return sortperm(real.(vals); rev=true)
    elseif which === :SR
        return sortperm(real.(vals); rev=false)
    elseif which === :LI
        return sortperm(imag.(vals); rev=true)
    elseif which === :SI
        return sortperm(imag.(vals); rev=false)
    end
    throw(ArgumentError("native_eigsolve v1 supports KrylovKit selectors which=:LM,:LR,:SR,:LI,:SI, got $which"))
end

function _native_validate_eig_selector(which::Symbol, ::Type{S}) where {S}
    which in (:LM, :LR, :SR, :LI, :SI) ||
        throw(ArgumentError("native_eigsolve v1 supports KrylovKit selectors which=:LM,:LR,:SR,:LI,:SI, got $which"))
    if S === Float64 && which in (:LI, :SI)
        throw(ArgumentError("which=$which requires ComplexF64 arithmetic; pass ComplexF64 x0 or scalar=:complexf64"))
    end
    return nothing
end

function _native_apply_for_info(A_or_f, y, shape::Tuple)
    if A_or_f isa AbstractMatrix
        return reshape(A_or_f * vec(y), shape)
    end
    return A_or_f(y)
end

function _native_arnoldi_ritz_residual(fact, coeff, λ, yn::Real,
                                       shape::Tuple, fallback)
    m = fact.m
    if m <= 0 || yn <= 0
        return fallback()
    end
    if size(fact.V, 2) >= m + 1 && size(fact.H, 1) >= m + 1
        small = fact.H[1:(m + 1), 1:m] * coeff
        small[1:m] .-= λ .* coeff
        rflat = (fact.beta / yn) .* (fact.V[:, 1:(m + 1)] * small)
        return reshape(rflat, shape)
    end
    return fallback()
end

function _native_schur_block_indices(S, idx::Integer)
    n = length(S.values)
    idx_i = Int(idx)
    if eltype(S.T) <: Real
        scale = max(1.0, opnorm(Matrix(S.T), Inf))
        tol = 64eps(Float64) * scale
        if idx_i > 1 && abs(S.T[idx_i, idx_i - 1]) > tol
            return (idx_i - 1, idx_i)
        elseif idx_i < n && abs(S.T[idx_i + 1, idx_i]) > tol
            return (idx_i, idx_i + 1)
        end
    end
    return (idx_i,)
end

function _native_schur_select_mask(S, which::Symbol, target::Integer)
    n = length(S.values)
    selected = falses(n)
    order = _native_eig_order(S.values, which)
    for idx in order
        for j in _native_schur_block_indices(S, idx)
            selected[j] = true
        end
        count(selected) >= target && break
    end
    return selected
end

function _native_ordered_schur_for_keep(Hm, which::Symbol, keep::Integer)
    m = size(Hm, 1)
    m > 0 || return nothing
    S = schur(Matrix(Hm))
    max_keep = min(Int(keep), m - 1)
    for target in max_keep:-1:1
        selected = _native_schur_select_mask(S, which, target)
        count(selected) < m || continue
        return ordschur(S, selected), count(selected)
    end
    return nothing
end

function _native_krylov_schur_keep(krylovdim::Integer, converged::Integer,
                                   howmany::Integer)
    k = Int(krylovdim)
    keep = max(Int(howmany), fld(3 * k + 2 * Int(converged), 5))
    return clamp(keep, 1, max(1, k - 1))
end

function _native_krylov_schur_restart(fact, which::Symbol, keep::Integer,
                                      tol::Real)
    m = fact.m
    max_k = size(fact.H, 2)
    m > 1 || return nothing
    ordered = _native_ordered_schur_for_keep(fact.H[1:m, 1:m], which, keep)
    ordered === nothing && return nothing
    S, actual_keep = ordered
    actual_keep > 0 && actual_keep < max_k || return nothing
    C = Matrix(S.Z[:, 1:actual_keep])
    Tsmall = Matrix(S.T[1:actual_keep, 1:actual_keep])
    initial_H = zeros(eltype(fact.H), max_k + 1, max_k)
    initial_H[1:actual_keep, 1:actual_keep] .= Tsmall
    tail = fact.final_resnorm .* S.Z[end, 1:actual_keep]
    tail_norm = Float64(norm(tail))
    initial_H[actual_keep + 1, 1:actual_keep] .= tail
    basis = fact.V[:, 1:m] * C
    if tail_norm > Float64(tol) && size(fact.V, 2) >= m + 1
        initial_V = Matrix{eltype(fact.V)}(undef, size(fact.V, 1), actual_keep + 1)
        initial_V[:, 1:actual_keep] .= basis
        initial_V[:, actual_keep + 1] .= fact.V[:, m + 1]
    else
        initial_V = Matrix{eltype(fact.V)}(undef, size(fact.V, 1), actual_keep)
        initial_V[:, 1:actual_keep] .= basis
    end
    return (;
        V=initial_V,
        H=initial_H,
        completed_cols=actual_keep,
        schur_keep=actual_keep,
        schur_tail_norm=tail_norm,
    )
end

function _native_linsolve_effective_tol(b::Array; tol, atol::Real, rtol::Real)
    atol64 = _nonnegative_float(atol, "atol")
    rtol64 = _nonnegative_float(rtol, "rtol")
    if tol === nothing
        return max(atol64, rtol64 * Float64(norm(b))), atol64, rtol64, :atol_rtol
    end
    return _nonnegative_float(tol, "tol"), atol64, rtol64, :tol
end

function _native_linsolve_enrich_info(info; algorithm::Symbol, maxiter::Integer,
                                      tol::Float64, atol::Float64,
                                      rtol::Float64, tol_source::Symbol)
    converged = Int(info.converged)
    numiter = Int(info.numiter)
    status = converged >= 1 ? :converged : :not_converged
    reason = if converged >= 1
        :converged
    elseif numiter >= Int(maxiter)
        :maxiter
    else
        :breakdown_or_stagnation
    end
    return merge(info, (;
        status,
        reason,
        algorithm,
        tol,
        atol,
        rtol,
        tol_source,
    ))
end

function native_eigsolve(A_or_f, x0::Array{T}, howmany::Integer=1,
                         which::Symbol=:LM;
                         krylovdim::Integer=30,
                         maxiter::Integer=100,
                         tol::Real=1e-12,
                         backend::Symbol=:auto,
                         algorithm::Symbol=:arnoldi,
                         scalar::Symbol=:auto,
                         ishermitian::Bool=false,
                         issymmetric::Bool=false,
                         lib=nothing) where {T<:Union{Float64,ComplexF64}}
    algorithm in (:arnoldi, :krylovschur) ||
        throw(ArgumentError("native_eigsolve v1 supports algorithm=:arnoldi or :krylovschur"))
    howmany > 0 || throw(ArgumentError("howmany must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    _native_krylov_backend(A_or_f, x0; backend)
    promote_complex = A_or_f isa AbstractMatrix && eltype(A_or_f) <: Complex
    S = _native_krylov_scalar(x0; scalar, promote_complex)
    _native_validate_eig_selector(which, S)
    xwork = Array{S}(x0)
    n = length(xwork)
    howmany <= n || throw(ArgumentError("howmany must be <= length(x0)=$n"))
    kdim = Int(min(max(Int(krylovdim), 1), n))
    howmany <= kdim ||
        throw(ArgumentError("howmany must be <= effective krylovdim=$kdim"))
    if howmany > 1 && kdim <= howmany && kdim < n
        throw(ArgumentError("restarted native_eigsolve requires krylovdim > howmany for howmany>1 unless krylovdim reaches the full problem dimension"))
    end
    xstart = copy(xwork)
    total_numops = 0
    last_result = nothing
    maxiter_i = Int(maxiter)
    tol64 = Float64(tol)
    restart_state = nothing
    last_schur_keep = 0
    last_schur_tail_norm = NaN
    used_thick_restart = false
    for iter in 1:maxiter_i
        fact = if restart_state === nothing
            _native_arnoldi_cpu(A_or_f, xstart; krylovdim=kdim, tol, lib)
        else
            _native_arnoldi_prefilled_cpu(
                A_or_f, restart_state.V, restart_state.H,
                restart_state.completed_cols, size(xwork); tol, lib)
        end
        total_numops += fact.numops
        m = fact.m
        m > 0 || throw(ArgumentError("native_eigsolve Arnoldi produced an empty basis"))
        Hm = fact.H[1:m, 1:m]
        F = eigen(Matrix(Hm))
        order = _native_eig_order(F.values, which)
        take = min(Int(howmany), length(order))
        vals = Vector{eltype(F.values)}(undef, take)
        vecs = Vector{Any}(undef, take)
        residuals = Vector{Any}(undef, take)
        normres = Vector{Float64}(undef, take)
        for out_i in 1:take
            idx = order[out_i]
            λ = F.values[idx]
            coeff = F.vectors[:, idx]
            yflat = fact.beta .* (fact.V[:, 1:m] * coeff)
            yn = norm(yflat)
            yn > 0 || throw(ArgumentError("native_eigsolve selected a zero Ritz vector"))
            yflat ./= yn
            y = reshape(yflat, size(x0))
            residual_fallback = () -> begin
                fy = _native_apply_for_info(A_or_f, y, size(xwork))
                fy .- λ .* y
            end
            residual = _native_arnoldi_ritz_residual(
                fact, coeff, λ, yn, size(x0), residual_fallback)
            if S === Float64 && norm(imag.(yflat)) <= 1e-10 * max(1.0, norm(real.(yflat))) &&
               abs(imag(λ)) <= 1e-10 * max(1.0, abs(real(λ)))
                y = reshape(real.(yflat), size(x0))
                λ = real(λ)
                if norm(imag.(vec(residual))) <= 1e-10 * max(1.0, norm(real.(vec(residual))))
                    residual = reshape(real.(vec(residual)), size(x0))
                end
            end
            vals[out_i] = λ
            vecs[out_i] = y
            residuals[out_i] = residual
            normres[out_i] = Float64(norm(residual))
        end
        converged = count(<=(tol64), normres)
        requested_done = converged >= Int(howmany) && length(vals) >= Int(howmany)
        info = (;
            converged,
            residual=residuals,
            normres,
            numops=total_numops,
            numiter=iter,
            num_restarts=max(iter - 1, 0),
            backend=:cpu,
            scalar=S === Float64 ? :float64 : :complexf64,
            path=fact.path,
            status=requested_done ? :converged : :not_converged,
            reason=requested_done ? :converged : :maxiter,
            algorithm=:krylovschur,
            requested_algorithm=algorithm,
            schur_keep=last_schur_keep,
            schur_tail_norm=last_schur_tail_norm,
            thick_restart=used_thick_restart,
            tol=tol64,
        )
        last_result = (vals, vecs, info)
        requested_done && return vals, vecs, info
        keep = _native_krylov_schur_keep(kdim, converged, howmany)
        next_restart = _native_krylov_schur_restart(fact, which, keep, tol64)
        if next_restart !== nothing
            restart_state = next_restart
            last_schur_keep = next_restart.schur_keep
            last_schur_tail_norm = next_restart.schur_tail_norm
            used_thick_restart = true
            continue
        end
        if fact.final_resnorm <= tol64 || fact.m < kdim
            reason = fact.final_resnorm <= tol64 ? :breakdown : :short_arnoldi
            last_result = (vals, vecs, merge(info, (;
                reason,
                status=:not_converged,
            )))
            return last_result
        end
        restart = vecs[1]
        restart_state = nothing
        used_thick_restart = false
        last_schur_keep = 0
        last_schur_tail_norm = NaN
        if S === Float64
            restart_vec = vec(restart)
            if eltype(restart_vec) <: Real
                xstart = reshape(Vector{Float64}(restart_vec), size(x0))
            else
                real_restart = real.(restart_vec)
                nrm = norm(real_restart)
                nrm > 0 || throw(ArgumentError("native_eigsolve restart vector has zero real part for Float64 arithmetic"))
                real_restart ./= nrm
                xstart = reshape(Vector{Float64}(real_restart), size(x0))
            end
        else
            xstart = reshape(Vector{ComplexF64}(vec(restart)), size(x0))
        end
    end
    return last_result
end

function _native_linsolve_dense_cpu(A::Matrix{T}, b::Array{T}, xstart::Array{T},
                                    a0::Number, a1::Number;
                                    algorithm::Symbol=:gmres,
                                    krylovdim::Integer=30,
                                    maxiter::Integer=100,
                                    tol::Real=1e-12,
                                    lib=nothing) where {T<:Union{Float64,ComplexF64}}
    n = length(b)
    size(A) == (n, n) ||
        throw(DimensionMismatch("matrix operator must have size ($n, $n), got $(size(A))"))
    k = Int64(min(max(Int(krylovdim), 1), n))
    normres = Ref{Float64}(0.0)
    converged = Ref{Int64}(0)
    numops = Ref{Int64}(0)
    numiter = Ref{Int64}(0)
    bv = Vector{T}(vec(b))
    xstartv = Vector{T}(vec(xstart))
    if T === Float64
        x = Array{Float64}(undef, size(b))
        residual = similar(x)
        if algorithm === :gmres
            sym = :tenet_native_krylov_gmres_dense_d_cpu
            fptr = _required_optional_symbol(lib, sym)
            status = GC.@preserve A bv xstartv x residual begin
                ccall(fptr, Cint,
                      (Int64, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64},
                       Float64, Float64, Int64, Int64, Float64, Ptr{Float64},
                       Ptr{Float64}, Ref{Float64}, Ref{Int64}, Ref{Int64},
                       Ref{Int64}),
                      Int64(n), A, Int64(size(A, 1)), bv, xstartv, Float64(a0),
                      Float64(a1), k, Int64(maxiter),
                      _nonnegative_float(tol, "tol"), x, residual, normres,
                      converged, numops, numiter)
            end
        elseif algorithm === :cg
            sym = :tenet_native_krylov_cg_dense_d_cpu
            fptr = _required_optional_symbol(lib, sym)
            status = GC.@preserve A bv xstartv x residual begin
                ccall(fptr, Cint,
                      (Int64, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64},
                       Float64, Float64, Int64, Float64, Ptr{Float64},
                       Ptr{Float64}, Ref{Float64}, Ref{Int64}, Ref{Int64},
                       Ref{Int64}),
                      Int64(n), A, Int64(size(A, 1)), bv, xstartv, Float64(a0),
                      Float64(a1), Int64(maxiter),
                      _nonnegative_float(tol, "tol"), x, residual, normres,
                      converged, numops, numiter)
            end
        else
            sym = :tenet_native_krylov_bicgstab_dense_d_cpu
            fptr = _required_optional_symbol(lib, sym)
            status = GC.@preserve A bv xstartv x residual begin
                ccall(fptr, Cint,
                      (Int64, Ptr{Float64}, Int64, Ptr{Float64}, Ptr{Float64},
                       Float64, Float64, Int64, Float64, Ptr{Float64},
                       Ptr{Float64}, Ref{Float64}, Ref{Int64}, Ref{Int64},
                       Ref{Int64}),
                      Int64(n), A, Int64(size(A, 1)), bv, xstartv, Float64(a0),
                      Float64(a1), Int64(maxiter),
                      _nonnegative_float(tol, "tol"), x, residual, normres,
                      converged, numops, numiter)
            end
        end
        _check_status(status, String(sym); lib)
        info = (; converged=Int(converged[]), residual, normres=normres[],
                numops=Int(numops[]), numiter=Int(numiter[]), backend=:cpu,
                scalar=:float64, path=:generic_cpu_dense)
        return x, info
    end
    x = Array{ComplexF64}(undef, size(b))
    residual = similar(x)
    if algorithm === :gmres
        sym = :tenet_native_krylov_gmres_dense_z_cpu
        fptr = _required_optional_symbol(lib, sym)
        status = GC.@preserve A bv xstartv x residual begin
            ccall(fptr, Cint,
                  (Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Ptr{ComplexF64},
                   _NativeComplex64, _NativeComplex64, Int64, Int64, Float64,
                   Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{Float64},
                   Ref{Int64}, Ref{Int64}, Ref{Int64}),
                  Int64(n), A, Int64(size(A, 1)), bv, xstartv,
                  _native_complex64(a0), _native_complex64(a1), k, Int64(maxiter),
                  _nonnegative_float(tol, "tol"), x, residual, normres,
                  converged, numops, numiter)
        end
    elseif algorithm === :cg
        sym = :tenet_native_krylov_cg_dense_z_cpu
        fptr = _required_optional_symbol(lib, sym)
        status = GC.@preserve A bv xstartv x residual begin
            ccall(fptr, Cint,
                  (Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Ptr{ComplexF64},
                   _NativeComplex64, _NativeComplex64, Int64, Float64,
                   Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{Float64},
                   Ref{Int64}, Ref{Int64}, Ref{Int64}),
                  Int64(n), A, Int64(size(A, 1)), bv, xstartv,
                  _native_complex64(a0), _native_complex64(a1), Int64(maxiter),
                  _nonnegative_float(tol, "tol"), x, residual, normres,
                  converged, numops, numiter)
        end
    else
        sym = :tenet_native_krylov_bicgstab_dense_z_cpu
        fptr = _required_optional_symbol(lib, sym)
        status = GC.@preserve A bv xstartv x residual begin
            ccall(fptr, Cint,
                  (Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Ptr{ComplexF64},
                   _NativeComplex64, _NativeComplex64, Int64, Float64,
                   Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{Float64},
                   Ref{Int64}, Ref{Int64}, Ref{Int64}),
                  Int64(n), A, Int64(size(A, 1)), bv, xstartv,
                  _native_complex64(a0), _native_complex64(a1), Int64(maxiter),
                  _nonnegative_float(tol, "tol"), x, residual, normres,
                  converged, numops, numiter)
        end
    end
    _check_status(status, String(sym); lib)
    info = (; converged=Int(converged[]), residual, normres=normres[],
            numops=Int(numops[]), numiter=Int(numiter[]), backend=:cpu,
            scalar=:complexf64, path=:generic_cpu_dense)
    return x, info
end

function native_linsolve(A_or_f, b::Array{T}, x0=nothing,
                         a0::Number=0, a1::Number=1;
                         algorithm::Symbol=:gmres,
                         krylovdim::Integer=30,
                         maxiter::Integer=100,
                         tol=nothing,
                         atol::Real=_NATIVE_KRYLOV_DEFAULT_TOL,
                         rtol::Real=_NATIVE_KRYLOV_DEFAULT_TOL,
                         backend::Symbol=:auto,
                         scalar::Symbol=:auto,
                         lib=nothing) where {T<:Union{Float64,ComplexF64}}
    algorithm in (:gmres, :cg, :bicgstab) ||
        throw(ArgumentError("native_linsolve CPU v1 supports algorithm=:gmres, :cg, or :bicgstab, got $algorithm"))
    if algorithm === :cg && (!(a0 isa Real) || !(a1 isa Real))
        throw(ArgumentError("native_linsolve algorithm=:cg requires real a0 and a1 so the shifted operator can remain Hermitian; use algorithm=:gmres for complex shifts"))
    end
    _native_krylov_backend(A_or_f, b; backend)
    promote_complex = (A_or_f isa AbstractMatrix && eltype(A_or_f) <: Complex) ||
                      !(a0 isa Real && a1 isa Real)
    S = _native_krylov_scalar(b; scalar, promote_complex)
    bwork = Array{S}(b)
    tol_eff, atol_eff, rtol_eff, tol_source = _native_linsolve_effective_tol(
        bwork; tol, atol, rtol)
    xstart = x0 === nothing ? zeros(S, size(bwork)) : Array{S}(x0)
    size(xstart) == size(bwork) ||
        throw(DimensionMismatch("x0 must have size $(size(bwork)), got $(size(xstart))"))
    dense_A = _native_dense_matrix(A_or_f, size(bwork), S)
    if dense_A !== nothing
        x, info = _native_linsolve_dense_cpu(
            dense_A, bwork, xstart, a0, a1;
            algorithm, krylovdim, maxiter, tol=tol_eff, lib)
        return x, _native_linsolve_enrich_info(
            info; algorithm, maxiter, tol=tol_eff, atol=atol_eff,
            rtol=rtol_eff, tol_source)
    end
    op = _native_krylov_operator(A_or_f, size(bwork), S)
    ctx = _NativeKrylovCallbackContext(op, size(bwork), nothing)
    ctxptr = pointer_from_objref(ctx)
    n = length(bwork)
    k = Int64(min(max(Int(krylovdim), 1), n))
    normres = Ref{Float64}(0.0)
    converged = Ref{Int64}(0)
    numops = Ref{Int64}(0)
    numiter = Ref{Int64}(0)
    bv = Vector{S}(vec(bwork))
    xstartv = Vector{S}(vec(xstart))
    if S === Float64
        x = Array{Float64}(undef, size(bwork))
        residual = similar(x)
        if algorithm === :gmres
            sym = :tenet_native_krylov_gmres_d_cpu
            fptr = _required_optional_symbol(lib, sym)
            status = GC.@preserve ctx bv xstartv x residual begin
                ccall(fptr, Cint,
                      (Int64, Ptr{Float64}, Ptr{Float64}, Float64, Float64,
                       Int64, Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Int64},
                       Ref{Int64}, Ref{Int64}),
                      Int64(n), bv, xstartv, Float64(a0), Float64(a1), k,
                      Int64(maxiter), tol_eff,
                      _matvec_d_cpu_c(), ctxptr, x, residual, normres,
                      converged, numops, numiter)
            end
        elseif algorithm === :cg
            sym = :tenet_native_krylov_cg_d_cpu
            fptr = _required_optional_symbol(lib, sym)
            status = GC.@preserve ctx bv xstartv x residual begin
                ccall(fptr, Cint,
                      (Int64, Ptr{Float64}, Ptr{Float64}, Float64, Float64,
                       Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Int64},
                       Ref{Int64}, Ref{Int64}),
                      Int64(n), bv, xstartv, Float64(a0), Float64(a1),
                      Int64(maxiter), tol_eff,
                      _matvec_d_cpu_c(), ctxptr, x, residual, normres,
                      converged, numops, numiter)
            end
        else
            sym = :tenet_native_krylov_bicgstab_d_cpu
            fptr = _required_optional_symbol(lib, sym)
            status = GC.@preserve ctx bv xstartv x residual begin
                ccall(fptr, Cint,
                      (Int64, Ptr{Float64}, Ptr{Float64}, Float64, Float64,
                       Int64, Float64, Ptr{Cvoid}, Ptr{Cvoid},
                       Ptr{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Int64},
                       Ref{Int64}, Ref{Int64}),
                      Int64(n), bv, xstartv, Float64(a0), Float64(a1),
                      Int64(maxiter), tol_eff,
                      _matvec_d_cpu_c(), ctxptr, x, residual, normres,
                      converged, numops, numiter)
            end
        end
        _throw_callback_failure(ctx)
        _check_status(status, String(sym); lib)
        info = (; converged=Int(converged[]), residual, normres=normres[],
                numops=Int(numops[]), numiter=Int(numiter[]), backend=:cpu,
                scalar=:float64, path=:generic_cpu_callback)
        return x, _native_linsolve_enrich_info(
            info; algorithm, maxiter, tol=tol_eff, atol=atol_eff,
            rtol=rtol_eff, tol_source)
    end
    x = Array{ComplexF64}(undef, size(bwork))
    residual = similar(x)
    if algorithm === :gmres
        sym = :tenet_native_krylov_gmres_z_cpu
        fptr = _required_optional_symbol(lib, sym)
        status = GC.@preserve ctx bv xstartv x residual begin
            ccall(fptr, Cint,
                  (Int64, Ptr{ComplexF64}, Ptr{ComplexF64},
                   _NativeComplex64, _NativeComplex64, Int64, Int64,
                   Float64, Ptr{Cvoid}, Ptr{Cvoid},
                   Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{Float64}, Ref{Int64},
                   Ref{Int64}, Ref{Int64}),
                  Int64(n), bv, xstartv, _native_complex64(a0),
                  _native_complex64(a1), k, Int64(maxiter), tol_eff,
                  _matvec_z_cpu_c(), ctxptr, x, residual, normres, converged,
                  numops, numiter)
        end
    elseif algorithm === :cg
        sym = :tenet_native_krylov_cg_z_cpu
        fptr = _required_optional_symbol(lib, sym)
        status = GC.@preserve ctx bv xstartv x residual begin
            ccall(fptr, Cint,
                  (Int64, Ptr{ComplexF64}, Ptr{ComplexF64},
                   _NativeComplex64, _NativeComplex64, Int64, Float64,
                   Ptr{Cvoid}, Ptr{Cvoid},
                   Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{Float64}, Ref{Int64},
                   Ref{Int64}, Ref{Int64}),
                  Int64(n), bv, xstartv, _native_complex64(a0),
                  _native_complex64(a1), Int64(maxiter), tol_eff,
                  _matvec_z_cpu_c(), ctxptr, x, residual, normres, converged,
                  numops, numiter)
        end
    else
        sym = :tenet_native_krylov_bicgstab_z_cpu
        fptr = _required_optional_symbol(lib, sym)
        status = GC.@preserve ctx bv xstartv x residual begin
            ccall(fptr, Cint,
                  (Int64, Ptr{ComplexF64}, Ptr{ComplexF64},
                   _NativeComplex64, _NativeComplex64, Int64, Float64,
                   Ptr{Cvoid}, Ptr{Cvoid},
                   Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{Float64}, Ref{Int64},
                   Ref{Int64}, Ref{Int64}),
                  Int64(n), bv, xstartv, _native_complex64(a0),
                  _native_complex64(a1), Int64(maxiter), tol_eff,
                  _matvec_z_cpu_c(), ctxptr, x, residual, normres, converged,
                  numops, numiter)
        end
    end
    _throw_callback_failure(ctx)
    _check_status(status, String(sym); lib)
    info = (; converged=Int(converged[]), residual, normres=normres[],
            numops=Int(numops[]), numiter=Int(numiter[]), backend=:cpu,
            scalar=:complexf64, path=:generic_cpu_callback)
    return x, _native_linsolve_enrich_info(
        info; algorithm, maxiter, tol=tol_eff, atol=atol_eff, rtol=rtol_eff,
        tol_source)
end

function tenet_native_arnoldi_two_layer_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    x0::Array{Float64,2};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    transpose::Bool=false,
    lib=nothing,
)
    chi, phys, len = _two_layer_inputs(Aup, Adn, x0)
    k = _resolved_max_k(max_k, len)
    V = Matrix{Float64}(undef, Int(len), Int(k) + 1)
    H = Matrix{Float64}(undef, Int(k) + 1, Int(k))
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    final_resnorm = Ref{Float64}(0.0)
    fptr = _symbol(lib, :tenet_native_arnoldi_two_layer_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Int64, Float64, Cint, Ptr{Float64}, Int64, Ptr{Float64},
         Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        chi, phys, Aup, Adn, x0, k, _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _transpose_flag(transpose), V, len, H, Int64(size(H, 1)), beta, m,
        final_resnorm,
    )
    _check_status(status, "tenet_native_arnoldi_two_layer_d_cpu"; lib)
    return _basis_result(V, H, beta, m, final_resnorm)
end

function tenet_native_arnoldi_two_layer_ritz_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    x0::Array{Float64,2};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    transpose::Bool=false,
    nvalues::Integer=2,
    lib=nothing,
)
    chi, phys, len = _two_layer_inputs(Aup, Adn, x0)
    k = _resolved_max_k(max_k, len)
    nvalues64 = Int64(nvalues)
    nvalues64 > 0 || throw(ArgumentError("nvalues must be positive, got $nvalues"))
    lambda_real = Vector{Float64}(undef, Int(nvalues64))
    lambda_imag = Vector{Float64}(undef, Int(nvalues64))
    m = Ref{Int64}(0)
    fptr = _symbol(lib, :tenet_native_arnoldi_two_layer_ritz_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Int64, Float64, Cint, Int64, Ptr{Float64}, Ptr{Float64},
         Ref{Int64}),
        chi, phys, Aup, Adn, x0, k, _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _transpose_flag(transpose), nvalues64, lambda_real, lambda_imag, m,
    )
    _check_status(status, "tenet_native_arnoldi_two_layer_ritz_d_cpu"; lib)
    return (values=complex.(lambda_real, lambda_imag), m=Int(m[]))
end

function _projected_two_layer_basis(
    symbol::Symbol,
    context::AbstractString,
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    rho::Array{Float64,2},
    x0::Array{Float64,2};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    transpose::Bool=false,
    lib=nothing,
)
    chi, phys, len = _projected_two_layer_inputs(Aup, Adn, rho, x0)
    k = _resolved_max_k(max_k, len)
    V = Matrix{Float64}(undef, Int(len), Int(k) + 1)
    H = Matrix{Float64}(undef, Int(k) + 1, Int(k))
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    final_resnorm = Ref{Float64}(0.0)
    fptr = _symbol(lib, symbol)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Int64,
         Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        chi, phys, Aup, Adn, rho, x0, k,
        _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _transpose_flag(transpose), V, len, H, Int64(size(H, 1)), beta, m,
        final_resnorm,
    )
    _check_status(status, context; lib)
    return _basis_result(V, H, beta, m, final_resnorm)
end

function tenet_native_arnoldi_projected_two_layer_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    rho::Array{Float64,2},
    x0::Array{Float64,2};
    kwargs...,
)
    return _projected_two_layer_basis(
        :tenet_native_arnoldi_projected_two_layer_d_cpu,
        "tenet_native_arnoldi_projected_two_layer_d_cpu",
        Aup, Adn, rho, x0;
        kwargs...,
    )
end

function tenet_native_arnoldi_qprojected_two_layer_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    rho::Array{Float64,2},
    x0::Array{Float64,2};
    kwargs...,
)
    return _projected_two_layer_basis(
        :tenet_native_arnoldi_qprojected_two_layer_d_cpu,
        "tenet_native_arnoldi_qprojected_two_layer_d_cpu",
        Aup, Adn, rho, x0;
        kwargs...,
    )
end

function tenet_native_arnoldi_three_layer_leg4_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    M::Array{Float64,4},
    x0::Array{Float64,3};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    transpose::Bool=false,
    lib=nothing,
)
    chi, phys, len = _three_layer_inputs(Aup, Adn, M, x0)
    k = _resolved_max_k(max_k, len)
    V = Matrix{Float64}(undef, Int(len), Int(k) + 1)
    H = Matrix{Float64}(undef, Int(k) + 1, Int(k))
    beta = Ref{Float64}(0.0)
    m = Ref{Int64}(0)
    final_resnorm = Ref{Float64}(0.0)
    fptr = _symbol(lib, :tenet_native_arnoldi_three_layer_leg4_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Int64,
         Ptr{Float64}, Int64, Ref{Float64}, Ref{Int64}, Ref{Float64}),
        chi, phys, Aup, Adn, M, x0, k, _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _transpose_flag(transpose), V, len, H, Int64(size(H, 1)), beta, m,
        final_resnorm,
    )
    _check_status(status, "tenet_native_arnoldi_three_layer_leg4_d_cpu"; lib)
    return _basis_result(V, H, beta, m, final_resnorm)
end

function _two_layer_vector(
    symbol::Symbol,
    context::AbstractString,
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    x0::Array{Float64,2};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    transpose::Bool=false,
    lib=nothing,
)
    chi, phys, len = _two_layer_inputs(Aup, Adn, x0)
    k = _resolved_max_k(max_k, len)
    y = Matrix{Float64}(undef, Int(chi), Int(chi))
    lambda = Ref{Float64}(0.0)
    fptr = _symbol(lib, symbol)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Int64, Float64, Cint, Ptr{Float64}, Ref{Float64}),
        chi, phys, Aup, Adn, x0, k, _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _transpose_flag(transpose), y, lambda,
    )
    _check_status(status, context; lib)
    return (lambda=lambda[], y=y)
end

function tenet_native_dominant_two_layer_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    x0::Array{Float64,2};
    kwargs...,
)
    return _two_layer_vector(
        :tenet_native_dominant_two_layer_d_cpu,
        "tenet_native_dominant_two_layer_d_cpu",
        Aup, Adn, x0;
        kwargs...,
    )
end

function tenet_native_smallest_real_two_layer_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    x0::Array{Float64,2};
    kwargs...,
)
    return _two_layer_vector(
        :tenet_native_smallest_real_two_layer_d_cpu,
        "tenet_native_smallest_real_two_layer_d_cpu",
        Aup, Adn, x0;
        kwargs...,
    )
end

function _three_layer_vector(
    symbol::Symbol,
    context::AbstractString,
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    M::Array{Float64,4},
    x0::Array{Float64,3};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    transpose::Bool=false,
    lib=nothing,
)
    chi, phys, len = _three_layer_inputs(Aup, Adn, M, x0)
    k = _resolved_max_k(max_k, len)
    y = Array{Float64}(undef, Int(chi), Int(phys), Int(chi))
    lambda = Ref{Float64}(0.0)
    fptr = _symbol(lib, symbol)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Int64, Float64, Cint, Ptr{Float64}, Ref{Float64}),
        chi, phys, Aup, Adn, M, x0, k, _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _transpose_flag(transpose), y, lambda,
    )
    _check_status(status, context; lib)
    return (lambda=lambda[], y=y)
end

function tenet_native_dominant_three_layer_leg4_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    M::Array{Float64,4},
    x0::Array{Float64,3};
    kwargs...,
)
    return _three_layer_vector(
        :tenet_native_dominant_three_layer_leg4_d_cpu,
        "tenet_native_dominant_three_layer_leg4_d_cpu",
        Aup, Adn, M, x0;
        kwargs...,
    )
end

function tenet_native_smallest_real_three_layer_leg4_d_cpu(
    Aup::Array{Float64,3},
    Adn::Array{Float64,3},
    M::Array{Float64,4},
    x0::Array{Float64,3};
    kwargs...,
)
    return _three_layer_vector(
        :tenet_native_smallest_real_three_layer_leg4_d_cpu,
        "tenet_native_smallest_real_three_layer_leg4_d_cpu",
        Aup, Adn, M, x0;
        kwargs...,
    )
end

function tenet_native_acc_to_alar_d_cpu(
    AC::Array{Float64,3},
    C::Array{Float64,2};
    lib=nothing,
)
    chi, phys, chi2 = size(AC)
    chi == chi2 || throw(DimensionMismatch("AC must have size chi x phys x chi, got $(size(AC))"))
    _check_matrix(C, (chi, chi), "C")
    AL = similar(AC)
    AR = similar(AC)
    err = Ref{Float64}(0.0)
    fptr = _symbol(lib, :tenet_native_acc_to_alar_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ref{Float64}),
        Int64(chi), Int64(phys), AC, C, AL, AR, err,
    )
    _check_status(status, "tenet_native_acc_to_alar_d_cpu"; lib)
    return (AL=AL, AR=AR, err=err[])
end

function tenet_native_ising_vumps_step_d_cpu(
    M::Array{Float64,4},
    AL::Array{Float64,3},
    AR::Array{Float64,3},
    C::Array{Float64,2},
    FL::Array{Float64,3},
    FR::Array{Float64,3};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    lib=nothing,
)
    chi, phys, len = _vumps_inputs(M, AL, AR, C, FL, FR)
    k = _resolved_max_k(max_k, len)
    err = Ref{Float64}(0.0)
    fptr = _symbol(lib, :tenet_native_ising_vumps_step_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Float64,
         Ref{Float64}),
        chi, phys, M, AL, AR, C, FL, FR, k,
        _nonnegative_float(breakdown_tol, "breakdown_tol"), err,
    )
    _check_status(status, "tenet_native_ising_vumps_step_d_cpu"; lib)
    return (AL=AL, AR=AR, C=C, FL=FL, FR=FR, err=err[])
end

function tenet_native_ising_vumps_step_checked_d_cpu(
    M::Array{Float64,4},
    AL::Array{Float64,3},
    AR::Array{Float64,3},
    C::Array{Float64,2},
    FL::Array{Float64,3},
    FR::Array{Float64,3};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    residual_tol::Real=1e-8,
    lib=nothing,
)
    chi, phys, len = _vumps_inputs(M, AL, AR, C, FL, FR)
    k = _resolved_max_k(max_k, len)
    err = Ref{Float64}(0.0)
    fptr = _symbol(lib, :tenet_native_ising_vumps_step_checked_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Float64,
         Float64, Ref{Float64}),
        chi, phys, M, AL, AR, C, FL, FR, k,
        _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _nonnegative_float(residual_tol, "residual_tol"), err,
    )
    _check_status(status, "tenet_native_ising_vumps_step_checked_d_cpu"; lib)
    return (AL=AL, AR=AR, C=C, FL=FL, FR=FR, err=err[])
end

function tenet_native_ising_vumps_run_d_cpu(
    M::Array{Float64,4},
    AL::Array{Float64,3},
    AR::Array{Float64,3},
    C::Array{Float64,2},
    FL::Array{Float64,3},
    FR::Array{Float64,3};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    tol::Real=1e-10,
    miniter::Integer=1,
    maxiter::Integer=100,
    lib=nothing,
)
    chi, phys, len = _vumps_inputs(M, AL, AR, C, FL, FR)
    k = _resolved_max_k(max_k, len)
    err = Ref{Float64}(0.0)
    iterations = Ref{Int64}(0)
    converged = Ref{Cint}(0)
    fptr = _symbol(lib, :tenet_native_ising_vumps_run_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Float64,
         Float64, Int64, Int64, Ref{Float64}, Ref{Int64}, Ref{Cint}),
        chi, phys, M, AL, AR, C, FL, FR, k,
        _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _nonnegative_float(tol, "tol"), Int64(miniter), Int64(maxiter), err,
        iterations, converged,
    )
    _check_status(status, "tenet_native_ising_vumps_run_d_cpu"; lib)
    return (AL=AL, AR=AR, C=C, FL=FL, FR=FR, err=err[],
            iterations=Int(iterations[]), converged=converged[] != 0)
end

function tenet_native_ising_vumps_run_checked_d_cpu(
    M::Array{Float64,4},
    AL::Array{Float64,3},
    AR::Array{Float64,3},
    C::Array{Float64,2},
    FL::Array{Float64,3},
    FR::Array{Float64,3};
    max_k=nothing,
    breakdown_tol::Real=1e-12,
    tol::Real=1e-10,
    miniter::Integer=1,
    maxiter::Integer=100,
    residual_tol::Real=1e-8,
    lib=nothing,
)
    chi, phys, len = _vumps_inputs(M, AL, AR, C, FL, FR)
    k = _resolved_max_k(max_k, len)
    err = Ref{Float64}(0.0)
    iterations = Ref{Int64}(0)
    converged = Ref{Cint}(0)
    fptr = _symbol(lib, :tenet_native_ising_vumps_run_checked_d_cpu)
    status = ccall(
        fptr,
        Cint,
        (Int64, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Int64, Float64,
         Float64, Int64, Int64, Float64, Ref{Float64}, Ref{Int64},
         Ref{Cint}),
        chi, phys, M, AL, AR, C, FL, FR, k,
        _nonnegative_float(breakdown_tol, "breakdown_tol"),
        _nonnegative_float(tol, "tol"), Int64(miniter), Int64(maxiter),
        _nonnegative_float(residual_tol, "residual_tol"), err, iterations,
        converged,
    )
    _check_status(status, "tenet_native_ising_vumps_run_checked_d_cpu"; lib)
    return (AL=AL, AR=AR, C=C, FL=FL, FR=FR, err=err[],
            iterations=Int(iterations[]), converged=converged[] != 0)
end

end # module
