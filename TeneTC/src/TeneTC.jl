module TeneTC

using FastTeneT:
    BoundaryResult,
    VUMPS,
    VUMPSEnv,
    VUMPSRuntime,
    build_native_arnoldi,
    critical_beta,
    energy_density,
    free_energy_density,
    ising_network,
    ising_tensor,
    log_partition_density,
    log_partition_density_exact,
    magnetization,
    magnetization_exact,
    run_boundary,
    vumps_algorithm

using KrylovKitC:
    build_native_krylov,
    native_krylov_library,
    native_krylov_capabilities,
    native_eigsolve,
    native_linsolve

export BoundaryResult,
       VUMPS,
       VUMPSEnv,
       VUMPSRuntime,
       build_native_arnoldi,
       build_native_krylov,
       critical_beta,
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
       native_krylov_library,
       native_linsolve,
       run_boundary,
       vumps_algorithm

end
