using FastTeneT

result = run_boundary(critical_beta(); chi=4, maxiter=25, maxiter_ad=0, verbosity=1)

println("beta = ", result.beta)
println("chi = ", result.chi)
println("logZ density = ", log_partition_density(result))
println("free energy density = ", free_energy_density(result))
println("energy density = ", energy_density(result))
println("magnetization = ", magnetization(result))
