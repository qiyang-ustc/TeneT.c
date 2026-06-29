using Test
using TeneTC

@testset "TeneTC API" begin
    @test critical_beta() > 0
    @test run_boundary isa Function
    @test native_eigsolve isa Function
    @test native_linsolve isa Function
end

if get(ENV, "TENETC_RUN_RELEASE_GATE", "0") == "1"
    ENV["FASTTENET_RUN_VUMPS_SMOKE"] = get(ENV, "FASTTENET_RUN_VUMPS_SMOKE", "1")
    include(joinpath(@__DIR__, "..", "..", "FastTeneT", "test", "runtests.jl"))
elseif get(ENV, "TENETC_RUN_FASTTENET_GATE", "0") == "1"
    include(joinpath(@__DIR__, "..", "..", "FastTeneT", "test", "runtests.jl"))
end
