using Test
using TeneTC

@testset "TeneTC API" begin
    @test critical_beta() > 0
    @test run_boundary isa Function
    @test native_eigsolve isa Function
    @test native_linsolve isa Function
end
