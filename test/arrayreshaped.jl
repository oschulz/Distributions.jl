using Distributions, Test, Random, LinearAlgebra
using Distributions: ArrayReshaped

rng = MersenneTwister(123456)

σ = rand(rng, 16, 16)
μ = rand(rng, 16)
d1 = MvNormal(μ, σ * σ')
x1 = rand(rng, d1)

sizes = [(4, 4), (8, 2), (2, 8), (1, 16), (16, 1), (2, 4, 2), (2, 1, 4, 2)]
ranks = [4, 2, 2, 1, 1, 2, 1]

d1s = [ArrayReshaped(d1, s...) for s in sizes]


@testset "ArrayReshaped MvNormal tests" begin
    @testset "ArrayReshaped constructor" begin
        for d in d1s
            @test d isa ArrayReshaped
        end
        @test MatrixReshaped(d1, 4) == MatrixReshaped(d1, 4, 4) == ArrayReshaped(d1, 4, 4)
    end
    @testset "ArrayReshaped constructor errors" begin
        @test_throws ArgumentError ArrayReshaped(d1, 4, 3)
        @test_throws ArgumentError ArrayReshaped(d1, 3)
        @test_throws ArgumentError ArrayReshaped(d1, -4, -4)
    end
    @testset "ArrayReshaped size" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test size(d) == s
        end
    end
    @testset "ArrayReshaped length" begin
        for d in d1s
            @test length(d) == length(μ)
        end
    end
    @testset "ArrayReshaped rank" begin
        for (d, r) in zip(d1s, ranks)
            @test rank(d) == r
        end
    end
    @testset "ArrayReshaped insupport" begin
        for (i, d) in enumerate(d1s[1:end-1])
            for (j, s) in enumerate(sizes[1:end-1])
                @test (i == j) ⊻ !insupport(d, reshape(x1, s))
            end
        end
    end
    @testset "ArrayReshaped mean" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test mean(d) == reshape(μ, s)
        end
    end
    @testset "ArrayReshaped mode" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test mode(d) == reshape(mode(d1), s)
        end
    end
    @testset "ArrayReshaped covariance" begin
        for (d, sz) in zip(d1s[1:end-1], sizes[1:end-1])
            @test cov(d) == σ * σ'
            @test cov(d, Val(false)) == reshape(σ * σ', sz..., sz...)
        end
    end
    @testset "ArrayReshaped variance" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test var(d) == reshape(var(d1), s)
        end
    end
    @testset "ArrayReshaped params" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test params(d) == (d1, s...)
        end
    end
    @testset "ArrayReshaped partype" begin
        for d in d1s
            @test partype(d) == Float64
        end
    end
    @testset "ArrayReshaped logpdf" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            x = reshape(x1, s)
            @test logpdf(d, x) == logpdf(d1, x1)
        end
    end
    @testset "ArrayReshaped rand" begin
        for d in d1s
            @info "xxx" d
            x = rand(rng, d)
            @test insupport(d, x)
            @test insupport(d1, vec(x))
            @test logpdf(d, x) == logpdf(d1, vec(x))
        end
    end
    @testset "ArrayReshaped vec" begin
        for d in d1s
            @test vec(d) == d1
        end
    end
end

α = rand(rng, 36)
d1 = Dirichlet(α)
x1 = rand(rng, d1)

sizes = [(6, 6), (4, 9), (9, 4), (3, 12), (12, 3), (1, 36), (36, 1), (6, 2, 3), (1, 2, 3, 6)]
ranks = [6, 4, 4, 3, 3, 1, 1, 2, 1]

d1s = [ArrayReshaped(d1, s...) for s in sizes]

@testset "ArrayReshaped Dirichlet tests" begin
    @testset "ArrayReshaped constructor" begin
        for d in d1s
            @test d isa ArrayReshaped
        end
    end
    @testset "ArrayReshaped constructor errors" begin
        @test_throws ArgumentError ArrayReshaped(d1, 4, 3)
        @test_throws ArgumentError ArrayReshaped(d1, 3)
    end
    @testset "ArrayReshaped size" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test size(d) == s
        end
    end
    @testset "ArrayReshaped length" begin
        for d in d1s
            @test length(d) == length(α)
        end
    end
    @testset "ArrayReshaped rank" begin
        for (d, r) in zip(d1s, ranks)
            @test rank(d) == r
        end
    end
    @testset "ArrayReshaped insupport" begin
        for (i, d) in enumerate(d1s[1:end-1])
            for (j, s) in enumerate(sizes[1:end-1])
                @test (i == j) ⊻ !insupport(d, reshape(x1, s))
            end
        end
    end
    @testset "ArrayReshaped mean" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test mean(d) == reshape(mean(d1), s)
        end
    end
    @testset "ArrayReshaped covariance" begin
        for (d, sz) in zip(d1s[1:end-1], sizes[1:end-1])
            @test cov(d) == cov(d1)
            @test cov(d, Val(false)) == reshape(cov(d1), sz..., sz...)
        end
    end
    @testset "ArrayReshaped variance" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test var(d) == reshape(var(d1), s)
        end
    end
    @testset "ArrayReshaped params" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            @test params(d) == (d1, s...)
        end
    end
    @testset "ArrayReshaped partype" begin
        for d in d1s
            @test partype(d) == Float64
        end
    end
    @testset "ArrayReshaped logpdf" begin
        for (d, s) in zip(d1s[1:end-1], sizes[1:end-1])
            x = reshape(x1, s)
            @test logpdf(d, x) == logpdf(d1, x1)
        end
    end
    @testset "ArrayReshaped rand" begin
        for d in d1s
            x = rand(rng, d)
            @test insupport(d, x)
            @test insupport(d1, vec(x))
            @test logpdf(d, x) == logpdf(d1, vec(x))
        end
    end
    @testset "ArrayReshaped vec" begin
        for d in d1s
            @test vec(d) == d1
        end
    end
end
