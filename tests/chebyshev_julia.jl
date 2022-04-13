
using BenchmarkTools
using Distributions
using Printf
using Random
using Test

include("../src/multivariate_chebyshev_julia.jl")

@testset "test_indexing" begin
    x = [1, 2]
    y = [1, 2, 3]
    z = [1, 2, 3, 4]
    p = cartesian_product([x, y, z]) # call with list and unpack
    # println(size(p))
    # println(p)
    v = 100*p[:,1] + 10*p[:,2] + p[:,3]
    # println(v)
    V = reshape(v, (size(x)[1], size(y)[1], size(z)[1]))
    # println(V)
    V_ref = [
        # [:,:,1]
        111 121 131;
        211 221 231;;;
        # [:,:,2]
        112 122 132;
        212 222 232;;;
        # [:,:,3]
        113 123 133;
        213 223 233;;; 
        # [:,:,4]
        114 124 134;
        214 224 234
    ]
    @test max(abs.(V-V_ref)...) == 0
end

@testset "test_batchmul" begin
    C_ref = 3.0*ones(4,2,5,5)
    @test batchmul(ones(4,3,5,5), ones(3,2,5,5)) == C_ref
    @test batchmul(ones(4,3,5,1), ones(3,2,5,5)) == C_ref
    @test batchmul(ones(4,3,1,5), ones(3,2,5,5)) == C_ref
    @test batchmul(ones(4,3,5,5), ones(3,2,5,1)) == C_ref
    @test batchmul(ones(4,3,5,5), ones(3,2,1,5)) == C_ref
    @test batchmul(ones(4,3,5,1), ones(3,2,1,5)) == C_ref
    @test batchmul(ones(4,3,1,5), ones(3,2,5,1)) == C_ref
    @test batchmul(ones(4,3,1,5), ones(3,2,1,5)) == 3.0*ones(4,2,1,5)
    @test batchmul(ones(4,3,5,1), ones(3,2,5,1)) == 3.0*ones(4,2,5,1)
    @test batchmul(ones(4,3,1,1), ones(3,2,1,1)) == 3.0*ones(4,2,1,1)
    @test batchmul(ones(4,3,5), ones(3,2,5)) == 3.0*ones(4,2,5)
    @test batchmul(ones(4,3), ones(3,2)) == 3.0*ones(4,2)
end

@testset "test_matmul" begin
    C_ref = 3.0*ones(5,5,4,2)
    @test matmul(ones(5,5,4,3), ones(5,5,3,2)) == C_ref
    @test matmul(ones(5,1,4,3), ones(5,5,3,2)) == C_ref
    @test matmul(ones(1,5,4,3), ones(5,5,3,2)) == C_ref
    @test matmul(ones(5,5,4,3), ones(5,1,3,2)) == C_ref
    @test matmul(ones(5,5,4,3), ones(1,5,3,2)) == C_ref
    @test matmul(ones(5,1,4,3), ones(1,5,3,2)) == C_ref
    @test matmul(ones(1,5,4,3), ones(5,1,3,2)) == C_ref
    @test matmul(ones(1,5,4,3), ones(1,5,3,2)) == 3.0*ones(1,5,4,2)
    @test matmul(ones(5,1,4,3), ones(5,1,3,2)) == 3.0*ones(5,1,4,2)
    @test matmul(ones(1,1,4,3), ones(1,1,3,2)) == 3.0*ones(1,1,4,2)
    @test matmul(ones(5,4,3), ones(5,3,2)) == 3.0*ones(5,4,2)
    @test matmul(ones(4,3), ones(3,2)) == 3.0*ones(4,2)
end

@testset "test_chebyshev_points" begin
    degrees = [ 1, 2, 3 ]
    points = chebyshev_points(degrees)
    @test size(points)==(3,)
    @test points[1]==[1.0, -1.0]
    @test isapprox(points[2], [1.0, 0.0, -1.0], atol=1.0e-16)
    @test isapprox(points[3], [1.0, 0.5, -0.5, -1.0], atol=5.0e-16)
    # println(points)
end

@testset "test_chebyshev_multi_points" begin
    degrees = [ 1, 2]
    points = chebyshev_multi_points(degrees)
    points_ref = [
      1.0  1.0; 
     -1.0  1.0;
      1.0  0.0; 
     -1.0  0.0;
      1.0 -1.0;
     -1.0 -1.0
    ]
    @test isapprox(points, points_ref, atol=1.0e-16)
end

@testset "test_chebyshev_polynomials" begin
    polys = chebyshev_polynomials(0.5, 3)
    # println(size(polys))
    # println(polys)
    @test size(polys)==(1,4)
    @test polys==[1.0 0.5 -0.5 -1.0]
    #
    x = collect(range(-1.0, 1.0, 5))
    polys = chebyshev_polynomials(x, 3)
    # println(size(polys))
    # println(polys)
    polys_ref = [
        1.0 -1.0  1.0 -1.0;
        1.0 -0.5 -0.5  1.0;
        1.0  0.0 -1.0 -0.0;
        1.0  0.5 -0.5 -1.0;
        1.0  1.0  1.0  1.0
    ]
    @test size(polys)==(5,4)
    @test polys==polys_ref
end

@testset "test_chebyshev_batch_call" begin
    C = ones(2, 3, 4)
    x = ones(5, 3)
    y = chebyshev_batch_call(C, x)
    # println(size(y))
    # println(y)
    @test y==24.0*ones(5)
    y = chebyshev_batch_call(C, x, matmul)
    @test y==24.0*ones(5)
end

function BlackOverK(x)
    @assert size(x)==(3,)
    moneyness = x[1]
    stdDev    = x[2]
    callOrPut = x[3]
    d1 = log(moneyness) / stdDev + stdDev / 2.0
    d2 = d1 - stdDev
    norm = Normal()
    return callOrPut * (moneyness*cdf(norm,callOrPut*d1)-cdf(norm,callOrPut*d2))
end

@testset "test_chebyshev_coefficients" begin
    degrees = [ 2, 3, 4 ]
    multi_points = chebyshev_multi_points(degrees)
    values = ones(size(multi_points, 1))
    coeff = chebyshev_coefficients(degrees, multi_points, values)
    coeff_ref = zeros(size(coeff))
    coeff_ref[(1 for d in degrees)...] = 1.0
    # println(coeff)
    @test isapprox(coeff, coeff_ref, atol=5.0e-16)
    #
    a = [ 0.5, 0.01, -1.0 ]'
    b = [ 2.0, 0.50, +1.0 ]'
    degrees = [ 3, 4, 5 ]
    #
    multi_points = chebyshev_multi_points(degrees)
    Y = chebyshev_transform(multi_points, a, b)
    values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
    C = chebyshev_coefficients(degrees, multi_points, values)
    # println(size(C))
    @test size(C)==(4,5,6)
end

@testset "test_chebyshev_coefficients_matmul" begin
    degrees = [ 2, 3, 4 ]
    multi_points = chebyshev_multi_points(degrees)
    values = ones(size(multi_points, 1))
    coeff = chebyshev_coefficients(degrees, multi_points, values, matmul)
    coeff_ref = zeros(size(coeff))
    coeff_ref[(1 for d in degrees)...] = 1.0
    # println(coeff)
    @test isapprox(coeff, coeff_ref, atol=5.0e-16)
    #
    a = [ 0.5, 0.01, -1.0 ]'
    b = [ 2.0, 0.50, +1.0 ]'
    degrees = [ 3, 4, 5 ]
    #
    multi_points = chebyshev_multi_points(degrees)
    Y = chebyshev_transform(multi_points, a, b)
    values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
    C = chebyshev_coefficients(degrees, multi_points, values, matmul)
    # println(size(C))
    @test size(C)==(4,5,6)
end

@testset "test_black_formula_chebyshev_points" begin
    a = [ 0.5, 0.01, -1.0 ]'
    b = [ 2.0, 0.50, +1.0 ]'
    degrees = [ 3, 4, 5 ]
    #
    multi_points = chebyshev_multi_points(degrees)
    Y = chebyshev_transform(multi_points, a, b)
    values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
    C = chebyshev_coefficients(degrees, multi_points, values)
    #
    z = chebyshev_interpolation(Y, C, a, b)
    # println(max(abs.(z - values)...))
    @test isapprox(z, values, atol=1.0e-14)
end

@testset "test_black_formula_chebyshev_points_matmul" begin
    a = [ 0.5, 0.01, -1.0 ]'
    b = [ 2.0, 0.50, +1.0 ]'
    degrees = [ 3, 4, 5 ]
    #
    multi_points = chebyshev_multi_points(degrees)
    Y = chebyshev_transform(multi_points, a, b)
    values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
    C = chebyshev_coefficients(degrees, multi_points, values, matmul)
    #
    z = chebyshev_interpolation(Y, C, a, b, matmul)
    # println(max(abs.(z - values)...))
    @test isapprox(z, values, atol=1.0e-14)
end

@testset "test_black_formula_random_points" begin
    a = [ 0.5, 0.50, -1.0 ]'
    b = [ 2.0, 2.50, +1.0 ]'
    degrees = [ 5, 5, 5 ]
    #
    multi_points = chebyshev_multi_points(degrees)
    Y = chebyshev_transform(multi_points, a, b)
    values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
    C = chebyshev_coefficients(degrees, multi_points, values)
    #
    rng = MersenneTwister(42)
    base2 = 13
    y = a .+ rand(rng, Float64, (2^base2, 3)) .* (b.-a)
    z = chebyshev_interpolation(y, C, a, b)
    z_ref = [ BlackOverK(y[i,:]) for i in 1:size(y,1) ]
    # println(max(abs.(z - z_ref)...))
    @test isapprox(z, z_ref, atol=1.0e-1)
    @test max(abs.(z - z_ref)...) < 7.0e-3
end

@testset "test_black_formula_random_points_matmul" begin
    a = [ 0.5, 0.50, -1.0 ]'
    b = [ 2.0, 2.50, +1.0 ]'
    degrees = [ 5, 5, 5 ]
    #
    multi_points = chebyshev_multi_points(degrees)
    Y = chebyshev_transform(multi_points, a, b)
    values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
    C = chebyshev_coefficients(degrees, multi_points, values, matmul)
    #
    rng = MersenneTwister(42)
    base2 = 13
    y = a .+ rand(rng, Float64, (2^base2, 3)) .* (b.-a)
    z = chebyshev_interpolation(y, C, a, b, matmul)
    z_ref = [ BlackOverK(y[i,:]) for i in 1:size(y,1) ]
    # println(max(abs.(z - z_ref)...))
    @test isapprox(z, z_ref, atol=1.0e-1)
    @test max(abs.(z - z_ref)...) < 7.0e-3
end

@testset "test_matrix_multiplication_performance" begin
    D = 5  # number of dimensions
    Nd = 5 # size per dimension
    degrees = ( Nd for d in 1:D )
    n_points = prod(degrees)
    rng = MersenneTwister(42)
    A = rand(rng, Float64, (degrees..., n_points))
    B = permutedims(A, length(size(A)):-1:1)
    b1 = @benchmark batchmul($A,$A)
    b2 = @benchmark matmul($B,$B)
    display(b1)
    println()
    display(b2)
    println()
end

@testset "test_black_formula_performance" begin
    a = [ 0.5, 0.50, -1.0 ]'
    b = [ 2.0, 2.50, +1.0 ]'
    # degrees = [ 5, 5, 5 ]
    degrees = [ 10, 10, 10 ]
    #
    multi_points = chebyshev_multi_points(degrees)
    Y = chebyshev_transform(multi_points, a, b)
    values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
    #
    C1 = chebyshev_coefficients(degrees, multi_points, values)
    C2 = chebyshev_coefficients(degrees, multi_points, values, matmul)
    # println(max(abs.(C1 - C2)...))
    @test max(abs.(C1 - C2)...) < 2.0e-16
    b1 = @benchmark chebyshev_coefficients($degrees, $multi_points, $values)
    b2 = @benchmark chebyshev_coefficients($degrees, $multi_points, $values, $matmul)
    display(b1)
    println()
    display(b2)
    println()
end
