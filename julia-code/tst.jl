# The test file
module Tst

include("Macaulay.jl")
using .Macaulay
using LinearAlgebra
using BlockDiagonals
using Random

# auxiliary functions for testing
DFT(n) = (1 / sqrt(n)) * [exp(-π * (i - 1) * (j - 1) * 2im / n) for i in 1:n, j in 1:n]
Φ(Δd, S) = BlockDiagonal([kron((DFT(k))', I(S)) for k = Δd+1:-1:1])
Ψ(φ) = BlockDiagonal([Diagonal(D(length(φ) - k + 1, φ_k)) * DFT(length(φ) - k + 1) for (k, φ_k) ∈ zip(1:length(φ), φ)])



function test_polynomialconstruction()
    coeff = [
        [4, 3, 1, -1],
        [2, 3, 2],
        [1, -2],
        [-3]
    ]
    p1 = BivariatePolynomial{Float64}(coeff)
    return true
end

function test_macaulayconstruction_square()
    p1 = BivariatePolynomial{Float64}([[-4, -3, 1],
        [5, 2],
        [-1]
    ])
    p2 = BivariatePolynomial{Float64}([[-1, 0, 1],
        [0, 2],
        [1]
    ])
    d = 6
    A = MacaulayBivariate{Float64}([p1, p2], d)
    no_roots = bezout_number(A)
    V_d = bivariate_vandermonde_basis([(1, -1, 0), (1, 0, 1), (1, -2, 3), (1, -5, 4)], d)
    Adense = Matrix(A)
    F = svd(Adense, full=true)
    rA_eps = sum(F.S / F.S[1] .> 5E-16)
    nullity_A = size(Adense, 2) - rA_eps

    @assert no_roots == 4
    @assert A.M == sum(A.block_i)
    @assert A.N == sum(A.block_j)
    @assert (maximum(norm.(eachcol(A * V_d), Inf)) <= 1E-16)
    @assert getindex(A, 1:A.M, 1:A.N) == Matrix(A)
    @assert nullity_A == no_roots

    return true
end

function test_macaulayconstruction_overdetermined()
    p1 = BivariatePolynomial{Float64}([[-4, -3, 1],
        [5, 2],
        [-1]
    ])
    p2 = BivariatePolynomial{Float64}([[-1, 0, 1],
        [0, 2],
        [1]
    ])
    p3 = BivariatePolynomial{Float64}([[-5, -3, 2],
        [5, 4],
        [0]
    ])
    d = 6
    A = MacaulayBivariate{Float64}([p1, p2, p3], d)
    no_roots = bezout_number(A)
    V_d = bivariate_vandermonde_basis([(1, -1, 0), (1, 0, 1), (1, -2, 3), (1, -5, 4)], d)
    Adense = Matrix(A)
    F = svd(Adense, full=true)
    rA_eps = sum(F.S / F.S[1] .> 5E-16)
    nullity_A = size(Adense, 2) - rA_eps

    @assert no_roots == 4
    @assert A.M == sum(A.block_i)
    @assert A.N == sum(A.block_j)
    @assert (maximum(norm.(eachcol(A * V_d), Inf)) <= 1E-16)
    @assert getindex(A, 1:A.M, 1:A.N) == Matrix(A)
    @assert nullity_A == no_roots

    return true
end

function test_cauchyconstruction()
    omega, lambda = Ω(8, 1), Ω(8, -1 + 0im)
    U, V = rand(8, 2), rand(8, 2)
    Acauchy = CauchyLike{ComplexF64}(omega, lambda, U, V)
    @assert Matrix(Acauchy) ≈ getindex(Acauchy, 1:Acauchy.m, 1:Acauchy.n)
    return true
end


function test_macaulaytocauchyconversion_square()
    p1 = BivariatePolynomial{Float64}([[-4, -3, 1, 3],
        [5, 2, -1],
        [-1, 2],
        [-3]
    ])
    p2 = BivariatePolynomial{Float64}([[-1, 0, 1, 3],
        [0, 2, 6],
        [1, -4],
        [2]
    ])
    d = 7
    A = MacaulayBivariate{Float64}([p1, p2], d)
    φ = generate_φ(A)
    Ahat = CauchyLike(A, φ)
    Ahatdense = Φ(A.Δd, A.S) * Matrix(A) * Ψ(φ)
    @assert Ahat ≈ Ahatdense
    return true

end

function test_macaulaytocauchyconversion_overdetermined()
    p1 = BivariatePolynomial{Float64}([[-4, -3, 1, 3],
        [5, 2, -1],
        [-1, 2],
        [-3]
    ])
    p2 = BivariatePolynomial{Float64}([[-1, 0, 1, 3],
        [0, 2, 6],
        [1, -4],
        [2]
    ])
    p3 = BivariatePolynomial{Float64}([[4, -2, 1, 3],
        [2, 2, 3],
        [1, -5],
        [-1]
    ])
    d = 7
    A = MacaulayBivariate{Float64}([p1, p2, p3], d)
    φ = generate_φ(A)
    Ahat = CauchyLike(A, φ)
    Ahatdense = Φ(A.Δd, A.S) * Matrix(A) * Ψ(φ)
    @assert Ahat ≈ Ahatdense
    return true
end


function test_schuralgorithm()
    A = CauchyLike{ComplexF64}(Ω(6, 1), Ω(6, -1 + 0im), rand(6, 2), rand(6, 2))
    K_stop = 3
    Π_1, Π_2, N, schur_complement = schur_algorithm(A; Kmax=K_stop, return_schur_complement=true, ABSTOL=-1)
    Aperm = A[Π_1, Π_2]
    A11, A12 = Aperm[1:K_stop, 1:K_stop], Aperm[1:K_stop, K_stop+1:end]
    A21, A22 = Aperm[K_stop+1:end, 1:K_stop], Aperm[K_stop+1:end, K_stop+1:end]
    schur_complement_dense = A22 - A21 * (A11 \ A12)
    N_dense = -(A11 \ A12)
    @assert schur_complement_dense ≈ Matrix(schur_complement)
    @assert N_dense ≈ Matrix(N)
    return true
end


function test_cauchynullbasis()
    m = 10
    n = 12
    omega = Ω(m, 1)
    lambda = Ω(n, -1 + 0im)
    U = [omega -ones(length(omega))]
    V = [ones(length(lambda)) conj(lambda)]
    N = CauchyLike{ComplexF64}(omega, lambda, U, V)
    Π_2 = randperm(size(N, 1) + size(N, 2))
    X = NullBasisCauchyLike{ComplexF64}(N, Π_2)
    @assert Matrix(X) == getindex(X, 1:size(X, 1), 1:size(X, 2))
    return true
end

function test_cauchynullspace()
    # A matrix of all ones (i.e., rank-one matrix)
    m = 10
    n = 12
    omega = Ω(m, 1)
    lambda = Ω(n, -1 + 0im)
    U = [omega -ones(length(omega))]
    V = [ones(length(lambda)) conj(lambda)]
    Acauchy = CauchyLike{ComplexF64}(omega, lambda, U, V)
    null_dim = size(Acauchy, 2) - rank(Acauchy)
    W = null(Acauchy; ABSTOL=1E-12)
    @assert rel_ind_err(Acauchy, W) < 5E-16
    @assert size(W, 2) == null_dim

    # a rank-two matrix
    m = 10
    n = 12
    omega = Ω(m, 1)
    lambda = Ω(n, -1 + 0im)
    U = [omega -ones(length(omega))]
    V = [ones(length(lambda)) conj(lambda)]
    V[1, 1] = 3
    Acauchy = CauchyLike{ComplexF64}(omega, lambda, U, V)
    null_dim = size(Acauchy, 2) - rank(Acauchy)
    W = null(Acauchy; ABSTOL=1E-12)
    @assert rel_ind_err(Acauchy, W) < 5E-16
    @assert size(W, 2) == null_dim

    # a random fat m X n Cauchy-like matrix. Expected nullity is n-m
    m = 5
    n = 10
    omega = exp.(π * 2im * 0.1 * rand(m))
    lambda = exp.(π * 2im * (0.1 * rand(n) .+ 0.5))
    U = randn(m, 2)
    V = randn(n, 2)
    Acauchy = CauchyLike{ComplexF64}(omega, lambda, U, V)
    W = null(Acauchy; ABSTOL=1E-12)
    @assert n - m > 0
    @assert size(W, 2) == n - m
    @assert rel_ind_err(Acauchy, W) < 5E-14

    return true
end

function test_macaulaynullspace_square()
    p1 = BivariatePolynomial{Float64}([[-4, -3, 1],
        [5, 2],
        [-1]
    ])
    p2 = BivariatePolynomial{Float64}([[-1, 0, 1],
        [0, 2],
        [1]
    ])
    d = 5
    A = MacaulayBivariate{Float64}([p1, p2], d)
    H = null(A; ABSTOL=1E-12)

    @assert size(H, 2) == 4
    @assert rel_ind_err(A, H) < 5E-14

    return true
end

function test_macaulaynullspace_overdetermined()
    p1 = BivariatePolynomial{Float64}([[-4, -3, 1],
        [5, 2],
        [-1]
    ])
    p2 = BivariatePolynomial{Float64}([[-1, 0, 1],
        [0, 2],
        [1]
    ])
    p3 = BivariatePolynomial{Float64}([[-5, -3, 2],
        [5, 4],
        [0]
    ])
    d = 6
    A = MacaulayBivariate{Float64}([p1, p2, p3], d)
    H = null(A; ABSTOL=1E-12)

    @assert size(H, 2) == 4
    @assert rel_ind_err(A, H) < 5E-14

    return true
end


end

using Test

#### Run all tests ####
# TEST 1: Constructing a bivariate polynomial
@test Tst.test_polynomialconstruction()
# TEST 2: Constructing a Macaulay matrix for a square polynomial system
@test Tst.test_macaulayconstruction_square()
# TEST 3: Constructing a Macaulay matrix for an overdetermined polynomial system
@test Tst.test_macaulayconstruction_overdetermined()
# TEST 4: Constructing a Cauchy-like matrix
@test Tst.test_cauchyconstruction()
# TEST 5: Converting the Macaulay matrix of a square system into a Cauchy-like matrix
@test Tst.test_macaulaytocauchyconversion_square()
# TEST 6: Converting the Macaulay matrix of an overdetermined system into a Cauchy-like matrix
@test Tst.test_macaulaytocauchyconversion_overdetermined()
# TEST 7: Running the Schur algorithm on a Cauchy matrix
@test Tst.test_schuralgorithm()
# TEST 8: Evaluating a Cauchy null-space basis
@test Tst.test_cauchynullbasis()
# TEST 9: Compute nullspaces of simple Cauchy-like matrices
@test Tst.test_cauchynullspace()
# TEST 10: Compute nullspaces of maucalay matrix for a square polunomial system
@test Tst.test_macaulaynullspace_square()
# TEST 11: Compute nullspaces of maucalay matrix for a square polunomial system
@test Tst.test_macaulaynullspace_overdetermined()