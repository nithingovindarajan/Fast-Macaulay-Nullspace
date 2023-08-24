module Macaulay

# exported functions
export BivariatePolynomial, MacaulayBivariate, bezout_number, bivariate_vandermonde_vec, bivariate_vandermonde_basis, CauchyLike, qr_initial, qr_initial_macaulay, qr_update, schur_algorithm, NullBasisCauchyLike, null, Ω, generate_φ_randomly, D, rel_ind_err, randBivariatePolynomial, randnBivariatePolynomial, add_noise, generate_φ_greedy

# required packages
using LinearAlgebra
using FFTW
using Random

# Auxiliary functions
D(n, φ::Complex) = [φ^(-(k - 1) / n) for k = 1:n]
Ω(n) = [exp(π * 2im * (k - 1) / n) for k = 1:n]
Ω(n, φ) = φ^(1 / n) * Ω(n)
function rel_ind_err(A, H)
    A = Matrix(A)
    H = Matrix(H)
    return(opnorm(A * H) / (opnorm(A) * opnorm(H)))
end

# the new functions
μ(d, S) = vcat((repeat(Ω(k + 1), inner=S) for k = d:-1:0)...)
ν(φ) = vcat((Ω(length(φ) - k + 1, φ_k) for (k, φ_k) ∈ zip(1:length(φ), φ))...)

# generating vandermonde bases
vandermonde(t, x, y, i, d) = y^(d - i) * (x .^ (0:i)) .* (t .^ (i:-1:0))
bivariate_vandermonde_vec(t, x, y, d) = vcat([vandermonde(t, x, y, i, d) for i = d:-1:0]...)
bivariate_vandermonde_basis(roots, d) = hcat([bivariate_vandermonde_vec(t, x, y, d) for (t, x, y) ∈ roots]...)


# Bivariate polynomial type
struct BivariatePolynomial{Scalar<:Number}
    coeff::Vector{Vector{Scalar}}
    d_Σ::Integer
    function BivariatePolynomial{Scalar}(coeff) where {Scalar<:Number}
        for i = 1:length(coeff)-1
            @assert length(coeff[i]) - 1 == length(coeff[i+1])
        end
        @assert length(coeff[1]) == length(coeff)
        new{Scalar}(convert(Vector{Vector{Scalar}}, coeff), length(coeff) - 1)
    end
end
function randBivariatePolynomial(d)
    p = Vector{Vector{Float64}}(undef, d + 1)
    for i = 1:d+1
        p[d+2-i] = rand(Float64, i)
    end
    return BivariatePolynomial{Float64}(p)
end
function randnBivariatePolynomial(d)
    p = Vector{Vector{Float64}}(undef, d + 1)
    for i = 1:d+1
        p[d+2-i] = randn(Float64, i)
    end
    return BivariatePolynomial{Float64}(p)
end
function Base.:+(p::BivariatePolynomial{T}, q::BivariatePolynomial{T}) where {T<:Number}
    if length(p.coeff) < length(q.coeff)
        p, q = q, p
    end
    r = deepcopy(p.coeff)
    n = length(q.coeff)
    for i = 1:n
        r[i][1:n-i+1] .+= q.coeff[i]
    end
    return BivariatePolynomial{T}(r)
end
function Base.:*(c::S, q::BivariatePolynomial{T}) where {T<:Number,S<:Number}
    r = deepcopy(q.coeff)
    for i = eachindex(r)
        r[i] .*= c
    end
    return BivariatePolynomial{T}(r)
end
Base.:*(q::BivariatePolynomial{T}, c::S) where {T<:Number,S<:Number} = c * q
function add_noise(p::BivariatePolynomial{T}, eta::S) where {T<:Number,S<:Number}
    r = deepcopy(p.coeff)
    for i = eachindex(r)
        r[i] .+= eta * randn(T, length(r[i]))
    end
    return BivariatePolynomial{T}(r)
end


# Macaulay matrices of bivariate polynomial systemss
struct MacaulayBivariate{Scalar<:Number} <: AbstractMatrix{Scalar}
    polynomials::Vector{BivariatePolynomial{Scalar}}
    d::Integer
    S::Integer
    d_Σ::Integer
    Δd::Integer
    M::Integer
    N::Integer
    block_i::Vector{Integer}
    block_j::Vector{Integer}
    off_i::Vector{Integer}
    off_j::Vector{Integer}
    disp_rank::Integer  # The displacement rank of M(d) as defined in the paper..

    function MacaulayBivariate{Scalar}(polynomials, d) where {Scalar<:Number}
        S = length(polynomials)
        @assert S >= 2       # We need at least a square system.
        d_Σ = polynomials[1].d_Σ
        @assert all([p.d_Σ == d_Σ for p ∈ polynomials]) # For now, all polynomials should be of the same degree.
        Δd = d - d_Σ
        @assert Δd >= 0    # The Macaulay matrix is only defined if d >= d_Σ.
        M = (S * (Δd + 1) * (Δd + 2)) ÷ 2
        N = ((d + 1) * (d + 2)) ÷ 2
        block_i = [S * (Δd + 1 - i) for i ∈ 0:Δd]
        block_j = [d + 1 - i for i ∈ 0:d]
        off_i = [0; cumsum(block_i)]
        off_j = [0; cumsum(block_j)]
        disp_rank = S * (Δd + 1)
        new{Scalar}(polynomials, d, S, d_Σ, Δd, M, N, block_i, block_j, off_i, off_j, disp_rank)
    end

end
Base.:size(A::MacaulayBivariate) = (A.M, A.N)::Tuple{Int64, Int64}
function Base.:getindex(A::MacaulayBivariate, i::Int, j::Int)
    i -= 1
    j -= 1
    I = floor(Int, ((2 * A.Δd + 3) - sqrt((2 * A.Δd + 3)^2 - 8 * i / A.S)) / 2)
    J = floor(Int, ((2 * A.d + 3) - sqrt((2 * A.d + 3)^2 - 8 * j)) / 2)
    if 0 <= J - I <= A.d_Σ
        v, u = divrem(i - (A.M - A.S * (A.Δd + 2 - I) * (A.Δd + 1 - I) ÷ 2), A.S)
        w = j - (A.N - (A.d + 2 - J) * (A.d + 1 - J) ÷ 2)
        if 0 <= w - v <= A.d_Σ - (J - I)
            return A.polynomials[u+1].coeff[J-I+1][w-v+1]
        else
            return 0
        end
    else
        return 0
    end
end
function Base.:Matrix(A::MacaulayBivariate)
    Adense = zeros(eltype(A), A.M, A.N)
    for I = 0:A.Δd
        for J = 0:A.d_Σ
            for s = 1:A.S
                for i = 0:(A.Δd-I)
                    x = A.off_i[I+1] + A.S * i + s
                    y = A.off_j[I+J+1] + i + 1
                    Adense[x, y:(y+A.d_Σ-J)] = A.polynomials[s].coeff[J+1]
                end
            end
        end
    end
    return Adense
end
bezout_number(A::MacaulayBivariate) = A.d_Σ^2    # Here, it is assumed that the two polynomials of the system generate the entire ideal



# Cauchy-like matrices
struct CauchyLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    omega::Vector{Scalar}
    lambda::Vector{Scalar}
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    m::Int64
    n::Int64
    r::Int64

    function CauchyLike{Scalar}(omega, lambda, U, V) where {Scalar<:Number}
        if size(U, 2) != size(V, 2)
            throw(DomainError("size(R, 2) != size(S, 2)"))
        end
        if length(omega) != size(U, 1)
            throw(DomainError("length(omega) != size(R, 1)"))
        end
        if length(lambda) != size(V, 1)
            throw(DomainError("length(lambda) != size(S, 1)"))
        end
        m = length(omega)
        n = length(lambda)
        r = size(U, 2)

        new{Scalar}(convert(Vector{Scalar}, omega), convert(Vector{Scalar}, lambda),
            convert(Matrix{Scalar}, U), convert(Matrix{Scalar}, V), m, n, r)
    end

end
Base.:size(A::CauchyLike) = (A.m, A.n)::Tuple{Int64, Int64}
Base.:getindex(A::CauchyLike, i::Int, j::Int) = @views dot(A.V[j, :], A.U[i, :]) / (A.omega[i] - A.lambda[j])
Base.:Matrix(A::CauchyLike{TC}) where {TC <: Number} = ((A.U * A.V') ./ (A.omega .- transpose(A.lambda)))::Matrix{TC}

function determine_rho(A::CauchyLike)
    X = abs.(A.omega .- transpose(A.lambda))
    return maximum(X) / minimum(X)
end

# Conversion of Macualay into Cauchy-like matrix 
function generate_φ_randomly(A::MacaulayBivariate)
    return exp.(π * 2im * rand(A.d + 1))    # to be improved later
end
function generate_φ_greedy(A::MacaulayBivariate{TF}) where {TF<:Number}
    # Compute α except on [0 1] instead of unity circle
    Ω(n) = [(k-1) / n for k = 1:n]
    α = vcat((Ω(k + 1) for k = A.Δd:-1:0)...)
    # Initialize output
    φ = Vector{Complex{TF}}(undef, A.d + 1)
    # Initialize already placed values
    j = length(α) # Number of already placed
    w = Vector{Float64}(undef,j + round(Int64,(A.d + 1)*(A.d + 2)/2))
    w[1:j] .= α # Alpha is already placed
    # Greedy addition
    for i = 1:A.d + 1
        # Number of points to place equidistantly
        k = A.d + 2 - i
        # Sort all current points between 0 and 1/k after modulo 1/k
        modk(x) = mod(x,1/k)
        @views sort!(w[1:j], by=modk)
        w[j+1] = 1/k-1E-16
        # Find largest gap
        @views mi = argmax(diff(modk.(w[1:j+1])))
        φi = (modk(w[mi]) + modk(w[mi+1]))/2 # Middle of largest gap
        # Add to already placed
        w[j+1:j+k] = [φi+(h - 1) / k for h = 1:k]
        j = j + k
        # Store φ^k
        φ[i] = exp(π * 2im * φi * k) 
    end
    return φ::Vector{Complex{TF}}
end
function CauchyLike(A::MacaulayBivariate, φ)

    if length(φ) != A.d + 1
        DomainError("length(φ) != A.d +1")
    end

    # constructing U
    U = zeros(ComplexF64, A.M, A.disp_rank)
    for i = 1:A.Δd+1
        entry = 1 / sqrt(A.Δd + 1 - (i - 1))
        for k = 1:(A.Δd+1-(i-1))
            for s = 1:A.S
                U[A.off_i[i]+(k-1)*A.S+s, (i-1)*A.S+s] = entry
            end
        end
    end

    # constructing V
    V = zeros(ComplexF64, A.N, A.disp_rank)
    for j = 1:A.d_Σ+1
        for k = 1:A.Δd+1
            diag_scaling = sqrt(A.d + 1 - ((j + k - 1) - 1)) * conj(D(A.d + 1 - ((j + k - 1) - 1), φ[j+k-1]))  # square-root term is needed to compensate for ifft
            for s = 1:A.S
                q = [zeros(A.d + 1 - ((j + k - 1) - 1) - (A.d_Σ + 1 - (j - 1)))
                    conj(A.polynomials[s].coeff[j])]
                w = [conj(A.polynomials[s].coeff[j][2:end])
                    zeros(A.d + 1 - ((j + k - 1) - 1) - (A.d_Σ + 1 - (j - 1)))
                    conj(φ[j+k-1] * A.polynomials[s].coeff[j][1])]
                V[A.off_j[j+k-1]+1:A.off_j[j+k], (k-1)*A.S+s] = ifft(diag_scaling .* (q - w), 1)
            end
        end
    end

    # mu
    mu = μ(A.Δd, A.S)
    # nu
    nu = ν(φ)

    return CauchyLike{ComplexF64}(mu, nu, U, V)
end


############ Schur updating ##############

function qr_initial(A, B)
    if size(A, 1) == 0
        Q = 0 * A
    else
        F = qr(A)
        Q = F.Q * Matrix(I, size(A, 1), min(size(A, 1), size(A, 2)))
        B = B * F.R'
    end
    return Q, B
end


function schur_update!(U::AbstractMatrix{TC}, V::AbstractMatrix{TC}, αinv::TC, g::AbstractVector{TC}, h::AbstractVector{TC}, 
    p::Integer, t::AbstractVector{TC}, Δt::AbstractVector{TC}, tmp::AbstractVector{TC}) where {TC <: Number}
    tol = 1E-8

    # Address empty cases.
    if size(U, 1) < 1
        return p
    end

    # Store rank.
    a = size(U, 2)

    # Deleting the first row, step 1: make sure U[1,:] = [c 0 0 ... 0] so all columns stay orthonormal.
    @views nrm = norm(U[1, p:end])
    if abs(U[1, p]) < 1E-32
        sgn = 1.0+0.0im
    else
        sgn = sign(U[1, p])
    end
    U[1, p] += sgn*nrm
    τ = 1/(conj(sgn*nrm)*U[1, p])
    @views U[1, p:end] .= conj.(U[1, p:end])
    # U[2:end, p:end] -= (U[2:end, p:end]*U[1, p:end])*(conj(τ)*U[1, p:end])';
    @views LinearAlgebra.BLAS.gemv!('N', true, U[2:end, p:end], U[1, p:end], false, tmp[1:size(U,1)-1])
    @views LinearAlgebra.BLAS.ger!(-τ, tmp[1:size(U,1)-1], U[1, p:end], U[2:end, p:end])
    # V[:, p:end] -= (V[:, p:end]*U[1, p:end])*(τ*U[1, p:end])';
    @views LinearAlgebra.BLAS.gemv!('N', true, V[:, p:end], U[1, p:end], false, tmp[1:size(V,1)])
    @views LinearAlgebra.BLAS.ger!(-τ, tmp[1:size(V,1)], U[1, p:end], V[:, p:end])
    U[1, p] = -sgn*nrm

    # Update U & V with G & H
    U[1, p] *= -αinv
    # @views U[2:end, p] .+= g.*U[1, p]
    @views LinearAlgebra.BLAS.axpy!(U[1, p], g, U[2:end, p])
    # @views V[2:end, :] .-= conj(αinv).*h.*transpose(V[1,:])
    @views t .= conj.(V[1, :]) # Can't use zgeru in blas so use zgerc instead
    @views LinearAlgebra.BLAS.ger!(conj(-αinv), h, t, V[2:end,:])

    @views U = U[2:end,:]
    @views V = V[2:end,:]

    # Reorthogonalize U: Gram-Schmidt
    @views LinearAlgebra.BLAS.gemv!('C', true, U[:, p+1:end], U[:, p], false, t[p+1:end])
    @views LinearAlgebra.BLAS.gemv!('N', -1.0+0.0im, U[:, p+1:end], t[p+1:end], true, U[:, p])
    # @views @inbounds for i = p+1:a
    #     @views t[i] = U[:, i] ⋅ U[:, p]
    #     @views U[:, p] .-= U[:, i] .* t[i]
    # end
    
    # Reorthogonalize U: Second time Gram-Schmidt for stability
    @views LinearAlgebra.BLAS.gemv!('C', true, U[:, p+1:end], U[:, p], false, Δt[p+1:end])
    @views LinearAlgebra.BLAS.gemv!('N', -1.0+0.0im, U[:, p+1:end], Δt[p+1:end], true, U[:, p])
    @views LinearAlgebra.BLAS.axpy!(true, Δt[p+1:end], t[p+1:end])
    # @views @inbounds for i = p+1:a
    #     @views tmp = U[:, i] ⋅ U[:, p]
    #     @views U[:, p] .-= U[:, i] .* tmp
    #     t[i] += tmp
    # end
    @views un = norm(U[:, p])

    # Compensate changes in V
    # @views @inbounds for i = p+1:a
    #     @views V[:, i] .+= V[:, p] .* conj(t[i])
    # end
    @views LinearAlgebra.BLAS.ger!(1.0+0.0im, V[:, p], t[p+1:a], V[:,p+1:a])
    
    # Normalize
    if un < tol || size(U,1)-1 < size(U,2)-p
        p += 1
    else
        # U[:, p] ./= un
        @views LinearAlgebra.BLAS.scal!(convert(typeof(U[1,p]), 1/un), U[:, p])
        # V[:, p] .*= un
        @views LinearAlgebra.BLAS.scal!(convert(typeof(U[1,p]), un), V[:, p])
    end

    return p::Integer
end

############ SCHUR ALGORITHM  ##############
function swaprows!(X::AbstractMatrix, i::Integer, j::Integer)
    @inbounds for k = axes(X, 2)
        X[i, k], X[j, k] = X[j, k], X[i, k]
    end
end

function maxrow(A::AbstractMatrix{TC}) where {TC <: Number}
    maxnrm = norm(view(A, 1, :))
    maxidx = 1
    for idx = 2:size(A,1)
        nrm = norm(view(A, idx, :))
        if nrm > maxnrm
            maxnrm = nrm
            maxidx = idx
        end
    end
    return maxnrm, maxidx
end

function schur_algorithm(A::CauchyLike{TC}; Kmax::Int64=minimum(size(A), init=0), qr_initialization=qr_initial, ABSTOL::TF=1E-12) where {TF <: Real, TC <: Number}
    Π_1 = collect(Int64, 1:size(A, 1))::Vector{Int64}
    Π_2 = collect(Int64, 1:size(A, 2))::Vector{Int64}
    μ = copy(A.omega)
    ν = copy(A.lambda)
    U = copy(A.U)
    V = copy(A.V)
    U2 = copy(U)
    V2 = copy(V)

    # Prealloc
    a = Vector{eltype(U)}(undef, size(U, 1))
    w = Vector{eltype(U)}(undef, size(U, 1))
    g = Vector{eltype(V)}(undef, size(V, 1))
    uvt = Vector{eltype(V)}(undef, size(V, 2))
    tmp = Vector{eltype(V)}(undef, size(V, 2))
    tmp2 = Vector{eltype(V)}(undef, maximum([size(V, 1), size(U, 1)]))

    # Initial orthonormalization
    U, V = qr_initialization(U, V)

    r = 0
    p = 1
    for k = 1:Kmax

        #### check stopping criteria ####
        @views max2nrm, j = maxrow(V[k:end, p:end])
        if max2nrm < ABSTOL   # stopping criteria  
            break
        end

        #### Pivot columns ####
        j += k .- 1
        if j != k
            swaprows!(V, j, k)
            swaprows!(V2, j, k)
            ν[k], ν[j] = ν[j], ν[k]
            Π_2[k], Π_2[j] = Π_2[j], Π_2[k]
        end

        #### Pivot rows ####
        @views uvt[p:end] .= conj.(V[k, p:end])
        @views LinearAlgebra.BLAS.gemv!('N', true, U[k:end, p:end], uvt[p:end], false, a[k:end])
        @views a[k:end] ./= (μ[k:end] .- ν[k])
        i = argmax(Iterators.map(abs, a[k:end]))
        i += k .- 1
        if i != k
            a[k], a[i] = a[i], a[k]
            swaprows!(U, k, i)
            swaprows!(U2, k, i)
            μ[k], μ[i] = μ[i], μ[k]
            Π_1[k], Π_1[i] = Π_1[i], Π_1[k]
        end

        ######################### VERSION 1 ORTHOGONAL ###########################
        #### Gauss step ####
        alphainv = 1 / a[k]
        # g[(k+1):end] = V[k+1:end, :]*conj.(U[k, :]) ./ conj.(μ[k] .- ν[k+1:end])
        @views uvt[p:end] .= conj.(U[k, p:end])
        @views LinearAlgebra.BLAS.gemv!('N', true, V[k+1:end, p:end], uvt[p:end], false, g[(k+1):end])
        @views g[(k+1):end] ./= conj.(μ[k] .- ν[k+1:end])
        ## QR step (update) ##
        @views p = schur_update!(U[k:end,:], V[k:end,:], alphainv, a[k+1:end], g[(k+1):end], p, uvt, tmp, tmp2)  

        ######################### VERSION 2 ACCURATE   ###########################
        # a = (U2[k:end,:]*conj(V2[k,:]))./(μ[k:end] .- ν[k])
        @views uvt .= conj.(V2[k, :])
        @views LinearAlgebra.BLAS.gemv!('N', true, U2[k:end, :], uvt, false, a[k:end])
        @views a[k:end] ./= (μ[k:end] .- ν[k])
        # w[1:k-1] = (U2[1:k-1, :]*conj(V2[k, :]))./(ν[1:k-1] .- ν[k])
        # @views uvt .= conj.(V2[k, :])
        @views LinearAlgebra.BLAS.gemv!('N', true, U2[1:k-1, :], uvt, false, w[1:k-1])
        @views w[1:k-1] ./= (ν[1:k-1] .- ν[k])
        # g[(k+1):end] = (V2[k+1:end, :]*conj.(U2[k, :])) ./ conj.(μ[k] .- ν[k+1:end])
        @views uvt .= conj.(U2[k, :])
        @views LinearAlgebra.BLAS.gemv!('N', true, V2[k+1:end, :], uvt, false, g[(k+1):end])
        @views g[(k+1):end] ./= conj.(μ[k] .- ν[k+1:end])
        #### Gauss step ####
        αinv = 1/a[k]
        # @views U2[k,:] .*= -αinv
        @views LinearAlgebra.BLAS.scal!(-αinv, U2[k,:])
        # @views U2[1:k-1,:] .+= w[1:k-1] .* transpose(U2[k,:])
        @views LinearAlgebra.BLAS.ger!(-αinv, w[1:k-1], uvt, U2[1:k-1,:])
        # @views U2[k+1:end,:] .+= a[2:end] .* transpose(U2[k,:])
        @views LinearAlgebra.BLAS.ger!(-αinv, a[k+1:end], uvt, U2[k+1:end,:])
        # @views V2[k+1:end,:] .-= conj(αinv) .* g[(k+1):end] .* transpose(V2[k,:])
        @views uvt .= conj.(V2[k, :]) # Can't use zgeru in blas so use zgerc instead
        @views LinearAlgebra.BLAS.ger!(-conj(αinv), g[(k+1):end], uvt, V2[k+1:end,:])

        #update rank
        r += 1
    end

    # Construct N
    N = CauchyLike{TC}(ν[1:r], ν[r+1:end], U2[1:r, :], V2[r+1:end, :])
    # if return_schur_complement
    #     # Construct Schur complement
    #     schur_complement = CauchyLike{TC}(μ[r+1:end], ν[r+1:end], U2[r+1:end, :], V2[r+1:end, :])
    #     return Π_1, Π_2, N, schur_complement
    # else
        return Π_1, Π_2, N
    # end

end


# Nullspace basis Cauchy-like matrix
struct NullBasisCauchyLike{Scalar<:Complex} <: AbstractMatrix{Scalar}
    N::CauchyLike{Scalar}
    Π_2_T::Vector{<:Integer}
    m::Integer
    n::Integer
    function NullBasisCauchyLike{Scalar}(N, Π_2) where {Scalar<:Complex}
        @assert size(N, 1) + size(N, 2) == length(Π_2)
        Π_2_T = zeros(Int, length(Π_2))
        Π_2_T[Π_2] = collect(1:length(Π_2))   # the inverse permutation
        new{Scalar}(N, Π_2_T, size(N, 1) + size(N, 2), size(N, 2))
    end
end
Base.:size(A::NullBasisCauchyLike{TC}) where {TC <: Number} = (A.m::Integer, A.n::Integer)
function Base.:getindex(A::NullBasisCauchyLike, i::Int, j::Int)
    i = A.Π_2_T[i]
    if i <= A.N.m
        return A.N[i, j]
    else
        return ((i - A.N.m) - j) == 0 ? 1 : 0
    end
end
function Base.:Matrix(A::NullBasisCauchyLike)
    X = [Matrix(A.N); I(size(A.N, 2))]
    X = X[A.Π_2_T, :]
    return X
end

# nullspace of CauchyLike matrices
function null(A::CauchyLike{TC}; ABSTOL::TF=1E-12, Kmax::Integer=minimum(size(A)), qr_initialization=qr_initial_macaulay) where {TF <: Real, TC <: Number}
    _, Π_2, N = schur_algorithm(A; ABSTOL=ABSTOL, qr_initialization=qr_initialization, Kmax)
    return NullBasisCauchyLike{TC}(N, Π_2)
end


# Nullspace basis Macaulay matrix


# Nullspace of Macaulay matrix
function qr_initial_macaulay(A, B)
    return A, B
end
function null(A::MacaulayBivariate{TC}; ABSTOL::TF=1E-12, Kmax::Integer=minimum(size(A)), gen_φ = generate_φ_greedy) where {TF <: Real, TC <: Number}

    # convert into Cauchy-Like
    φ = gen_φ(A)
    Ahat = CauchyLike(A, φ)

    # call nullspace algorithm for cauchy-like matrix
    Hhat = null(Ahat; ABSTOL=ABSTOL, qr_initialization=qr_initial_macaulay, Kmax)

    # inverse FFT transformation to get H
    H = Matrix(Hhat)
    for j = 1:A.d+1
        diag_scaling = 1 / sqrt(A.d + 2 - j) * D(A.d + 2 - j, φ[j])
        @views H[(A.off_j[j]+1):A.off_j[j+1], :] = diag_scaling .* fft(H[(A.off_j[j]+1):A.off_j[j+1], :], 1)
    end

    return H
end




end












