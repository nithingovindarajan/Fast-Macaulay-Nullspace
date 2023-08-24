include("Macaulay.jl")
using .Macaulay
using Plots
using BenchmarkTools
using LinearAlgebra
using Random
using DelimitedFiles

# Parameters
Random.seed!(1234)
d_Σs = round.(Int64, 10 .^ (1.4:.025:2.7))
eta = 0     # Noise to add
abstol = 0.0::Float64  # 0 to avoid early stop
S = 2       # Number of polynomials
repeats = 5

## PRECOMPILE
function prep_exp(A::CauchyLike{ComplexF64})
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

    # Return all
    return Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2 
end

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

function null_exp(Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2, ABSTOL, Kmax)
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
end
precompile(null_exp,(CauchyLike{ComplexF64}, Float64, Int64))



## WARMUP RUN
# for i=1:10
#     global d_Σ = 100
#     global d = 2*d_Σ-2

#     # Generate polynomials
#     p = Vector{BivariatePolynomial{Float64}}(undef, S)
#     p[1] = randBivariatePolynomial(d_Σ)
#     p[2] = randBivariatePolynomial(d_Σ)
#     for i = 3:S
#         p[i] = add_noise(rand() * p[1] + rand() * p[2], eta)
#     end

#     # Create Macaulay matrix
#     M = MacaulayBivariate{Float64}(p, d)
#     φ = generate_φ_randomly(M)
#     global C = CauchyLike(M, φ)
#     # global MM = Matrix(M)

#     # Benchmark svd
#     # a = @elapsed nullspace(MM)

#     # Benchmark our method
#     global Kmax = 1::Int64
#     Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2 = prep_exp(C)
#     global Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2
#     a = @elapsed null_exp(Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2, abstol, Kmax)
# end



## EXPERIMENT COMPLEXITY
ressvd = zeros(length(d_Σs),repeats)
resmcl = zeros(length(d_Σs),repeats)
for rep = 1:repeats
    for d_Σi = eachindex(d_Σs)
        global d_Σ = d_Σs[d_Σi]
        global d = 2*d_Σ-2
        display("-------------------------------------------------------------")
        println([d_Σ, rep])
        
        # Generate polynomials
        p = Vector{BivariatePolynomial{Float64}}(undef, S)
        p[1] = randBivariatePolynomial(d_Σ)
        p[2] = randBivariatePolynomial(d_Σ)
        for i = 3:S
            p[i] = add_noise(rand() * p[1] + rand() * p[2], eta)
        end

        # Create Macaulay matrix
        M = MacaulayBivariate{Float64}(p, d)
        φ = generate_φ_randomly(M)
        global C = CauchyLike(M, φ)
        # global MM = Matrix(M)
        display(size(C))
        
        # SVD
        # a = @benchmark nullspace(MM)
        # display(a)
        # ressvd[d_Σi, rep] = median(a.times)

        # Benchmark our method
        global Kmax = 1::Int64
        Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2 = prep_exp(C)
        global Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2
        a = @benchmark null_exp(Π_1, Π_2, μ, ν, U, V, U2, V2, a, w, g, uvt, tmp, tmp2, abstol, Kmax)
        display(a)
        resmcl[d_Σi, rep] = median(a.times)
    end
    writedlm("figures/complexity_1it.dat",hcat(string.(d_Σs),string.(resmcl)))
    writedlm("figures/complexity_1it_svd.dat",hcat(string.(d_Σs),string.(ressvd)))
end

# ressvd = ressvd/1E9 #Transform from ns to s
# resmcl = resmcl/1E9 #Transform from ns to s

# plot(d_Σs, ressvd, yaxis=:log, xaxis=:log, label="timing svd", xlabel="Degree", ylabel="Time (s)")
# plot!(d_Σs, resmcl, label="timing our method")
# plot!(d_Σs,d_Σs.^4*resmcl[1]/d_Σs[1].^4, label= "O(d^4)")
# plot!(d_Σs,d_Σs.^5*resmcl[1]/d_Σs[1].^5, label= "O(d^5)")
# plot!(d_Σs,d_Σs.^6*resmcl[1]/d_Σs[1].^6, label= "O(d^6)")
# plot!(legend=:bottomright)

writedlm("figures/complexity_1it.dat",hcat(string.(d_Σs),string.(resmcl)))
writedlm("figures/complexity_1it_svd.dat",hcat(string.(d_Σs),string.(ressvd)))