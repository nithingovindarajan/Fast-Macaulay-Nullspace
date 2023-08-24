include("Macaulay.jl")
using .Macaulay
using Plots
using BenchmarkTools
using LinearAlgebra
using Random
using DelimitedFiles

# Parameters
Random.seed!(1234)
d_Σs = round.(Int64, 10 .^ (1.4:.025:2.175))
eta = 0     # Noise to add
abstol = 0.0::Float64  # 0 to avoid early stop
S = 2       # Number of polynomials
repeats = 5

## PRECOMPILE
function null_exp(C::CauchyLike{ComplexF64}, abstol, Kmax)
    null(C::CauchyLike{ComplexF64}; ABSTOL = abstol, Kmax = Kmax, qr_initialization = qr_initial_macaulay)
end
precompile(null_exp,(CauchyLike{ComplexF64}, Float64, Int64))

## WARMUP RUN
d_Σi = 1
global d_Σ = d_Σs[d_Σi]
global d = 2*d_Σ-2

# Generate polynomials
p = Vector{BivariatePolynomial{Float64}}(undef, S)
p[1] = randBivariatePolynomial(d_Σ)
p[2] = randBivariatePolynomial(d_Σ)
for i = 3:S
    p[i] = add_noise(rand() * p[1] + rand() * p[2], eta)
end

# Create Macaulay matrix
M = MacaulayBivariate{Float64}(p, d)
φ = generate_φ_greedy(M)
global C = CauchyLike(M, φ)
global MM = Matrix(M)

# Benchmark svd
a = @elapsed nullspace(MM)

# Benchmark our method
global Kmax = (size(C,2)-d_Σ^2)::Int64
a = @elapsed null_exp(C::CauchyLike{ComplexF64}, abstol, Kmax)

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
        φ = generate_φ_greedy(M)
        global C = CauchyLike(M, φ)
        global MM = Matrix(M)
        display(size(C))
        
        if d_Σ < 60
            re = round(Int64, 2*60^5/d_Σ^5)
            a = @elapsed for i=1:re
                nullspace(MM)
            end 
            a = a/re
            println(a, " ", re)
        else
            a = @elapsed nullspace(MM)
            display(a)
        end
        ressvd[d_Σi, rep] = a

        # Benchmark our method
        global Kmax = (size(C,2)-d_Σ^2)::Int64
        if d_Σ < 60
            re = round(Int64, 2*60^5/d_Σ^5)
            a = @elapsed for i=1:re
                null_exp(C::CauchyLike{ComplexF64}, abstol, Kmax)
            end 
            a = a/re
            println(a, " ", re)
        else
            a = @elapsed null_exp(C::CauchyLike{ComplexF64}, abstol, Kmax)
            println(a)
        end
        resmcl[d_Σi, rep] = a
    end
    writedlm("figures/complexity.dat",hcat(string.(d_Σs),string.(resmcl)))
    writedlm("figures/complexity_svd.dat",hcat(string.(d_Σs),string.(ressvd)))
end

# ressvd = ressvd/1E9 #Transform from ns to s
# resmcl = resmcl/1E9 #Transform from ns to s

# plot(d_Σs, ressvd, yaxis=:log, xaxis=:log, label="timing svd", xlabel="Degree", ylabel="Time (s)")
# plot!(d_Σs, resmcl, label="timing our method")
# plot!(d_Σs,d_Σs.^4*resmcl[1]/d_Σs[1].^4, label= "O(d^4)")
# plot!(d_Σs,d_Σs.^5*resmcl[1]/d_Σs[1].^5, label= "O(d^5)")
# plot!(d_Σs,d_Σs.^6*resmcl[1]/d_Σs[1].^6, label= "O(d^6)")
# plot!(legend=:bottomright)

writedlm("figures/complexity.dat",hcat(string.(d_Σs),string.(resmcl)))
writedlm("figures/complexity_svd.dat",hcat(string.(d_Σs),string.(ressvd)))