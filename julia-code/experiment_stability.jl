include("Macaulay.jl")
using .Macaulay
using Statistics
using JLD
using Random
using MAT

reps = 100
minSizePower = 1
maxSizePower = 6
nProblems = 5
results = zeros(nProblems, maxSizePower-minSizePower+1, reps)
for rep = 1:reps
    for (i, d_Σ) in enumerate(2 .^(minSizePower:maxSizePower))
        # Problem 1: 2 uniformly random polynomials
        println("Problem 1, size ", d_Σ,", rep ", rep)
        Random.seed!(i*reps+rep+1)
        p1 = randBivariatePolynomial(d_Σ)
        p2 = randBivariatePolynomial(d_Σ)
        A = MacaulayBivariate{Float64}([p1, p2], 2*d_Σ-2)
        H = null(A::MacaulayBivariate{Float64}; ABSTOL=0, Kmax = size(A,2)-d_Σ^2, gen_φ = generate_φ_greedy)
        results[1, i, rep] = rel_ind_err(A, H)
        display(results[1, i, rep])
        # display(rel_ind_err_old(A, H))

        # Problem 2: 2 uniformly random polynomials but φ chosen randomly instead of greedy
        println("Problem 2, size ", d_Σ,", rep ", rep)
        Random.seed!(i*reps+rep+1)
        p1 = randBivariatePolynomial(d_Σ)
        p2 = randBivariatePolynomial(d_Σ)
        A = MacaulayBivariate{Float64}([p1, p2], 2*d_Σ-2)
        H = null(A::MacaulayBivariate{Float64}; ABSTOL=0, Kmax = size(A,2)-d_Σ^2, gen_φ = generate_φ_randomly)
        results[2, i, rep] = rel_ind_err(A, H)
        display(results[2, i, rep])
        # display(rel_ind_err_old(A, H))

        # Problem 3: 10 uniformly random polynomials
        println("Problem 3, size ", d_Σ,", rep ", rep)
        Random.seed!(i*reps+rep+1)
        p = Vector{BivariatePolynomial{Float64}}(undef, 10)
        p[1] = randBivariatePolynomial(d_Σ)
        p[2] = randBivariatePolynomial(d_Σ)
        for i = 3:10
            p[i] = rand() * p[1] + rand() * p[2]
        end
        A = MacaulayBivariate{Float64}(p, 2*d_Σ-2)
        H = null(A::MacaulayBivariate{Float64}; ABSTOL=0, Kmax = size(A,2)-d_Σ^2, gen_φ = generate_φ_greedy)
        results[3, i, rep] = rel_ind_err(A, H)
        display(results[3, i, rep])
        # display(rel_ind_err_old(A, H))
        
        # Problem 4: 10 noisy (eta = 1E-8) uniformly random polynomials
        # println("Problem 4, size ", d_Σ,", rep ", rep)
        # Random.seed!(i*reps+rep+1)
        # p = Vector{BivariatePolynomial{Float64}}(undef, 10)
        # p[1] = randBivariatePolynomial(d_Σ)
        # p[2] = randBivariatePolynomial(d_Σ)
        # for i = 3:10
        #     p[i] = add_noise(rand() * p[1] + rand() * p[2], 1E-8)
        # end
        # A = MacaulayBivariate{Float64}(p, 2*d_Σ-2)
        # H = null(A::MacaulayBivariate{Float64}; ABSTOL=0, Kmax = size(A,2)-d_Σ^2, gen_φ = generate_φ_greedy)
        # results[4, i, rep] = rel_ind_err(A, H)
        # display(results[4, i, rep])

        # Problem 5: 2 uniformly random polynomials in huge Macaulay matrix
        if d_Σ < 25
            println("Problem 5, size ", d_Σ,", rep ", rep)
            Random.seed!(i*reps+rep+1)
            p1 = randBivariatePolynomial(d_Σ)
            p2 = randBivariatePolynomial(d_Σ)
            A = MacaulayBivariate{Float64}([p1, p2], 4*d_Σ)
            H = null(A::MacaulayBivariate{Float64}; ABSTOL=0, Kmax = size(A,2)-d_Σ^2, gen_φ = generate_φ_greedy)
            results[5, i, rep] = rel_ind_err(A, H)
            display(results[5, i, rep])
            # display(rel_ind_err_old(A, H))
        end
    end
end
save("figures/stability.jld", "data", results)
file = matopen("figures/stability.mat", "w")
write(file, "results", results)
close(file)
display(mean(results; dims=3))
display(median(results; dims=3))
display(std(results; dims=3))