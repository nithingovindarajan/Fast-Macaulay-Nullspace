include("Macaulay.jl")
using .Macaulay
using Statistics
using JLD
using Random
using LinearAlgebra

reps = 100
minSizePower = 1
maxSizePower = 5
noises = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
subsp_errors = zeros(length(noises), maxSizePower-minSizePower+1, reps)
singval_errors = zeros(length(noises), maxSizePower-minSizePower+1, reps)
nullsp_erros = zeros(length(noises), maxSizePower-minSizePower+1, reps)
nullsp_frob_erros = zeros(length(noises), maxSizePower-minSizePower+1, reps)

function subspace_angle(A, B)
    # Assumes A and B have the same size.

    # Make A and B unitary bases for their column space.
    A = Matrix(qr(A, ColumnNorm()).Q)
    B = Matrix(qr(B, ColumnNorm()).Q)

    # Compute subspace angle.
    return asin(min(1, norm(B - A*(A'*B))))
end

for rep = 1:reps
    for (i, d_Σ) in enumerate(2 .^(minSizePower:maxSizePower))
        # Problem creation
        Random.seed!(i*reps+rep+1)
        p = Vector{BivariatePolynomial{Float64}}(undef, 10)
        p[1] = randBivariatePolynomial(d_Σ)
        p[2] = randBivariatePolynomial(d_Σ)
        for i = 3:10
            p[i] = rand() * p[1] + rand() * p[2]
        end
        
        for (j, η) in enumerate(noises)
            println("Noise ", η, ", size ", d_Σ,", rep ", rep)
            pη = deepcopy(p)
            for i = 3:10
                pη[i] = add_noise(p[i], η)
            end
            Aη = MacaulayBivariate{Float64}(pη, 2*d_Σ-2)
            Kmax = size(Aη,2)-d_Σ^2
            Hη = Matrix(null(Aη::MacaulayBivariate{Float64}; ABSTOL=0, Kmax = Kmax, gen_φ = generate_φ_greedy))
            Aη = Matrix(Aη)
            SVDAη = svd(Aη)
            singval_errors[j, i, rep] = SVDAη.S[Kmax+1]/SVDAη.S[1]
            Hη = svd(Hη).U[:,1:d_Σ^2]
            nullsp_erros[j, i, rep] = opnorm(Aη * Hη)/SVDAη.S[1]
            println("σ ",  singval_errors[j, i, rep], "   nullsp ", nullsp_erros[j, i, rep])
        end
        
    end
    
    save("figures/stability_noisy.jld", "data", [singval_errors, nullsp_erros])
end
display(mean(singval_errors; dims=3))
display(mean(nullsp_erros; dims=3))
display(median(singval_errors; dims=3))
display(median(nullsp_erros; dims=3))
display(std(singval_errors; dims=3))
display(std(nullsp_erros; dims=3))