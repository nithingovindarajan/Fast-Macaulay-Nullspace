include("Macaulay.jl")
using .Macaulay
using LinearAlgebra
using Plots

gaps = Vector{Float64}(undef, 46)
# Set parameters
for ds = 5:50
eta = 0
abstol = 0
S = 8 # Number of additional polynomials 

# Generate polynomials
p = Vector{Macaulay.BivariatePolynomial{Float64}}(undef, S + 2)
p[1] = Macaulay.randnBivariatePolynomial(ds)
p[2] = Macaulay.randnBivariatePolynomial(ds)
for i = 1:S
    p[i+2] = rand() * p[1] + rand() * p[2] + eta * Macaulay.randBivariatePolynomial(ds)
end

# Compute needed size of matrix, matrix and no_roots
d = 2 * ds
M = Macaulay.MacaulayBivariate{Float64}(p, d)
no_roots = Macaulay.bezout_number(M)
ERM = size(M, 2)-no_roots

# Look at singular values
F = svd(Matrix(M), full=true)
nullity_gap = F.S[ERM] / F.S[ERM+1]
println("ds: ", ds, " ------\nnoise: ", eta, "\ninv_gap: ", 1 / nullity_gap)

# Plot singular values
plot(F.S, yaxis=:log)

# # Look at max2nrms
# φ = Macaulay.generate_φ(M)
# C = Macaulay.CauchyLike(M, φ)
# _, _, _, max2nrms = Macaulay.schur_algorithm(C;  return_schur_complement=false, ABSTOL=abstol, Kmax=size(M, 2)-no_roots+1)
# nrm_gap = minimum(max2nrms[1:ERM]) / max2nrms[ERM+1]
# gaps[ds-4] = 1/nrm_gap
# println("nrm_1:   ", max2nrms[1], "\nnrm_max: ", maximum(max2nrms[1:ERM]), "\nnrm_min: ", minimum(max2nrms[1:ERM]), "\nnrm_gap: ", 1/nrm_gap)

# Plot max2nrms
plot!(max2nrms, yaxis=:log, reuse=false)
end