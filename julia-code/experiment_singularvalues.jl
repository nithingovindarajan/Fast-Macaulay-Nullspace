include("Macaulay.jl")
using .Macaulay
using LinearAlgebra
using Plots


ds = 16
d = 2 * ds
p = [Macaulay.randnBivariatePolynomial(ds), Macaulay.randnBivariatePolynomial(ds)]
Md = Macaulay.MacaulayBivariate{Float64}(p, d)
F = svd(Matrix(Md), full=true)

normalized_singular_values = [F.S / F.S[1]
    zeros(size(Md, 2) - size(Md, 1))]
normalized_nonzero_singular_values = normalized_singular_values[1:end-ds^2]

b_range = range(0, 1, length=ds)

p1 = scatter(normalized_nonzero_singular_values, legend=false)
p2 = histogram(normalized_nonzero_singular_values, legend=false, bins=b_range)
plot(p1, p2, layout=2)
