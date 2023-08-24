include("Macaulay.jl")
using .Macaulay

# The simple polynomial system
p1 = BivariatePolynomial{Float64}([[-4, -3, 1],
    [5, 2],
    [-1]
])
p2 = BivariatePolynomial{Float64}([[-1, 0, 1],
    [0, 2],
    [1]
])


# nullity is 4
ϵ = 1E-10        # the way how we now specify tolerance and stopping criteria is non-intuitive!
d_max = 7
numerical_nullity = Array{Int}(undef, (d_max))
rel_err = Array{Float64}(undef, (d_max))
for (i, d) ∈ enumerate(2:d_max)
    A = MacaulayBivariate{Float64}([p1, p2], d)
    H = null(A; ABSTOL=1E-10)

    # save results
    numerical_nullity[i] = size(H, 2)
    rel_err[i] = rel_ind_err(A, H)
end