using LinearAlgebra
using Plots
using Statistics
using Profile
include("Macaulay.jl")


# SIMPLE EXPERIMENT (COMPLEX)
n = 10000
r = 1000

A = rand(Complex{Float64}, n, r);
Q, R = Macaulay.qr_initial(A)
u = rand(Complex{Float64}, n - 1)
v = rand(Complex{Float64}, r)
R1 = u * transpose(v)

Q2, R2 = Macaulay.qr_update!(Q, R, u, v)
display(norm(Q2' * Q2 - I))
display(norm(tril(R2, -1)))
display(norm(A[2:end, :] + R1 - Q2 * R2))
@profview for i = 1:10
    Macaulay.qr_update!(Q, R, u, v)
end
@time for i = 1:10
    Macaulay.qr_update!(Q, R, u, v)
end

# Find complexity
# NS = 1000:500:5000
# RS = 100:100:900
# res = Array{Float64,3}(undef, length(NS), length(RS), 5)
# for i in eachindex(NS)
#     for j in eachindex(RS)
#         for k = 1:5
#             n = NS[i]
#             r = RS[j]

#             A = rand(Complex{Float64}, n, r)
#             Q, R = qr_initial(A)
#             u = rand(Complex{Float64}, n - 1)
#             v = rand(Complex{Float64}, r)

#             res[i, j, k] = @elapsed qr_update(Q, R, u, v)
#             println(NS[i], " ", RS[j], " ", k, " ", res[i, j, k])
#         end
#     end
# end
# display(median(res, dims=3))
# surface(NS, RS, reshape(median(res, dims=3), length(NS), length(RS)))

# # Is complexity O(rn)? Check for n

# vals = 100:100:5000
# res = Matrix{Float64}(undef, length(vals),5)
# r = 50

# for j = 1:5
#     for i in eachindex(vals)
#         A = rand(Complex{Float64}, vals[i], r);
#         Q,R = qr_initial(A)
#         u = rand(Complex{Float64}, vals[i]-1)
#         v = rand(Complex{Float64}, r)

#         res[i,j] = @elapsed qr_update!(Q, R, u, v)
#         println(vals[i]," ", j," ", res[i,j])
#     end
# end
# plot(vals,median(res,dims=2))

# # Is complexity O(rn)? Check for r -> It seems O(r^2n) or O(rn+r^2) -> need further check

# vals = 10:10:490
# res = Matrix{Float64}(undef, length(vals),5)
# n = 500

# for j = 1:5
#     for i in eachindex(vals)
#         A = rand(Complex{Float64}, n, vals[i]);
#         Q,R = qr_initial(A)
#         u = rand(Complex{Float64}, n-1)
#         v = rand(Complex{Float64}, vals[i])

#         res[i,j] = @elapsed qr_update!(Q, R, u, v)
#         println(vals[i]," ", j," ", res[i,j])
#     end
# end
# plot(vals,median(res,dims=2))
# plot!(vals,(vals.^2)./50000)

