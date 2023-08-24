include("Macaulay.jl")
using .Macaulay
using Statistics
using JLD
using Random
using MAT
using LinearAlgebra
using FFTW

function cplu!(A)
    m = size(A,1);
    n = size(A,2);
    p = collect(1:m)
    q = collect(1:n)
    L = Matrix{eltype(A)}(I, m, minimum([n, m]))
    U = Matrix{eltype(A)}(I, minimum([n, m]), n)
    AC = Matrix{Float64}(undef, m, n)

    for i = 1:minimum([n, m])
        @views AC[i:end, i:end] .= abs.(A[i:end, i:end])
        @views ind = argmax(AC[i:end, i:end])
        pv1 = ind[1]+i-1
        pv2 = ind[2]+i-1
        A[[i, pv1], :] = A[[pv1, i], :]
        A[:, [i, pv2]] = A[:, [pv2, i]]
        L[[i, pv1], 1:i-1] = L[[pv1, i], 1:i-1]
        U[1:i-1, [i, pv2]] = U[1:i-1, [pv2, i]]
        p[[i, pv1]] = p[[pv1, i]]
        q[[i, pv2]] = q[[pv2, i]]

        @views L[i+1:end, i] = A[i+1:end, i]/A[i, i]
        @views U[i, i:end] = A[i, i:end]
        @views A[i+1:end, i+1:end] .= A[i+1:end, i+1:end] .- (L[i+1:end, i].*transpose(U[i, i+1:end]))
    end
    ip = zeros(Int, m)
    ip[p] = collect(1:m)
    iq = zeros(Int, n)
    iq[q] = collect(1:n)
    return L, U, ip, iq
end


function generate_φ_maliciously(A::MacaulayBivariate{TF}, gap::Float64) where {TF<:Number}
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
        φi = modk(w[mi]) + gap # Make gap required size
        # Add to already placed
        w[j+1:j+k] = [φi+(h - 1) / k for h = 1:k]
        j = j + k
        # Store φ^k
        φ[i] = exp(π * 2im * φi * k) 
    end
    return φ::Vector{Complex{TF}}
end

function CauchyMatrix(A, φ)
    Ahatmat = Matrix{ComplexF64}(A)
    idx = 1
    for i = A.d+1:-1:1
        Ahatmat[:, idx:idx+i-1] = fft(Ahatmat[:, idx:idx+i-1].*transpose(D(i, φ[A.d+2-i]))/sqrt(i), 2)
        idx += i
    end
    idx = 1
    for i = A.Δd+1:-1:1
        for s = 0:A.S-1
            Ahatmat[idx+s:A.S:idx+A.S*i+s-1, :] = ifft(Ahatmat[idx+s:A.S:idx+A.S*i+s-1, :]*sqrt(i), 1)
        end
        idx += i*A.S 
    end
    return Ahatmat
end


reps = 100
sizes = [16, ]
nMethods = 22
results = zeros(nMethods, length(sizes), reps)
for (i, d_Σ) in enumerate(sizes)
    for rep = 1:reps
        println("d_Σ ", d_Σ,", rep ", rep)
        Random.seed!(rep+1)
        p1 = randnBivariatePolynomial(d_Σ)
        p2 = randnBivariatePolynomial(d_Σ)
        p = [p1, p2]
        A = MacaulayBivariate{Float64}(p, 2*d_Σ-2)

        Kmax = size(A,2)-d_Σ^2

        ####### Method 1: theoretical optimum (lower bound)
        svdA = svd!(Matrix(A), full=true)
        # results[1, i, rep] = svdA.S[Kmax+1]/svdA.S[1]

        ####### Method 2: LU on full matrix
        # _, U, _, q = cplu!(Matrix(A))
        # H = [inv(U[1:Kmax,1:Kmax])*U[1:Kmax,Kmax+1:end]; -Matrix{Float64}(I,d_Σ^2,d_Σ^2)]
        # H = H[q, :]
        # results[2, i, rep] = rel_ind_err(A, H)

        for j = 1:5
            ####### Create Cauchy matrix
            if j == 1
                φ = generate_φ_greedy(A)
            elseif j == 2
                φ = generate_φ_randomly(A)
            else
                φ = generate_φ_maliciously(A, 10.0.^(-2*j+2)/pi)
            end

            Ahat = CauchyLike(A, φ)
            Ahatmat = CauchyMatrix(A, φ) # = Matrix(Ahat) but more directly computed

            ####### Method 3: LU on Cauchy matrix
            # LU
            _, U, _, q = cplu!(Ahatmat)
            H = [inv(U[1:Kmax,1:Kmax])*U[1:Kmax,Kmax+1:end]; -Matrix{ComplexF64}(I,d_Σ^2,d_Σ^2)]
            H = H[q, :]
            # Transform back
            for j = 1:A.d+1
                diag_scaling = 1 / sqrt(A.d + 2 - j) * D(A.d + 2 - j, φ[j])
                @views H[(A.off_j[j]+1):A.off_j[j+1], :] = diag_scaling .* fft(H[(A.off_j[j]+1):A.off_j[j+1], :], 1)
            end
            results[4*j-1, i, rep] = rel_ind_err(A, H)

            ####### Method 4: Schur
            # Schur with complete pivoting
            Π_1, Π_2, N = schur_algorithm(Ahat; ABSTOL=0, qr_initialization=qr_initial_macaulay, Kmax=size(A,2)-d_Σ^2, CP=true)
            # Transform back to full
            Hhat = NullBasisCauchyLike{ComplexF64}(N, Π_2)
            H = Matrix(Hhat)
            for j = 1:A.d+1
                diag_scaling = 1 / sqrt(A.d + 2 - j) * D(A.d + 2 - j, φ[j])
                @views H[(A.off_j[j]+1):A.off_j[j+1], :] = diag_scaling .* fft(H[(A.off_j[j]+1):A.off_j[j+1], :], 1)
            end
            # Compute error
            results[4*j, i, rep] = rel_ind_err(A, H)

            ####### Method 5: Schur with cheap pivoting
            # Schur without complete pivoting
            Π_1, Π_2, N = schur_algorithm(Ahat; ABSTOL=0, qr_initialization=qr_initial_macaulay, Kmax=size(A,2)-d_Σ^2, CP=false)
            # Transform back to full
            Hhat = NullBasisCauchyLike{ComplexF64}(N, Π_2)
            H = Matrix(Hhat)
            for j = 1:A.d+1
                diag_scaling = 1 / sqrt(A.d + 2 - j) * D(A.d + 2 - j, φ[j])
                @views H[(A.off_j[j]+1):A.off_j[j+1], :] = diag_scaling .* fft(H[(A.off_j[j]+1):A.off_j[j+1], :], 1)
            end
            # Compute error
            results[4*j+1, i, rep] = rel_ind_err(A, H)

            # Save gap
            results[4*j+2, i, rep] = minimum([minimum(abs.(Ahat.omega.-transpose(Ahat.lambda))), minimum(abs.(Ahat.lambda.-transpose(Ahat.lambda)+Inf*I))])/
                                     maximum([maximum(abs.(Ahat.omega.-transpose(Ahat.lambda))), maximum(abs.(Ahat.lambda.-transpose(Ahat.lambda)))])
        end
        display(results[:, i, rep])
    end
    # file = matopen("figures/stability_mu_nu_gap.mat", "w")
    # write(file, "results", results)
    # close(file)
    display(mean(results./results[:,:,:]; dims=3))
    display(median(results./results[:,:,:]; dims=3))
    display(std(results./results[:,:,:]; dims=3))
end