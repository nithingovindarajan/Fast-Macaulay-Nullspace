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
d_Σs = [2, 4, 8, 16, 32]
nMethods = 7
results = zeros(nMethods, length(d_Σs), reps)
for (i, d_Σ) in enumerate(d_Σs)
    for rep = 1:reps
        println("d_Σ ", d_Σ,", rep ", rep)
        Random.seed!(rep+1)
        p1 = randnBivariatePolynomial(d_Σ)
        p2 = randnBivariatePolynomial(d_Σ)
        A = MacaulayBivariate{Float64}([p1, p2], 2*d_Σ-2)

        Kmax = size(A,2)-d_Σ^2

        ####### Method 1: theoretical optimum (lower bound)
        svdA = svd!(Matrix(A), full=true)
        # results[1, i, rep] = svdA.S[Kmax+1]/svdA.S[1]

        ####### Method 2: SVD on full matrix
        H = svdA.V[:, Kmax+1:end]
        results[2, i, rep] = rel_ind_err(A, H)

        ####### Method 3: SVD on Cauchy matrix
        # Transform to Cauchy
        φ = generate_φ_greedy(A)
        Ahat = CauchyLike(A, φ)
        Ahatmat = CauchyMatrix(A, φ) # = Matrix(Ahat) but more directly computed
        # SVD
        svdA = svd!(copy(Ahatmat), full=true)
        H = svdA.V[:, Kmax+1:end]
        # Transform back
        for j = 1:A.d+1
            diag_scaling = 1 / sqrt(A.d + 2 - j) * D(A.d + 2 - j, φ[j])
            @views H[(A.off_j[j]+1):A.off_j[j+1], :] = diag_scaling .* fft(H[(A.off_j[j]+1):A.off_j[j+1], :], 1)
        end
        results[3, i, rep] = rel_ind_err(A, H)

        ####### Method 4: LU on full matrix
        _, U, _, q = cplu!(Matrix(A))
        H = [inv(U[1:Kmax,1:Kmax])*U[1:Kmax,Kmax+1:end]; -Matrix{Float64}(I,d_Σ^2,d_Σ^2)]
        H = H[q, :]
        results[4, i, rep] = rel_ind_err(A, H)

        ####### Method 5: LU on Cauchy matrix
        # LU
        _, U, _, q = cplu!(Ahatmat)
        H = [inv(U[1:Kmax,1:Kmax])*U[1:Kmax,Kmax+1:end]; -Matrix{ComplexF64}(I,d_Σ^2,d_Σ^2)]
        H = H[q, :]
        # Transform back
        for j = 1:A.d+1
            diag_scaling = 1 / sqrt(A.d + 2 - j) * D(A.d + 2 - j, φ[j])
            @views H[(A.off_j[j]+1):A.off_j[j+1], :] = diag_scaling .* fft(H[(A.off_j[j]+1):A.off_j[j+1], :], 1)
        end
        results[5, i, rep] = rel_ind_err(A, H)

        #######  Method 6: Schur
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
        results[6, i, rep] = rel_ind_err(A, H)

        ####### Method 7: Schur with cheap pivoting
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
        results[7, i, rep] = rel_ind_err(A, H)
        display(results[:, i, rep])
    end
    # file = matopen("figures/methods_dS.mat", "w")
    # write(file, "results", results)
    # close(file)
    display(mean(results; dims=3))
    display(median(results; dims=3))
    display(std(results; dims=3))
end