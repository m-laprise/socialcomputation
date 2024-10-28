using LinearAlgebra
using Random
using Distributions
using SparseArrays
#using Statistics
#using DataFrames

# Generate a random m x n matrix with rank r
function gen_matrix(m, n, r; 
                    seed::Int = 0,
                    noisy::Bool = false, 
                    sparse::Bool = false, 
                    noisevar::Float64 = 1e-2, 
                    sparsity::Float64 = 1/3, 
                    spatial::Bool = false,
                    symmetric::Bool = false)
    if seed == 0
        seed = Int(round(time()))
    end
    rng = MersenneTwister(seed)
    if sparse
        B = sprandn(rng, m, r, sparsity)
        C = sprandn(rng, r, n, sparsity)
    else
        B = randn(rng, m, r)
        C = randn(rng, r, n)
    end
    if spatial
        B = cumsum(B, dims=1)
        C = cumsum(C, dims=2)
    end
    if symmetric
        A = B * B'
    else
        A = B * C
    end
    if noisy
        A += noisevar * randn(rng, m, n)
    end
    return A
end

@benchmark gen_matrix(100, 100, 10; seed=1234, sparse=false, sparsity=1/3)

#A = (A .* 0.1) + diagm(0 => diag(A))

# Generate a random m x n matrix with rank r and given singular values
function gen_matrix_singvals(m, n, r, singvals)
    A = gen_matrix(m, n, r)
    U, S, Vt = svd(A)
    # Pad singular values with zeros
    if length(singvals) < n
        singvals = [singvals; zeros(n - length(singvals))]
    end
    S = diagm(0 => singvals)
    A = U * S * Vt
    return A
end

# Unit test of matrix generation functions
function test_matrix_gen()
    m = 10
    n = 5
    r = 3
    singvals = [1.0, 2.0, 3.0]
    A = gen_matrix(m, n, r)
    B = gen_symm_matrix(n, r)
    C = gen_matrix_singvals(m, n, r, singvals)
    println("A = ", A)
    println("B = ", B)
    println("D = ", C)
    # Check the dimensions
    @assert size(A) == (m, n)
    @assert size(B) == (n, n)
    @assert size(C) == (m, n)
    # Check the rank
    @assert rank(A) == r
    @assert rank(B) == r
    @assert rank(C) == r
    # Check the first r singular values
    U, S, Vt = svd(C)
    @assert S[1:r] ≈ sort(singvals, rev=true)
    # Check the symmetry
    @assert B ≈ B'
    # Check the positive semidefiniteness
    @assert all(eigvals(B) .≥ -1e-10)
end

test_matrix_gen()

function gen_data_from_vcov(Σ, d)
    rng = MersenneTwister(1234)
    n = size(Σ, 1)
    Y = zeros(n, d)
    
    if isposdef(Σ)
        L = cholesky(Σ).L
        for i in 1:d
            x = randn(rng, n)
            y = L * x
            @assert length(y) == n
            Y[:, i] = y
        end
    elseif issymmetric(Σ)
        U, S, Ut = svd(Σ)
        S = diagm(sqrt.(S))
        for i in 1:d
            x = randn(rng, n)
            y = U * S * x
            @assert length(y) == n
            Y[:, i] = y
        end
    else
        error("The input matrix is not symmetric or positive definite.")
    end
    return Y'
end

function test_gen_data_from_vcov()
    n = 5
    r = 3
    d = 1000

    # Low rank case
    Σ = gen_symm_matrix(n, r)
    Y = gen_data_from_vcov(Σ, d)
    @assert size(Y) == (d, n)
    Σhat = cov(Y)
    @assert isapprox(Σ, Σhat, atol=10)

    # Full rank case
    Σ = gen_symm_matrix(n, n)
    Y = gen_data_from_vcov(Σ, d)
    @assert size(Y) == (d, n)
    Σhat = cov(Y)
    @assert isapprox(Σ, Σhat, atol=10)
end

test_gen_data_from_vcov()

function zscoremat(A; by="columns")
    if by == "columns"
        μ = mean(A, dims=1)
        σ = std(A, dims=1)
        A = (A .- μ) ./ σ
    elseif by == "rows"
        μ = mean(A, dims=2)
        σ = std(A, dims=2)
        A = (A .- μ) ./ σ
    elseif by == "all"
        μ = mean(A)
        σ = std(A)
        A = (A .- μ) ./ σ
    else
        error("Invalid option for 'by'. Choose from 'columns', 'rows', or 'all'.")
    end
    return A
end

function test_zscore()
    A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    B = zscoremat(A, by="columns")
    C = zscoremat(A, by="rows")
    D = zscoremat(A, by="all")
    @assert isapprox(mean(B, dims=1), zeros(1, 3))
    @assert isapprox(std(B, dims=1), ones(1, 3))
    @assert isapprox(mean(C, dims=2), zeros(3, 1))
    @assert isapprox(std(C, dims=2), ones(3, 1))
    @assert isapprox(mean(D), 0.0)
    @assert isapprox(std(D), 1.0)
end

test_zscore()

function submatrix(A, minval=0, maxval=0; seed=1234, type="partition")
    m, n = size(A)
    if m < n
        A = A'
        m, n = n, m
    end
    if minval == 0
        minval = 2 * rank(A)
    end
    if maxval == 0
        maxval = m/2
    end
    blocks = []
    if type == "partition"
        # Deterministically partition the matrix in blocks of min x min
        # and return all blocks
        for i in 1:minval:m
            for j in 1:minval:n
                push!(blocks, A[i:min(i+minval-1, m), j:min(j+minval-1, n)])
            end
        end
    elseif type == "random"
        k = m * n / (minval * minval)
        # Randomly sample a block of size min x min, k times
        rng = MersenneTwister(seed)
        for _ in 1:k
            i = rand(rng, 1:m-minval+1)
            j = rand(rng, 1:n-minval+1)
            push!(blocks, A[i:min(i+minval-1, m), j:min(j+minval-1, n)])
        end
    else 
        error("Invalid type")
    end
    return blocks
end

# Generate collection of all possible sensing masks (sensing one entry only) for a given matrix size
# Randomly permute them and return the first k masks, stored sparsely as a list of (i, j) tuples
function sensingmasks(A; k=0, seed=1234)
    m, n = size(A)
    if k == 0
        k = m * n
    end
    maskij = [(i, j) for i in 1:m for j in 1:n]

    rng = MersenneTwister(seed)
    shuffle!(rng, maskij)
    return maskij[1:k]
end

function apply_mask!(B, A, maskij; addoutcome=true, outcomecol=1)
    i,j = maskij
    B[i,j] += A[i,j]
    if addoutcome
        if i != 1
            B[i, outcomecol] = A[i, outcomecol]
        end
    end
    return B
end

function svd_recon(A, k)
    U, S, Vt = svd(A)
    Ahat = U[:, 1:k] * diagm(0 => S[1:k]) * Vt[:, 1:k]'
    return Ahat
end

function trace_product(X::AbstractMatrix, O::AbstractMatrix)
    return sum(diag(X * O))
end

function trace_product(X::AbstractMatrix, O::Array)
    nb_measurements = size(O, 3)
    result = Vector{Float64}(undef, nb_measurements)
    temp_matrix = similar(X, size(X, 1), size(O, 2))
    diag_temp = Vector{Float64}(undef, min(size(X, 1), size(O, 2)))
    
    @inbounds for i in 1:nb_measurements
        @views mul!(temp_matrix, X, O[:, :, i])
        @inbounds for j in 1:length(diag_temp)
            diag_temp[j] = temp_matrix[j, j]
        end
        result[i] = sum(diag_temp)
    end
    return result
end

function get_data_densematrix(dataset_size::Int, m::Int, n::Int; 
        seed::Int, 
        measurement_type::String = "mask",
        mask_prob::Float64 = 0.33, 
        traces_nb::Int = 50,
        traces_projsize::Int = 25,
        include_X::Bool = false, 
        rank_threshold::Float64 = 0.1)

    cutoff = Int(round(min(m, n) * rank_threshold))
    maxrank = Int(round(min(m, n) / 2))

    rng = MersenneTwister(seed)
    # Generate matrix ranks
    indexablecollection = vcat(1:min(10,cutoff-2), max(maxrank-10, cutoff+2):maxrank)
    r = rand(rng, indexablecollection, dataset_size)
    n_seen = round(Int, (1 - mask_prob) * m * n)
    r = min.(r, fill(n_seen ÷ 2, dataset_size))

    # Preallocate the array for low-rank matrices
    X = Array{Float64, 3}(undef, dataset_size, m, n)
    # Generate (dataset_size) low-rank matrices from the rank vector
    for i in 1:dataset_size
        X[i, :, :] .= gen_matrix(m, n, r[i], seed=seed+i)
    end

    # Generate labels with shape (dataset_size, 1)
    Y = reshape(r .< cutoff, dataset_size, 1)

    if measurement_type == "mask"
        # Generate measurements of each X by masking some of the entries; dataset_size x m x n
        masks = rand(rng, Bool, dataset_size, m, n) .< (1 - mask_prob)
        Xhat = X .* masks
    elseif measurement_type == "trace"
        # Preallocate the array for trace measurements
        Xhat = Array{Float64, 2}(undef, dataset_size, traces_nb)
        # Generate measurements of each X by taking Trace(X * O_i) for (nb_traces) random matrices O_i
        O = randn(rng, dataset_size, n, traces_projsize, traces_nb)
        @inbounds for i in 1:dataset_size
            #Xhat[i, :] .= trace_product(X[i, :, :], O[i, :, :, :])
            X_i = view(X, i, :, :)
            for j in 1:traces_nb
                O_ij = view(O, i, :, :, j)
                Xhat[i, j] = trace_product(X_i, O_ij)
            end
        end
    else
        error("Measurement type not recognized. Choose from 'mask' or 'trace'.")
    end
    if include_X
        return X, Xhat, Y, r
    else
        return Xhat, Y, r
    end
end


function generate_matrix_set(dataset_size::Int,
                             m::Int,
                             n::Int,
                             rankset::Vector{Int},
                             sparsity=1.0,
                             seed=0)
    if seed == 0
        seed = Int(round(time()))
    end
    if sparsity == 1.0
        sparse = false
    else
        sparse = true
    end
    rng = MersenneTwister(seed)
    X = Array{Float64, 3}(undef, m, n, dataset_size)
    ranks = rand(rng, rankset, dataset_size)
    for i in 1:dataset_size
        X[:, :, i] .= gen_matrix(m, n, ranks[i], seed=seed+i, sparse=sparse, sparsity=sparsity)
    end
end

using BenchmarkTools
@benchmark generate_matrix_set(500, 100, 100, [2,4,6,8], 1.0, 1234)