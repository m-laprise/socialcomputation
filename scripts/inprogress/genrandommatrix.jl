using LinearAlgebra
using Random
using Distributions
using SparseArrays
#using Statistics


function gen_matrix(m, n, r; 
                    seed::Int = 0,
                    noisy::Bool = false, 
                    noisevar::Float64 = 1e-2, 
                    sparse::Bool = false, 
                    sparsity::Float64 = 1/3, 
                    spatial::Bool = false,
                    symmetric::Bool = false,
                    dtype = Float64)
    seed = seed == 0 ? Int(round(time())) : seed
    rng = MersenneTwister(seed)

    if sparse
        B = randn(rng, dtype, m, r) .* (rand(rng, dtype, m, r) .< sparsity)
        C = randn(rng, dtype, r, n) .* (rand(rng, dtype, r, n) .< sparsity)
    else
        B = randn(rng, dtype, m, r)
        C = randn(rng, dtype, r, n)
    end
    if spatial
        B = cumsum(B, dims=1)
        C = cumsum(C, dims=2)
    end
    # Normalize factors 
    # (Otherwise variance of the entries of A will be proportional to r)
    B = B ./ sqrt(sqrt(r))
    C = C ./ sqrt(sqrt(r))
    if symmetric
        A = B * B'
    else
        A = B * C
    end
    if noisy
        A .+= (noisevar * randn(rng, dtype, m, n))
    end
    return A
end

#= TEST OF NORMALIZATION FACTOR
ms = [20, 200, 800]
for m in ms
    rs = [1, Int(round(m/10)), Int(round(m/5)), Int(round(m/2)), m]
    for r in rs
        maxs1 = Float32[]
        maxs4 = Float32[]
        vars1 = Float32[]
        vars4 = Float32[] 
        for _ in 1:500
            push!(maxs1, maximum(abs.(randn(Float32, m,r) * randn(Float32, r,m))))
            push!(maxs4, maximum(abs.((randn(Float32, m,r) ./ sqrt(sqrt(r))) * (randn(Float32, r,m) ./ sqrt(sqrt(r))) ) ))
        end
        for _ in 1:500
            push!(vars1, var(randn(Float32, m,r) * randn(Float32, r,m)))
            push!(vars4, var((randn(Float32, m,r) ./ sqrt(sqrt(r))) * (randn(Float32, r,m) ./ sqrt(sqrt(r)))  ))
        end
        println("Maxima and variances for m=$m, r=$r")
        println(mean(maxs1), " ", mean(maxs4))
        println(mean(vars1), " ", mean(vars4))
        println("---------------------------------")
    end
    println("=====================================")
end =#

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
    B = gen_matrix(n, n, r; symmetric=true)
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
    Σ = gen_matrix(n, n, r; symmetric=true)
    Y = gen_data_from_vcov(Σ, d)
    @assert size(Y) == (d, n)
    Σhat = cov(Y)
    @assert isapprox(Σ, Σhat, atol=10)

    # Full rank case
    Σ = gen_matrix(n, n, n; symmetric=true)
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

function generate_matrix_set(m::Int,
                             n::Int,
                             dataset_size::Int,
                             rankset::Vector{Int};
                             sparsity=1.0,
                             dtype=Float32,
                             seed=0)
    seed = seed == 0 ? Int(round(time())) : seed
    sparse = sparsity != 1.0
    rng = MersenneTwister(seed)
    ranks = Int32.(rand(rng, rankset, dataset_size))
    X = Array{dtype, 3}(undef, m, n, dataset_size)
    for i in 1:dataset_size
        X[:, :, i] = gen_matrix(m, n, ranks[i], seed=seed+i, sparse=sparse, sparsity=sparsity, dtype=dtype)
    end
    return X, ranks
end

function trace_measurements(X::AbstractArray{Float32, 3}, O::AbstractArray{Float32, 4})
    m, n, dataset_size = size(X)
    traces_nb = size(O, 3)
    Xhat = Array{Float32, 2}(undef, traces_nb, dataset_size)
    for i in 1:dataset_size
        X_i = view(X, :, :, i)
        for j in 1:traces_nb
            O_ij = view(O, :, :, j, i)
            Xhat[j, i] = trace_product(X_i, O_ij)
        end
    end
    return Xhat
end

function trace_product(X::AbstractMatrix, O::AbstractMatrix)
    return sum(diag(X * O))
end
