#=
This script defines function to generate random matrices with specific properties, including 
both ground truth matrices and measurements, as well as other utilities for data generation.
=#

using LinearAlgebra
using Random
using Distributions
using SparseArrays

"""
    gen_matrix(m, n, r; seed=0, noisy=false, sparse=false, noisevar=1e-2, sparsity=1/3, 
    spatial=false, symmetric=false, dtype=Float64)

Generate a random m x n matrix A with rank r. The matrix is generated as a product of two 
random matrices B and C of sizes m x r and r x n, respectively. The entries of B and C are 
drawn from a normal distribution with mean 0 and variance sqrt(1/r), so that the variance of
the entries of the product matrix A is independent of the rank r and equal to 1. 

The seed parameter can be used to set the seed of the random number generator; if no seed is
provided, the current time is used as a seed.

The matrix A can be made noisy by adding Gaussian noise with variance `noisevar`. 
The matrix A can be made sparse by setting the `sparse` flag to true. In this case, the entries 
of B and C are multiplied by a binary mask with sparsity `sparsity`. 
The matrix A can be made spatial by cumulatively summing the entries of B and C along the rows 
and columns, respectively. 
The matrix A can be made symmetric by setting the `symmetric` flag to true. In this case, the 
matrix A is computed as A = B * B'. 

The function returns the generated matrix A.

# Arguments
- `m::Int`: number of rows of the matrix
- `n::Int`: number of columns of the matrix
- `r::Int`: rank of the matrix

# Optional Keyword Arguments
- `seed::Int=0`: seed of the random number generator. If 0, the current time is used as a seed.
Default is 0.
- `noisy::Bool=false`: whether to add Gaussian noise to the matrix. Default is false.
- `noisevar::Float64=1e-2`: variance of the Gaussian noise, if any. Default is 1e-2.
- `sparse::Bool=false`: whether to generate a sparse matrix. Default is false.
- `sparsity::Float64=1/3`: sparsity of the matrix, if sparse; corresponds to the proportion 
of zero entries. Default is 1/3.
- `spatial::Bool=false`: whether to generate a smoother matrix by using cumulative sums (random 
walk) instead of random entries. Default is false.
- `symmetric::Bool=false`: whether to generate a symmetric matrix. Default is false.
- `dtype=Float64`: data type of the matrix. Default is Float64.

# Examples
```julia
A = gen_matrix(10, 5, 3)
B = gen_matrix(5, 5, 3; symmetric=true)
C = gen_matrix(10, 5, 3, noisy=true)
D = gen_matrix(10, 5, 3, seed=1234, sparse=true, sparsity=0.2)
```
"""
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

"""
    gen_matrix_singvals(m, n, r, singvals)

Generate a random m x n matrix A with rank r and given singular values. The matrix is generated
as a product of two random matrices U and Vt of sizes m x r and r x n, respectively, and a diagonal
matrix S of size r x r with the given singular values. The matrix A is computed as A = U * S * Vt.
"""
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

"""
    gen_data_from_vcov(Σ, d)

Generate d samples from a multivariate normal distribution with covariance matrix Σ. The function
first checks if the input matrix Σ is symmetric or positive definite. If Σ is positive definite,
the function generates d samples by drawing a random vector x from a standard normal distribution
and computing y = L * x, where L is the Cholesky factor of Σ. If Σ is symmetric but not positive
definite, the function generates d samples by drawing a random vector x from a standard normal
distribution and computing y = U * S * x, where U and S are the singular vectors and values of Σ,
respectively. If Σ is neither symmetric nor positive definite, the function throws an error.

The function returns the samples as a d x n matrix, where n is the size of Σ. 

# Arguments
- `Σ::AbstractMatrix`: covariance matrix
- `d::Int`: number of samples to generate

# Examples
```julia
Σ = [1.0 0.5; 0.5 1.0]
Y = gen_data_from_vcov(Σ, 1000)
```
"""
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

"""
    zscoremat(A; by="columns")

Standardize the matrix A by subtracting the mean and dividing by the standard deviation, either
by columns, by rows, or across all elements. The `by` parameter can be set to "columns", "rows", 
or "all". 
"""
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

"""
    submatrix(A::AbstractArray, blocksize::Int = 0; 
              seed::Int = Int(round(time())), type::String="partition")

Generate a collection of submatrices of A, either by partitioning the matrix into blocks of 
a given size or by randomly sampling blocks of a given size. The function returns a list of
submatrices.

The `blocksize` parameter sets the size of the blocks. If `blocksize` is 0, the function uses 
twice the rank of A as the block size. If `blocksizes` is larger than half of the smallest 
dimension of A, the function reduces it to this value.
"""
function submatrix(A::AbstractArray, blocksize::Int = 0; 
                   seed::Int = Int(round(time())), type::String="partition")
    m, n = size(A)
    if m < n
        A = A'
        m, n = n, m
    end
    if blocksize == 0
        blocksize = 2 * rank(A)
    end
    blocksize = min(blocksize, Int(round(m/2)))
    blocks = []
    if type == "partition"
        # Deterministically partition the matrix in blocks of blocksize x blocksize
        for i in 1:blocksize:m
            for j in 1:blocksize:n
                push!(blocks, A[i:min(i+blocksize-1, m), j:min(j+blocksize-1, n)])
            end
        end
    elseif type == "random"
        k = m * n / (blocksize * blocksize)
        # Randomly sample a block of size blocksize x blocksize, k times
        rng = MersenneTwister(seed)
        for _ in 1:k
            i = rand(rng, 1:m-blocksize+1)
            j = rand(rng, 1:n-blocksize+1)
            push!(blocks, A[i:min(i+blocksize-1, m), j:min(j+blocksize-1, n)])
        end
    else 
        error("Invalid type")
    end
    return blocks
end

"""
    sensingmasks(m::Int, n::Int; k::Int = 0, seed::Int = Int(round(time())))

Generate the collection of all possible sensing masks for the matrix A, shuffle them randomly, and
return the first k masks. The masks are stored sparsely as a list of (i, j) tuples where i and j
are the row and column indices of the sensed entry.
"""
function sensingmasks(m::Int, n::Int; k::Int = 0, seed::Int = Int(round(time())))
    if k == 0
        k = m * n
    end
    maskij = [(i, j) for i in 1:m for j in 1:n]
    rng = MersenneTwister(seed)
    shuffle!(rng, maskij)
    return maskij[1:k]
end


"""
    generate_matrix_set(m::Int, n::Int, dataset_size::Int, rankset::Vector{Int};
                        sparsity=1.0, dtype=Float32, seed=0)
    
Generate a dataset of matrices of size m x n with varying ranks. The dataset consists of
`dataset_size` matrices, each with a rank drawn with equal probability from the `rankset` 
vector of possible ranks. The matrices are generated using the `gen_matrix` function with 
the given parameters. The function returns the dataset as a 3D array of size m x n x 
dataset_size and the corresponding ranks as a vector.

# Arguments
- `m::Int`: number of rows of the matrices
- `n::Int`: number of columns of the matrices
- `dataset_size::Int`: number of matrices in the dataset
- `rankset::Vector{Int}`: vector of possible ranks for the matrices

# Optional Keyword Arguments
- `sparsity::Float64=1.0`: sparsity of the matrices. If less than 1.0, the matrices are
generated as sparse matrices with the given sparsity. If 1.0, the matrices are dense.
Default is 1.0.
- `dtype::DataType=Float32`: data type of the matrices. Default is Float32.
- `seed::Int=0`: seed of the random number generator. If 0, the current time is used as a 
seed. Default is 0.

# Examples
```julia
X, ranks = generate_matrix_set(10, 5, 100, [1, 2, 3, 4, 5])
```
"""
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

"""
    trace_measurements(X::AbstractArray{Float32, 3}, O::AbstractArray{Float32, 4})

Compute the trace of the product of each matrix in X with each projection matrix in O. The
dimensions of X should be m x n x dataset_size, and the dimensions of O should be m x n x
traces_nb x dataset_size. The function returns a matrix of size traces_nb x dataset_size with
the trace of each product of X and O for each matrix X in the dataset.

# Arguments
- `X::AbstractArray{Float32, 3}`: dataset of matrices
- `O::AbstractArray{Float32, 4}`: dataset of dense random projection matrices
"""
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

trace_product(X::AbstractMatrix, O::AbstractMatrix) = sum(diag(X * O))

"""
    train_val_test_split(X::AbstractArray,
                         train_prop::Float64, val_prop::Float64, test_prop::Float64)

Split any array into training, validation, and test sets. The function takes the input data
and the proportions to allocate to the training, validation, and test sets. The function returns 
the split data as separate matrices or vectors.
"""
function train_val_test_split(X::AbstractArray, 
                              train_prop::Float64, val_prop::Float64, test_prop::Float64)
    @assert train_prop + val_prop + test_prop == 1.0
    dimsX = length(size(X))
    dataset_size = size(X, dimsX)
    train_nb = Int(train_prop * dataset_size)
    val_nb = Int(val_prop * dataset_size)
    train_idxs = 1:train_nb
    val_idxs = train_nb+1:train_nb+val_nb
    test_idxs = train_nb+val_nb+1:dataset_size
    if X isa Vector
        Xtrain, Xval, Xtest = X[train_idxs], X[val_idxs], X[test_idxs]
    elseif dimsX == 2
        Xtrain, Xval, Xtest = X[:,train_idxs], X[:,val_idxs], X[:,test_idxs]
    elseif dimsX == 3
        Xtrain, Xval, Xtest = X[:,:,train_idxs], X[:,:,val_idxs], X[:,:,test_idxs]
    else
        error("Invalid number of dimensions for X: $dimsX")
    end
    @assert size(Xtrain, dimsX) == train_nb
    @assert size(Xval, dimsX) == val_nb
    @assert size(Xtest, dimsX) == dataset_size - train_nb - val_nb
    return Xtrain, Xval, Xtest
end
