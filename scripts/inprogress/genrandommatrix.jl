using LinearAlgebra
using Random
using Distributions
using SparseArrays
#using Statistics
#using DataFrames
#using CairoMakie

# Generate a random m x n matrix with rank r
function gen_matrix(m, n, r; noisy=false, sparse=false, noisevar=1e-2, sparsity=1/3, seed=1234)
    rng = MersenneTwister(seed)
    if sparse
        B = sprandn(rng, m, r, sparsity)
        C = sprandn(rng, r, n, sparsity)
    else
        B = randn(rng, m, r)
        C = randn(rng, r, n)
    end
    A = B * C
    if noisy
        A += noisevar * randn(rng, m, n)
    end
    return A
end

# Generate a random n x n symmetric positive semidefinite matrix of rank r
function gen_symm_matrix(n, r; noisy=false, sparse=false, noisevar=1e-2, sparsity=1/3, seed=1234)
    rng = MersenneTwister(seed)
    if sparse
        B = sprandn(rng, n, r, sparsity)
    else
        B = randn(rng, n, r)
    end
    A = B * B'
    if noisy
        A += noisevar * randn(rng, m, n)
    end
    return A
end

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

# Find each agent's best guess for the target entry, before communication,
# by applying matrix reconstruction heuristics to their observations
using LinearAlgebra

#function single_agent_guess(agentobs, targetidx)
#     m, n = size(agentobs)
#     # Create a mask with a 1 at the target index
#     mask = zeros(m, n)
#     i,j = targetidx
#     mask[i,j] = 1
#     # Apply the mask to the agent's observations
#     masked = agentobs .* mask
#     # Find the best rank-1 approximation to the masked observations
#     u, s, vt = svd(masked)
#     return u[:,1] * s[1] * vt[1,:]
# end

using GLM
using DataFrames

# For each pair of X, y: sample a training set and a test set. 
# Fit a linear model on the training set and evaluate on the test set. 
# Report model R-squared and average test error.
# Compare with the null model that predicts the mean of y.
function fit_linear_model(X, y)
    n = size(X, 1)
    ntrain = Int(round(n * 0.8))
    ntest = n - ntrain
    permuted = randperm(n)
    trainidx = permuted[1:ntrain]
    testidx = permuted[ntrain+1:end]
    Xtrain = X[trainidx, :]
    ytrain = y[trainidx]
    Xtest = X[testidx, :]
    ytest = y[testidx]
    k = size(X, 2)
    # Create data 
    train_data = DataFrame(y=ytrain)
    for i in 1:k
        train_data[!, Symbol("X$i")] = Xtrain[:, i]
    end
    test_data = DataFrame(y=ytest)
    for i in 1:k
        test_data[!, Symbol("X$i")] = Xtest[:, i]
    end
    formula = Term(:y) ~ sum(Term(Symbol("X$i")) for i in 1:k)
    model = lm(formula, train_data)
    rsquared = r2(model)
    ypred = predict(model, test_data)
    testerror = sum((ytest - ypred).^2) / ntest
    return rsquared, testerror, model
end