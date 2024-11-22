#= Julia implementation of various state-of-the-art matrix completion algorithms:
1) Nuclear norm minimization (NNM), using interior-point methods.
2) Scaled Alternating Steepest Descent (ScaledASD).
See Jared Tanner and Ke Wei. Low rank matrix completion by alternating 
steepest descent methods. Applied and Computational Harmonic Analysis, 40(2):417-429, 2016.
3) Iteratively Reweighted Least Squares (IRLS). See 
Massimo Fornasier, Holger Rauhut, and Rachel Ward. 
Low-rank Matrix Recovery via Iteratively Reweighted Least Squares Minimization,
SIAM Journal on Optimization 2011 21:4, 1614-1640.
=#
using LinearAlgebra
using SparseArrays
using JuMP, SCS

#=
ENV["PYTHON"] = "/Users/mlaprise/.pyenv/versions/socialcomputation/bin/python"
using PyCall
py"""
import cvxpy as cp
import numpy as np
def nuclear_norm_minimization(A, mask):
    X = cp.Variable(A.shape)
    objective = cp.Minimize(cp.norm(X, "nuc"))
    constraints = [X[mask] == A[mask]]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return X.value
"""

function PYnnm(A::AbstractArray, mask::AbstractArray)
    return py"nuclear_norm_minimization"(A, mask)
end
=#
PYnnm(A::AbstractArray, mask::AbstractArray) = A

import MathOptInterface as MOI
function SCSnnm(A, mask; verbose=false)
    model = Model(SCS.Optimizer)
    MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), 1e-5)
    MOI.set(model, MOI.RawOptimizerAttribute("eps_rel"), 1e-5)
    n = size(A, 1)
    @variable(model, X[1:n, 1:n])
    @constraint(model, X[mask] .== A[mask])
    @variable(model, t)
    @constraint(model, [t; vec(X)] in MOI.NormNuclearCone(n, n))
    @objective(model, Min, t)
    if !verbose
        set_silent(model)
    end
    optimize!(model)
    return value.(X)
end

function sparse2idx(A::AbstractArray)
    A = sparse(A)
    I_idx, J_idx, V = findnz(A)
    return I_idx, J_idx, V
end

function partXY(Xt::AbstractArray, Y::AbstractArray, I_idx::Vector{Int}, J_idx::Vector{Int}, p::Int)
    result = zeros(eltype(Xt), p)
    for k in 1:p
        result[k] = dot(Xt[:, I_idx[k]], Y[:, J_idx[k]])
    end
    return result
end

function trace(A::AbstractArray)
    return sum(diag(A))
end

function ScaledASD(
    m::Int, # Number of rows
    n::Int, # Number of columns
    r::Int, # Rank of the factorization
    I_idx::Vector{Int}, # I indices of the observed entries
    J_idx::Vector{Int}, # J indices of the observed entries
    data::Vector{Float64}, # Observed entries
    opts; # Options: rel_res_tol (Float64), maxit (Int), verbosity (Int or Bool), rel_res_change_tol (Float64)
    soln_only::Bool = false, # Return only the reconstructed matrix
    start = nothing, # Optional starting point: initial factor matrices L (m x r) and R (r x n)
)
    @assert length(data) == length(I_idx)
    @assert length(data) == length(J_idx)
    @assert r < min(m, n)
    p = length(data) # Number of observed entries
    @assert p > 0 && p < m * n

    reltol = opts[:rel_res_tol] # Relative tolerance for stopping criterion
    maxiter = opts[:maxit] # Maximum number of iterations
    @assert maxiter > 0
    verbosity = opts[:verbosity] 
    rate_limit = 1 - opts[:rel_res_change_tol] # Relative change tolerance for stopping criterion
    relres = reltol * norm(data) # Relative tolerance for stopping criterion (scaled by the norm of the data)

    # Create a sparse matrix
    diff_matrix = sparse(I_idx, J_idx, data, m, n) # Sparse matrix of the observed entries (no prediction yet, so the difference is the data itself)

    if !isnothing(start) # Check if an initial point is provided
        X = start[:L] 
        Y = start[:R]
        if verbosity
            println("initial point provided")
        end
    else # If not, initialize with SVD
        U, S, V = svd(Matrix(diff_matrix))
        X = U[:, 1:r] * sqrt(Diagonal(S[1:r]))
        Y = sqrt(Diagonal(S[1:r])) * V[:, 1:r]'
        if verbosity
            println("no initial point provided; initializing with SVD")
        end
    end

    identity = I(r) # Identity matrix of size r
    Xt = X' 

    known_diffs = data - partXY(Xt, Y, I_idx, J_idx, p) # Diff between observed and predicted entries
    res = norm(known_diffs) # Initial residual (Norm of the difference)

    iter = 1 # Iteration counter
    itres = zeros(maxiter+1) # Array to store the residuals at each iteration
    itres[iter] = res # Store the initial residual value
    Factors = Vector{Any}(undef, maxiter+1) # Array to store the factor matrices at each iteration

    conv_rate = 0

    # Iteration
    while iter <= maxiter && res >= relres && conv_rate <= rate_limit
        iter += 1 # Increment the iteration counter
        # Gradient for X
        @assert length(diff_matrix.nzval) == p
        @assert length(known_diffs) == p
        # Update the sparse matrix for gradient computation
        diff_matrix.nzval .= known_diffs
        grad_X = diff_matrix * Y'  # Gradient of the loss function wrt X

        # Scaled gradient and stepsize for X
        scale = (Y * Y') \ identity # Scale matrix
        dx = grad_X * scale # Scaled gradient
        delta_XY = partXY(dx', Y, I_idx, J_idx, p)  
        tx = trace(dx' * grad_X) / norm(delta_XY)^2 # Step size

        # Update X
        X += tx * dx  
        Xt = X'
        known_diffs -= tx * delta_XY 

        # Gradient for Y
        diff_matrix.nzval .= known_diffs
        grad_Y = Xt * diff_matrix

        # Scaled gradient and stepsize for Y
        scale = (Xt * X) \ identity
        dy = scale * grad_Y
        delta_XY = partXY(Xt, dy, I_idx, J_idx, p)
        ty = trace(dy * grad_Y') / norm(delta_XY)^2

        # Update Y
        Y += ty * dy
        known_diffs -= ty * delta_XY
        res = norm(known_diffs)

        itres[iter] = res # Store the residual at the current iteration
        conv_rate = (itres[iter] / itres[max(1, iter - 15)])^(1 / min(15, iter - 1))

        Factors[iter] = (X, Y) # Store the factor matrices at the current iteration
        if verbosity
            println("Iteration $iter: residual = $(round(res, digits=2)), "*
                    "conv_rate = $(round(conv_rate, digits=2))")
        end
    end
    if iter < 2
        # If no iteration is performed, throw an error
        error("No iterations performed; initial residual = $(round(res, digits=2))")
    end
    if iter == maxiter
        println("Warning: maximum number of iterations reached. Algorithm may not have converged.")
    end

    Factors = Factors[1:iter] # Factor matrices at each iteration up to the current iteration
    Out = Dict(
        :itrelres => itres[1:iter] / norm(data),  # Relative residuals at each iteration
        :iter => iter,  # Number of iterations
        :reschg => abs(1 - conv_rate)  # Change in the residual
    )
    if soln_only
        final_X, final_Y = Factors[end]
        soln = final_X * final_Y
        return soln, Out
    else
        return Factors, Out
    end
end

function scaled_asd_performance(X_dataset, Y_dataset, I_idx, J_idx, opts, r)
    dataset_size = size(Y_dataset, 3)
    mse_losses = zeros(Float32, dataset_size)
    spectral_dists = zeros(Float32, dataset_size)
    m, n = size(Y_dataset, 1), size(Y_dataset, 2)
    for i in 1:dataset_size
        #X = reshape(X_dataset[:,i], m, n)
        #Y = reshape(Y_dataset[:,i], m, n)
        #r = rank(Y)
        #I_idx, J_idx, knownentries = sparse2idx(Float64.(X))
        knownentries = Float64.(X_dataset[:,i])
        soln, _ = ScaledASD(m, n, r, I_idx, J_idx, knownentries, opts; 
                            soln_only = true)
        Y = Float64.(Y_dataset[:,:,i])
        mse_losses[i] = sum((Y .- soln) .^ 2) / (m * n)
        spectral_dists[i] = norm(svdvals(Y) .- svdvals(soln)) / length(svdvals(Y))
    end
    return mean(mse_losses), mean(spectral_dists)
end

#----------------#
#A_true = reshape(Ytrain[:,1], 8, 8)
#nonzeroindices = findall(x -> x != 0, A)
#A_mask = zeros(Float64, 8, 8)
#A_mask[nonzeroindices] .= 1.0
#A_mask = BitMatrix(A_mask)

#= m = 8
n = 8
r = 2

A_true = randn(m, r) * randn(r, n)
mask = rand(0:1, m, n)
A_obs = A_true .* mask

I_idx, J_idx, data = sparse2idx(A_obs)
start = Dict(:L => randn(m, r), :R => randn(r, n))  # Optional starting point
opts = Dict(
    :rel_res_tol => 1e-5,           # Float64
    :maxit => 1000,                 # Int
    :verbosity => true,                # Int or Bool
    :rel_res_change_tol => 1e-4     # Float64
)
Factors, Out = ScaledASD(m, n, r, I_idx, J_idx, data, start, opts)
final_X, final_Y = Factors[end]
soln = final_X * final_Y
=#

"""
    IRLS_M(m, n, I_idx, J_idx, data, q, alpha, tol, maxit)

Julia implementation of the Iteratively Reweighted Least Squares (IRLS) algorithm for matrix completion.
See M. Fronnasier, H. Rauhut and R. Ward. Low-rank matrix recovery via iteratively reweighted least squares 
minimization. SIAM Journal on Optimization, 21(4): 1614-1640, 2011.

Pseudo-code:
Input: a constant q >= r; a scaling parameter alpha > 0; a stopping criterion T
Initialize: an iteration counter k = 0; a regularizing sequence eps0 = 1; W0 = I
Algorithm:
While T = false do
    k = k + 1
    X_k = argmin_{known entries} ||W_{k-1}^(1/2) * X||_F^2  ; a WLS problem solved by updating each column of X_k
    eps_k = min(eps_{k-1}, alpha * sigma_{q+1}(X_k) )
    Compute SVD perturbation version tildeX_k of X_k  (to avoid misbehavior in the subsequent inversion)
    W_k = (tildeX_k * tildeX_k^T)^(-1/2)
End
Output: a matrix X
"""
function IRLS_M(
    m::Int, # Number of rows
    n::Int, # Number of columns
    I_idx::Vector{Int}, # I indices of the observed entries
    J_idx::Vector{Int}, # J indices of the observed entries
    data::Vector{Float64}, # Observed entries
    q::Int, # Constant q >= r
    alpha::Float64, # Scaling parameter alpha > 0
    tol::Float64 = 1e-5, # Stopping criterion
    maxit::Int = 5000, # Maximum number of iterations
)
    @assert length(data) == length(I_idx)
    @assert length(data) == length(J_idx)
    p = length(data) # Number of observed entries
    @assert p > 0 && p < m * n
    @assert q < min(m, n)

    # Initialize
    iter = 0 # Iteration counter
    eps = 1 # Regularizing sequence
    W = I(n) # Weight matrix
    println("Shape of W: ", size(W))

    diff_matrix = sparse(I_idx, J_idx, data, m, n)
    diff_matrix.nzval .= data
    diff_matrix = Matrix(diff_matrix)
    println("Shape of diff_matrix: ", size(diff_matrix))

    converged = false
    local X = diff_matrix * W
    # While stopping criterion is not met
    while iter < maxit && !converged
        old_W = copy(W)        
        println("Shape of X at iteration $(iter): ", size(X))
        try
            # SVD perturbation
            U, S, V = svd(X)
            println("Shape of U, S, V:", size(U), size(S), size(V))
            X_tilde = U[:, 1:q] * sqrt(Diagonal(S[1:q])) * V[:, 1:q]'
            println("Shape of X_tilde: ", size(X_tilde))
            W = inv(X_tilde * X_tilde')^(1/2)
            println("Shape of W: ", size(W))
            eps = min(eps, alpha * S[q+1])
            println("eps: ", eps)
        catch e
            @error("Numerical issue encountered: $e")
            break
        end
        X = diff_matrix * W
        iter += 1
        if eps <= tol || norm(W - old_W) < tol
            converged = true
        end
    end
    # Issue warning if appropriate
    if iter <= 1
        @warn("No iterations performed.")
    end
    if eps <= tol || norm(W - old_W) < tol
        @info("Convergence reached after $iter iterations.")
    else
        @info("Convergence not reached after $iter iterations, with eps = $eps.")
    end
    println("Shape of X after convergence: ", size(X))
    return X
end

function IRLS_performance(X_dataset, Y_dataset, I_idx, J_idx)
    dataset_size = size(Y_dataset, 3)
    mse_losses = zeros(Float32, dataset_size)
    spectral_dists = zeros(Float32, dataset_size)
    m, n = size(Y_dataset, 1), size(Y_dataset, 2)
    for i in 1:dataset_size
        #X = reshape(X_dataset[:,i], m, n)
        #Y = reshape(Y_dataset[:,i], m, n)
        #r = rank(Y)
        #I_idx, J_idx, knownentries = sparse2idx(Float64.(X))
        knownentries = Float64.(X_dataset[:,i])
        soln = IRLS_M(m, n, I_idx, J_idx, knownentries, 8, 1.0, 1e-5, 5000)
        Y = Float64.(Y_dataset[:,:,i])
        mse_losses[i] = norm(Y .- soln, 2)^2 / (m*n)
        spectral_dists[i] = norm(svdvals(Y) .- svdvals(soln),2)^2 / length(svdvals(Y))
    end
    return mean(mse_losses), mean(spectral_dists)
end

##############
#=
m, n, dataset_size = size(Ytest)
mse_losses = zeros(Float32, dataset_size)
spectral_dists = zeros(Float32, dataset_size)

Xtest[:,1]
#for i in 1:dataset_size
knownentries = Float64.(Xtest[:,1])
soln = IRLS_M(m, n, I_idx, J_idx, knownentries, 8, 1.0, 1e-5, 500)

Y = Float64.(Y_dataset[:,:,i])
mse_losses[i] = norm(Y .- soln, 2)^2 / (m*n)
spectral_dists[i] = norm(svdvals(Y) .- svdvals(soln),2)^2 / length(svdvals(Y))

IRLS_performance(Xtest, Ytest, I_idx, J_idx)

##############

p = length(data) # Number of observed entries
k = 0 # Iteration counter
eps = 1 # Regularizing sequence
W = I(n) # Weight matrix
diff_matrix = sparse(I_idx, J_idx, data, m, n)
diff_matrix.nzval .= data

# LOOP
k += 1
# Weighted least squares problem
X = Matrix(diff_matrix) * W
# SVD perturbation
U, S, V = svd(X)
X_tilde = U[:, 1:q] * Diagonal(S[1:q]) * V[:, 1:q]'
W = inv(X_tilde * X_tilde')^(0.5)
eps = min(eps, alpha * S[q+1])


# Test
A = gen_matrix(30, 30, 2)
m, n = size(A)
# randomly sample 30% of the entries
rng = MersenneTwister(1234)
B = A .* (rand(rng, 30, 30) .< 1/3)
I_idx, J_idx, data = sparse2idx(B)
q=20
alpha=1.0
tol=1e-5
maxit=5000
=#


# Marchenko Pastur bounds on largest and smalled eigvals if normal
function mpbound(m, n, var)
    λmax = (1 + sqrt(m/n))^2 * var
    λmin = (1 - sqrt(m/n))^2 * var
    # Warn if m and n too small for asymptotic bounds
    if m < 50 || n < 50
        @warn("m and n too small for asymptotic bounds to be accurate.")
    end
    return λmin, λmax
end
#mpbound(30, 30, 1.0)
