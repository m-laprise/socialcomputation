# Julia implementation of Scaled Alternating Steepest Descent (ScaledASD) for matrix completion.
# For the algorithm itself, see Jared Tanner and Ke Wei. Low rank matrix completion by alternating 
# steepest descent methods. Applied and Computational Harmonic Analysis, 40(2):417-429, 2016.

using LinearAlgebra
using SparseArrays

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
        
    end
    if iter <= 2
        # If no iteration is performed, throw an error
        error("No iterations performed; initial residual = $res")
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