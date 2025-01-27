using LinearAlgebra
using Random

function power_iter(A, num_iterations; eigvec = false)
    # Randomly initialize a vector
    b_k = rand(size(A, 2))
    b_k /= norm(b_k)
    for _ in 1:num_iterations
        # Calculate the matrix-by-vector product Ab
        b_k1 = A * b_k
        # Re normalize the vector
        b_k = b_k1 / norm(b_k1)
    end
    # Approximate the largest eigenvalue using the Rayleigh quotient
    largest_eig = b_k' * (A * b_k)
    if eigvec
        return abs(largest_eig), b_k
    else
        return abs(largest_eig)
    end
end

function setup(m, n, r, alpha, seed)
    rng = Random.MersenneTwister(seed)
    A = (randn(rng, m, r) ./ sqrt(sqrt(r))) * (randn(rng, r, n) ./ sqrt(sqrt(r)))
    B = A .* (rand(rng, m, n) .< alpha)
    mask = B .!= 0
    return A, B, mask
end

m = 64
n = 64
r = 1

A, M, mask = setup(m, n, r, 0.2, 1)

rank(A)
rank(M)
sdvval1 = sqrt(power_iter(A' * A, 1))
svdval1m = sqrt(power_iter(M' * M, 1))

eigvals(A' * A)
svdvals(A)
eigvals(M' * M)
svdvals(M)

# All equivalent (spectral norm or largest singular value):
sqrt(power_iter(A' * A, 100))
sqrt(eigvals(A' * A)[end])
svdvals(A)[1]

# Frobenius norm (for rank-1, equality; otherwise, larger than spectral norm)
sqrt(sum(svdvals(A).^2))
sqrt(sum(diag(A' * A)))

# Nuclear norm aka trace norm aka Schatten 1-norm
sum(svdvals(A))
norm(svdvals(A), 1)
sum(diag(sqrt(A' * A))) # requires a matrix square root

# Spectral radius
maximum(abs.(eigvals(A)))
power_iter(A, 100)

#=
As rank increases,
Spectral norm decreases
Frobenius norm fluctuates
Nuclear norm increases
Spectral radius fluctuates
=#


function trim(A::AbstractMatrix)
    m, n = size(A)
    k = sum(A .!= 0)
    for col in eachcol(A)
        if sum(col .!= 0)  > (2*k) / n
            col .= 0
        end
    end
    for row in eachrow(A)
        if sum(row .!= 0) > (2*k) / m
            row .= 0
        end
    end
    return A
end


Mtilde = trim(M)

U, S, V = svd(Mtilde)


#======Rank estimation=======#
degree(A::AbstractMatrix) = ceil(sum(A .!= 0) / sqrt(size(A, 1) * size(A, 2)) )
degree(Mtilde)

function obj_rank_est(sig, i, degree)
    N = length(sig)
    if i == N
        @warn "i == N"
    end
    return (sig[i+1] + sig[1]*sqrt(i/degree)) / sig[i]
end

function min_obj(Atilde::AbstractArray)
    sigmasA = svdvals(Atilde)
    degA = degree(Atilde)
    is = 1:length(sigmasA)-1
    rs = []
    for i in is
        push!(rs, obj_rank_est(sigmasA, i, degA))
    end
    return argmin(rs)
end

function estimate_rank_M(A::AbstractArray)
    Atilde = trim(A)
    return min_obj(Atilde)
end


n = 1000
r = 30
A = randn(n, r) * randn(r, n)
B = A .* (rand(n, n) .< 0.2)
estimate_rank_M(A)
estimate_rank_M(B)

sigmasA = svdvals(A)
degA = degree(A)
is = 1:length(sigmasA)-1
rs = []
for i in is
    push!(rs, obj_rank_est(sigmasA, i, degA))
end
rs
argmin(rs)

rankhat = estimate_rank_M(M)

#=====================#

function project_r(Atilde::AbstractArray, r::Int)
    u, s, v = svd(Atilde)
    return u[:, 1:r] * Diagonal(s[1:r]) * v[:, 1:r]'
end

PrMtilde = project_r(Mtilde, 1)



#Generate the Grassmann manifold Gr(n,k), real-valued is default
Gr = Grassmann(n, r)
X_0, S_0, Yt_0 = svd(PrMtilde)

nzmask = Mtilde .!= 0

using Zygote: gradient
f(X, Yt, S) = 0.5 * sum( (M .- (X*diagm(S)*Yt) .* nzmask)^2 )
grad_f(X, Yt, S) = Zygote.gradient((X, Yt, S) -> f(X, Yt, S), X, Yt, S)

f(X_0, Yt_0, S_0)

#========#
using Manopt, Manifolds, ManifoldDiff
using ManifoldDiff: grad_distance, prox_distance
Random.seed!(42);
# Manopt.set_parameter!(:Mode, "Tutorial")
# Manopt.set_parameter!(:Mode, "")

# Center of mass of a sphere
n = 100
σ = π / 8
M = Sphere(2)
# Initial point
p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
# Random points around the initial point
data = [exp(M, p, σ * rand(M; vector_at=p)) for i in 1:n];
# Cost function f and its gradient for Riemannian center of mass
f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)));

# Call gradient descent, provide manifold, cost, gradient, and a starting point (here the first data point)
m3 = gradient_descent(M, f, grad_f, data[1];
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.11f | "), "\n", :Stop, 25], # Output every 25 iterations only
    stopping_criterion = StopWhenGradientNormLess(1e-14) | StopAfterIteration(400),
)

# Same but determine stepsize adaptively
m4 = gradient_descent(M, f, grad_f, data[1];
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.11f | "), "\n", :Stop, 2],
      stepsize = ArmijoLinesearch(; contraction_factor=0.999, sufficient_decrease=0.5),
    stopping_criterion = StopWhenGradientNormLess(1e-14) | StopAfterIteration(400),
)

[f(M, m3)-f(M,m4), distance(M, m3, m4)]

# Median of the manifold of 3 x 3 sym PD matrices example
N = SymmetricPositiveDefinite(3)
m = 100
σ = 0.005
q = Matrix{Float64}(I, 3, 3)
data2 = [exp(N, q, σ * rand(N; vector_at=q)) for i in 1:m];

# Generalized median as the minimiser of the sum of distances
g(N, q) = sum(1 / (2 * m) * distance.(Ref(N), Ref(q), data2))

# non-smooth, so need to use cyclic proximal point algorithm
# define vector of proximal maps
proxes_g = Function[(N, λ, q) -> prox_distance(N, λ / m, di, q, 1) for di in data2];

res = cyclic_proximal_point(N, g, proxes_g, data2[1];
  debug=[:Iteration," | ",:Change," | ",(:Cost, "F(x): %1.12f"),"\n", 1000, :Stop,
        ],
        record=[:Iteration, :Change, :Cost, :Iterate],
        return_state=true,
    );
median = get_solver_result(res)