using LinearAlgebra
using Random
import Distributions: Dirichlet, mean, std, var
using Plots
using Lux

function trim_matrix(A::AbstractMatrix)
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

function project_r(A::AbstractArray, r::Int)
    u, s, v = svd(A)
    return u[:, 1:r] * Diagonal(s[1:r]) * v[:, 1:r]'
end

function project_r!(A::AbstractArray, r::Int)
    try
        u, s, v = svd(A)
        A .= u[:, 1:r] * Diagonal(s[1:r]) * v[:, 1:r]'
    catch 
        u, s, v = svd(A; alg = LinearAlgebra.QRIteration())
        A .= u[:, 1:r] * Diagonal(s[1:r]) * v[:, 1:r]'
    end
end

function correct_known!(grassman_proj, sparsematrix)
    for i = axes(sparsematrix, 1)
        for j = axes(sparsematrix, 2)
            if sparsematrix[i, j] != 0
                grassman_proj[i, j] = sparsematrix[i, j]
            end
        end
    end
end

function indiv_embed(sparsevec::AbstractVector, r, iter=5)
    n2 = length(sparsevec)
    n = Int(sqrt(n2))
    sparsematrix = reshape(sparsevec, n, n)
    grassman_proj = project_r(trim_matrix(sparsematrix), r)
    # For iter times, correct the known entries and project back to the Grassman manifold again
    for _ in 1:iter
        correct_known!(grassman_proj, sparsematrix)
        project_r!(grassman_proj, r)
    end
    correct_known!(grassman_proj, sparsematrix)
    return grassman_proj
end

function indiv_embed(sparsematrix::AbstractMatrix, r, iter=5)
    grassman_proj = project_r(trim_matrix(sparsematrix), r)
    # For iter times, correct the known entries and project back to the Grassman manifold again
    for _ in 1:iter
        correct_known!(grassman_proj, sparsematrix)
        project_r!(grassman_proj, r)
    end
    correct_known!(grassman_proj, sparsematrix)
    return grassman_proj
end

# Hyperparameters and constants
K::Int = 100 # Try 9 (182 entries per agent) or 8 (205 entries per agent)
N::Int = 128
const DATASETSIZE::Int = 1
const KNOWNENTRIES::Int = 2500

mylw = 2
datarng = Random.MersenneTwister(4021)
k = K

_3D(y) = reshape(y, N, N, size(y, 2))
_2D(y) = reshape(y, N*N, size(y, 3))
_3Dslices(y) = eachslice(_3D(y), dims=3)
l(f, y) = mean(f.(_3Dslices(y)))

include("../datacreation_LuxCPU.jl")

RANK::Int = 1
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng)
#maskmatrix = masktuple2array(masktuples, N, N)
#nonzeroidx = findall((vec(maskmatrix)) .!= 0)
size(dataX), size(dataY)
X = dataX[:,:,1]
Y = dataY[:,:,1]
#=
A = indiv_embed(vec(sum(X[:,1:2], dims=2)), 1, 100)
MAELoss()(A, Y)=#

is = 1:k
MAEs1 = []
MAEs10 = []
MAEs100 = []
MAEs1000 = []
for i in is
    A1 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 1)
    A10 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10)
    A100 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 100)
    A1000 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10000)
    push!(MAEs1, MAELoss()(A1, Y))
    push!(MAEs10, MAELoss()(A10, Y))
    push!(MAEs100, MAELoss()(A100, Y))
    push!(MAEs1000, MAELoss()(A1000, Y))
end

x = cumsum(knowledgedistr[1:k])
p1 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "10,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)
#====================#
RANK::Int = 2
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng)
X = dataX[:,:,1]
Y = dataY[:,:,1]
MAEs1 = []
MAEs10 = []
MAEs100 = []
MAEs1000 = []
for i in is
    A1 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 1)
    A10 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10)
    A100 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 100)
    A1000 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10000)
    push!(MAEs1, MAELoss()(A1, Y))
    push!(MAEs10, MAELoss()(A10, Y))
    push!(MAEs100, MAELoss()(A100, Y))\
    push!(MAEs1000, MAELoss()(A1000, Y))
end

x = cumsum(knowledgedistr[1:k])
p2 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "10,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
rnlogn = N * RANK * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.vline!([rnlogn], color=:grey, linestyle=:dash, label="x = r n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)
#====================#
RANK::Int = 4
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng)
X = dataX[:,:,1]
Y = dataY[:,:,1]
MAEs1 = []
MAEs10 = []
MAEs100 = []
MAEs1000 = []
for i in is
    A1 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 1)
    A10 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10)
    A100 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 100)
    A1000 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10000)
    push!(MAEs1, MAELoss()(A1, Y))
    push!(MAEs10, MAELoss()(A10, Y))
    push!(MAEs100, MAELoss()(A100, Y))
    push!(MAEs1000, MAELoss()(A1000, Y))
end

x = cumsum(knowledgedistr[1:k])
p3 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "10,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)
#====================#
RANK::Int = 8
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng)
X = dataX[:,:,1]
Y = dataY[:,:,1]
MAEs1 = []
MAEs10 = []
MAEs100 = []
MAEs1000 = []
for i in is
    A1 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 1)
    A10 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10)
    A100 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 100)
    A1000 = indiv_embed(vec(sum(X[:,1:i], dims=2)), RANK, 10000)
    push!(MAEs1, MAELoss()(A1, Y))
    push!(MAEs10, MAELoss()(A10, Y))
    push!(MAEs100, MAELoss()(A100, Y))
    push!(MAEs1000, MAELoss()(A1000, Y))
end

x = cumsum(knowledgedistr[1:k])
p4 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw,
                legend_position = :bottomleft)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "10,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)

#====================#

fig = plot(p1, p2, p3, p4, layout=(2,2), size=(850,600),
            suptitle = "Completion of one random $(N)x$(N) matrix (Optspace)")
savefig(fig, "optspace_$(N)x$(N)_r1r2r4r8.png")
#, title = )

#====================#

#is = 20:25
res = []
Alist = []
for i in is
    As = zeros(N, N, i)
    for agent in 1:i
        As[:, :, agent] .= indiv_embed(X[:,agent], 1, 100)
    end
    A = sum(As, dims=3)
    push!(Alist, As)
    push!(res, MAELoss()(A, Y))
end

#===================================#
using LinearAlgebra
using Manopt, Manifolds, ManifoldDiff

# Define the Grassmann manifold
n, k = 64, 1
Gr = Grassmann(n, k)
# Random initial pair of points on the Grassmann manifold
(X0a, X0b) = (rand(Gr), rand(Gr))

sparseM = reshape(X[:,1], n, n)
u, s, v = svd(sparseM)

# Define the cost function as the Frobenius norm of 
# the difference between points and their projections onto the Grassmann manifold
function cost_function((Ma, Mb), X)
    a = norm(Ma - X * X' * Ma, Fro)^2
    b = norm(Mb - X * X' * Mb, Fro)^2
    return (a + b) / 2
end

# Define the gradient of the cost function
function grad_cost_function((Ma, Mb), X)
    a = Ma - X * X' * Ma
    b = Mb - X * X' * Mb
    return (a * Ma' + b * Mb') / 2
end

# Perform gradient descent
X_closest = gradient_descent(Gr, 
    cost_function,
    grad_cost_function,
    (X0a, X0b)
)