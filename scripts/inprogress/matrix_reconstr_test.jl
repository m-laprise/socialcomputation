include("genrandommatrix.jl")
using StatsBase

m = 5000
n = 100
r = 4
tot = m*n
L = Int(floor(r * (m+n) * log(m)) + 1)
L / tot

A=gen_matrix(m, n, r, noisy=false)
rank(A) == r
B=gen_matrix(m, n, r, noisy=true)
rank(B) == n

masksB = sensingmasks(B)
toobserve = masksB[1:L]
observed_maskmat = zeros(m, n)
for mask in toobserve
    i,j = mask
    observed_maskmat[i,j] += 1
    if i != 1
        observed_maskmat[i,1] = 1
    end
end
# matrix with 1s where observed_maskmat has 0s and 0s where observed_maskmat has 1s
secret_maskmat = 1 .- observed_maskmat

d = 500
agentids = [i for i in 1:d]

# There are L observations to distribute among d agents.

function equal_split(L, d)
    splits = fill(Int(floor(L/d)), d)
    for i in 1:(L%d)
        splits[i] += 1
    end
    return splits
end

function random_split(L, d)
    splits = [0 for i in 1:d]
    for i in 1:L
        splits[rand(1:d)] += 1
    end
    return splits
end

function unequal_split(L, d; α=1.0)
    splits = [0 for i in 1:d]
    # alpha is the critical exponent for the power law distribution of the splits
    biases = [rand()^(-1/α) for i in 1:d]
    biases = biases ./ sum(biases)
    for i in 1:L
        splits[sample(1:d, Weights(biases))] += 1
    end
    return splits
end

function splits_to_maskijs(splits, masks)
    return [masks[1:splits[i]] for i in 1:length(splits)]
end

agentsplits = unequal_split(L, d; α=1.0)
agentmasks = splits_to_maskijs(agentsplits, toobserve)

# d matrices of observations, one for each agent
agentobs = Dict{Int, Matrix{Float64}}(
    agent => zeros(m, n) for agent in agentids
)

using BenchmarkTools
for agent in agentids
    for maskij in agentmasks[agent]
        agentobs[agent] = apply_mask!(agentobs[agent], B, maskij, addoutcome=true, outcomecol=1)
    end
end

# Create new matrices for each agent by dropping all zeros rows or columns
agentobs_small = deepcopy(agentobs)
for agent in agentids
    A = sparse(agentobs_small[agent])
    rowidx, colidx, _ = findnz(A)
    A = A[unique(rowidx), unique(colidx)]
    agentobs_small[agent] = Matrix(A)
end


# Create communication graph
using Graphs

g = newman_watts_strogatz(d, 5, 0.3)

# randomly select a secret entry to be the target
# by sampling a 1 from the secret matrix and finding its index
targetidx = findfirst(secret .== 1)
target = B[targetidx]


## OLS BEST
X = B[2:end,2:end]
y = B[2:end,1]
rsquared, testerror, olsfit = fit_linear_model(X, y)

k = size(X, 2)
Xtest = DataFrame(y=y)
for i in 1:k
    Xtest[!, Symbol("X$i")] = X[:, i]
end
yhat = predict(olsfit, DataFrame(Xtest[1,:]))[1]
ynot = Xtest[1,1]
prederr = (yhat - ynot)^2


