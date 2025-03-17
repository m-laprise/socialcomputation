
function lowrankmatrix(m, n, r, rng; datatype=Float32, std::Float32 = 1f0)
    std/sqrt(datatype(r)) * randn(rng, datatype, m, r) * randn(rng, datatype, r, n)
end

function sensingmasks(m::Int, n::Int, k::Int, rng)
    @assert k <= m * n
    maskij = Set{Tuple{Int, Int}}()
    while length(maskij) < k
        i = rand(rng, 1:m)
        j = rand(rng, 1:n)
        push!(maskij, (i, j))
    end
    return collect(maskij)
end

function masktuple2array(masktuples::Vector{Tuple{Int, Int}}, m::Int, n::Int)
    maskmatrix = zeros(Float32, m, n)
    for (i, j) in masktuples
        maskmatrix[i, j] = 1f0
    end
    return maskmatrix
end

"""Create a vector of length k with the number of known entries for each agent, based on the
alpha concentration parameter. The vector should sum to the total number of known entries."""
function allocateentries(k::Int, knownentries::Int, alpha::Float32, rng)
    @assert alpha >= 0 && alpha <= 50
    if alpha == 0
        # One agent knows all entries; others know none
        entries_per_agent = zeros(Int, k)
        entries_per_agent[rand(rng, 1:k)] = knownentries
    else
        # Distribute entries among agents with concentration parameter alpha
        @fastmath dirichlet_dist = Dirichlet(alpha * ones(Float32, k))
        @fastmath proportions = rand(rng, dirichlet_dist)
        @fastmath entries_per_agent = round.(Int, proportions * knownentries)
        # Adjust to ensure the sum is exactly knownentries after rounding
        while sum(entries_per_agent) != knownentries
            diff = knownentries - sum(entries_per_agent)
            # If the difference is negative (positive), add (subtract) one to (from) a random agent
            entries_per_agent[rand(rng, 1:k)] += 1 * sign(diff)
            # Check that no entry is negative, and if so, replace by zero
            entries_per_agent = max.(0, entries_per_agent)
        end
    end
    return entries_per_agent
end

"Populate ground truth low rank matrices Y"
function populateY!(Y::AbstractArray{Float32, 3}, rank::Int, rng)
    @inbounds for i in axes(Y, 3)
        @fastmath Y[:, :, i] .= lowrankmatrix(size(Y,1), size(Y,2), rank, rng)
    end
end

function populateX!(X::AbstractArray{Float32, 3}, 
                    Y::AbstractArray{Float32}, 
                    knowledgedistribution::Vector{Int}, 
                    masktuples::Vector{Tuple{Int, Int}})
    @inbounds for i in axes(X, 3)
        globalcount = 1
        @inbounds for agent in axes(X, 2)
            @inbounds for _ in 1:knowledgedistribution[agent]
                row, col = masktuples[globalcount]
                flat_index = size(Y, 1) * (col - 1) + row
                X[flat_index, agent, i] = Y[row, col, i]
                globalcount += 1
            end
        end
    end
end

function datasetgeneration(m, n, rank, dataset_size, nbknownentries, k, rng;
                           alpha::Float32 = 50f0)
    Y = Array{Float32, 3}(undef, m, n, dataset_size)
    populateY!(Y, rank, rng)
    masktuples = sensingmasks(m, n, nbknownentries, rng)
    knowledgedistribution = allocateentries(k, nbknownentries, alpha, rng)
    X = zeros(Float32, m * n, k, dataset_size)
    populateX!(X, Y, knowledgedistribution, masktuples)
    return X, Y, masktuples, knowledgedistribution
end
