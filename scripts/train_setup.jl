using Random, Distributions
using SparseArrays
#=
The script includes functions required to run the training loop under various experimental conditions.
It also includes helper functions for inspecting and diagnosing gradients during training.
=#

"""
    train_setup(data::Dict{String, Array}, TASKCAT::String, TASK::String)

Setup the training data for the specified task category and task. 
For reconstruction tasks, the function returns the input data. For binary classification tasks, 
the function converts the ranks to binary labels of whether the matrix is low rank. For multi-class 
classification tasks, the function converts the ranks to one-hot encoded vectors.
"""
function label_setup(data::Dict{String, Array}, 
                     TASKCAT::String, TASK::String)
    @assert haskey(data, "ranks")
    @assert haskey(data, "X")
    if TASKCAT == "reconstruction"
        return data["X"]
    elseif TASKCAT == "classification"
        if TASK == "classif1a" || TASK == "classif1b"
            return data["ranks"] .< 100
        elseif TASK == "classif2a" || TASK == "classif2b"
            ranks = data["ranks"]
            # Convert vector of possible ranks to multi-class one-hot encoded matrix
            nb_classes = length(unique(ranks))
            ordered_classes = sort(unique(ranks))
            Y = zeros(Int, nb_classes, dataset_size)
            for i in 1:dataset_size
                Y[findall(x -> x == ranks[i], ordered_classes)[1], i] = 1
            end
            return Y
        else return @warn("Invalid TASK value.")
        end
    else return @warn("Invalid TASKCAT value. Choose from 'reconstruction' or 'classification'.")
    end
end

"""
    input_setup(Y::AbstractArray, MEASCAT::String, m::Int, n::Int, dataset_size::Int)

Setup the input data for the specified measurement category, for vector-state cells 
    taking a vector of known entries as input.
"""
function input_setup(Y::AbstractArray, 
                     m::Int, n::Int,
                     dataset_size::Int,
                     knownentries::Int, masks::Vector)
    X = zeros(Float32, knownentries, dataset_size)
    for k in 1:dataset_size
        for l in 1:knownentries
            i, j = masks[l]
            X[l, k] = Y[i, j, k]
        end
    end
    #= X = zeros(Float32, knownentries, 3, dataset_size)
    for k in 1:dataset_size
        for l in 1:knownentries
            i, j = masks[l]
            X[l, 1, k] = Y[i, j, k]
            X[l, 2, k] = i
            X[l, 3, k] = j
        end
    end =#
    return X
end

"""
    matinput_setup(Y::AbstractArray, m::Int, n::Int, dataset_size::Int)

Setup the input data for the specified measurement category, for matrix-state cells
    taking a k-collection of sparse M x N matrices as input, structured as a
    3D tensor of size k x (M x N) x dataset_size (k is the number of agents or net_width).

Within each dataset_size element, k indexes sparse matrices representing the knowledge of each agent.

`masks` is the output of the function `sensingmasks` and is a vector of tuples of 
the form (i, j) where i and j are the row and column indices of the known entries.

The `alpha` parameter determines how uniformly the known entries are distributed across k agents.
When `alpha = 50`, the known entries are almost surely distributed uniformly across the agents. When `alpha = 0`,
one agent has all the known entries and the others have none.
"""
function matinput_setup(Y::AbstractArray, k::Int,
                        M::Int, N::Int, dataset_size::Int,
                        knownentries::Int, masks::Vector;
                        alpha::Float64 = 50.0)
    @assert knownentries == length(masks)
    @assert alpha >= 0.0 && alpha <= 50.0
    knownentries_per_agent = zeros(Int, k)
    # Create a vector of length k with the number of known entries for each agent,
    # based on the alpha concentration parameter. The vector should sum to the total number of known entries.
    if alpha == 0
        knownentries_per_agent[1] = knownentries
    else
        dirichlet_dist = Dirichlet(alpha * ones(minimum([k, knownentries])))
        proportions = rand(dirichlet_dist)
        knownentries_per_agent = round.(Int, proportions * minimum([k, knownentries]))
        # If knownentries < k, pad the vector with zeros
        if knownentries < k
            knownentries_per_agent = vcat(knownentries_per_agent, zeros(Int, k - knownentries))
        end
        # Adjust to ensure the sum is exactly knownentries
        while sum(knownentries_per_agent) != knownentries
            diff = knownentries - sum(knownentries_per_agent)
            # If the difference is negative (positive), add (subtract) one to (from) a random agent
            knownentries_per_agent[rand(1:k)] += 1 * sign(diff)
            # Check that no entry is negative, and if so, replace by zero
            knownentries_per_agent = max.(0, knownentries_per_agent)
        end
    end
    #X = zeros(Float32, k, M*N, dataset_size)
    X = []
    for i in 1:dataset_size
        inputmat = spzeros(k, M*N)
        entry_count = 1
        for agent in 1:k
            for l in 1:knownentries_per_agent[agent]
                row, col = masks[entry_count]
                flat_index = M * (col - 1) + row
                inputmat[agent, flat_index] = Y[row, col, i]
                entry_count += 1
            end
        end
        push!(X, inputmat)
    end
    return X
end

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
        return @warn("Invalid number of dimensions for X: $dimsX")
    end
    @assert size(Xtrain, dimsX) == train_nb
    @assert size(Xval, dimsX) == val_nb
    @assert size(Xtest, dimsX) == dataset_size - train_nb - val_nb
    return Xtrain, Xval, Xtest
end
