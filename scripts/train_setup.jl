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

Setup the input data for the specified measurement category.
"""
function input_setup(Y::AbstractArray, 
                     MEASCAT::String, 
                     m::Int, n::Int, dataset_size::Int,
                     knownentries::Int)
    if MEASCAT == "masks"
        masks = sensingmasks(m, n; k=knownentries, seed=9632)
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
    elseif MEASCAT == "traces"
        return @warn("Not implemented yet.")
    elseif MEASCAT == "blocks"
        return @warn("Not implemented yet.")
    else return @warn("Invalid MEASCAT value. Choose from 'masks', 'traces', or 'blocks'.")
    end
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
