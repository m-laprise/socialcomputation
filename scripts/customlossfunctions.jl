#= 
This script defines the loss functions for the RNN experiments, for 
the reconstruction and classification tasks. The loss functions are
logit binary cross-entropy, classification accuracy, mean squared 
reconstruction error, and spectral distance of singular values.
=#
using Random
using Distributions
using Flux
using Zygote
using LinearAlgebra
using SparseArrays: sparse

MSE(A::AbstractArray, B::AbstractArray) = abs(mean((A .- B).^2))
RMSE(A::AbstractArray, B::AbstractArray) = abs(sqrt(MSE(A, B)))

spectdist(A::AbstractArray, B::AbstractArray) = norm(abs.(svdvals(A)) - abs.(svdvals(B))) / length(svdvals(A))
nuclearnorm(A::AbstractArray) = sum(abs.(svdvals(A))) 

# Use Flux.logitbinarycrossentropy instead, same operation but more numerically stable
# mylogitbinarycrossentropy(ŷ, y) = mean(@.((1 - y) * ŷ - logσ(ŷ)))

"""
    predict_through_time(m, xs, turns)

    Takes a recurrent model `m`, a set of inputs `xs`, and the number of time steps `turns` to recur the model.
    Returns the predicted outputs for each input after `turns` steps.
    It uses Zygote buffers and is autodifferentiable.
"""
function predict_through_time(m::Chain, 
                              xs::AbstractArray{Float32}, 
                              turns::Int)
    @assert ndims(xs) == 2
    if isa(m[:dec], Dense)
        output_size = length(m[:dec].bias)
    elseif isa(m[:dec], Split)
        output_size = length(m[:dec].paths[1].bias)
    elseif isa(m[:dec], BasisChange)
        output_size = size(m[:dec].bias, 1)^2
    else
        @error("Model decoder layer not implemented for prediction through time.")
    end
    nb_examples = size(xs)[2]
    # For each example, read in the input, recur for `turns` steps, 
    # and predict the label, then reset the state
    zs = eachcol(xs)
    preds = Zygote.Buffer(zeros(Float32, output_size, nb_examples))
    for (i, example) in enumerate(zs)
        reset!(m)
        for _ in 1:turns
            m(example)
        end
        pred = m(example)
        # If any predicted entry is NaN or infinity, replace with 0
        if any(isnan.(pred)) || any(isinf.(pred))
            @warn("NaN or Inf detected in prediction during computation of loss. Replacing with 0.")
            pred = replace(y -> isfinite(y) ? y : 0.0f0, pred)
        end
        if output_size == 1
            preds[:,i] = pred[1]
        else
            preds[:,i] = pred
        end
    end
    reset!(m)
    return copy(preds)
end

function logitbinarycrossent(m, 
                xs::AbstractArray{Float32}, 
                ys::AbstractArray{Float32}; 
                turns::Int = TURNS)
    #println(size(xs))
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 1)
    end
    ys_hat = predict_through_time(m, xs, turns)
    #println("Predicted: ", ys_hat)
    #println("True: ", ys)
    loss = Flux.logitbinarycrossentropy(ys_hat, ys)
    #println("Losses: ", losses)
    #println("Size losses: $(size(losses))")
    return loss #, losses, ys_hat
end

function masktuple2array(fixedmask::Vector{Tuple{Int, Int}})
    k = length(fixedmask)    
    is = [x[1] for x in fixedmask]
    js = [x[2] for x in fixedmask]
    sparsemat = sparse(is, js, ones(k))
    return Matrix(sparsemat)
end

function recon_losses(m::Chain, 
                    xs::AbstractArray{Float32}, 
                    ys_mat::AbstractArray{Float32},
                    mask_mat = nothing,
                    groundtruth::Bool = false; 
                    turns::Int = TURNS, 
                    mode::String = "training", 
                    incltrainloss::Bool = false, 
                    type::String = "l2nnm")
    if !groundtruth
        @assert !isnothing(fixedmask)
    end
    @assert ndims(ys_mat) == 3
    l, n, nb_examples = size(ys_mat)
    if ndims(xs) == 1
        xs = reshape(xs, length(xs), 2)
    end
    @assert ndims(xs) == 2

    ys = reshape(ys_mat, l * n, nb_examples)
    ys_hat = predict_through_time(m, xs, turns)

    if !groundtruth
        totN = sum(vec(mask_mat))
    else
        totN = l * n
    end

    # If the ground truth is given, compare y (full ground truth matrix) to y_hat (full estimated matrix)
    # If not, compare x (masked ground truth matrix) to x_hat (masked estimated matrix)
    if mode == "training"
        # When training, return only the training loss for gradient computation
        errors = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        for i in 1:nb_examples
            if !groundtruth
                diff = vec(mask_mat) .* (ys[:,i] .- ys_hat[:,i])
            else
                diff = ys[:,i] .- ys_hat[:,i]
            end
            l2normsq = norm(diff, 2)^2
            if type == "l2l1"
                l1norm = norm(diff, 1)
                errors[i] = (l2normsq + l1norm) / totN
            elseif type == "l2"
                errors[i] = l2normsq / totN
            elseif type == "l2nnm"
                theta = 0.7
                nnm = nuclearnorm(ys_hat[:,i]) / l
                errors[i] = theta * (l2normsq / totN) + (1 - theta) * nnm
            else 
                error("Invalid type of loss function declared in training branch.")
            end
        end
        return mean(copy(errors))
    elseif mode == "testing"
        # When testing or validating, return all relevant metrics
        l2errors = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        RMSerrors_all = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        for i in 1:nb_examples
            # Include training loss with or without ground truth
            if incltrainloss
                if !groundtruth
                    diff = vec(mask_mat) .* (ys[:,i] .- ys_hat[:,i])
                else
                    diff = ys[:,i] .- ys_hat[:,i]
                end
                l2normsq = norm(diff, 2)^2
                if type == "l2l1"
                    l1norm = norm(diff, 1)
                    l2errors[i] = (l2normsq + l1norm) / totN
                elseif type == "l2"
                    l2errors[i] = l2normsq / totN
                elseif type == "l2nnm"
                    theta = 0.7
                    nnm = nuclearnorm(ys_hat[:,i]) / l
                    l2errors[i] = theta * (l2normsq / totN) + (1 - theta) * nnm    
                else 
                    error("Invalid type of loss function declared in testing branch.")
                end
            end
            # Include RMSE over all entries (in all cases)
            RMSerrors_all[i] = RMSE(ys[:,i], ys_hat[:,i])
        end
        return Dict("l2" => mean(copy(l2errors)), 
                    "RMSE" => mean(copy(RMSerrors_all)))
    end
end

# Include spectral distance with ground truth (in all cases)
#yhat_mat = reshape(ys_hat, l, n, nb_examples)
#spectdists[i] = spectdist(ys_mat[:,:,i], yhat_mat[:,:,i])

function classification_accuracy(m, 
                xs::AbstractArray{Float32}, 
                ys::AbstractArray{Float32}; 
                turns::Int = TURNS)
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 2)
    end
    ys_hat = predict_through_time(m, xs, turns)
    ys_hat_bin = ys_hat .> 0.5f0
    labels = Bool.(ys)
    iscorrect = ys_hat_bin .== labels
    return sum(iscorrect) / length(labels)
end

########## Functions from Flux

#= function __check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y)) 
     size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
        "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
      ))
    end
end

function xlogy(x, y)
    result = x * log(y)
    ifelse(iszero(x), zero(result), result)
end

function mybinarycrossentropy(ŷ, y; agg = mean, eps::Real = eps(float(eltype(ŷ))))
    __check_sizes(ŷ, y)
    agg(@.(-xlogy(y, ŷ + eps) - xlogy(1 - y, 1 - ŷ + eps)))
end

function mylogitbinarycrossentropy(ŷ, y; agg = mean)
    __check_sizes(ŷ, y)
    agg(@.((1 - y) * ŷ - logσ(ŷ)))
end =#
