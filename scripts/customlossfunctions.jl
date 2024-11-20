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

function logitbinarycrossent(m, 
                xs::AbstractArray{Float32}, 
                ys::AbstractArray{Float32}; 
                turns::Int = TURNS)
    #println(size(xs))
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 1)
    end
    zs = eachcol(xs)
    # For each example, read in the input, recur for `turns` steps, 
    # and predict the label, then reset the state
    preds = Zygote.Buffer(ys)
    #println("Slice iterator size: ", size(zs))
    #println("Buffer size: ", size(preds))
    for (i, example) in enumerate(zs)
        reset!(m)
        for _ in 1:turns
            m(example)
        end
        pred = m(example)[1]
        preds[i] = pred
    end
    ys_hat = copy(preds)
    #println("Predicted: ", ys_hat)
    #println("True: ", ys)
    losses = @.((1 - ys) * ys_hat - logσ(ys_hat))
    #println("Losses: ", losses)
    #println("Size losses: $(size(losses))")
    return mean(losses)#, losses, ys_hat
end


function recon_mse(m, 
                xs::AbstractArray{Float32}, 
                ys::AbstractArray{Float32}; 
                turns::Int = TURNS)
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 2)
    end
    if length(size(ys)) == 3
        l, n, nb_examples = size(ys)
        ys = reshape(ys, l * n, nb_examples)
    end
    zs = eachcol(xs)
    # For each example, read in the input, recur for `turns` steps, 
    # and predict the label, then reset the state
    preds = Zygote.Buffer(ys)
    for (i, example) in enumerate(zs)
        reset!(m)
        for _ in 1:turns
            m(example)
        end
        pred = m(example)
        preds[:,i] = pred
    end
    ys_hat = copy(preds)
    errors = Zygote.Buffer(ys[1,:])
    nb_examples = size(ys, 2)
    for i in 1:nb_examples
        errors[i] = norm(ys[:,i] .- ys_hat[:,i], 2)^2 / length(ys[:,i])
    end
    return mean(copy(errors))
end


function classification_accuracy(m, 
                xs::AbstractArray{Float32}, 
                ys::AbstractArray{Float32}; 
                turns::Int = TURNS)
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 2)
    end
    zs = eachcol(xs)
    # For each example, read in the input, recur for `turns` steps, 
    # and predict the label, then reset the state
    preds = Zygote.Buffer(ys)
    for (i, example) in enumerate(zs)
        reset!(m)
        for _ in 1:turns
            m(example)
        end
        pred = m(example)[1]
        preds[i] = pred
    end
    ys_hat = copy(preds)
    ys_hat_bin = ys_hat .> 0.5f0
    labels = Bool.(ys)
    iscorrect = ys_hat_bin .== labels
    return sum(iscorrect) / length(labels)
end

function spectral_distance(m, 
                            xs::AbstractArray{Float32}, 
                            ys::AbstractArray{Float32}; 
                            turns::Int = TURNS)
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 2)
    end
    zs = eachcol(xs)
    # For each example, read in the input, recur for `turns` steps, 
    # and predict the label, then reset the state
    preds = Zygote.Buffer(ys)
    for (i, example) in enumerate(zs)
        reset!(m)
        for _ in 1:turns
            m(example)
        end
        pred = m(example)
        # If any predicted entry is NaN or infinity, replace with 0
        if any(isnan.(pred)) || any(isinf.(pred))
            pred[findall(isinf.(pred))] .= 0.0
            pred[findall(isnan.(pred))] .= 0.0
        end
        if length(size(ys)) == 2
            preds[:,i] = pred
        elseif length(size(ys)) == 3
            l, n, _ = size(ys)
            preds[:,:,i] = reshape(pred, l, n)
        else
            error("Invalid size of ys.")
        end
    end
    ys_hat = copy(preds)
    nb_examples = size(ys)[end]
    # Difference between the spectrums
    errors = Zygote.Buffer(ys[1,1,:])
    for i in 1:nb_examples
        if length(size(ys)) == 2
            l2, _ = size(ys)
            l = Int(sqrt(l2))
            ys_m = reshape(ys[:,i], l, l)
            ys_hat_m = reshape(ys_hat[:,i], l, l)
        elseif length(size(ys)) == 3
            ys_m = ys[:,:,i]
            ys_hat_m = ys_hat[:,:,i]
        else
            error("Invalid size of ys.")
        end
        svdvals_hat = svdvals(ys_hat_m)
        svdvals_true = svdvals(ys_m)
        errors[i] = norm(svdvals_true .- svdvals_hat, 2)^2 / length(svdvals_true)
    end
    spectral_distances = copy(errors)
    return mean(spectral_distances)
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