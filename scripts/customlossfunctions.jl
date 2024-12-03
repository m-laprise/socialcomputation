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

MSE(A::AbstractArray, B::AbstractArray) = abs(mean((A .- B).^2))
RMSE(A::AbstractArray, B::AbstractArray) = abs(sqrt(MSE(A, B)))

spectdist(A, B) = norm(abs.(svdvals(A)) - abs.(svdvals(B))) / length(svdvals(A))

# Use Flux.logitbinarycrossentropy instead, same operation but more numerically stable
# mylogitbinarycrossentropy(ŷ, y) = mean(@.((1 - y) * ŷ - logσ(ŷ)))

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
    loss = Flux.logitbinarycrossentropy(ys_hat, ys)
    #println("Losses: ", losses)
    #println("Size losses: $(size(losses))")
    return loss #, losses, ys_hat
end

function recon_losses(m::Chain, 
                   xs::AbstractArray{Float32}, 
                   ys_init::AbstractArray{Float32}; 
                   turns::Int = TURNS, 
                   mode::String = "training", 
                   set::String = "train", 
                   type::String = "l2")
    @assert length(size(ys_init)) == 3
    l, n, nb_examples = size(ys_init)
    totN = l * n
    ys = reshape(ys_init, totN, nb_examples)
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 2)
    end
    @assert length(size(xs)) == 2
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
            @warn("NaN or Inf detected in prediction during computation of loss. Replacing with 0.")
            #pred[findall(isinf.(pred))] .= 0.0
            #pred[findall(isnan.(pred))] .= 0.0
            pred = replace(y -> isfinite(y) ? y : 0.0f0, pred)
        end
        preds[:,i] = pred
    end
    ys_hat = copy(preds)
    if mode == "training"
        errors = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        if type == "l2l1"
            for i in 1:nb_examples
                diff = ys[:,i] .- ys_hat[:,i]
                l2normsq = norm(diff, 2)^2
                l1norm = norm(diff, 1)
                errors[i] = (l2normsq + l1norm) / totN
            end
        elseif type == "l2"
            for i in 1:nb_examples
                errors[i] = norm(ys[:,i] .- ys_hat[:,i], 2)^2 / totN
            end
        else 
            error("Invalid type.")
        end
        return mean(copy(errors))
    elseif mode == "testing"
        l2errors = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        RMSEerrors = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        spectdists = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        reshaped_yhat = reshape(ys_hat, l, n, nb_examples)
        for i in 1:nb_examples
            if set == "full"
                diff = ys[:,i] .- ys_hat[:,i]
                if type == "l2l1"
                    l2normsq = norm(diff, 2)^2
                    l1norm = norm(diff, 1)
                    l2errors[i] = (l2normsq + l1norm) / totN
                elseif type == "l2"
                    l2errors[i] = norm(diff, 2)^2 / totN
                else 
                    error("Invalid type.")
                end
            end
            RMSEerrors[i] = RMSE(ys[:,i], ys_hat[:,i])
            spectdists[i] = spectdist(ys_init[:,:,i], reshaped_yhat[:,:,i])
        end
        return Dict("l2" => mean(copy(l2errors)), 
                    "RMSE" => mean(copy(RMSEerrors)), 
                    "spectdist" => mean(copy(spectdists)))
    end
end

#= function spectral_distance(m, 
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
        errors[i] = spectdist(ys_hat_m, ys_m)
    end
    return mean(copy(errors))
end =#

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