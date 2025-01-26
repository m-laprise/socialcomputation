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

MSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs.((A .- B).^2), dims=1)
RMSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = sqrt.(MSE(A, B))

spectdist(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = norm(abs.(svdvals(A)) - abs.(svdvals(B))) / length(svdvals(A))
nuclearnorm(A::AbstractArray{Float32}) = sum(abs.(svdvals(A))) 

# Use Flux.logitbinarycrossentropy instead, same operation but more numerically stable
# mylogitbinarycrossentropy(ŷ, y) = mean(@.((1 - y) * ŷ - logσ(ŷ)))

"""
    predict_through_time(m, xs, turns)

    Takes a recurrent model `m`, a set of inputs `xs`, and the number of time steps `turns` to recur the model.
    Returns the predicted outputs for each input after `turns` steps.

    For vector-state models (type Chain), it uses Zygote buffers and is autodifferentiable with Zygote.
    For matrix-state models (type matnet), it uses the GPU if available, in which case it requires Enzyme for autodiff.
"""
# dispatch to either cpu or gpu version
predict_through_time(m::Chain, xs::Vector, turns::Int) = cpu_predict_through_time(m, xs, turns)
predict_through_time(m::matnet, xs::Vector{SparseMatrixCSC}, turns::Int) = cpu_predict_through_time(m, xs, turns::Int)
predict_through_time(m::matnet, xs::Vector{CUDA.CUSPARSE.CuSparseMatrixCSC}, turns::Int) = gpu_predict_through_time(m, xs, turns::Int)
function predict_through_time(m::matnet, xs::AbstractArray{Float32}, turns::Int)
    if CUDA.functional() && isa(xs, CuArray)
        return gpu_predict_through_time(m, xs, turns)
    else
        return cpu_predict_through_time(m, xs, turns)
    end
end

function cpu_predict_through_time(m::Chain, 
                              xs::AbstractArray{Float32}, 
                              turns::Int)
    @assert ndims(xs) == 2
    trial_output = m(xs[:,1])
    output_size = length(trial_output)
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

function cpu_predict_through_time(m::matnet, 
                              xs::Vector, 
                              turns::Int)
    trial_output = m(xs[1])
    output_size = length(trial_output)
    nb_examples = length(xs)
    # Pre-allocate the output array
    preds = Zygote.Buffer(zeros(Float32, output_size, nb_examples)) 
    # For each example, read in the input, recur for `turns` steps, 
    # predict the label, then reset the state
    for (i, example) in enumerate(xs)
        reset!(m)
        if turns > 0
            for _ in 1:turns
                m(example)
            end
        end
        pred = m(example)
        # If any predicted entry is NaN or infinity, replace with 0
        Zygote.ignore() do
            if any(isnan.(pred)) || any(isinf.(pred))
                @warn("NaN or Inf detected in prediction during computation of loss. Replacing with 0.")
                pred = replace(y -> isfinite(y) ? y : 0.0f0, pred)
            end
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

function gpu_predict_through_time(m::matnet, 
                              xs::AbstractArray{Float32}, 
                              turns::Int)
    examples = eachslice(xs; dims=3)
    # For each example, reset the state, recur for `turns` steps, 
    # predict the label, store it
    if turns == 0
        reset!(m)
        preds = stack(m.(examples; selfreset = true))
    elseif turns > 0    
        trial_output = m(examples[1])
        output_size = length(trial_output)
        @assert output_size > 1
        nb_examples = length(examples)
        # This loop can run in parallel; order does not matter
        preds = device(zeros(Float32, output_size, nb_examples))
        @inbounds for (i, example) in enumerate(examples)
            reset!(m)
            repeatedexample = [example for _ in 1:turns+1]
            successive_answers = stack(m.(repeatedexample; selfreset = false))
            pred = successive_answers[:,end]
            #CUDA.unsafe_free!(repeatedexample)
            #CUDA.unsafe_free!(successive_answers)
            preds[:,i] .+= pred
        end
    else
        error("Invalid number of turns specified.")
    end
    if any(isnan.(preds)) || any(isinf.(preds))
        @warn("NaN or Inf detected in prediction during computation of loss at turn $i.")
    end
    reset!(m)
    return preds
end

#= function gpu_predict_through_time(m::matnet, xs::Vector{CUDA.CUSPARSE.CuSparseMatrixCSC}, 
                              turns::Int)
    # For each example, reset the state, recur for `turns` steps, 
    # predict the label, store it
    if turns == 0
        reset!(m)
        preds = stack(m.(xs; selfreset = true))
    elseif turns > 0    
        trial_output = m(xs[1])
        output_size = length(trial_output)
        @assert output_size > 1
        nb_examples = length(xs)
        # This loop can run in parallel; order does not matter
        preds = device(zeros(Float32, output_size, nb_examples))
        @inbounds for (i, example) in enumerate(xs)
            reset!(m)
            repeatedexample = [example for _ in 1:turns+1]
            successive_answers = stack(m.(repeatedexample; selfreset = false))
            pred = successive_answers[:,end]
            #CUDA.unsafe_free!(repeatedexample)
            #CUDA.unsafe_free!(successive_answers)
            preds[:,i] .+= pred
        end
    else
        error("Invalid number of turns specified.")
    end
    if any(isnan.(preds)) || any(isinf.(preds))
        @warn("NaN or Inf detected in prediction during computation of loss at turn $i.")
    end
    reset!(m)
    return preds
end =#

function binary_classif_losses(m, 
                xs::AbstractArray{Float32}, 
                ys::AbstractArray{Float32}; 
                turns::Int = TURNS, include_accuracy::Bool = false)
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 1)
    end
    ys_hat = predict_through_time(m, xs, turns)
    loss = Flux.logitbinarycrossentropy(ys_hat, ys)
    if include_accuracy
        ys_hat_bin = ys_hat .> 0.5f0
        acc = sum(ys_hat_bin .== Bool.(ys)) / length(ys)
        return Dict("cross-entropy loss" => loss, "accuracy" => acc)
    else
        return Dict("cross-entropy loss" => loss)
    end
end

function multiclass_classif_losses(m,
                xs::AbstractArray{Float32}, 
                ys::AbstractArray{Float32}; 
                turns::Int = TURNS, include_accuracy::Bool = false)
    if length(size(xs)) == 1
        xs = reshape(xs, length(xs), 1)
    end
    ys_hat = predict_through_time(m, xs, turns)
    # Use label_smoothing to smooth the true labels as preprocessing before computing the loss
    ys_smooth = Flux.label_smoothing(ys, 0.15f0)
    # For multiclass loss, use Flux.crossentropy
    loss = Flux.logitcrossentropy(ys_hat, ys_smooth)
    if include_accuracy
        ys_hat_int = map(i -> i[1], vec(argmax(ys_hat, dims=1)))
        ys_label_int = map(i -> i[1], vec(argmax(ys, dims=1)))
        acc = sum(ys_hat_int .== ys_label_int) / size(ys, 2)
        return Dict("cross-entropy loss" => loss, "accuracy" => acc)
    else
        return loss
    end
end

#=
y_label = Float32.(Flux.onehotbatch([2, 9, 5, 7, 6], 0:9))
y_model = softmax(reshape(-14:35, 10, 5) .* 1f0)
sum(y_model; dims=1)
Flux.crossentropy(y_model, y_label)
y_smooth = Flux.label_smoothing(y_label, 0.15f0)
Flux.crossentropy(y_model, y_smooth)
multiclass_classif_losses(m_vanilla, randn(Float32, 100, 5), y_label; turns=1)
=#

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
        @assert !isnothing(mask_mat)
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
            l2normsq = norm(diff, 2)^2 /0.1
            if type == "l2l1"
                l1norm = norm(diff, 1)
                errors[i] = (l2normsq + l1norm) / totN
            elseif type == "l2"
                errors[i] = l2normsq / totN
            elseif type == "l2nnm"
                theta = 0.8
                nnm = nuclearnorm(ys_hat[:,i]) / l
                errors[i] = (theta * (l2normsq / totN) + (1 - theta) * nnm) /0.1
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
                l2normsq = norm(diff, 2)^2 /0.1
                if type == "l2l1"
                    l1norm = norm(diff, 1)
                    l2errors[i] = (l2normsq + l1norm) / totN
                elseif type == "l2"
                    l2errors[i] = l2normsq / totN
                elseif type == "l2nnm"
                    theta = 0.8
                    nnm = nuclearnorm(ys_hat[:,i]) / l
                    l2errors[i] = (theta * (l2normsq / totN) + (1 - theta) * nnm) /0.1
                else 
                    error("Invalid type of loss function declared in testing branch.")
                end
            end
            # Include RMSE over all entries (in all cases)
            RMSerrors_all[i] = RMSE(ys[:,i], ys_hat[:,i]) /0.1
        end
        return Dict("l2" => mean(copy(l2errors)), 
                    "RMSE" => mean(copy(RMSerrors_all)))
    end
end

function gpu_l2nnm_loss(m::matnet, 
                            xs::AbstractArray{Float32}, 
                            ys_mat::AbstractArray{Float32},
                            mask_mat::AbstractArray{Float32}; 
                            turns::Int = TURNS, 
                            #mode::String = "training", 
                            theta::Float32 = 0.8f0,
                            datascale::Float32 = 0.1f0)::Float32
    @assert CUDA.functional()
    #@assert isa(xs[1], CUDA.CUSPARSE.CuSparseMatrixCSC)
    @assert ndims(ys_mat) == 3
    l, n, nb_examples = size(ys_mat)
    totN = sum(vec(mask_mat))

    ys = reshape(ys_mat, l * n, nb_examples) 
    ys_hat = predict_through_time(m, xs, turns) 
    @assert size(ys_hat) == size(ys) 

    # Compute the nuclear norm for each matrix in ys_hat
    sq_ys_hat = reshape(ys_hat, l, n, nb_examples)
    @inbounds nnm = device([sum(abs.(svdvals(sq_ys_hat[:,:,i]))) for i in 1:nb_examples])

    # Create diff matrix, n2 x nb_examples;
    # Multiply by mask vector len n2 broadcasted to hide entries in each example
    hidden_diff = vec(mask_mat) .* (ys .- ys_hat)
    # Compute the L2 norm squared for each column of hidden_diff
    l2normsq = device(sum(hidden_diff.^2, dims=1) / datascale)
    
    errors = theta * (l2normsq' / totN) .+ (1f0 - theta) * nnm / datascale
    #= if mode == "testing"
        RMSerrors = RMSE(ys, ys_hat) / datascale
    end
    if mode == "testing"
        return Dict("l2" => mean(errors), 
                    "RMSE" => mean(RMSerrors))
    else
        return mean(errors)
    end =#
    return mean(errors)
end

function cpu_l2nnm_loss(m::matnet, 
                            xs::Vector, 
                            ys_mat::AbstractArray{Float32},
                            mask_mat::AbstractArray{Float32}; 
                            turns::Int = TURNS, 
                            mode::String = "training", 
                            theta::Float32 = 0.8f0,
                            datascale::Float32 = 0.1f0)
    @assert ndims(ys_mat) == 3
    l, n, nb_examples = size(ys_mat)
    totN = sum(vec(mask_mat))

    ys = reshape(ys_mat, l * n, nb_examples) 
    ys_hat = predict_through_time(m, xs, turns) 
    @assert size(ys_hat) == size(ys) 

    # Compute the nuclear norm for each matrix in ys_hat
    #sq_ys_hat = reshape(ys_hat, l, n, nb_examples)
    @inbounds nnm = [sum(abs.(svdvals(ys_hat[:,i]))) for i in 1:nb_examples]

    # Create diff matrix, n2 x nb_examples;
    # Multiply by mask vector len n2 broadcasted to hide entries in each example
    hidden_diff = vec(mask_mat) .* (ys .- ys_hat)
    # Compute the L2 norm squared for each column of hidden_diff
    l2normsq = sum(hidden_diff.^2, dims=1) / datascale
    
    errors = theta * (l2normsq' / totN) .+ (1f0 - theta) * nnm / datascale
    if mode == "testing"
        RMSerrors = RMSE(ys, ys_hat) / datascale
    end
    if mode == "testing"
        return Dict("l2" => mean(errors), 
                    "RMSE" => mean(RMSerrors))
    else
        return mean(errors)
    end
end

#using BenchmarkTools
#@benchmark cpu_l2nnm_nogt_loss(activemodel, x, y, mask_mat; turns = 0)
#@benchmark recon_losses(activemodel, x, y, mask_mat; turns = 0)

function recon_losses(m::matnet, 
                    xs::Vector, 
                    ys_mat::AbstractArray{Float32},
                    mask_mat = nothing,
                    groundtruth::Bool = false; 
                    turns::Int = TURNS, 
                    mode::String = "training", 
                    incltrainloss::Bool = false, 
                    type::String = "l2nnm", 
                    theta::Float32 = 0.8f0,
                    datascale::Float32 = 0.1f0)
    @assert !isnothing(mask_mat)
    @assert ndims(ys_mat) == 3
    l, n, nb_examples = size(ys_mat)

    @assert isa(xs[1], SparseMatrixCSC) #|| isa(xs[1], CUDA.CUSPARSE.CuSparseMatrixCSC)

    ys = reshape(ys_mat, l * n, nb_examples) 
    ys_hat = predict_through_time(m, xs, turns)  

    totN = sum(vec(mask_mat))
    
    # If the ground truth is given, compare y (full ground truth matrix) to y_hat (full estimated matrix)
    # If not, compare x (masked ground truth matrix) to x_hat (masked estimated matrix)
    if CUDA.functional()
        errors = zeros(Float32, 1, nb_examples)
    else
        errors = Zygote.Buffer(zeros(Float32, 1, nb_examples))
    end
    if mode == "testing"
        if CUDA.functional()
            RMSerrors_all = zeros(Float32, 1, nb_examples)
        else
            RMSerrors_all = Zygote.Buffer(zeros(Float32, 1, nb_examples))
        end
    end
    for i in 1:nb_examples
        diff = vec(mask_mat) .* (ys[:,i] .- ys_hat[:,i])
        
        l2normsq = norm(diff, 2)^2 /datascale

        if type == "l2l1"
            l1norm = norm(diff, 1)
            @inbounds errors[i] = (l2normsq + l1norm) / totN
        elseif type == "l2"
            @inbounds errors[i] = l2normsq / totN
        elseif type == "l2nnm"
            nnm = Float32.(nuclearnorm(ys_hat[:,i]) / l)
            @inbounds errors[i] = (theta * (l2normsq / totN) + (1f0 - theta) * nnm) /datascale
        else 
            error("Invalid type of loss function declared.")
        end
        if mode == "testing"
            @inbounds RMSerrors_all[i] = RMSE(ys[:,i], ys_hat[:,i]) /datascale
        end
    end
    # When training, return only the training loss for gradient computation
    if mode == "training"
        return mean(copy(errors))
    elseif mode == "testing"
        # When testing or validating, return all relevant metrics
        return Dict("l2" => mean(copy(l2errors)), 
                    "RMSE" => mean(copy(RMSerrors_all)))
    end
end
# Include spectral distance with ground truth (in all cases)
#yhat_mat = reshape(ys_hat, l, n, nb_examples)
#spectdists[i] = spectdist(ys_mat[:,:,i], yhat_mat[:,:,i])

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
