#== Helper functions for inference ==#

    "Populate a matrix with the predictions of the RNN for each example in a 3D array, with no time steps."
    function populatepreds!(preds, st, ps, m, xs::AbstractArray{Float32, 3})::Nothing
        @inbounds for i in axes(xs, 3)
            example = @view xs[:,:,i]
            preds[:,i] .= Luxapply!(st, ps, m, example; selfreset = true)
        end
    end
    
    "Populate a matrix with the predictions of the RNN for each example in a 3D array, with a given number of time steps."
    function populatepreds!(preds, st, ps, m, xs::AbstractArray{Float32, 3}, turns)::Nothing
        @inbounds for i in axes(xs, 3)
            reset!(st, m)
            example = @view xs[:,:,i]
            preds[:,i] .= Luxapply!(st, ps, m, example; selfreset = false, turns = turns)
        end
    end
    
    "Predict the output for a single input matrix, with no time steps."
    function predict_through_time(st, ps, m, 
                                  x::AbstractArray{Float32, 2})::AbstractArray{Float32, 2}
        preds = Luxapply!(st, ps, m, x; selfreset = true)
        return reshape(preds, :, 1)
    end
    
    "Predict the output for a single input matrix, with a given number of time steps."
    function predict_through_time(st, ps, m, 
                                  x::AbstractArray{Float32, 2}, 
                                  turns::Int)::AbstractArray{Float32, 2}
        if st.cell.H != st.cell.init
            reset!(st, m)
        end
        preds = Luxapply!(st, ps, m, x; selfreset = false, turns = turns)
        return reshape(preds, :, 1)
    end
    
    "Predict the outputs for an array of input matrices, with no time steps."
    function predict_through_time(st, ps, m::ComposedRNN, 
                                  xs::AbstractArray{Float32, 3})::AbstractArray{Float32, 2}
        preds = Array{Float32}(undef, m.cell.n2, size(xs, 3))
        populatepreds!(preds, st, ps, m, xs)
        return preds
    end
    
    "Predict the outputs for an array of input matrices, with a given number of time steps."
    function predict_through_time(st, ps, m::ComposedRNN, 
                                  xs::AbstractArray{Float32, 3}, 
                                  turns::Int)::AbstractArray{Float32, 2}
        preds = Array{Float32}(undef, m.cell.n2, size(xs, 3))
        populatepreds!(preds, st, ps, m, xs, turns)
        return preds
    end
    
    # Wrapper for prediction and loss
    "Compute predictions with no time steps, and use them to compute the training loss."
    function trainingloss(m, ps, st, (xs, ys, nonzeroidx))::Float32
        ys_hat = predict_through_time(st, ps, m, xs)
        return mainloss(ys, ys_hat, nonzeroidx)
    end
    "Compute predictions with a given number of time steps, and use them to compute the training loss."
    function trainingloss(m, ps, st, (xs, ys, nonzeroidx, turns))::Float32
        ys_hat = predict_through_time(st, ps, m, xs, turns)
        return mainloss(ys, ys_hat, nonzeroidx)
    end
    
    # Rules for autodiff backend
    #EnzymeRules.inactive(::typeof(reset!), args...) = nothing
    Enzyme.@import_rrule typeof(svdvals) AbstractMatrix{<:Number}
    #Enzyme.@import_rrule typeof(svd) AbstractMatrix{<:Number}


#== Helper functions for training ==#

# Compute the initial training and validation loss and other metrics with forward passes
function recordmetrics!(metricsdict, st, ps, activemodel, X, Y, nonzeroidx, TURNS, subset=200; split="train")
    if split == "test"
        subset = size(X, 3)
    end
    reset!(st, activemodel)
    Ys_hat = predict_through_time(st, ps, activemodel, X[:,:,1:subset], TURNS)
    push!(metricsdict[:loss], mainloss(Y[:,:,1:subset], Ys_hat, nonzeroidx))
    push!(metricsdict[:all_rmse], batchmeanloss(RMSE, Y[:,:,1:subset], Ys_hat))
    push!(metricsdict[:all_mae], batchmeanloss(MAE, Y[:,:,1:subset], Ys_hat))
    push!(metricsdict[:known_rmse], meanknownentriesloss(RMSE, Y[:,:,1:subset], Ys_hat, nonzeroidx))
    push!(metricsdict[:known_mae], meanknownentriesloss(MAE, Y[:,:,1:subset], Ys_hat, nonzeroidx))
    if split == "train"
        push!(metricsdict[:nuclearnorm], l(nuclearnorm, Ys_hat))
        push!(metricsdict[:spectralnorm], l(spectralnorm, Ys_hat))
        push!(metricsdict[:spectralgap], l(spectralgap, Ys_hat))
        push!(metricsdict[:variance], var(Ys_hat))
    end
    if split == "test"
        return Ys_hat
    end
end

function inspect_and_repare_gradients!(grads, ::MatrixVlaCell)
    g = [grads.cell.Wx_in, grads.cell.Whh, grads.cell.Bh, 
         grads.dec.Wx_out]
    tot = sum(length.(g))
    nan_params = sum(sum(isnan, gi) for gi in g)
    vanishing_params = sum(sum(abs.(gi) .< 1f-6) for gi in g)
    exploding_params = sum(sum(abs.(gi) .> 1f6) for gi in g)
    if nan_params > 0
        for gi in g
            gi[isnan.(gi)] .= 0f0
        end
        @warn("$(round(nan_params/tot*100, digits=0)) % NaN gradients detected and replaced with 0.")
    end
    return vanishing_params/tot, exploding_params/tot
end
function inspect_and_repare_gradients!(grads, ::MatrixGatedCell)
    g = [grads.cell.Wx_in, grads.cell.Whh, grads.cell.Bh,
         grads.cell.Wa, grads.cell.Wah, grads.cell.Wax, grads.cell.Ba,
         grads.dec.Wx_out]
    tot = sum(length.(g))
    nan_params = sum(sum(isnan, gi) for gi in g)
    vanishing_params = sum(sum(abs.(gi) .< 1f-6) for gi in g)
    exploding_params = sum(sum(abs.(gi) .> 1f6) for gi in g)
    if nan_params > 0
        for gi in g
            gi[isnan.(gi)] .= 0f0
        end
        @warn("$(round(nan_params/tot*100, digits=0)) % NaN gradients detected and replaced with 0.")
    end
    return vanishing_params/tot, exploding_params/tot
end
function inspect_and_repare_gradients!(grads, ::MatrixGatedCell2)
    g = [grads.cell.Whh, grads.cell.Bh,
         grads.cell.Wah, grads.cell.Wax, grads.cell.Ba,
         grads.dec.U, grads.dec.V]
    tot = sum(length.(g))
    nan_params = sum(sum(isnan, gi) for gi in g)
    vanishing_params = sum(sum(abs.(gi) .< 1f-6) for gi in g)
    exploding_params = sum(sum(abs.(gi) .> 1f6) for gi in g)
    if nan_params > 0
        for gi in g
            gi[isnan.(gi)] .= 0f0
        end
        @warn("$(round(nan_params/tot*100, digits=0)) % NaN gradients detected and replaced with 0.")
    end
    return vanishing_params/tot, exploding_params/tot
end

function diagnose_gradients(v, e)
    if v >= 0.1
        @info("$(round(v*100, digits=0)) % vanishing gradients detected")
    end
    if e >= 0.1
        @info("$(round(e*100, digits=0)) % exploding gradients detected")
    end
    if v < 0.1 && e < 0.1
        @info("Gradients well-behaved.")
    end
end

function inspect_and_repare_ps!(ps)
    if sum(sum(isnan, p) for p in ps.cell) > 0
        for p in ps.cell
            p[isnan.(p)] .= 0f0
        end
        @warn("$(sum(sum(isnan, p))) NaN parameters detected in CELL after update and replaced with 0.")
    end
    if sum(sum(isnan, p) for p in ps.dec) > 0
        for p in ps.dec
            p[isnan.(p)] .= 0f0
        end
        @warn("$(sum(sum(isnan, p))) NaN parameters detected in DEC after update and replaced with 0.")
    end
end
