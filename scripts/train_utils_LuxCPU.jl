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
         grads.dec.Wu1, grads.dec.Wu2, grads.dec.Wv1, grads.dec.Wv2]
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

#== Main training loop ==#

function main_training_loop!(opt_state, stateful_s, 
                             train_metrics, val_metrics, 
                             dataloader, dataX, dataY, valX, valY,
                             nonzeroidx, turns, epochs)
    starttime = time()
    println("===================")
    println("Initial training loss: " , train_metrics[:loss][1])
    println("Initial training MAE, all entries: ", train_metrics[:all_mae][1], 
            "; known entries: ", train_metrics[:known_mae][1])
    @info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
    for epoch in 1:epochs
        reset!(opt_state.states, opt_state.model)
        eta = next!(stateful_s)
        println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
        adjust!(opt_state, eta)
        # Iterate over minibatches
        mb = 1
        for (x, y) in dataloader
            # Forward pass (to compute the loss) and backward pass (to compute the gradients)
            grads = Enzyme.make_zero(ps)
            Δstates = Enzyme.make_zero(st)
            _, train_loss_value = autodiff(
                set_runtime_activity(Enzyme.ReverseWithPrimal), 
                trainingloss, Const(opt_state.model), 
                Duplicated(opt_state.parameters, grads), 
                Duplicated(opt_state.states, Δstates),  
                Const((x, y, nonzeroidx, turns)))
            if epoch == 1 && mb == 1
                @info("Time to first gradient: $(round(time() - starttime, digits=2)) seconds")
            end
            _v, _e = inspect_and_repare_gradients!(grads, opt_state.model.cell)
            # Detect loss of Inf or NaN. Print a warning, and then skip update
            if !isfinite(train_loss_value)
                @warn "Loss is $train_loss_value on minibatch $(epoch)--$(mb)" 
                diagnose_gradients(_v, _e)
                mb += 1
                continue
            end
            # During training, use the backward pass to store the training loss after the previous epoch
            push!(train_metrics[:loss], train_loss_value)
            if mb % 25 == 0
                # Diagnose the gradients every 25 minibatches
                diagnose_gradients(_v, _e)
            end
            if mb == 1 || mb % 5 == 0
                println("Minibatch ", epoch, "--", mb, ": loss of ", round(train_loss_value, digits=4))
            end
            # Use the optimizer and grads to update the trainable parameters and the optimizer states
            Training.apply_gradients!(opt_state, grads)
            # Check for NaN parameters and replace with zeros
            #inspect_and_repare_ps!(opt_state.parameters)
            mb += 1
        end
        # Compute training metrics -- expensive operation with a forward pass over the entire training set
        # but we restrict to a few mini-batches only
        recordmetrics!(train_metrics, opt_state.states, opt_state.parameters, 
                       opt_state.model, dataX, dataY, nonzeroidx, turns)
        push!(train_metrics[:Whh_spectra], eigvals(opt_state.parameters.cell.Whh))
        # Compute validation metrics
        recordmetrics!(val_metrics, opt_state.states, opt_state.parameters, 
                       opt_state.model, valX, valY, nonzeroidx, turns, split="val")
        println("Epoch ", epoch, 
                ": Train loss: ", train_metrics[:loss][end], 
                "; Val loss: ", val_metrics[:loss][end])
        println("Train MAE, all entries: ", train_metrics[:all_mae][end], 
                "; known entries: ", train_metrics[:known_mae][end])
        println("Val MAE, all entries: ", val_metrics[:all_mae][end], 
                "; known entries: ", val_metrics[:known_mae][end])
        # Check if validation loss has increased for 2 epochs in a row; if so, stop training
        if length(val_metrics[:loss]) > 2
            if val_metrics[:loss][end] > val_metrics[:loss][end-1] && val_metrics[:loss][end-1] > val_metrics[:loss][end-2] && val_metrics[:loss][end-2] > val_metrics[:loss][end-3]
                @warn("Early stopping at epoch $epoch")
                break
            end
        end
    end
    endtime = time()
    println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
end 