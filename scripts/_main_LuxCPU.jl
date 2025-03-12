#= Testing using: 
Julia 1.10.5 
Flux 0.16.3
Enzyme 0.13.30
=#
if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using Lux, Enzyme
using LinearAlgebra
using Random
using ChainRules
import NNlib
import Distributions: Dirichlet, mean, std, var
import ParameterSchedulers: CosAnneal, Stateful, next!
import WeightInitializers: glorot_uniform, glorot_normal, zeros32
import MLUtils: DataLoader, splitobs
import Optimisers: Adam, adjust!

BLAS.set_num_threads(4)

# Hyperparameters and constants

const K::Int = 100
const N::Int = 64
const HIDDEN_DIM::Int = 8
const RANK::Int = 1
const DATASETSIZE::Int = 8000
const KNOWNENTRIES::Int = 1600

const MINIBATCH_SIZE::Int = 64
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

INIT_ETA = 1f-3
END_ETA = 1f-6
DECAY = 0.7f0
ETA_PERIOD = 10

EPOCHS::Int = 10
TURNS::Int = 50

THETA::Float32 = 0.9f0

datarng = Random.MersenneTwister(Int(round(time())))
trainrng = Random.MersenneTwister(0)

include("rnncells_LuxCPU.jl")
include("lossfunctions_LuxCPU.jl")

# Helper functions for data 
_3D(y) = reshape(y, N, N, size(y, 2))
_2D(y) = reshape(y, N*N, size(y, 3))
_3Dslices(y) = eachslice(_3D(y), dims=3)
l(f, y) = mean(f.(_3Dslices(y)))

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
function trainingloss(m, ps, st, (xs, ys, nonzeroidx))
    ys_hat = predict_through_time(st, ps, m, xs)
    return spectrum_penalized_huber(ys, ys_hat, nonzeroidx)
    #return allentriesMAE(ys, ys_hat)
end
"Compute predictions with a given number of time steps, and use them to compute the training loss."
function trainingloss(m, ps, st, (xs, ys, nonzeroidx, turns))
    ys_hat = predict_through_time(st, ps, m, xs, turns)
    return spectrum_penalized_huber(ys, ys_hat, nonzeroidx)
    #return allentriesMAE(ys, ys_hat)
end

# Rules for autodiff backend
#EnzymeRules.inactive(::typeof(reset!), args...) = nothing
Enzyme.@import_rrule typeof(svdvals) AbstractMatrix{<:Number}
#Enzyme.@import_rrule typeof(svd) AbstractMatrix{<:Number}

#====CREATE DATA====#

function lowrankmatrix(m, n, r, rng; datatype=Float32, std::Float32=1f0)
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
    X = zeros(Float32, m*n, k, dataset_size)
    populateX!(X, Y, knowledgedistribution, masktuples)
    return X, Y, masktuples
end

dataX, dataY, masktuples = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng)
const maskmatrix = masktuple2array(masktuples, N, N)
const nonzeroidx = findall((vec(maskmatrix)) .!= 0)
# (Initially tried to create X with sparse arrays, but it made pullbacks slower
# and I am not RAM constrained, so focusing on regular arrays for now)

size(dataX), size(dataY)

dataX, valX, testX = splitobs(datarng, dataX, at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
dataY, valY, testY = splitobs(datarng, dataY, at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
size(dataX), size(dataY)
dataloader = DataLoader(
    (data = dataX, label = dataY), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true, parallel=true, rng=datarng)
@info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)

# Get reference value from ground truth
refspectralnorm = mean(spectralnorm.(eachslice(dataY, dims = 3)))
refspectralgap = mean(spectralgap.(eachslice(dataY, dims = 3)))
refnuclearnorm = mean(nuclearnorm.(eachslice(dataY, dims = 3)))
maxnn = maximum(nuclearnorm.(eachslice(dataY, dims = 3)))
refspectralgapovernorm = refspectralgap / refspectralnorm
refvariance = var(dataY)

#====INITIALIZE MODEL====#
activemodel = ComposedRNN(
    MatrixGatedCell(K, N^2, HIDDEN_DIM), 
    N2DecodingLayer(K, N^2, HIDDEN_DIM)
)
ps, st = Lux.setup(trainrng, activemodel)

#display(activemodel)
println("Parameter Length: ", LuxCore.parameterlength(activemodel), "; State Length: ",
    LuxCore.statelength(activemodel))

X = randn(datarng, Float32, N^2, K)
#y = activemodel(x, ps, st)[1]
Y = Luxapply!(st, ps, activemodel, X; selfreset=true, turns=20)

# Optimizer
opt = Adam(INIT_ETA)
opt_state = Training.TrainState(activemodel, ps, st, opt)

# Learning rate schedule

# cosine annealing (with warm restarts)
s = CosAnneal(INIT_ETA, END_ETA, ETA_PERIOD, true)

Base.IteratorSize(s)
println("Learning rate schedule:")
for i in 1:EPOCHS
    println(" - Epoch $(i): eta = $(s(i))")
end
stateful_s = Stateful(s)

# STORE INITIAL METRICS
train_metrics = Dict(
    :loss => Float32[],
    :all_rmse => Float32[],
    :all_mae => Float32[],
    :known_rmse => Float32[],
    :known_mae => Float32[],
    :nuclearnorm => Float32[],
    :spectralnorm => Float32[],
    :spectralgap => Float32[],
    :variance => Float32[],
    :Whh_spectra => []
)
val_metrics = Dict(
    :loss => Float32[],
    :all_rmse => Float32[],
    :all_mae => Float32[],
    :known_rmse => Float32[],
    :known_mae => Float32[],
)
# Compute the initial training and validation loss and other metrics with forward passes
function recordmetrics!(metricsdict, st, ps, activemodel, X, Y, nonzeroidx, TURNS, subset=200; split="train")
    if split == "test"
        subset = size(X, 3)
    end
    reset!(st, activemodel)
    Ys_hat = predict_through_time(st, ps, activemodel, X[:,:,1:subset], TURNS)
    push!(metricsdict[:loss], spectrum_penalized_huber(Y[:,:,1:subset], Ys_hat, nonzeroidx))
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
push!(train_metrics[:Whh_spectra], eigvals(ps.cell.Whh))
recordmetrics!(train_metrics, st, ps, activemodel, dataX, dataY, nonzeroidx, TURNS)
recordmetrics!(val_metrics, st, ps, activemodel, valX, valY, nonzeroidx, TURNS, split="val")

##================##
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

#reset!(stateful_s)
starttime = time()
println("===================")
println("Initial training loss: " , train_metrics[:loss][1])
println("Initial training MAE, all entries: ", train_metrics[:all_mae][1], 
        "; known entries: ", train_metrics[:known_mae][1])
@info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
for epoch in 1:EPOCHS
    reset!(st, activemodel)
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
            Const((x, y, nonzeroidx, TURNS)))
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
        inspect_and_repare_ps!(opt_state.parameters)
        mb += 1
    end
    # Compute training metrics -- expensive operation with a forward pass over the entire training set
    # but we restrict to two mini-batches only
    recordmetrics!(train_metrics, st, ps, activemodel, dataX, dataY, nonzeroidx, TURNS)
    push!(train_metrics[:Whh_spectra], eigvals(ps.cell.Whh))
    # Compute validation metrics
    recordmetrics!(val_metrics, st, ps, activemodel, valX, valY, nonzeroidx, TURNS, split="val")
    println("Epoch ", epoch, ": Train loss: ", train_metrics[:loss][end], "; Val loss: ", val_metrics[:loss][end])
    println("Train MAE, all entries: ", train_metrics[:all_mae][end], "; known entries: ", train_metrics[:known_mae][end])
    println("Val MAE, all entries: ", val_metrics[:all_mae][end], "; known entries: ", val_metrics[:known_mae][end])
    # Check if validation loss has increased for 2 epochs in a row; if so, stop training
    #=if length(val_metrics[:loss]) > 2
        if val_metrics[:loss][end] > val_metrics[:loss][end-1] && val_metrics[:loss][end-1] > val_metrics[:loss][end-2] && val_metrics[:loss][end-2] > val_metrics[:loss][end-3]
            @warn("Early stopping at epoch $epoch")
            break
        end
    end=#
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Assess on testing set
test_metrics = Dict(:loss => Float32[], 
                    :all_rmse => Float32[], 
                    :all_mae => Float32[],
                    :known_rmse => Float32[],
                    :known_mae => Float32[])
testYs_hat = recordmetrics!(test_metrics, st, ps, activemodel, testX, testY, nonzeroidx, TURNS, split="test")


##############################
#======explore results=======#
##############################
var(dataY), var(testYs_hat)

reshape(testYs_hat[:,10], N, N)

using CairoMakie
#plot(svdvals(dataY[:,:,10]))
plot(svdvals(reshape(testYs_hat[:,10], N, N)))
#plot(svdvals(rand(Float32, N, 5) * rand(Float32, 5, N) .+ 0.1f0 * rand(Float32, N, N)))

batchmeanloss(RMSE, testY, testYs_hat)

#======generate training plots=======#

using CairoMakie

taskfilename = "$(N)recon_rank$(RANK)"
tasklab = "Reconstructing $(N)x$(N) rank-$(RANK) matrices from $(KNOWNENTRIES) of their entries"
modlabel = "Matrix Gated"

fig = Figure(size = (850, 1200))
epochs = length(val_metrics[:loss])
train_metrics[:spectralgapovernorm] = train_metrics[:spectralgap] ./ train_metrics[:spectralnorm]
train_metrics[:logloss] = log.(train_metrics[:loss])
val_metrics[:logloss] = log.(val_metrics[:loss])
test_metrics[:logloss] = log.(test_metrics[:loss])
metrics = [
    (1, 1:2, "Log Loss", "logloss", "Spectral-gap and norm penalized Huber / std (known entries)"),
    #(1, 2, "RMSE", "all_rmse", "RMSE / std (all entries)"),
    (2, 1, "MAE", "known_mae", "MAE / std (known entries)"),
    (2, 2, "MAE", "all_mae", "MAE / std (all entries)"),
    (3, 1, "Spectral gap / spectral norm", "spectralgapovernorm", "Mean spectral gap / mean spectral norm"),
    (3, 2, "Spectral gap", "spectralgap", "Mean spectral gap"),
    (4, 1, "Nuclear norm", "nuclearnorm", "Mean nuclear norm"),
    (4, 2, "Variance", "variance", "Mean variance of matrix entries")
]
for (row, col, ylabel, key, title) in metrics
    ax = Axis(fig[row, col], xlabel = "Epochs", ylabel = ylabel, title = title)
    if key in ["logloss"]
        lines!(ax, [i for i in range(1, epochs-1, length(train_metrics[Symbol(key)][length(dataloader)+1:end]))], 
               train_metrics[Symbol(key)][length(dataloader)+1:end], 
               color = :blue, label = "Training")
        scatter!(ax, 1:epochs-1, val_metrics[Symbol(key)][2:end], 
               color = :red, markersize = 4, label = "Validation")
        lines!(ax, 1:epochs-1, [test_metrics[Symbol(key)][end]], color = :green, linestyle = :dash)
        scatter!(ax, epochs-1, test_metrics[Symbol(key)][end], color = :green, label = "Final Test")
    elseif key in ["known_mae", "all_mae", "all_rmse"]
        lines!(ax, 1:epochs-1, train_metrics[Symbol(key)][2:end], 
        color = :blue, label = "Training")
        lines!(ax, 1:epochs-1, val_metrics[Symbol(key)][2:end], 
        color = :red, label = "Validation")
        lines!(ax, 1:epochs-1, [test_metrics[Symbol(key)][end]], color = :green, linestyle = :dash)
        scatter!(ax, epochs-1, test_metrics[Symbol(key)][end], color = :green, label = "Final Test")
        #lines!(ax, 1:epochs-1, [1-(KNOWNENTRIES / N^2)], color = :black, linestyle = :dash, label = "Knowledge Threshold*")
    else
        lines!(ax, 1:epochs-1, train_metrics[Symbol(key)][2:end], 
        color = :blue, label = "Reconstructed")
        lines!(ax, 1:epochs-1, [eval(Symbol("ref$key"))], color = :orange, linestyle = :dash, label = "Mean ground truth")
    end
    axislegend(ax, backgroundcolor = :transparent)
end
Label(
    fig[begin-1, 1:2],
    "$(tasklab)\n$(modlabel) RNN of $K units, $TURNS dynamic steps"*
    "\nwith training loss based on known entries",
    fontsize = 20,
    padding = (0, 0, 0, 0),
)
Label(
    fig[end+1, 1:2],
    "Optimizer: Adam with schedule CosAnneal(start = $(INIT_ETA), period = $(ETA_PERIOD))\n"*
    "for $(epochs-1) epochs over $(size(dataX, 3)) examples, minibatch size $(MINIBATCH_SIZE).\n"*
    "Hidden internal state dimension: $(HIDDEN_DIM).\n"*
    "Test loss (known entries): $(round(test_metrics[:loss][end], digits=4)). "*
    "Test MAE (known entries): $(round(test_metrics[:known_mae][end], digits=4)).\n"*
    "Test MAE (all entries): $(round(test_metrics[:all_mae][end], digits=4)).\n",
    #"The knowledge threshold is the MAE that would result from zero error for the\n"*
    #"known entries and errors equal to the standard deviation for unknown entries.",
    fontsize = 14,
    padding = (0, 0, 0, 0),
)
fig

save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_WvecX_pHuber(CosAn-theta100).png", fig)


include("plot_utils.jl")
ploteigvals(ps.cell.Whh)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_WvecX_pHuber(CosAn-theta100)_Whheigvals.png", 
     ploteigvals(ps.cell.Whh))

include("inprogress/helpersWhh.jl")
g_end = adj_to_graph(ps.cell.Whh; threshold = 0.01)
figdegdist = plot_degree_distrib(g_end)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_WvecX_pHuber(CosAn-theta100)_degdist.png", figdegdist)

# Save Whh
using JLD2
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_WvecX_pHuber(CosAn-theta100)_Whh.jld2", "Whh", 
    ps.cell.Whh)


#====TEST FWD PASS AND LOSS====#

#using BenchmarkTools
#@btime Luxapply!($st, $ps, $activemodel, $X; selfreset=false, turns=20)
#=
starttime = time()
trainingloss_c = Reactant.@compile trainingloss(activemodel, dataX[:,:,1:2], dataY[:,:,1:2], maskmatrix)
endtime = time()
println("Compilation time: ", (endtime - starttime)/60, " minutes")
trainingloss_c = Reactant.@compile trainingloss(activemodel, dataX[:,:,1:2], dataY[:,:,1:2], maskmatrix, 2)
@benchmark trainingloss_c($activemodel, $dataX[:,:,1:32], $dataY[:,:,1:32], $maskmatrix)
=#
#==== GRADIENT COMPUTATION====#
# TRAINING LOSS
#@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), 
 #                               $dataX[:,:,1], $dataY[:,:,1], $maskmatrix)
# 122.946 ms (182 allocations: 114.95 MiB)
#=
fclosure(ps) = trainingloss(activemodel, ps, st, (dataX[:,:,1:3], dataY[:,:,1:3], maskmatrix, 2))

using DifferentiationInterface
DifferentiationInterface.value_and_gradient(fclosure, AutoEnzyme(), ps) 

using Fluxperimental, Mooncake
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(reset!), Any}
Mooncake.@mooncake_overlay norm(x) = sqrt(sum(abs2, x))
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(svdvals), AbstractMatrix{<:Number}}
dup_model = Moonduo(activemodel)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
loss, grads = Flux.withgradient(fclosure, dup_model)=#


_grads = Enzyme.make_zero(ps)
_dstates = Enzyme.make_zero(st)
_, loss = autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), 
        trainingloss,
        Const(opt_state.model), Duplicated(opt_state.parameters, _grads), Duplicated(opt_state.states, _dstates),  
        Const((dataX[:,:,1:32],dataY[:,:,1:32],maskmatrix,TURNS)))

m_test = MatrixGatedCell(K, N^2, HIDDEN_DIM)
ps_test, st_test = Lux.setup(trainrng, m_test)
m_test(X, ps_test, st_test)

_grads.cell.Wx_in
_grads.cell.Whh
_grads.cell.Bh
_grads.cell.Wa
_grads.cell.Wah
_grads.cell.Wax
_grads.cell.Ba

_grads.dec.Wx_out

inspect_and_repare_gradients!(_grads, activemodel.cell)

Training.apply_gradients!(opt_state, _grads)

inspect_and_repare_ps!(opt_state.parameters)

@btime autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), 
            $trainingloss,
            $(Const(activemodel)), $(Duplicated(ps, _grads)), $(Duplicated(st, _dstates)),  
            $(Const((dataX[:,:,1:64],dataY[:,:,1:64],maskmatrix,TURNS))))
            
#========#
function f(m, ps, st, (xs, ys, turns))
    ys_hat, _ = m(xs[:,:,1], ps, st)
    return sum(ys_hat)
end
_grads = Enzyme.make_zero(ps_test)
_dstates = Enzyme.make_zero(st_test)
_, loss = autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), 
    f,
    Const(m_test), Duplicated(ps_test, _grads), Duplicated(st_test, _dstates),  
    Const((dataX[:,:,1:32],dataY[:,:,1:32],TURNS)))