#= Testing using: 
Julia 1.10.5 
Flux 0.16.3
Enzyme 0.13.30=#

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

include("rnncells_LuxCPU.jl")
include("lossfunctions_LuxCPU.jl")
include("train_utils_LuxCPU.jl")

BLAS.set_num_threads(4)

# Hyperparameters and constants
const K::Int = 200
const N::Int = 64
const HIDDEN_DIM::Int = 16
const RANK::Int = 1
const DATASETSIZE::Int = 8000
const KNOWNENTRIES::Int = 1640
DEC_RANK::Int = 1

const MINIBATCH_SIZE::Int = 64
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

INIT_ETA = 1f-3
END_ETA = 1f-6
DECAY = 0.7f0
ETA_PERIOD = 10

EPOCHS::Int = 15
TURNS::Int = 50

THETA::Float32 = 0.9f0
mainloss = spectrum_penalized_huber

datarng = Random.MersenneTwister(Int(round(time())))
trainrng = Random.MersenneTwister(0)

taskfilename = "$(N)recon_rank$(RANK)"
tasklab = "Reconstructing $(N)x$(N) rank-$(RANK) matrices from $(KNOWNENTRIES) of their entries"
modlabel = "Matrix Gated"

# Helper functions for data 
_3D(y) = reshape(y, N, N, size(y, 2))
_2D(y) = reshape(y, N*N, size(y, 3))
_3Dslices(y) = eachslice(_3D(y), dims=3)
l(f, y) = mean(f.(_3Dslices(y)))

#====CREATE DATA====#
include("datacreation_LuxCPU.jl")

dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng)
const maskmatrix = masktuple2array(masktuples, N, N)
const nonzeroidx = findall((vec(maskmatrix)) .!= 0)

size(dataX), size(dataY)

dataX, valX, testX = splitobs(datarng, dataX, at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
dataY, valY, testY = splitobs(datarng, dataY, at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
size(dataX), size(dataY)

const dataloader = DataLoader(
    (data = dataX, label = dataY), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true, parallel=true, rng=datarng)
@info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)

# Get reference value from ground truth
const refspectralnorm = mean(spectralnorm.(eachslice(dataY, dims = 3)))
const refspectralgap = mean(spectralgap.(eachslice(dataY, dims = 3)))
const refnuclearnorm = mean(nuclearnorm.(eachslice(dataY, dims = 3)))
const maxnn = maximum(nuclearnorm.(eachslice(dataY, dims = 3)))
const refspectralgapovernorm = refspectralgap / refspectralnorm
const refvariance = var(dataY)

#====INITIALIZE MODEL====#
activemodel = ComposedRNN(
    MatrixGatedCell2(K, N^2, HIDDEN_DIM, knowledgedistr), 
    FactorDecodingLayer(K, N, HIDDEN_DIM, DEC_RANK)
)
ps, st = Lux.setup(trainrng, activemodel)

#display(activemodel)
println("Parameter Length: ", LuxCore.parameterlength(activemodel), "; State Length: ",
    LuxCore.statelength(activemodel))

# In the new setup, DOES NOT work with random matrix not exactly nl sparse in each column
X = dataX[:,:,1]
#y = activemodel(x, ps, st)[1]
Y = Luxapply!(st, ps, activemodel, X; selfreset=false, turns=20)

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

push!(train_metrics[:Whh_spectra], eigvals(ps.cell.Whh))
recordmetrics!(train_metrics, st, ps, activemodel, dataX, dataY, nonzeroidx, TURNS)
recordmetrics!(val_metrics, st, ps, activemodel, valX, valY, nonzeroidx, TURNS, split="val")

##================##

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
        #inspect_and_repare_ps!(opt_state.parameters)
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
    if length(val_metrics[:loss]) > 2
        if val_metrics[:loss][end] > val_metrics[:loss][end-1] && val_metrics[:loss][end-1] > val_metrics[:loss][end-2] && val_metrics[:loss][end-2] > val_metrics[:loss][end-3]
            @warn("Early stopping at epoch $epoch")
            break
        end
    end
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

include("plot_utils.jl")

fig = main_training_figure(
    train_metrics, val_metrics, test_metrics, 
    dataX, tasklab, modlabel, 
    HIDDEN_DIM, DEC_RANK, INIT_ETA, 
    ETA_PERIOD, MINIBATCH_SIZE, K, TURNS
)

save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_FactorR1Dec_pHuber(CosAn-theta90-sn).png", fig)


ploteigvals(ps.cell.Whh)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_FactorR1Dec_pHuber(CosAn-theta100)_Whheigvals.png", 
     ploteigvals(ps.cell.Whh))

include("inprogress/helpersWhh.jl")
g_end = adj_to_graph(ps.cell.Whh; threshold = 0.01)
figdegdist = plot_degree_distrib(g_end)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_FactorR1Dec_pHuber(CosAn-theta100)_degdist.png", figdegdist)

# Save Whh
using JLD2
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_FactorR1Dec_pHuber(CosAn-theta100)_Whh.jld2", "Whh", 
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