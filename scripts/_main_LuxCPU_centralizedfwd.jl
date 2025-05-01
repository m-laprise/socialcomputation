#= 
Julia 1.10.5 
Lux 1.12.4
Enzyme 0.13.38
Mooncake 0.4.117
=#
if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using Lux
#using Enzyme
using LinearAlgebra
using Random
using ChainRules
using DifferentiationInterface
import Mooncake
import NNlib
import Distributions: Dirichlet, mean, std, var
import ParameterSchedulers: CosAnneal, Stateful, next!
import WeightInitializers: glorot_uniform, glorot_normal, zeros32
import MLUtils: DataLoader, splitobs
import Optimisers: Adam, setup, adjust!, update!

include("rnncells_LuxCPU.jl")
include("lossfunctions_LuxCPU.jl")
include("train_utils_LuxCPU.jl")

BLAS.set_num_threads(4)

# Hyperparameters and constants
const N::Int = 20
const NETWIDTH::Int = N^2 
const HIDDEN_DIM::Int = 1
const RANK::Int = 1
const DATASETSIZE::Int = 2000
const KNOWNENTRIES::Int = Int(round(0.8*N^2))
DEC_RANK::Int = 1

const MINIBATCH_SIZE::Int = 32
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

INIT_ETA = 1f-3
END_ETA = 1f-4
DECAY = 0.7f0
ETA_PERIOD = 200

EPOCHS::Int = 100
TURNS::Int = 0

THETA::Float32 = 0.5f0
mainloss = normal_l2 #spectrum_penalized_l2

datarng = Random.MersenneTwister(Int(round(time())))
trainrng = Random.MersenneTwister(0)

taskfilename = "$(N)recon_rank$(RANK)"
tasklab = "Reconstructing $(N)x$(N) rank-$(RANK) matrices from $(KNOWNENTRIES) of their entries"
modlabel = "Centralized Attention Layers"

# Helper functions for data 
_3D(y) = reshape(y, N, N, size(y, 2))
_2D(y) = reshape(y, N*N, size(y, 3))
_3Dslices(y) = eachslice(_3D(y), dims=3)
l(f, y) = mean(f.(_3Dslices(y)))

#====CREATE DATA====#

include("datacreation_LuxCPU.jl")

dataX, dataY, mask = datasetgeneration_centralized(N, N, RANK, DATASETSIZE, KNOWNENTRIES, datarng)
const nonzeroidx = findall((vec(mask)) .!= 0)

size(dataX), size(dataY)

dataX, valX, testX = splitobs(datarng, dataX, at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
dataY, valY, testY = splitobs(datarng, dataY, at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
size(dataX), size(dataY)

const dataloader = DataLoader(
    (data = dataX, label = dataY), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true, parallel=true, rng=datarng)
@info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)

# Get reference values from ground truth
const refspectralnorm = mean(spectralnorm.(eachslice(dataY, dims = 3)))
const refspectralgap = mean(spectralgap.(eachslice(dataY, dims = 3)))
const refnuclearnorm = mean(nuclearnorm.(eachslice(dataY, dims = 3)))
const maxnn = maximum(nuclearnorm.(eachslice(dataY, dims = 3)))
const refspectralgapovernorm = refspectralgap / refspectralnorm
const refvariance = var(dataY)

#====INITIALIZE MODEL====#

function predict_through_time(st, ps, m::Chain,
                              xs::AbstractArray{Float32, 3}, turns; n = N)::AbstractArray{Float32, 2}
    preds = Array{Float32}(undef, n^2, size(xs, 3))
    populatepreds!(preds, st, ps, m, xs, turns)
    return preds
end

function Luxapply!(st, ps, m::A, x; 
        selfreset::Bool = false, 
        turns::Int = 1) where A <: Lux.Chain
    Lux.apply(m, x, ps, st)[1]
end

function sparsity_penalty(ps; λ=10f-4, power=1f0)
    ps = destructure(ps)
    # Extract weights (excluding biases)
    all_weights = []
    for p in ps
        if ndims(p) > 1 && size(p, 1) > 1
            push!(all_weights, vec(p))
        end
    end
    if isempty(all_weights)
        return 0f0
    end
    weights_concat = abs.(vcat(all_weights...))
    penalty = λ * sum(weights_concat .^ power)
    return penalty
end

function trainingloss(m::C, ps, st, (xs, ys, nonzeroidx, turns))::Float32 where C <: Chain
    ys_hat = predict_through_time(st, ps, m, xs, turns)
    return mainloss(ys, ys_hat, nonzeroidx) + sparsity_penalty(ps; λ=10f-4, power=3f0)
end

struct DecodingLayer{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n::Int
    r::Int
    init::F1
    init_zeros::F2
    init_ones::F3
end

function DecodingLayer(k::Int, n::Int, r::Int; 
                       init=glorot_uniform, 
                       init_zeros=zeros32, 
                       init_ones=ones32)
    DecodingLayer{typeof(init), typeof(init_zeros), typeof(init_ones)}(
        k, n, r, init, init_zeros, init_ones
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::DecodingLayer)
    (Wu=l.init(rng, l.n, l.k; gain = 0.1f0),
    Wv=l.init(rng, l.n, l.k; gain = 0.1f0),
    #Wv=l.init(rng, l.n; gain = 1f0),
    γ=l.init_ones(1), 
    α=l.init_ones(1),
    β=l.init_zeros(1))
end
Lux.initialstates(::AbstractRNG, ::DecodingLayer) = NamedTuple()

function (l::DecodingLayer)(cellh, ps, st)
    u = ps.Wu * cellh
    v = ps.Wv * cellh
    ans = u * v'
    #DyT!(ans, ps.γ[1], ps.α[1], ps.β[1])
    #mylayernorm!(ans, ps.γ[1], ps.β[1])
    return vec(ans), st
end

#= activemodel = Chain(
    x -> reshape(x, N^2, 1),
    SkipConnection(Chain(
        Dense(N^2, NETWIDTH, tanh),
        LayerNorm((NETWIDTH,)),
        SkipConnection(
            Dense(NETWIDTH, NETWIDTH, tanh),+), 
        LayerNorm((NETWIDTH,)),
        #SkipConnection(
        #    Dense(NETWIDTH, NETWIDTH, tanh),+),
        #LayerNorm((NETWIDTH,)),
        DecodingLayer(NETWIDTH, N, DEC_RANK),
        x -> reshape(x, N^2, 1),
        LayerNorm((N^2,)),
    ), +),
    DecodingLayer(N^2, N, DEC_RANK),
    x -> reshape(x, N^2, 1),
    #LayerNorm((N^2,)),
) =#

#=activemodel = Chain(
    SkipConnection(
        Chain(
            MultiHeadAttention(N => NETWIDTH => N),
            x -> x[1],
            LayerNorm((N,)),
            MultiHeadAttention(N => NETWIDTH => N),
            x -> x[1],
            LayerNorm((N,)),
        ),+),
    SkipConnection(
        Chain(
            Dense(N, N, relu),
            LayerNorm((N,)),
        ),+),
    SkipConnection(
        Chain(
            MultiHeadAttention(N => NETWIDTH => N),
            x -> x[1],
            LayerNorm((N,)),
            MultiHeadAttention(N => NETWIDTH => N),
            x -> x[1],
            LayerNorm((N,)),
        ),+),
    SkipConnection(
        Chain(
            Dense(N, N, relu),
            LayerNorm((N,)),
        ),+),
    x -> vec(x),
    DecodingLayer(N^2, N, DEC_RANK),
)=#

activemodel = Chain(
    SkipConnection(
        Chain(
            MultiHeadAttention(N => NETWIDTH => N; nheads=2),
            x -> x[1],
            LayerNorm((N,)),
            MultiHeadAttention(N => NETWIDTH => N; nheads=2),
            x -> x[1],
            LayerNorm((N,)),
        ),+),
    Dense(N, N, swish),
    LayerNorm((N,)),
    x -> vec(x),
    DecodingLayer(N^2, N, DEC_RANK),
)

# elu: 10, 21; 28-27
# gelu_tanh: 11, 21; 28-27
# lisht: 09, 19; 26-25
# relu6: 10, 18; 27-26
# softplus: 11, 18; 26-26
# gelu_erf: 09, 18; 26-25
# mish: 13, 17; 26-26
# swish: 10, 15; 26-24

ps, st = Lux.setup(trainrng, activemodel)
activemodel(dataX[:,:,1], ps, st)[1] #[1]

#display(activemodel)
nparams::Int = LuxCore.parameterlength(activemodel)
println("Parameter Length: ", nparams, "; State Length: ",
    LuxCore.statelength(activemodel))


X = dataX[:,:,1]
Y = Luxapply!(st, ps, activemodel, X)
println("Output variance at init: ", var(Y))

# Optimizer
opt = Adam(INIT_ETA)
opt_state = setup(opt, ps)

# Learning rate schedule: cosine annealing (with warm restarts)
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

#push!(train_metrics[:Whh_spectra], eigvals(ps.cell.Whh))
recordmetrics!(train_metrics, st, ps, activemodel, dataX, dataY, nonzeroidx, TURNS)
recordmetrics!(val_metrics, st, ps, activemodel, valX, valY, nonzeroidx, TURNS, split="val")

##================##

#reset!(stateful_s)

main_training_loop!(activemodel, ps, st, opt_state, stateful_s,
                    train_metrics, val_metrics, 
                    dataloader, dataX, dataY, valX, valY,
                    nonzeroidx, TURNS, EPOCHS, MINIBATCH_SIZE)

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
K = NETWIDTH
fig = main_training_figure(
    train_metrics, val_metrics, test_metrics, 
    dataX, tasklab, modlabel, 
    HIDDEN_DIM, DEC_RANK, INIT_ETA, 
    ETA_PERIOD, MINIBATCH_SIZE, K, TURNS, nparams
)

save("data/$(taskfilename)_$(modlabel)width$(K)"*
     "_knownentries_uvDec_l2(CosAn)_1x2x2.png", fig)


ploteigvals(ps.cell.Whh)
save("data/$(taskfilename)_$(modlabel)width$(K)_$(TURNS)turns"*
     "_knownentries_uvDec_l2(CosAn)_Whheigvals.png", 
     ploteigvals(ps.cell.Whh))

include("inprogress/helpersWhh.jl")
g_end = adj_to_graph(ps.cell.Whh; threshold = 0.01)
figdegdist = plot_degree_distrib(g_end)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(TURNS)turns"*
     "_knownentries_uvDec_l2(CosAn)_degdist.png", figdegdist)

# Save Whh
using JLD2
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(TURNS)turns"*
     "_knownentries_uvDec_l2(CosAn)_Whh.jld2", "Whh", 
    ps.cell.Whh)

