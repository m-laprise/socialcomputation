#= 
Julia 1.10.5 
Lux 1.9
Enzyme 0.13.33
Mooncake 0.4.108
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
const NETWIDTH::Int = 512
const N::Int = 16
const HIDDEN_DIM::Int = 1
const RANK::Int = 1
const DATASETSIZE::Int = 1000
const KNOWNENTRIES::Int = 200
DEC_RANK::Int = 1

const MINIBATCH_SIZE::Int = 32
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

INIT_ETA = 1f-3
END_ETA = 1f-5
DECAY = 0.7f0
ETA_PERIOD = 50

EPOCHS::Int = 20
TURNS::Int = 70

THETA::Float32 = 0.5f0
mainloss = normal_l2 #spectrum_penalized_l2

datarng = Random.MersenneTwister(Int(round(time())))
trainrng = Random.MersenneTwister(0)

taskfilename = "$(N)recon_rank$(RANK)"
tasklab = "Reconstructing $(N)x$(N) rank-$(RANK) matrices from $(KNOWNENTRIES) of their entries"
modlabel = "Centralized Gated"

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

struct CentralizedGatedCell{F1, F2, F3, F4} <: Lux.AbstractLuxLayer
    k::Int
    n::Int
    init_params::F1 
    init_states::F2
    init_zeros::F3
    init_ones::F4
end
MatrixCell = Union{MatrixVlaCell, MatrixGatedCell, MatrixGatedCell2, CentralizedGatedCell}

function CentralizedGatedCell(k::Int, n::Int; 
                              init_params=glorot_uniform, 
                              init_states=glorot_uniform, 
                              init_zeros=zeros32, 
                              init_ones=ones32)
    CentralizedGatedCell{typeof(init_params), typeof(init_states), typeof(init_zeros), typeof(init_ones)}(
        k, n, init_params, init_states, init_zeros, init_ones
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::CentralizedGatedCell)
    (# Recurrent weights and biases
     Whh=l.init_params(rng, l.k, l.k; gain = 1f0),
     Bh=l.init_zeros(l.k),
     # Gating weights and biases
     #Wa=l.init_ones(l.k),
     Wah=l.init_params(rng, l.k, l.k),
     Wax=l.init_params(rng, l.k, l.k),
     Ba=l.init_zeros(l.k),
     # Params of layernorm
     γ=l.init_ones(1), 
     β=l.init_zeros(1),
     # Params of sigmoid inside update gate
     α=l.init_ones(1),
     ξ=l.init_zeros(1))
end

function Lux.initialstates(rng::AbstractRNG, l::CentralizedGatedCell)
    h = l.init_states(rng, l.k)
    (H=h,
     A=l.init_ones(l.k),
     oldH=deepcopy(h),
     selfreset=[false],
     turns=[1],
     init=deepcopy(h)) 
end

function updategate2!(A, H, X, Wah, Wax, Ba, α, ξ)
    # Formula: st.A = σ.(ps.Wah * st.H .+ ps.Wax * st.Xproj .+ ps.Ba)
    A .= Wah * H .+ Wax * X
    @. A += Ba
    @. A = NNlib.sigmoid_fast(α * A + ξ)
end

function updategatedstate2!(H, oldH, A, X, Whh, Bh)
    # Formula: st.H = st.A .* tanh.(st.H * ps.Whh .+ ps.Bh .+ st.Xproj) + (1f0 .- st.A) .* st.H
    H .= Whh * oldH
    @. H = (A * NNlib.tanh_fast(H + Bh + X)) + ((1f0 - A) * oldH)
end

function (l::CentralizedGatedCell)(X, ps, st)
    if st.selfreset[1]
        reset!(st)
    end
    @inbounds for _ = 1:st.turns[1]
        updategate2!(st.A, st.H, X, ps.Wah, ps.Wax, ps.Ba,
                     ps.α[1], ps.ξ[1])
        @. st.oldH = deepcopy(st.H)
        updategatedstate2!(st.H, st.oldH, st.A, X, ps.Whh, ps.Bh)
        mylayernorm!(st.H, 
                     ps.γ[1], ps.β[1])
    end
    return st.H, st
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
    #mylayernorm!(ans, ps.γ[1], ps.β[1])
    return vec(ans), st
end

g(x) = reshape(x, size(x,1), 1)

activemodel = Chain(
    enc = Chain(
        WrappedFunction(vec),
        WrappedFunction(x -> x[nonzeroidx]),
        Dense(length(nonzeroidx), NETWIDTH, swish),
        WrappedFunction(g), 
        LayerNorm((NETWIDTH,))
    ),
    cell = SkipConnection(
        CentralizedGatedCell(NETWIDTH, N),
    +),
    #=ln2 = Chain(
        WrappedFunction(g), 
        LayerNorm((NETWIDTH,))
    ),=#
    proj = Chain(
        Dense(NETWIDTH, N^2, swish),
        WrappedFunction(g), 
        LayerNorm((N^2,))
    ),
    dec = Chain(
        WrappedFunction(vec),
        DecodingLayer(N^2, N, DEC_RANK)
    )
)


function predict_through_time(st, ps, m::Chain,
                              xs::AbstractArray{Float32, 3}, turns; n = N)::AbstractArray{Float32, 2}
    preds = Array{Float32}(undef, n^2, size(xs, 3))
    populatepreds!(preds, st, ps, m, xs, turns)
    return preds
end

function Luxapply!(st, ps, m::A, x; 
                    selfreset::Bool = false, 
                    turns::Int = 1) where A <: Lux.Chain
    setup!(st; selfreset = selfreset, turns = turns)
    Lux.apply(m, x, ps, st)[1]
end

ps, st = Lux.setup(trainrng, activemodel)
activemodel(dataX[:,:,1], ps, st)[1]

#display(activemodel)
nparams::Int = LuxCore.parameterlength(activemodel)
println("Parameter Length: ", nparams, "; State Length: ",
    LuxCore.statelength(activemodel))

X = dataX[:,:,1]
Y = Luxapply!(st, ps, activemodel, X; selfreset=false, turns=20)
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

save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(TURNS)turns"*
     "_knownentries_uvDec_l2(CosAn)_2.png", fig)


ploteigvals(ps.cell.Whh)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(TURNS)turns"*
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
