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
const K::Int = 8 # Try 9 (182 entries per agent) or 8 (205 entries per agent)
const NETWIDTH::Int = 32
const N::Int = 64
const HIDDEN_DIM::Int = 164
const RANK::Int = 1
const DATASETSIZE::Int = 2000
const KNOWNENTRIES::Int = 1640
DEC_RANK::Int = 1

const MINIBATCH_SIZE::Int = 64
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

INIT_ETA = 1f-4
END_ETA = 1f-7
DECAY = 0.7f0
ETA_PERIOD = 10

EPOCHS::Int = 3
TURNS::Int = 70

THETA::Float32 = 0.5f0
mainloss = spectrum_penalized_l2

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

agent_nonzeroidx = Vector{Vector{Int}}(undef, K)
for i in 1:K
    agent_nonzeroidx[i] = zeros(Int, knowledgedistr[i])
    findnz!(agent_nonzeroidx[i], @view dataX[:, i, 1])
end

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
activemodel = ComposedRNN(
    MatrixGatedCell2(NETWIDTH, N^2, HIDDEN_DIM, knowledgedistr, agent_nonzeroidx), 
    FactorDecodingLayer(NETWIDTH, N, HIDDEN_DIM, DEC_RANK)
)

ps, st = Lux.setup(trainrng, activemodel)
activemodel(dataX[:,:,1], ps, st)[1]

#display(activemodel)
println("Parameter Length: ", LuxCore.parameterlength(activemodel), "; State Length: ",
    LuxCore.statelength(activemodel))

# In the new setup, DOES NOT work with random matrix not exactly nl sparse in each column
X = dataX[:,:,1]
#y = activemodel(x, ps  , st)[1]
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

fig = main_training_figure(
    train_metrics, val_metrics, test_metrics, 
    dataX, tasklab, modlabel, 
    HIDDEN_DIM, DEC_RANK, INIT_ETA, 
    ETA_PERIOD, MINIBATCH_SIZE, K, TURNS
)

save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_Factor1Dec_pHuber(CosAn-theta90-sn)_layernormrow.png", fig)


ploteigvals(ps.cell.Whh)
save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(HIDDEN_DIM)_$(TURNS)turns"*
     "_knownentries_Factor1Dec_pHuber(CosAn-theta100-sn)_Whheigvals.png", 
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

#==== GRADIENT COMPUTATION WITH MOONCAKE====#

# Toy examples
#=f(x) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
cache = Mooncake.prepare_gradient_cache(f, x)
Mooncake.value_and_gradient!!(cache, f, x)

# Lux toy examples - setup
x = rand(Float32, 1, 100)
y = rand(Float32, 1, 100)
model = Chain(Dense(1, 32, tanh), Dense(32, 1))
mps, mst = Lux.setup(Random.default_rng(), model)
myloss(ps, st, m, x, y) = Lux.MSELoss()(m(x,ps,st)[1], y)
# Mooncake direct
cache = Mooncake.prepare_gradient_cache(myloss, mps, mst, model, x, y);
@btime Mooncake.prepare_gradient_cache($(myloss), $(mps), $(mst), $(model), $(x), $(y));
val, grad = Mooncake.value_and_gradient!!(cache, myloss, mps, mst, model, x, y)
@btime val, grad = Mooncake.value_and_gradient!!($(cache), $(myloss), $(mps), $(mst), $(model), $(x), $(y))
grad = deepcopy(grad[2])
# DI wrapper

val, grad = value_and_gradient(myloss, AutoMooncake(;config=nothing), mps, Constant(mst), Constant(model), Constant(x), Constant(y))
@btime val, grad = value_and_gradient($(myloss), $(AutoMooncake(;config=nothing)), $(mps), $(Constant(mst)), $(Constant(model)), $(Constant(x)), $(Constant(y)))
=#

# Real example
using BenchmarkTools

backend = Mooncake.AutoMooncake(;config=nothing)
starttime = time()
_val, _grad = value_and_gradient(trainingloss, backend, ps, 
                               Constant(activemodel), Cache(st), 
                               Constant((dataX[:,:,1:2],dataY[:,:,1:2],nonzeroidx,TURNS)))
endtime = time()
println("Compilation and computation time: ", (endtime - starttime))
# 10.81 s

@btime prep = prepare_gradient(
    myloss, backend, ps, 
    Constant(activemodel), Cache(st), 
    Constant((dataX[:,:,1:3],dataY[:,:,1:3],nonzeroidx,TURNS))
)
# 3.188 s (11845 allocations: 3.22 GiB)
@btime _val, _grad = value_and_gradient(
    myloss, prep, AutoMooncake(;config=nothing), ps, 
    Cache(st), Constant(activemodel),
    Constant((dataX[:,:,1:3],dataY[:,:,1:3],nonzeroidx,TURNS))
)
#  299.311 ms (4258 allocations: 37.30 MiB)
_grad

_v, _e = inspect_and_repare_gradients!(_grad, activemodel)
diagnose_gradients(_v, _e)

#=
fclosure(ps) = trainingloss(activemodel, ps, st, (dataX[:,:,1:3], dataY[:,:,1:3], maskmatrix, 2))
using DifferentiationInterface
DifferentiationInterface.value_and_gradient(fclosure, AutoEnzyme(), ps) 
using Fluxperimental, Mooncake
dup_model = Moonduo(activemodel)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
loss, grads = Flux.withgradient(fclosure, dup_model)=#


@btime Luxapply!($st, $ps, $activemodel, $dataX[:,:,1]; 
                 selfreset=false, turns=1)
@btime Luxapply!($st, $ps, $activemodel, $dataX[:,:,1]; 
                 selfreset=false, turns=10)

m_test = MatrixGatedCell2(K, N^2, HIDDEN_DIM, knowledgedistr, agent_nonzeroidx)
ps_test, st_test = Lux.setup(trainrng, m_test)
a = m_test(dataX[:,:,1], ps_test, st_test)
@btime gatedtimemovement!($(st_test), $(ps_test), $(1))
@btime m_test($(dataX[:,:,1]), $(ps_test), $(st_test))

dec = FactorDecodingLayer(K, N, HIDDEN_DIM, DEC_RANK)
psdec, stdec = Lux.setup(trainrng, dec)
@btime dec($a[1], $psdec, $stdec, $a[2].Xproj)



_grads = Enzyme.make_zero(ps)
_dstates = Enzyme.make_zero(st)
_, loss = autodiff(
    set_runtime_activity(Enzyme.ReverseWithPrimal), 
    trainingloss, Const(activemodel), 
    Duplicated(ps, _grads), 
    Duplicated(st, _dstates),  
    Const((dataX[:,:,1:30], dataY[:,:,1:30], nonzeroidx, TURNS)))

gradlist = destructure(_grads)

@btime autodiff(
    set_runtime_activity(Enzyme.ReverseWithPrimal), 
    $(trainingloss), $(Const(activemodel)), 
    $(Duplicated(ps, _grads)), 
    $(Duplicated(st, _dstates)),  
    $(Const((dataX[:,:,1:30], dataY[:,:,1:30], nonzeroidx, TURNS))))

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

m_test = MatrixGatedCell2(K, N^2, HIDDEN_DIM, knowledgedistr)
ps_test, st_test = Lux.setup(trainrng, m_test)
m_test(X, ps_test, st_test)
