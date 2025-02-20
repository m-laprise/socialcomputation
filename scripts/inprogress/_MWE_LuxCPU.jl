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
import ParameterSchedulers: Exp
import WeightInitializers: glorot_uniform, zeros32
import MLUtils: DataLoader, splitobs
import Optimisers: Adam, adjust!

BLAS.set_num_threads(4)

# Hyperparameters and constants

const K::Int = 200
const N::Int = 64
const HIDDEN_DIM::Int = 16 #2*N
const RANK::Int = 1
const DATASETSIZE::Int = 5000
const KNOWNENTRIES::Int = 1600

const MINIBATCH_SIZE::Int = 64
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

INIT_ETA = 1f-4
DECAY = 0.7f0
EPOCHS::Int = 10
TURNS::Int = 50

datarng = Random.MersenneTwister(Int(round(time())))
trainrng = Random.MersenneTwister(0)

#====CREATE CUSTOM RNN====#

struct MatrixVlaCell{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init_params::F1 
    init_states::F2
    init_rep::F3
end

struct DecodingLayer{F1} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init::F1
end

# Custom chain
struct ComposedRNN{L1, L2} <: Lux.AbstractLuxContainerLayer{(:cell, :dec)}
    cell::L1
    dec::L2
end

#"Parent type for both types of RNN cells"
#MatrixCell = Union{MatrixVlaCell, MatrixGRUCell}

# LAYER INITIALIZERS
function MatrixVlaCell(k::Int, n2::Int, m::Int; 
                       init_params=glorot_uniform, init_states=glorot_uniform, init_rep=zeros32)
    MatrixVlaCell{typeof(init_params), typeof(init_states), typeof(init_rep)}(
        k, n2, m, init_params, init_states, init_rep
    )
end
function Lux.initialparameters(rng::AbstractRNG, l::MatrixVlaCell)
    (Wx_in=l.init_params(rng, l.m, l.n2),
     Whh=l.init_params(rng, l.k, l.k),
     Bh=l.init_params(rng, l.m, l.k))
end
function Lux.initialstates(rng::AbstractRNG, l::MatrixVlaCell)
    h = l.init_states(rng, l.m, l.k)
    (H=h,
     Xproj=l.init_rep(l.m, l.k),
     selfreset=[false],
     turns=[1],
     init=deepcopy(h)) 
end

function DecodingLayer(k::Int, n2::Int, m::Int; 
                       init=glorot_uniform)
    DecodingLayer{typeof(init)}(k, n2, m, init)
end
function Lux.initialparameters(rng::AbstractRNG, l::DecodingLayer)
    (Wx_out=l.init(rng, l.n2, l.m*l.k),
     #β=ones(Float32, l.k)
     )
end
Lux.initialstates(::AbstractRNG, ::DecodingLayer) = NamedTuple()

# FORWARD PASSES
function timemovement!(st, ps, turns)
    # Each agent takes a col of the n2 x k input matrix, 
    # which is a sparse (n2 x 1) vector, and projects it to a m x 1 vector, with m << n2
    # (They all share the same projection / compression stragegy Wx_in in R^{m x n2})
    # Agents consult their compressed input, exchange compressed information, and update their state.
    # Repeat for a given number of time steps.
    @inbounds for _ in 1:turns
        st.H .= NNlib.tanh_fast.(st.H * ps.Whh .+ ps.Bh .+ st.Xproj)
    end
end

function (l::MatrixVlaCell)(X, ps, st)
    if st.selfreset[1]
        reset!(st)
    end
    # To avoid recomputing the projection for each time step,
    # store it as a hidden state
    st.Xproj .= ps.Wx_in * X
    timemovement!(st, ps, st.turns[1])
    return st.H, st
end

function (l::DecodingLayer)(cellh, ps, st)
    #return ps.Wx_out * cellh * ps.β, st
    return ps.Wx_out * vec(cellh), st
end

function (c::ComposedRNN)(x::AbstractMatrix, ps, st::NamedTuple)
    h, st_l1 = c.cell(x, ps.cell, st.cell)
    y, st_l2 = c.dec(h, ps.dec, st.dec)
    # Return the new state which has the same structure as `st`
    return y, (cell = st_l1, dec = st_l2)
end

# Replace Lux.apply with Luxapply! to allow for custom state handling
function setup!(st; selfreset, turns)
    st.cell.selfreset .= selfreset
    st.cell.turns .= turns
end

function Luxapply!(st, ps, m::A, x; 
                   selfreset::Bool = false, 
                   turns::Int = 1) where A <: Lux.AbstractLuxContainerLayer
    setup!(st; selfreset = selfreset, turns = turns)
    Lux.apply(m, x, ps, st)[1]
end

# HELPER FUNCTIONS

# *NOTE*: Careful with this; it grabs it from the globals
state(m::ComposedRNN) = st.cell.H
reset!(st, m::ComposedRNN) = (st.cell.H .= deepcopy(st.cell.init); st.cell.Xproj .= 0f0)
reset!(st, m::MatrixVlaCell) = (st.H .= deepcopy(st.init); st.Xproj .= 0f0)
reset!(st) = (st.H .= deepcopy(st.init))

# Helper functions for data 
_3D(y) = reshape(y, N, N, size(y, 2))
_2D(y) = reshape(y, N*N, size(y, 3))
_3Dslices(y) = eachslice(_3D(y), dims=3)
l(f, y) = mean(f.(_3Dslices(y)))

#====DEFINE LOSS FUNCTIONS====#

# Nuclear norm, but scaled to avoid the norm going to zero simply by scaling the matrix
"Nuclear norm of a matrix (sum of singular values), scaled by the standard deviation of its entries"
scalednuclearnorm(A::AbstractArray{Float32, 2})::Float32 = sum(svdvals(A)) / (size(A, 1) * std(A))

"Spectral norm of a matrix (largest singular value)"
spectralnorm(A::AbstractArray{Float32, 2})::Float32 = svdvals(A)[1]

"Spectral gap of a matrix (largest singular value - second largest singular value)"
function spectralgap(A::AbstractArray{Float32, 2})::Float32
    vals = svdvals(A)
    return vals[1] - vals[2] 
end

"Scaled spectral gap of a matrix (largest singular value - second largest singular value) / largest singular value"
function scaledspectralgap(A::AbstractArray{Float32, 2})::Float32
    vals = svdvals(A)
    return (vals[1] - vals[2]) / vals[1] 
end

"Populates a vector with the scaled spectral gap of each matrix in a 3D array"
function populatepenalties!(penalties, ys_hat::AbstractArray{Float32, 3})::Nothing
    @inbounds for i in axes(ys_hat, 3)
        nn = scalednuclearnorm(@view ys_hat[:,:,i])
        gap = scaledspectralgap(@view ys_hat[:,:,i]) - 1f0
        penalties[i] = nn >= 1f0 ? nn-gap : 1f0-gap
        #penalties[i] = -scaledspectralgap(@view ys_hat[:,:,i]) + 1f0 
    end
end

# Training loss - Many data points
""" 
    Training loss given a 3D array of true matrices, a matrix where each row is a vectorized 
    predicted matrix, and the mask matrix with information about which entries are known.
    The loss is a weighted sum of the L1 loss on known entries and a scaled spectral gap penalty.
"""
function spectrum_penalized_l1(ys::AbstractArray{Float32, 3}, 
                          ys_hat::AbstractArray{Float32, 2}, 
                          maskmatrix::AbstractArray{Float32, 2};
                          theta::Float32 = 1f0,#0.8f0,
                          datascale::Float32 = 0.1f0)::Float32
    nb_examples = size(ys, 3)
    # L1 loss on known entries
    diff = vec(maskmatrix) .* (_2D(ys) .- ys_hat)
    l1_known = vec(sum(abs, diff, dims = 1)) / sum(maskmatrix)
    # Spectral norm penalty
    penalties = Array{Float32}(undef, nb_examples)
    populatepenalties!(penalties, _3D(ys_hat))
    # Training loss
    left = theta * l1_known / datascale
    right = (1f0 - theta) * penalties 
    errors = left .+ right
    return mean(errors)
end

function spectrum_penalized_l2(ys::AbstractArray{Float32, 3}, 
                          ys_hat::AbstractArray{Float32, 2}, 
                          maskmatrix::AbstractArray{Float32, 2};
                          theta::Float32 = 1f0,#0.8f0,
                          datascale::Float32 = 0.1f0)::Float32
    nb_examples = size(ys, 3)
    # L1 loss on known entries
    diff = vec(maskmatrix) .* (_2D(ys) .- ys_hat)
    l2_known = sqrt(vec(sum(abs2, diff, dims = 1)) / sum(maskmatrix))
    # Spectral norm penalty
    penalties = Array{Float32}(undef, nb_examples)
    populatepenalties!(penalties, _3D(ys_hat))
    # Training loss
    left = theta * l2_known / datascale
    right = (1f0 - theta) * penalties 
    errors = left .+ right
    return mean(errors)
end

# Training loss - Single datum 
#=
""" 
    Training loss for a single true matrix, a predicted matrix, and the mask matrix with information 
    about which entries are known.
    The loss is a weighted sum of the L1 loss on known entries and a scaled spectral gap penalty.
"""
function spectrum_penalized_l2(ys::AbstractArray{Float32, 2}, 
                          ys_hat::AbstractArray{Float32, 2}, 
                          maskmatrix::AbstractArray{Float32, 2};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.1f0)::Float32
    l, n = size(ys)
    # L1 loss on known entries
    ys2 = reshape(ys, l * n, 1)
    diff = vec(maskmatrix) .* (ys2 .- ys_hat)
    l1_known = sum(abs2, diff) / sum(maskmatrix)
    # Spectral norm penalty
    ys_hat3 = reshape(ys_hat, l, n)
    penalty = scalednuclearnorm(ys_hat3)
    #penalty = -scaledspectralgap(ys_hat3)
    # Training loss
    left = theta * l1_known / datascale^2
    right = (1f0 - theta) * penalty
    return left .+ right
end =#

MSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs2, A .- B, dims=1)
RMSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = sqrt.(MSE(A, B))
MAE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs, A .- B, dims=1)

function allentriesRMSE(ys::AbstractArray{Float32, 3}, 
                        ys_hat::AbstractArray{Float32, 2};
                        datascale::Float32 = 0.1f0)::Float32
    mean(RMSE(_2D(ys), ys_hat) / datascale)
end

function allentriesMAE(ys::AbstractArray{Float32, 3}, 
                        ys_hat::AbstractArray{Float32, 2};
                        datascale::Float32 = 0.1f0)::Float32
    mean(MAE(_2D(ys), ys_hat) / datascale)
end

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
function trainingloss(m, ps, st, (xs, ys, maskmatrix))
    ys_hat = predict_through_time(st, ps, m, xs)
    #return spectrum_penalized_l2(ys, ys_hat, maskmatrix)#, st, NamedTuple()
    return allentriesMAE(ys, ys_hat)
end
"Compute predictions with a given number of time steps, and use them to compute the training loss."
function trainingloss(m, ps, st, (xs, ys, maskmatrix, turns))
    ys_hat = predict_through_time(st, ps, m, xs, turns)
    #return spectrum_penalized_l2(ys, ys_hat, maskmatrix)#, st, NamedTuple()
    return allentriesMAE(ys, ys_hat)
end

# Rules for autodiff backend
EnzymeRules.inactive(::typeof(reset!), args...) = nothing
Enzyme.@import_rrule typeof(svdvals) AbstractMatrix{<:Number}

#====CREATE DATA====#

function lowrankmatrix(m, n, r, rng; datatype=Float32, std::Float32=0.1f0)
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
maskmatrix = masktuple2array(masktuples, N, N)
# (Initially tried to create X with sparse arrays, but it made pullbacks slower
# and I am not RAM constrained, so focusing on regular arrays for now)

size(dataX), size(dataY)

dataX, valX, testX = splitobs(datarng, dataX, 
                 at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
dataY, valY, testY = splitobs(datarng, dataY, 
                 at = (TRAIN_PROP, VAL_PROP, TEST_PROP))
size(dataX), size(dataY)

dataloader = DataLoader(
    (data = dataX, label = dataY), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true, parallel=true, rng=datarng)

@info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)

# Get reference value from ground truth
refspectralnorm = mean(spectralnorm.(eachslice(dataY, dims = 3)))
refspectralgap = mean(spectralgap.(eachslice(dataY, dims = 3)))
refnuclearnorm = mean(scalednuclearnorm.(eachslice(dataY, dims = 3)))
refvariance = var(dataY)

#====INITIALIZE MODEL====#

activemodel = ComposedRNN(
    MatrixVlaCell(K, N^2, HIDDEN_DIM), 
    DecodingLayer(K, N^2, HIDDEN_DIM)
)
ps, st = Lux.setup(trainrng, activemodel)

#display(activemodel)
println("Parameter Length: ", LuxCore.parameterlength(activemodel), "; State Length: ",
    LuxCore.statelength(activemodel))

X = randn(datarng, Float32, N^2, K)
#y = activemodel(x, ps, st)[1]
Y, H = Luxapply!(st, ps, activemodel, X; selfreset=false, turns=20)

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
loss, grads = Flux.withgradient(fclosure, dup_model)

_grads = Enzyme.make_zero(ps)
_dstates = Enzyme.make_zero(st)
_, loss = autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), 
        trainingloss,
        Const(opt_state.model), Duplicated(opt_state.parameters, _grads), Duplicated(opt_state.states, _dstates),  
        Const((dataX[:,:,1:64],dataY[:,:,1:64],maskmatrix,2)))
Training.apply_gradients!(opt_state, grads)

@btime autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), 
            $trainingloss,
            $(Const(activemodel)), $(Duplicated(ps, grads)), $(Const(st)),  
            $(Const((dataX[:,:,1:64],dataY[:,:,1:64],maskmatrix,2))))
            =#
#========#

# Optimizer
opt = Adam(INIT_ETA)
opt_state = Training.TrainState(activemodel, ps, st, opt)

# Learning rate schedule
s = Exp(start = INIT_ETA, decay = DECAY)
println("Learning rate schedule:")
for (eta, epoch) in zip(s, 1:EPOCHS)
    println(" - Epoch $epoch: eta = $eta")
end

# STORE INITIAL METRICS
train_metrics = Dict(
    :loss => Float32[],
    :rmse => Float32[],
    :nuclearnorm => Float32[],
    :spectralnorm => Float32[],
    :spectralgap => Float32[],
    :variance => Float32[],
    :Whh_spectra => Array{Complex{Float32}}[]
)
val_metrics = Dict(
    :loss => Float32[],
    :rmse => Float32[],
)
# Compute the initial training and validation loss and other metrics with forward passes
function recordmetrics!(metricsdict, st, ps, activemodel, X, Y, maskmatrix, TURNS, subset=200; split="train")
    if split == "test"
        subset = size(X, 3)
    end
    Ys_hat = predict_through_time(st, ps, activemodel, X[:,:,1:subset], TURNS)
    #push!(metricsdict[:loss], spectrum_penalized_l2(Y[:,:,1:subset], Ys_hat, maskmatrix))
    #push!(metricsdict[:rmse], allentriesRMSE(Y[:,:,1:subset], Ys_hat))
    push!(metricsdict[:rmse], spectrum_penalized_l1(Y[:,:,1:subset], Ys_hat, maskmatrix))
    push!(metricsdict[:loss], allentriesMAE(Y[:,:,1:subset], Ys_hat))
    if split == "train"
        push!(metricsdict[:nuclearnorm], l(scalednuclearnorm, Ys_hat))
        push!(metricsdict[:spectralnorm], l(spectralnorm, Ys_hat))
        push!(metricsdict[:spectralgap], l(spectralgap, Ys_hat))
        push!(metricsdict[:variance], var(Ys_hat))
    end
    if split == "test"
        return Ys_hat
    end
end
train_metrics[:Whh_spectra] = eigvals(ps.cell.Whh)
recordmetrics!(train_metrics, st, ps, activemodel, dataX, dataY, maskmatrix, TURNS)
recordmetrics!(val_metrics, st, ps, activemodel, valX, valY, maskmatrix, TURNS, split="val")

##================##
function inspect_gradients(grads)
    # Convert a named tuple to a vector
    g = [grads.cell.Wx_in, grads.cell.Whh, grads.cell.Bh, grads.dec.Wx_out]
    tot = sum(length.(g))
    nan_params = sum(sum(isnan, gi) for gi in g)
    vanishing_params = sum(sum(abs.(gi) .< 1e-6) for gi in g)
    exploding_params = sum(sum(abs.(gi) .> 1e6) for gi in g)
    return nan_params/tot, vanishing_params/tot, exploding_params/tot
end

function diagnose_gradients(n, v, e)
    if n > 0
        @info("$(round(n*100, digits=0)) % NaN gradients detected")
    end
    if v > 0 
        @info("$(round(v*100, digits=0)) % vanishing gradients detected")
    end
    if e > 0
        @info("$(round(e*100, digits=0)) % exploding gradients detected")
    end
    #Otherwise, report that no issues were found
    if n < 0.1 && v < 0.1 && e < 0.1
        println("Gradients appear well-behaved.")
    end
end

starttime = time()
println("===================")
println("Initial training loss: " , train_metrics[:loss][1], "; initial training RMSE: ", train_metrics[:rmse][1]*10)
@info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
for (eta, epoch) in zip(s, 1:EPOCHS)
    reset!(st, activemodel)
    println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
    adjust!(opt_state, eta)
    # Initialize counters for gradient diagnostics
    mb = 1
    # Iterate over minibatches
    for (x, y) in dataloader
        # Forward pass (to compute the loss) and backward pass (to compute the gradients)
        grads = Enzyme.make_zero(ps)
        Δstates = Enzyme.make_zero(st)
        _, train_loss_value = autodiff(
            set_runtime_activity(Enzyme.ReverseWithPrimal), 
            trainingloss, Const(opt_state.model), 
            Duplicated(opt_state.parameters, grads), 
            Duplicated(opt_state.states, Δstates),  
            Const((x, y, maskmatrix, TURNS)))
        if epoch == 1 && mb == 1
            @info("Time to first gradient: $(round(time() - starttime, digits=2)) seconds")
        end
        _n, _v, _e = inspect_gradients(grads)
        # If any gradients are NaN, print a warning and skip the update
        if _n > 0
            @warn "NaN gradients detected on minibatch $(epoch)--$(mb)"
            diagnose_gradients(_n, _v, _e)
            mb += 1
            continue
        end
        # Detect loss of Inf or NaN. Print a warning, and then skip update
        if !isfinite(train_loss_value)
            @warn "Loss is $train_loss_value on minibatch $(epoch)--$(mb)" 
            diagnose_gradients(_n, _v, _e)
            mb += 1
            continue
        end
        # During training, use the backward pass to store the training loss after the previous epoch
        push!(train_metrics[:loss], train_loss_value)
        if mb % 25 == 0
            # Diagnose the gradients
            if _v > 0.1 || _e > 0.1
                diagnose_gradients(_n, _v, _e)
            end
        end
        if mb == 1 || mb % 5 == 0
            println("Minibatch ", epoch, "--", mb, ": loss of ", round(train_loss_value, digits=4))
        end
        # Use the optimizer and grads to update the trainable parameters and the optimizer states
        Training.apply_gradients!(opt_state, grads)
        mb += 1
    end
    # Compute training metrics -- expensive operation with a forward pass over the entire training set
    # but we restrict to two mini-batches only
    recordmetrics!(train_metrics, st, ps, activemodel, dataX, dataY, maskmatrix, TURNS)
    # Compute validation metrics
    recordmetrics!(val_metrics, st, ps, activemodel, valX, valY, maskmatrix, TURNS, split="val")
    println("Epoch ", epoch, ": Train loss: ", train_metrics[:loss][end], "; train RMSE: ", train_metrics[:rmse][end]*10)
    println("Epoch ", epoch, ": Val loss: ", val_metrics[:loss][end], "; val RMSE: ", val_metrics[:rmse][end]*10)
    # Check if validation loss has increased for 2 epochs in a row; if so, stop training
    if length(val_metrics[:loss]) > 2
        if val_metrics[:loss][end] > val_metrics[:loss][end-1] && val_metrics[:loss][end-1] > val_metrics[:loss][end-2]
            @warn("Early stopping at epoch $epoch")
            break
        end
    end
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Assess on testing set
test_metrics = Dict(:loss => Float32[], :rmse => Float32[])
testYs_hat = recordmetrics!(test_metrics, st, ps, activemodel, testX, testY, maskmatrix, TURNS, split="test")

#======explore results=======#
var(dataY), var(testYs_hat)

reshape(testYs_hat[:,10], N, N)

using CairoMakie
#plot(svdvals(dataY[:,:,10]))
plot(svdvals(reshape(testYs_hat[:,10], N, N)))
#plot(svdvals(rand(Float32, N, 5) * rand(Float32, 5, N) .+ 0.1f0 * rand(Float32, N, N)))

mean(RMSE(testYs_hat, _2D(testY))) / 0.1f0

include("../plot_utils.jl")
ploteigvals(ps.cell.Whh)

#======generate training plots=======#

using CairoMakie

tasklab = "Reconstructing $(N)x$(N) rank-$(RANK) matrices from $(KNOWNENTRIES) of their entries"
taskfilename = "matrix$(N)recon_rank$(RANK)"
modlabel = "Matrix Vanilla"

CairoMakie.activate!()
fig = Figure(size = (820, 1000))
epochs = length(val_metrics[:loss])
train_metrics[:spectralgapovernorm] = train_metrics[:spectralgap] ./ train_metrics[:spectralnorm]
refspectralgapovernorm = refspectralgap / refspectralnorm
metrics = [
    #(1, 1, "Loss", "loss", "Spectral-gap and norm penalized RMSE / std (known entries)"),
    #(1, 2, "RMSE", "rmse", "RMSE / std (all entries)"),
    (1, 1, "Loss", "loss", "MAE / std (all entries)"),
    (1, 2, "MAE", "rmse", "MAE / std (known entries)"),
    (2, 1, "Spectral gap / spectral norm", "spectralgapovernorm", "Mean spectral gap / mean spectral norm"),
    (2, 2, "Spectral gap", "spectralgap", "Mean spectral gap"),
    (3, 1, "Nuclear norm", "nuclearnorm", "Mean scaled nuclear norm"),
    (3, 2, "Variance", "variance", "Mean variance of matrix entries")
]
for (row, col, ylabel, key, title) in metrics
    ax = Axis(fig[row, col], xlabel = "Epochs", ylabel = ylabel, title = title)
    if key in ["loss"]
        lines!(ax, [i for i in range(2, epochs, length(train_metrics[Symbol(key)][length(dataloader)+1:end]))], 
               train_metrics[Symbol(key)][length(dataloader)+1:end], color = :blue, label = "Training")
        lines!(ax, 2:epochs, val_metrics[Symbol(key)][2:end], color = :red, label = "Validation")
        #lines!(ax, 2:epochs, [test_metrics[Symbol(key)][end]], color = :green, linestyle = :dash)
        #scatter!(ax, epochs, test_metrics[Symbol(key)][end], color = :green, label = "Final Test")
    elseif key in ["rmse"]
        lines!(ax, 2:epochs, train_metrics[Symbol(key)][2:end]*10, color = :blue, label = "Training")
        lines!(ax, 2:epochs, val_metrics[Symbol(key)][2:end]*10, color = :red, label = "Validation")
        #lines!(ax, 2:epochs, [test_metrics[Symbol(key)][end]*10], color = :green, linestyle = :dash)
        #scatter!(ax, epochs, test_metrics[Symbol(key)][end]*10, color = :green, label = "Final Test")
    else
        lines!(ax, 2:epochs, train_metrics[Symbol(key)][2:end], color = :blue, label = "Reconstructed")
        lines!(ax, 2:epochs, [eval(Symbol("ref$key"))], color = :orange, linestyle = :dash, label = "Mean ground truth")
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
    "Optimizer: Adam with schedule Exp(start = $(INIT_ETA), decay = $(DECAY), restart = 10)\n"*
    "for $(epochs-1) epochs over $(size(dataX, 3)) examples, minibatch size $(MINIBATCH_SIZE).\n"*
    "Hidden internal state dimension: $(HIDDEN_DIM).\n",#*
    #"Test MAE (all entries): $(round(test_metrics[:loss][end], digits=4)). "*
    #"Test MAE (known entries): $(round(test_metrics[:rmse][end]*10, digits=4)).",
    fontsize = 14,
    padding = (0, 0, 0, 0),
)
fig

save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(TURNS)turns_knownentries_WvecX_allMAE_moreepochs_failure.png", fig)
