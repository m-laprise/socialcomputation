#= Testing using: 
Julia 1.10.5 
Flux 0.16.3
Enzyme 0.13.30
=#
if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using Lux
using Enzyme
using LinearAlgebra
using Random
using ChainRules
import NNlib
import Distributions: Dirichlet, mean, std, var
import ParameterSchedulers: Exp
import WeightInitializers: glorot_uniform, zeros32
import MLUtils: DataLoader

BLAS.set_num_threads(4)

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
    (Wx_out=l.init(rng, l.n2, l.m),
     β=ones(Float32, l.k))
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
        st.H .= tanh.(st.H * ps.Whh .+ ps.Bh .+ st.Xproj)
    end
end

function (l::MatrixVlaCell)(X, ps, st)
    if st.selfreset[1]
        st.H .= st.init
    end
    # To avoid recomputing the projection for each time step,
    # store it as a hidden state
    st.Xproj .= ps.Wx_in * X
    timemovement!(st, ps, st.turns[1])
    return st.H, st
end

function (l::DecodingLayer)(cellh, ps, st)
    return ps.Wx_out * cellh * ps.β, st
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
        #penalties[i] = scalednuclearnorm(@view ys_hat[:,:,i])
        penalties[i] = -scaledspectralgap(@view ys_hat[:,:,i]) + 1f0 
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
                          theta::Float32 = 0.8f0,
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

# Training loss - Single datum
""" 
    Training loss for a single true matrix, a predicted matrix, and the mask matrix with information 
    about which entries are known.
    The loss is a weighted sum of the L1 loss on known entries and a scaled spectral gap penalty.
"""
function spectrum_penalized_l1(ys::AbstractArray{Float32, 2}, 
                          ys_hat::AbstractArray{Float32, 2}, 
                          maskmatrix::AbstractArray{Float32, 2};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.1f0)::Float32
    l, n = size(ys)
    # L1 loss on known entries
    ys2 = reshape(ys, l * n, 1)
    diff = vec(maskmatrix) .* (ys2 .- ys_hat)
    l1_known = sum(abs, diff) / sum(maskmatrix)
    # Spectral norm penalty
    ys_hat3 = reshape(ys_hat, l, n)
    #penalty = scalednuclearnorm(ys_hat3)
    penalty = -scaledspectralgap(ys_hat3)
    # Training loss
    left = theta * l1_known / datascale
    right = (1f0 - theta) * penalty
    error = left .+ right
    return error
end

MSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs2, A .- B, dims=1)
RMSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = sqrt.(MSE(A, B))

function allentriesRMSE(ys::AbstractArray{Float32, 3}, 
                        ys_hat::AbstractArray{Float32, 2};
                        datascale::Float32 = 0.1f0)::Float32
    mean(RMSE(_2D(ys), ys_hat) / datascale)
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

#= Wrapper for prediction and loss
To use the Lux Training API, the loss function must take 4 inputs – model, parameters, states and data. 
It must return 3 values – loss, updated_state, and any computed statistics. 
=#
"Compute predictions with no time steps, and use them to compute the training loss."
function trainingloss(m, ps, st, (xs, ys, maskmatrix))
    ys_hat = predict_through_time(st, ps, m, xs)
    return spectrum_penalized_l1(ys, ys_hat, maskmatrix)#, st, NamedTuple()
end
"Compute predictions with a given number of time steps, and use them to compute the training loss."
function trainingloss(m, ps, st, (xs, ys, maskmatrix, turns))
    ys_hat = predict_through_time(st, ps, m, xs, turns)
    return spectrum_penalized_l1(ys, ys_hat, maskmatrix)#, st, NamedTuple()
end

# Rules for autodiff backend
EnzymeRules.inactive(::typeof(reset!), args...) = nothing
Enzyme.@import_rrule typeof(svdvals) AbstractMatrix{<:Number}

#====CREATE DATA====#
const K::Int = 200
const N::Int = 64
const RANK::Int = 1
const DATASETSIZE::Int = 8000
const KNOWNENTRIES::Int = 1600

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

myseed = Int(round(time()))
rng = Random.MersenneTwister(myseed)

dataX, dataY, masktuples = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, rng)
maskmatrix = masktuple2array(masktuples, N, N)
@info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)
# (Initially tried to create X with sparse arrays, but it made pullbacks slower
# and I am not RAM constrained, so focusing on regular arrays for now)

size(dataX), size(dataY)

#====INITIALIZE MODEL====#

Random.seed!(rng, 0)

activemodel = ComposedRNN(
    MatrixVlaCell(K, N^2, 4*N), 
    DecodingLayer(K, N^2, 4*N)
)
ps, st = Lux.setup(rng, activemodel)
ps_ra = ps |> xdev
st_ra = st |> xdev

#display(activemodel)
println("Parameter Length: ", LuxCore.parameterlength(activemodel), "; State Length: ",
    LuxCore.statelength(activemodel))

X = randn(rng, Float32, 64^2, 400)
#y = activemodel(x, ps, st)[1]
Y, H = Luxapply!(st, ps, activemodel, x; selfreset=false, turns=20)
#gradient(ps -> sum(first(activemodel(x, ps, st))), ps)

#====TEST FWD PASS AND LOSS====#
#=
using BenchmarkTools

m_c = Reactant.@compile activemodel(dataX[:,:,1])
# Forward pass, one datum, no time step
@benchmark activemodel($dataX[:,:,1]) 
@benchmark m_c($dataX[:,:,1])

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

grads = Enzyme.make_zero(ps)
_, loss = autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), 
        trainingloss,
        Const(opt_state.model), Duplicated(opt_state.parameters, grads), Const(opt_state.states),  
        Const((dataX[:,:,1:64],dataY[:,:,1:64],maskmatrix,2)))
Training.apply_gradients!(opt_state, grads)

@btime autodiff(set_runtime_activity(Enzyme.ReverseWithPrimal), 
            $trainingloss,
            $(Const(activemodel)), $(Duplicated(ps, grads)), $(Const(st)),  
            $(Const((dataX[:,:,1:64],dataY[:,:,1:64],maskmatrix,2))))
            =#
#========#

const MINIBATCH_SIZE::Int = 64
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

function train_val_test_split(X::AbstractArray, 
                              train_prop::Real, val_prop::Real, test_prop::Real)
    @assert train_prop + val_prop + test_prop == 1
    dimsX = length(size(X))
    dataset_size = size(X, dimsX)
    train_nb = Int(train_prop * dataset_size)
    val_nb = Int(val_prop * dataset_size)
    train_idxs = 1:train_nb
    val_idxs = train_nb+1:train_nb+val_nb
    test_idxs = train_nb+val_nb+1:dataset_size
    if dimsX == 2
        Xtrain, Xval, Xtest = X[:,train_idxs], X[:,val_idxs], X[:,test_idxs]
    elseif dimsX == 3
        Xtrain, Xval, Xtest = X[:,:,train_idxs], X[:,:,val_idxs], X[:,:,test_idxs]
    end
    @assert size(Xtrain, dimsX) == train_nb
    @assert size(Xval, dimsX) == val_nb
    @assert size(Xtest, dimsX) == dataset_size - train_nb - val_nb
    return Xtrain, Xval, Xtest
end

dataX, valX, testX = train_val_test_split(dataX, TRAIN_PROP, VAL_PROP, TEST_PROP)
dataY, valY, testY = train_val_test_split(dataY, TRAIN_PROP, VAL_PROP, TEST_PROP)
size(dataX), size(dataY)

dataloader = DataLoader(
    (data = dataX, label = dataY), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true, parallel=true, rng=rng)

# Optimizer
INIT_ETA = 1f-3
DECAY = 0.7f0
EPOCHS::Int = 5
TURNS::Int = 1

using Optimisers
opt = Adam(INIT_ETA)
opt_state = Training.TrainState(activemodel, ps, st, opt)

# Learning rate schedule
s = Exp(start = INIT_ETA, decay = DECAY)
println("Learning rate schedule:")
for (eta, epoch) in zip(s, 1:EPOCHS)
    println(" - Epoch $epoch: eta = $eta")
end

# STORE INITIAL METRICS
train_loss = Float32[]
train_rmse = Float32[]
train_nuclearnorm = Float32[]
train_spectralnorm = Float32[]
train_spectralgap = Float32[]
train_variance = Float32[]
val_loss = Float32[]
val_rmse = Float32[]
val_nuclearnorm = Float32[]
val_spectralnorm = Float32[]
val_spectralgap = Float32[]
val_variance = Float32[]
Whh_spectra = []
push!(Whh_spectra, eigvals(activemodel.rnn.Whh))

# Compute the initial training and validation loss and other metrics with forward passes
dataYs_hat = predict_through_time(st, ps, activemodel, dataX[:,:,1:128], TURNS)
initloss_train = spectrum_penalized_l1(dataY[:,:,1:128], dataYs_hat, maskmatrix)
initRMSE_train = allentriesRMSE(dataY[:,:,1:128], dataYs_hat)
push!(train_loss, initloss_train)
push!(train_rmse, initRMSE_train)
push!(train_nuclearnorm, l(scalednuclearnorm, dataYs_hat))
push!(train_spectralnorm, l(spectralnorm, dataYs_hat))
push!(train_spectralgap, l(spectralgap, dataYs_hat))
push!(train_variance, var(dataYs_hat))

valYs_hat = predict_through_time(st, ps, activemodel, valX[:,:,1:128], TURNS)
initloss_val = spectrum_penalized_l1(valY[:,:,1:128], valYs_hat, maskmatrix)
initRMSE_val = allentriesRMSE(valY[:,:,1:128], valYs_hat)
push!(val_loss, initloss_val)
push!(val_rmse, initRMSE_val)
push!(val_nuclearnorm, l(scalednuclearnorm, valYs_hat))
push!(val_spectralnorm, l(spectralnorm, valYs_hat))
push!(val_spectralgap, l(spectralgap, valYs_hat))
push!(val_variance, var(valYs_hat))

##================##
function inspect_gradients(grads)
    # Convert a named tuple to a vector
    g = [grads.cell.Wx_in, grads.cell.Whh, grads.dec.Wx_out]
    a, b = size(g[1])
    c, d = size(g[2])
    tot = 2*a*b + c*d
    nan_params, vanishing_params, exploding_params = 0, 0, 0
    if any(isnan.(g[1])) || any(isnan.(g[2])) || any(isnan.(g[3]))
        nan_params = sum(isnan.(g[1])) + sum(isnan.(g[2])) + sum(isnan.(g[3]))
    end
    if any(abs.(g[1]) .< 1e-6) || any(abs.(g[2]) .< 1e-6) || any(abs.(g[3]) .< 1e-6)
        vanishing_params = sum(abs.(g[1]) .< 1e-6) + sum(abs.(g[2]) .< 1e-6) + sum(abs.(g[3]) .< 1e-6)
    end
    if any(abs.(g[1]) .> 1e6) || any(abs.(g[2]) .> 1e6) || any(abs.(g[3]) .> 1e6)
        exploding_params = sum(abs.(g[1]) .> 1e6) + sum(abs.(g[2]) .> 1e6) + sum(abs.(g[3]) .> 1e6)
    end
    return nan_params/tot, vanishing_params/tot, exploding_params/tot
end

function diagnose_gradients(n, v, e)
    if n > 0
        @info(round(n, digits=2), " % NaN gradients detected")
    end
    if v > 0 
        @info(round(v, digits=2), " % vanishing gradients detected")
    end
    if e > 0
        @info(round(e, digits=2), " % exploding gradients detected")
    end
    #Otherwise, report that no issues were found
    if n < 0.1 && v < 0.1 && e < 0.1
        println("Gradients appear well-behaved.")
    end
end

starttime = time()
println("===================")
println("Initial training loss: " , initloss_train, "; initial training RMSE: ", initRMSE_train)
@info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
for (eta, epoch) in zip(s, 1:EPOCHS)
    reset!(st, activemodel)
    println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
    Optimisers.adjust!(opt_state, eta)
    # Initialize counters for gradient diagnostics
    mb = 1
    # Iterate over minibatches
    for (x, y) in dataloader
        # Forward pass (to compute the loss) and backward pass (to compute the gradients)
        grads = Enzyme.make_zero(ps)
        _, train_loss_value = autodiff(
            set_runtime_activity(Enzyme.ReverseWithPrimal), 
            trainingloss, Const(opt_state.model), 
            Duplicated(opt_state.parameters, grads), 
            Const(opt_state.states),  
            Const((x, y, maskmatrix, TURNS)))
        if epoch == 1 && mb == 1
            @info("Time to first gradient: ", round(time() - starttime, digits=2), " seconds")
        end
        # During training, use the backward pass to store the training loss after the previous epoch
        push!(train_loss, train_loss_value)
        # Diagnose the gradients
        _n, _v, _e = inspect_gradients(grads)
        if _n > 0.1 || _v > 0.1 || _e > 0.1
            diagnose_gradients(_n, _v, _e)
        end
        # Detect loss of Inf or NaN. Print a warning, and then skip update!
        if !isfinite(train_loss_value)
            @warn "Loss is $val on minibatch $(epoch)--$(mb)" 
            mb += 1
            continue
        end
        # Use the optimizer and grads to update the trainable parameters; update the optimizer states
        Training.apply_gradients!(opt_state, grads)
        if mb == 1 || mb % 5 == 0
            println("Minibatch ", epoch, "--", mb, ": loss of ", round(train_loss_value, digits=4))
        end
        mb += 1
    end
    push!(Whh_spectra, eigvals(activemodel.rnn.Whh))
    # Compute training metrics -- expensive operation with a forward pass over the entire training set
    dataYs_hat = predict_through_time(st, ps, activemodel, dataX[:,:,1:64], TURNS)
    trainloss = spectrum_penalized_l1(dataY[:,:,1:64], dataYs_hat, maskmatrix)
    trainRMSE = allentriesRMSE(dataY[:,:,1:64], dataYs_hat)
    push!(train_loss, trainloss)
    push!(train_rmse, trainRMSE)
    push!(train_nuclearnorm, l(scalednuclearnorm, dataYs_hat))
    push!(train_spectralnorm, l(spectralnorm, dataYs_hat))
    push!(train_spectralgap, l(spectralgap, dataYs_hat))
    push!(train_variance, var(dataYs_hat))
    # Compute validation metrics
    valYs_hat = predict_through_time(st, ps, activemodel, valX, TURNS)
    push!(val_loss, spectrum_penalized_l1(valY, valYs_hat, maskmatrix))
    push!(val_rmse, allentriesRMSE(valY, valYs_hat))
    push!(val_nuclearnorm, l(scalednuclearnorm, valYs_hat))
    push!(val_spectralnorm, l(spectralnorm, valYs_hat))
    push!(val_spectralgap, l(spectralgap, valYs_hat))
    push!(val_variance, var(valYs_hat))
    println("Epoch $epoch: Train loss: $(train_loss[end]); train RMSE: $(train_rmse[end])")
    println("Epoch $epoch: Val loss: $(val_loss[end]); val RMSE: $(val_rmse[end])")
    # Check if validation loss has increased for 2 epochs in a row; if so, stop training
    if length(val_loss) > 2
        if val_loss[end] > val_loss[end-1] && val_loss[end-1] > val_loss[end-2]
            @warn("Early stopping at epoch ", epoch)
            break
        end
    end
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Assess on testing set
testYs_hat = predict_through_time(st, ps, activemodel, testX, TURNS)
testloss = spectrum_penalized_l1(testY, testYs_hat, maskmatrix)
testRMSE = allentriesRMSE(testY, testYs_hat)
println("Test loss: $(testloss)")
println("Test RMSE: $(testRMSE)")

testspectralnorm = l(spectralnorm, testYs_hat)
testspectralgap = l(spectralgap, testYs_hat)
testnuclearnorm = l(scalednuclearnorm, testYs_hat)
testvariance = var(testYs_hat)
refspnorm = mean(spectralnorm.(eachslice(dataY, dims = 3)))
refgap = mean(spectralgap.(eachslice(dataY, dims = 3)))
refncn = mean(scalednuclearnorm.(eachslice(dataY, dims = 3)))

#======explore results=======#
var(dataY), var(testYs_hat)

reshape(testYs_hat[:,10], N, N) * 1000

using CairoMakie
plot(svdvals(dataY[:,:,10]))
plot(svdvals(reshape(testYs_hat[:,10], N, N)))
plot(svdvals(rand(Float32, N, 5) * rand(Float32, 5, N) .+ 0.1f0 * rand(Float32, N, N)))

mean(MSE(testYs_hat, _2D(testY)))

include("../plot_utils.jl")
ploteigvals(activemodel.rnn.Whh)

#======generate training plots=======#
using CairoMakie

lossname = "Mean spectral-gap and norm penalized L1 loss (known entries)"
rmsename = "Root mean squared reconstr. error / std (all entries)"
tasklab = "Reconstructing $(N)x$(N) rank-$(RANK) matrices from $(KNOWNENTRIES) of their entries"
taskfilename = "matrix$(N)recon_rank$(RANK)"
modlabel = "Matrix Vanilla"

CairoMakie.activate!()
fig = Figure(size = (820, 1000))
#epochs = length(train_loss) - 1
train_l = train_loss#[126:end]
val_l = val_loss#[2:end]
train_r = train_rmse#[2:end]
val_r = val_rmse#[2:end]
epochs = length(val_l)
ax_loss = Axis(fig[1, 1], xlabel = "Epochs", ylabel = "Loss", title = lossname)
# There are many samples of train loss (init + every mini batch) and few samples of val_loss (init + every epoch)
lines!(ax_loss, [i for i in range(1, epochs, length(train_l))], train_l, color = :blue, label = "Training")
lines!(ax_loss, 1:epochs, val_l, color = :red, label = "Validation")
lines!(ax_loss, 1:epochs, [testloss], color = :green, linestyle = :dash)
scatter!(ax_loss, epochs, testloss, color = :green, label = "Final Test")
axislegend(ax_loss, backgroundcolor = :transparent)
ax_rmse = Axis(fig[1, 2], xlabel = "Epochs", ylabel = "RMSE", title = rmsename)
#band!(ax_rmse, 1:epochs, 0.995 .* ones(epochs), 1.005 .* ones(epochs), label = "Random guess", color = :gray, alpha = 0.25)
lines!(ax_rmse, 1:epochs, train_r, color = :blue, label = "Training")
lines!(ax_rmse, 1:epochs, val_r, color = :red, label = "Validation")
lines!(ax_rmse, 1:epochs, [testRMSE], color = :green, linestyle = :dash)
scatter!(ax_rmse, epochs, testRMSE, color = :green, label = "Final Test")
axislegend(ax_rmse, backgroundcolor = :transparent)
# Also plot spectral norm, nuclear norm, spectral gap, and variance
ax_3 = Axis(fig[2, 1], xlabel = "Epochs", ylabel = "Spectral norm", title = "Mean spectral gap / mean spectral norm")
lines!(ax_3, 1:epochs, train_spectralgap ./ train_spectralnorm, color = :blue, label = "Reconstructed")
lines!(ax_3, 1:epochs, [refgap / refspnorm], color = :orange, linestyle = :dash, label = "Mean ground truth")
axislegend(ax_3, backgroundcolor = :transparent)
ax_4 = Axis(fig[2, 2], xlabel = "Epochs", ylabel = "Spectral gap", title = "Mean spectral gap")
lines!(ax_4, 1:epochs, train_spectralgap, color = :blue, label = "Reconstructed")
lines!(ax_4, 1:epochs, [refgap], color = :orange, linestyle = :dash, label = "Mean ground truth")
axislegend(ax_4, backgroundcolor = :transparent)
ax_5 = Axis(fig[3, 1], xlabel = "Epochs", ylabel = "Nuclear norm", title = "Mean nuclear norm")
lines!(ax_5, 1:epochs, train_nuclearnorm, color = :blue, label = "Reconstructed")
lines!(ax_5, 1:epochs, [refncn], color = :orange, linestyle = :dash, label = "Mean ground truth")
axislegend(ax_5, backgroundcolor = :transparent)
ax_6 = Axis(fig[3, 2], xlabel = "Epochs", ylabel = "Variance", title = "Mean variance of matrix entries")
lines!(ax_6, 1:epochs, train_variance, color = :blue, label = "Reconstructed")
lines!(ax_6, 1:epochs, [var(dataY)], color = :orange, linestyle = :dash, label = "Mean ground truth")
axislegend(ax_6, backgroundcolor = :transparent)
Label(
    fig[begin-1, 1:2],
    "$(tasklab)\n$(modlabel) RNN of $K units, $TURNS dynamic steps"*
    "\nwith training loss based on known entries",
    fontsize = 20,
    padding = (0, 0, 0, 0),
)
# Add notes to the bottom of the figure
Label(
    fig[end+1, 1:2],
    "Optimizer: Adam with schedule Exp(start = $(INIT_ETA), decay = $(DECAY))\n"*
    #"Training time: $(round((endtime - starttime) / 60, digits=2)) minutes; "*
    "Test loss: $(round(testloss,digits=4))."*
    "Test RMSE: $(round(testRMSE,digits=4)).",
    #"Optimizer: AdamW(eta=$eta, beta=$beta, decay=$decay)",
    fontsize = 14,
    padding = (0, 0, 0, 0),
)
fig

save("data/$(taskfilename)_$(modlabel)RNNwidth$(K)_$(TURNS)turns_knownentries_l1thenl2.png", fig)
