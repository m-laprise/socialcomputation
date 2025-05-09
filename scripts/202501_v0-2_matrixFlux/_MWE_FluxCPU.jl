#= Testing using: 
Julia 1.10.5 
Flux 0.16.3
Enzyme 0.13.30
=#
if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using Flux
using Enzyme
import NNlib
using LinearAlgebra
using Distributions
using Random
using ChainRules
using ParameterSchedulers

BLAS.set_num_threads(4)

#====CREATE CUSTOM RNN====#

# CREATE STATEFUL, MATRIX-VALUED RNN LAYER
"Parametric type for Vanilla RNN cell with matrix-valued states"
mutable struct MatrixVlaCell{A<:AbstractArray{Float32, 2}}
    # Parameters
    Wx_in::A # Input weights n2 x m
    Whh::A # Recurrent weights k x k
    bh::A  # Recurrent bias k x m
    # State 
    h::A   # Current state k x m
    const init::A # Store initial state h for reset
end

"Constructor for MatrixVlaCell"
function MatrixVlaCell(n2::Int, k::Int, m::Int)::MatrixVlaCell
    Wx_in = randn(Float32, n2, m) / sqrt(Float32(n2))
    Whh = randn(Float32, k, k) / sqrt(Float32(k))
    bh = randn(Float32, k, m) / sqrt(Float32(k))
    h = randn(Float32, k, m) * 0.01f0
    MatrixVlaCell(Wx_in, Whh, bh, h, h)
end

"Parametric type for Gated RNN cell with matrix-valued states and distributed inputs (incorrect implementation)"
mutable struct MatrixGRUCell{A<:AbstractArray{Float32, 2}, V<:AbstractArray{Float32, 1}}
    # Parameters
    Whh::A # Recurrent weights k x k
    bh::A  # Recurrent bias k x n2
    g1::V  # n2 x 1
    g2::V  # n2 x 1
    bz::A  # Update gate bias k x 1
    # State
    h::A   # Current state k x n2
    z::V   # Update gate k x 1
    const init::A # Store initial state h for reset 
end
"Constructor for MatrixGRUCell"
function MatrixGRUCell(n::Int, k::Int)::MatrixGRUCell
    Whh = randn(Float32, k, k) / sqrt(Float32(k))
    bz = randn(Float32, k, 1) / sqrt(Float32(k))
    bh = randn(Float32, k, n) / sqrt(Float32(k))
    g1 = ones(Float32, n)
    g2 = ones(Float32, n)
    h = randn(Float32, k, n) * 0.01f0
    z = rand(Float32, k) * 0.01f0
    MatrixGRUCell(Whh, bh, g1, g2, bz, h, z, h)
end

"Stateful RNN cell forward pass for distributed inputs"
function(m::MatrixVlaCell)(state::AbstractArray{Float32, 2}, 
                           I::AbstractArray{Float32, 2}; 
                           selfreset::Bool=false)::AbstractArray{Float32, 2}
    if selfreset
        h = m.init
    else
        h = state
    end
    # Each agent takes a row of the k x n2 input matrix, 
    # which is a sparse (1 x n2) vector, and projects it to a m x 1 vector, with m << n2
    # (They all share the same projection matrix / compressed storage 
    # stragegy Wx_in in R^{m x n2})
    # Agents consult their compressed input, exchange compressed information, and update their state
    m.h .= NNlib.tanh_fast.(m.Whh * h .+ m.bh .+ I * m.Wx_in)
    return m.h 
end 

"Stateful GRU cell forward pass for distributed inputs (incorrect implementation)"
function(m::MatrixGRUCell)(state::AbstractArray{Float32, 2}, 
                           I::AbstractArray{Float32, 2}; 
                           selfreset::Bool=false)::AbstractArray{Float32, 2}
    if selfreset
        h_old = m.init
    else
        h_old = state
    end
    hmul = m.Whh * h_old    # k x n2
    h_new = NNlib.tanh_fast.(hmul .+ m.bh .+ I)   # k x n2
    m.z .= NNlib.sigmoid_fast.((hmul * m.g1) .+ ((m.Whh * I) * m.g2) .+ m.bz)  # k x 1
    m.h .= ((1f0 .- m.z) .* h_new) .+ (m.z .* h_old)   # k x n2
    return m.h
end
"Parent type for both types of RNN cells"
MatrixCell = Union{MatrixVlaCell, MatrixGRUCell}

state(m::MatrixCell) = m.h
reset!(m::MatrixCell) = (m.h = m.init)

# CREATE DECODING LAYER WITH TRAINABLE WEIGHTS
"Parametric type for weighted mean layer"
struct WeightedMeanLayer{A<:AbstractArray{Float32, 2}, V<:AbstractArray{Float32, 1}}
    Wx_out::A
    weight::V
end

"Constructor for WeightedMeanLayer"
function WeightedMeanLayer(n2::Int, k::Int, m::Int;
                           init = ones)
    Wx_out = randn(Float32, n2, m) / sqrt(Float32(n2))
    weight_init = init(Float32, k) * 5f0
    WeightedMeanLayer(Wx_out, weight_init)
end

"""Forward pass for decoding layer to project each agent's guess back 
to n2 and create a linear combination of the guess of each agent"""
function (a::WeightedMeanLayer)(x::AbstractArray{Float32, 2})
    # Ensure weights are positive, between 0 and 1, and normalized
    # Sig is a vector with one entry per agent, representing the importance of each agent's guess
    sig = NNlib.sigmoid_fast.(a.weight)
    # Agents project their guess (state) back to n2; then the guesses are combined by weighted mean
    # All agents share the same projection matrix Wx_out in R^{n2 x m}
    return a.Wx_out * x' * sig / sum(sig)
end

# CHAIN RNN AND WEIGHTED MEAN LAYERS
" Parametric type for RNN with a cell with matrix-valued states and a weighted mean layer"
struct MatrixRNN{M<:MatrixCell, D<:WeightedMeanLayer}
    rnn::M
    dec::D
end
state(m::MatrixRNN) = state(m.rnn)
reset!(m::MatrixRNN) = reset!(m.rnn)

"Forward pass for the RNN means composing the cell and the weighted mean layer"
(m::MatrixRNN)(I::AbstractArray{Float32, 2}; 
               selfreset::Bool = false) = (m.dec ∘ m.rnn)(
                state(m), I; selfreset = selfreset)

# Tell Flux what parameters are trainable
Flux.@layer MatrixVlaCell trainable=(Wx_in, Whh, bh)
Flux.@layer MatrixGRUCell trainable=(Whh, bh, g1, g2, bz)
Flux.@layer WeightedMeanLayer trainable=(Wx_out)
Flux.@layer :expand MatrixRNN trainable=(rnn, dec)
Flux.@non_differentiable reset!(m::MatrixCell)
Flux.@non_differentiable reset!(m::MatrixRNN)

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
                          mask_mat::AbstractArray{Float32, 2};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.1f0)::Float32
    nb_examples = size(ys, 3)
    # L1 loss on known entries
    diff = vec(mask_mat) .* (_2D(ys) .- ys_hat)
    l1_known = vec(sum(abs, diff, dims = 1)) / sum(mask_mat)
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
                          mask_mat::AbstractArray{Float32, 2};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.1f0)::Float32
    l, n = size(ys)
    # L1 loss on known entries
    ys2 = reshape(ys, l * n, 1)
    diff = vec(mask_mat) .* (ys2 .- ys_hat)
    l1_known = sum(abs, diff) / sum(mask_mat)
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

#== Helper functions for inference and backprop through time ==#

"Move the RNN through time for a given number of steps, repeating the input x each time."
function timemovement!(m, x, turns)::Nothing
    @assert turns > 0
    @inbounds for _ in 1:turns
        m(x; selfreset = false)
    end
end

"Populate a matrix with the predictions of the RNN for each example in a 3D array, with no time steps."
function populatepreds!(preds, m, xs::AbstractArray{Float32, 3})::Nothing
    @inbounds for i in axes(xs, 3)
        reset!(m)
        example = @view xs[:,:,i]
        preds[:,i] .= m(example; selfreset = false)
    end
end

"Populate a matrix with the predictions of the RNN for each example in a 3D array, with a given number of time steps."
function populatepreds!(preds, m, xs::AbstractArray{Float32, 3}, turns)::Nothing
    @inbounds for i in axes(xs, 3)
        reset!(m)
        example = @view xs[:,:,i]
        timemovement!(m, example, turns)
        preds[:,i] .= m(example; selfreset = false)
    end
end

"Predict the output for a single input matrix, with no time steps."
function predict_through_time(m, 
                              x::AbstractArray{Float32, 2})::AbstractArray{Float32, 2}
    if m.rnn.h != m.rnn.init
        reset!(m)
    end
    preds = m(x; selfreset = false)
    return reshape(preds, :, 1)
end

"Predict the output for a single input matrix, with a given number of time steps."
function predict_through_time(m, 
                              x::AbstractArray{Float32, 2}, 
                              turns::Int)::AbstractArray{Float32, 2}
    if m.rnn.h != m.rnn.init
        reset!(m)
    end
    timemovement!(m, x, turns)
    preds = m(x; selfreset = false)
    return reshape(preds, :, 1)
end

"Predict the outputs for an array of input matrices, with no time steps."
function predict_through_time(m, 
                              xs::AbstractArray{Float32, 3})::AbstractArray{Float32, 2}
    output_size = size(xs, 2)
    nb_examples = size(xs, 3)
    preds = Array{Float32}(undef, output_size, nb_examples)
    populatepreds!(preds, m, xs)
    return preds
end

"Predict the outputs for an array of input matrices, with a given number of time steps."
function predict_through_time(m, 
                              xs::AbstractArray{Float32, 3}, 
                              turns::Int)::AbstractArray{Float32, 2}
    output_size = size(xs, 2)
    nb_examples = size(xs, 3)
    preds = Array{Float32}(undef, output_size, nb_examples)
    populatepreds!(preds, m, xs, turns)
    return preds
end

# Wrapper for prediction and loss
"Compute predictions with no time steps, and use them to compute the training loss."
function trainingloss(m, xs, ys, mask_mat)
    ys_hat = predict_through_time(m, xs)
    return spectrum_penalized_l1(ys, ys_hat, mask_mat)
end
"Compute predictions with a given number of time steps, and use them to compute the training loss."
function trainingloss(m, xs, ys, mask_mat, turns)
    ys_hat = predict_through_time(m, xs, turns)
    return spectrum_penalized_l1(ys, ys_hat, mask_mat)
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

function creatematrix(m, n, r, seed; datatype=Float32)
    rng = Random.MersenneTwister(seed)
    A = (randn(rng, datatype, m, r) ./ sqrt(sqrt(Float32(r)))) * (randn(rng, datatype, r, n) ./ sqrt(sqrt(Float32(r))))
    return A * 0.1f0
end

function sensingmasks(m::Int, n::Int; k::Int = 0, seed::Int = Int(round(time())))
    if k == 0
        k = m * n
    end
    maskij = [(i, j) for i in 1:m for j in 1:n]
    rng = MersenneTwister(seed)
    shuffle!(rng, maskij)
    return maskij[1:k]
end

function masktuple2array(fixedmask::Vector{Tuple{Int, Int}}, m::Int, n::Int)
    mask_mat = zeros(Float32, m, n)
    for (i, j) in fixedmask
        mask_mat[i, j] = 1.0
    end
    return mask_mat
end

function matrixinput_setup(Y::AbstractArray{Float32}, 
                        k::Int,
                        M::Int, 
                        N::Int, 
                        dataset_size::Int,
                        knownentries::Int, 
                        masks::Vector{Tuple{Int,Int}};
                        alpha::Float32 = 50f0)
    @assert knownentries == length(masks)
    @assert alpha >= 0 && alpha <= 50
    knownentries_per_agent = zeros(Int, k)
    # Create a vector of length k with the number of known entries for each agent,
    # based on the alpha concentration parameter. The vector should sum to the total number of known entries.
    if alpha == 0
        knownentries_per_agent[1] = knownentries
    else
        dirichlet_dist = Dirichlet(alpha * ones(Float32, minimum([k, knownentries])))
        proportions = rand(dirichlet_dist)
        knownentries_per_agent = round.(Int, proportions * minimum([k, knownentries]))
        # If knownentries < k, pad the vector with zeros
        if knownentries < k
            knownentries_per_agent = vcat(knownentries_per_agent, zeros(Int, k - knownentries))
        end
        # Adjust to ensure the sum is exactly knownentries
        while sum(knownentries_per_agent) != knownentries
            diff = knownentries - sum(knownentries_per_agent)
            # If the difference is negative (positive), add (subtract) one to (from) a random agent
            knownentries_per_agent[rand(1:k)] += 1 * sign(diff)
            # Check that no entry is negative, and if so, replace by zero
            knownentries_per_agent = max.(0, knownentries_per_agent)
        end
    end
    X = zeros(Float32, k, M*N, dataset_size)
    for i in 1:dataset_size
        inputmat = zeros(Float32, k, M*N)
        entry_count = 1
        for agent in 1:k
            for l in 1:knownentries_per_agent[agent]
                row, col = masks[entry_count]
                flat_index = M * (col - 1) + row
                inputmat[agent, flat_index] = Y[row, col, i]
                entry_count += 1
            end
        end
        X[:, :, i] = inputmat
    end
    return X
end

function datasetgeneration(m, n, rank, dataset_size, knownentries, net_width)
    Y = Array{Float32, 3}(undef, m, n, dataset_size)
    for i in 1:dataset_size
        Y[:, :, i] = creatematrix(m, n, rank, 1131+i)
    end
    fixedmask = sensingmasks(m, n; k=knownentries, seed=9632)
    mask_mat = masktuple2array(fixedmask, m, n)
    @assert size(mask_mat) == (m, n)
    X = matrixinput_setup(Y, net_width, m, n, dataset_size, knownentries, fixedmask)
    @info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)
    return X, Y, fixedmask, mask_mat
end

dataX, dataY, fixedmask, mask_mat = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K)
# (Initially tried to create X with sparse arrays, but it made pullbacks slower
# and I am not RAM constrained, so focusing on regular arrays for now)

size(dataX), size(dataY)
#dataXt = NNlib.batched_adjoint(dataX)

#====INITIALIZE MODEL====#
activemodel = MatrixRNN(
    MatrixVlaCell(N^2, K, 2*N), 
    WeightedMeanLayer(N^2, K, 2*N)
)

#====TEST FWD PASS AND LOSS====#
#=
using BenchmarkTools

m_c = Reactant.@compile activemodel(dataX[:,:,1])
# Forward pass, one datum, no time step
@benchmark activemodel($dataX[:,:,1]) 
@benchmark m_c($dataX[:,:,1])

# Forward pass, one datum, no time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1]) 
# Forward pass, one datum, 2 time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1], 2) 

# Forward pass, all data points, no time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1:32]) 
# Forward pass, all data points, 2 time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1:32], 2) 

ys_hat = predict_through_time(activemodel, dataX)
@benchmark spectrum_penalized_l1($dataY[:,:,1:2], $ys_hat[:,1:2], $mask_mat)
@benchmark trainingloss($activemodel, $dataX[:,:,1:32], $dataY[:,:,1:32], $mask_mat)

starttime = time()
trainingloss_c = Reactant.@compile trainingloss(activemodel, dataX[:,:,1:2], dataY[:,:,1:2], mask_mat)
endtime = time()
println("Compilation time: ", (endtime - starttime)/60, " minutes")
trainingloss_c = Reactant.@compile trainingloss(activemodel, dataX[:,:,1:2], dataY[:,:,1:2], mask_mat, 2)
@benchmark trainingloss_c($activemodel, $dataX[:,:,1:32], $dataY[:,:,1:32], $mask_mat)
=#
#====BEST GRADIENT COMPUTATION====#
#=
# Enzyme
# MODEL ONLY
@btime loss, grads = Flux.withgradient((m,x) -> sum(m(x)), 
                                Duplicated(activemodel), dataX[:,:,1])
# 123.524 ms (117 allocations: 114.67 MiB)
@btime loss, grads = Flux.withgradient((m,x) -> sum(predict_through_time(m, x)), 
                                Duplicated(activemodel), dataX)
# 9.072 s (1277 allocations: 1.65 GiB)

# TRAINING LOSS
@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), 
                                $dataX[:,:,1], $dataY[:,:,1], $mask_mat)
# 122.946 ms (182 allocations: 114.95 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), 
                                $dataX[:,:,1:3], $dataY[:,:,1:3], $mask_mat)
# 366.954 ms (666 allocations: 177.70 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), 
                                $dataX, $dataY, $mask_mat)
# 8.564 s (8836 allocations: 1.65 GiB)

@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), 
                                $dataX[:,:,1], $dataY[:,:,1], $mask_mat, 2)
# 364.710 ms (358 allocations: 164.96 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), 
                                $dataX[:,:,1:3], $dataY[:,:,1:3], $mask_mat, 2)
# 1.107 s (826 allocations: 327.97 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), 
                                $dataX, $dataY, $mask_mat, 2)
# 27.567 s (11070 allocations: 4.79 GiB)

#====TEST OTHER GRADIENT COMPUTATIONS====#

using DifferentiationInterface
fclosure(m) = trainingloss(m, dataX[:,:,1:3], dataY[:,:,1:3], mask_mat)
backend = AutoEnzyme()
@btime DifferentiationInterface.value_and_gradient($fclosure, $backend, $activemodel) 

using Fluxperimental, Mooncake
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(reset!), Any}
Mooncake.@mooncake_overlay norm(x) = sqrt(sum(abs2, x))
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(svdvals), AbstractMatrix{<:Number}}

dup_model = Moonduo(activemodel)

# MODEL ONLY
fclosure(m) = sum(m(dataX[:,:,1]))
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
# 1.089 s (4922099 allocations: 1.49 GiB)
fclosure(m) = sum(predict_through_time(m, dataX))
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
# 

# TRAINING LOSS
fclosure(m) = trainingloss(m, dataX[:,:,1], dataY[:,:,1], mask_mat)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)

fclosure(m) = trainingloss(m, dataX[:,:,1:3], dataY[:,:,1:3], mask_mat)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)

fclosure(m) = trainingloss(m, dataX[:,:,1], dataY[:,:,1], mask_mat, 2)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)

fclosure(m) = trainingloss(m, dataX[:,:,1:3], dataY[:,:,1:3], mask_mat, 2)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)

# set_runtime_activity(Enzyme.Reverse)
test = autodiff(Enzyme.Reverse, 
    trainingloss, Active, Duplicated(activemodel), 
    Const(dataX), Const(dataY), Const(mask_mat))
=#
#========#
MSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs2.(A .- B), dims=1)
RMSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = sqrt.(MSE(A, B))

function allentriesRMSE(ys::AbstractArray{Float32, 3}, 
                        ys_hat::AbstractArray{Float32, 2};
                        datascale::Float32 = 0.1f0)::Float32
    mean(RMSE(_2D(ys), ys_hat) / datascale)
end

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

dataloader = Flux.DataLoader(
    (data=dataX, label=dataY), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true)

# Optimizer
INIT_ETA = 5e-4
DECAY = 0.7
EPOCHS::Int = 5
TURNS::Int = 1

function prepareoptimizer(eta, decay, model)
    # Learning rate schedule
    s = Exp(start = eta, decay = decay)
    # Optimizer
    opt = Adam()
    # Tree of states
    opt_state = Flux.setup(opt, model)
    return s, opt, opt_state
end
s, opt, opt_state = prepareoptimizer(INIT_ETA, DECAY, activemodel)

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
dataYs_hat = predict_through_time(activemodel, dataX[:,:,1:128], TURNS)
initloss_train = spectrum_penalized_l1(dataY[:,:,1:128], dataYs_hat, mask_mat)
initRMSE_train = allentriesRMSE(dataY[:,:,1:128], dataYs_hat)
push!(train_loss, initloss_train)
push!(train_rmse, initRMSE_train)
push!(train_nuclearnorm, l(scalednuclearnorm, dataYs_hat))
push!(train_spectralnorm, l(spectralnorm, dataYs_hat))
push!(train_spectralgap, l(spectralgap, dataYs_hat))
push!(train_variance, var(dataYs_hat))

valYs_hat = predict_through_time(activemodel, valX[:,:,1:128], TURNS)
initloss_val = spectrum_penalized_l1(valY[:,:,1:128], valYs_hat, mask_mat)
initRMSE_val = allentriesRMSE(valY[:,:,1:128], valYs_hat)
push!(val_loss, initloss_val)
push!(val_rmse, initRMSE_val)
push!(val_nuclearnorm, l(scalednuclearnorm, valYs_hat))
push!(val_spectralnorm, l(spectralnorm, valYs_hat))
push!(val_spectralgap, l(spectralgap, valYs_hat))
push!(val_variance, var(valYs_hat))

##================##
function inspect_gradients(grads)
    g, _ = Flux.destructure(grads)
    
    nan_params = [0]
    vanishing_params = [0]
    exploding_params = [0]

    if any(isnan.(g))
        push!(nan_params, 1)
    end
    if any(abs.(g) .< 1e-6)
        push!(vanishing_params, 1)
    end
    if any(abs.(g) .> 1e6)
        push!(exploding_params, 1)
    end
    return sum(nan_params), sum(vanishing_params), sum(exploding_params)
end

function diagnose_gradients(n, v, e)
    if n > 0
        println(n, " NaN gradients detected")
    end
    if v > 0 && v != 125
        println(v, " vanishing gradients detected")
    end
    if e > 0
        println(e, " exploding gradients detected")
    end
    #Otherwise, report that no issues were found
    if n == 0 && v == 0 && e == 0
        println("Gradients appear well-behaved.")
    end
end

starttime = time()
println("===================")
println("Initial training loss: " , initloss_train, "; initial training RMSE: ", initRMSE_train)
@info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
for (eta, epoch) in zip(s, 1:EPOCHS)
    reset!(activemodel)
    println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
    Flux.adjust!(opt_state, eta = eta)
    # Initialize counters for gradient diagnostics
    mb, n, v, e = 1, 0, 0, 0
    # Iterate over minibatches
    for (x, y) in dataloader
        # Forward pass (to compute the loss) and backward pass (to compute the gradients)
        train_loss_value, grads = Flux.withgradient(trainingloss, Duplicated(activemodel), x, y, mask_mat, TURNS)
        if epoch == 1 && mb == 1
            @info("Time to first gradient: ", round(time() - starttime, digits=2), " seconds")
        end
        # During training, use the backward pass to store the training loss after the previous epoch
        push!(train_loss, train_loss_value)
        # Diagnose the gradients
        _n, _v, _e = inspect_gradients(grads[1])
        n += _n
        v += _v
        e += _e
        # Detect loss of Inf or NaN. Print a warning, and then skip update!
        if !isfinite(train_loss_value)
            @warn "Loss is $val on minibatch $(epoch)--$(mb)" 
            mb += 1
            continue
        end
        # Use the optimizer and grads to update the trainable parameters; update the optimizer states
        Flux.update!(opt_state, activemodel, grads[1])
        if mb == 1 || mb % 5 == 0
            println("Minibatch ", epoch, "--", mb, ": loss of ", round(train_loss_value, digits=4))
            #@info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
        end
        mb += 1
    end
    push!(Whh_spectra, eigvals(activemodel.rnn.Whh))
    # Print a summary of the gradient diagnostics for the epoch
    diagnose_gradients(n, v, e)
    # Compute training metrics -- expensive operation with a forward pass over the entire training set
    dataYs_hat = predict_through_time(activemodel, dataX[:,:,1:64], TURNS)
    trainloss = spectrum_penalized_l1(dataY[:,:,1:64], dataYs_hat, mask_mat)
    trainRMSE = allentriesRMSE(dataY[:,:,1:64], dataYs_hat)
    push!(train_loss, trainloss)
    push!(train_rmse, trainRMSE)
    push!(train_nuclearnorm, l(scalednuclearnorm, dataYs_hat))
    push!(train_spectralnorm, l(spectralnorm, dataYs_hat))
    push!(train_spectralgap, l(spectralgap, dataYs_hat))
    push!(train_variance, var(dataYs_hat))
    # Compute validation metrics
    valYs_hat = predict_through_time(activemodel, valX, TURNS)
    push!(val_loss, spectrum_penalized_l1(valY, valYs_hat, mask_mat))
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
testYs_hat = predict_through_time(activemodel, testX, TURNS)
testloss = spectrum_penalized_l1(testY, testYs_hat, mask_mat)
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
