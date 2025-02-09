#= Testing using: 
Julia 1.10.5 (constrained to this version due to module availability on my HPC)
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
using SparseArrays

#====CREATE CUSTOM RNN====#

# CREATE STATEFUL, MATRIX-VALUED RNN LAYER
mutable struct MatrixRnnCell{A<:Matrix{Float32}}
    Wx_in::A
    bx_in::A
    Whh::A # Recurrent weights
    bh::A  # Recurrent bias
    h::A   # Current state
    const init::A # Store initial state h for reset
end

function matrnn_constructor(n::Int, 
                     k::Int,
                     m::Int)::MatrixRnnCell
    Whh = randn(Float32, k, k) / sqrt(Float32(k))
    Wx_in = randn(Float32, m, n) / sqrt(Float32(m))
    bh = randn(Float32, k, n) / sqrt(Float32(k))
    bx_in = randn(Float32, k, m) / sqrt(Float32(k))
    h = randn(Float32, k, n) * 0.01f0
    return MatrixRnnCell(
        Wx_in, bx_in, 
        Whh, bh,
        h, h
    )
end

function(m::MatrixRnnCell)(state::Matrix{Float32}, 
                           I::AbstractArray{Float32, 2}; 
                           selfreset::Bool=false)::Matrix{Float32}
    if selfreset
        h = m.init
    else
        h = state
    end
    m.h .= NNlib.tanh_fast.(m.Whh * h .+ m.bh .+ I * m.Wx_in' .+ m.bx_in)
    return m.h 
end

state(m::MatrixRnnCell) = m.h
reset!(m::MatrixRnnCell) = (m.h = m.init)

# CREATE WEIGHTED MEAN LAYER WITH TRAINABLE WEIGHTS
struct WeightedMeanLayer{V<:Vector{Float32}}
    weight::V
end
function WeightedMeanLayer(num::Int;
                           init = ones)
    weight_init = init(Float32, num) * 5f0
    WeightedMeanLayer(weight_init)
end

function (a::WeightedMeanLayer)(X::Matrix{Float32})
    sig = NNlib.sigmoid_fast.(a.weight)
    X' * (sig / sum(sig))
end

# CHAIN RNN AND WEIGHTED MEAN LAYERS
struct MatrixRNN{M<:MatrixRnnCell, D<:WeightedMeanLayer}
    rnn::M
    dec::D
end
state(m::MatrixRNN) = state(m.rnn)
reset!(m::MatrixRNN) = reset!(m.rnn)

(m::MatrixRNN)(x::AbstractArray{Float32,2}; 
               selfreset::Bool = false)::Vector{Float32} = (m.dec ∘ m.rnn)(
                state(m), x; 
                selfreset = selfreset)
            
Flux.@layer MatrixRnnCell trainable=(Wx_in, bx_in, Whh, bh)
Flux.@layer WeightedMeanLayer
Flux.@layer :expand MatrixRNN trainable=(rnn, dec)
Flux.@non_differentiable reset!(m::MatrixRnnCell)
Flux.@non_differentiable reset!(m::MatrixRNN)

#====DEFINE LOSS FUNCTIONS====#

# Generate random vector for power iteration; must not get differentiated
# (Probably not the best way to do this!)
function genbk(A)::Vector{Float32}
    b_k = rand(Float32, size(A, 1))
    return b_k 
end
Flux.@non_differentiable genbk(A)

# Do power iteration and return approximate value of largest eigenvalue
function bkiter!(b_k::Vector{Float32}, A::AbstractArray{Float32, 2}, iterations::Int)
    @inbounds for _ in 1:iterations
        b_k .= A * b_k
        b_k ./= norm(b_k)
    end
end
function power_iter(A::AbstractArray{Float32, 2}, b_k::Vector{Float32}; iterations::Int = 50)::Float32    
    bkiter!(b_k, A, iterations)
    largest_eig = dot(b_k, A, b_k)
    return largest_eig
end

# Approximate spectral norm of a matrix (largest singular value) using power iteration
approxspectralnorm(A::AbstractArray{Float32, 2}, b_k::Vector{Float32} = genbk(A))::Float32 = sqrt(power_iter(A' * A, b_k / norm(b_k)))

# Training loss - Single datum
function spectrum_penalized_l2(ys::Matrix{Float32}, 
                          ys_hat::Matrix{Float32}, 
                          mask_mat::Matrix{Float32};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.01f0)::Float32
    l, n = size(ys)
    # L2 loss on known entries
    ys2 = reshape(ys, l * n, 1)
    diff = vec(mask_mat) .* (ys2 .- ys_hat)
    sql2_known = sum(diff.^2) / sum(mask_mat)
    # Spectral norm penalty
    ys_hat3 = reshape(ys_hat, l, n)
    penalty = approxspectralnorm(ys_hat3)
    # Training loss
    left = theta * (sql2_known / datascale)
    right = (1f0 - theta) * -(penalty .+ 2f0)
    error = left .+ right
    return error
end

# Spectral norm penalty 
# For N = 64, our rank 1 matrices have an average spectral norm of 6 or 7, 
# and full rank matrices, an average spectral norm of 0.1 to 0.2,
# so we add the negative of the spectral norm to the loss.
# We want rank 1 to be around zero so we add an offset. 
# This will start around 5.8 (-0.2 + 6) and go down to 0 (-6 + 6).
function populatepenalties!(penalties, ys_hat::Array{Float32, 3})::Nothing
    @inbounds for i in axes(ys_hat, 3)
        penalties[i] = -approxspectralnorm(@view ys_hat[:,:,i]) + 6f0
    end
end

# Training loss - Many data points
function spectrum_penalized_l2(ys::Array{Float32, 3}, 
                          ys_hat::Matrix{Float32}, 
                          mask_mat::Matrix{Float32};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.01f0)::Float32
    l, n, nb_examples = size(ys)
    # L2 loss on known entries
    ys2 = reshape(ys, l * n, nb_examples)
    diff = vec(mask_mat) .* (ys2 .- ys_hat)
    l2_known = vec(sum(abs2, diff, dims = 1)) / sum(mask_mat)
    # Spectral norm penalty
    ys_hat3 = reshape(ys_hat, l, n, nb_examples)
    penalties = Array{Float32}(undef, nb_examples)
    populatepenalties!(penalties, ys_hat3)
    # Training loss
    left = theta / datascale * l2_known 
    right = (1f0 - theta)/2f0 * penalties # Scale to bring to the same order of magnitude as L2 loss
    errors = left .+ right
    return mean(errors)
end

#== Helper function for prediction loops ==#
function timemovement!(m, x, turns)::Nothing
    #@assert turns > 0
    @inbounds for _ in 1:turns
        m(x; selfreset = false)
    end
end
# No time steps, many examples
function populatepreds!(preds, m, xs::Array{Float32, 3})::Nothing
    @inbounds for i in axes(xs, 3)
        reset!(m)
        example = @view xs[:,:,i]
        preds[:,i] .= m(example; selfreset = false)
    end
end
function populatepreds!(preds, m, xs::Vector)::Nothing
    for i in eachindex(xs)
        reset!(m)
        preds[:,i] .= m(xs[i]; selfreset = false)
    end
end
# Many time steps, many examples
function populatepreds!(preds, m, xs::Array{Float32, 3}, turns)::Nothing
    @inbounds for i in axes(xs, 3)
        reset!(m)
        example = @view xs[:,:,i]
        timemovement!(m, example, turns)
        preds[:,i] .= m(example; selfreset = false)
    end
end
function populatepreds!(preds, m, xs::Vector, turns)::Nothing
    for i in eachindex(xs)
        reset!(m)
        timemovement!(m, xs[i], turns)
        preds[:,i] .= m(xs[i]; selfreset = false)
    end
end

# Predict - Single datum; no time step
function predict_through_time(m, 
                              x::Matrix{Float32})::Matrix{Float32}
    if m.rnn.h != m.rnn.init
        reset!(m)
    end
    preds = m(x; selfreset = false)
    return reshape(preds, :, 1)
end
# Predict - Single datum; with time steps
function predict_through_time(m, 
                              x::Matrix{Float32}, 
                              turns::Int)::Matrix{Float32}
    if m.rnn.h != m.rnn.init
        reset!(m)
    end
    timemovement!(m, x, turns)
    preds = m(x; selfreset = false)
    return reshape(preds, :, 1)
end

# Predict - Many data points in an array; no time step
function predict_through_time(m, 
                              xs::Array{Float32, 3})::Matrix{Float32}
    output_size = size(xs, 2)
    nb_examples = size(xs, 3)
    preds = Array{Float32}(undef, output_size, nb_examples)
    populatepreds!(preds, m, xs)
    return preds
end
# Predict - Many data points in an array; with time steps
function predict_through_time(m, 
                              xs::Array{Float32, 3}, 
                              turns::Int)::Matrix{Float32}
    output_size = size(xs, 2)
    nb_examples = size(xs, 3)
    preds = Array{Float32}(undef, output_size, nb_examples)
    populatepreds!(preds, m, xs, turns)
    return preds
end
#=
predict_through_time(activemodel, dataX[:,:,1:2], 1)
@benchmark predict_through_time($activemodel, $dataX[:,:,1])
@benchmark predict_through_time($activemodel, $dataX[:,:,1], 2)
# Forward pass, data points, no time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1:2]) 
# Forward pass, data points, 2 time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1:2], 2) =#

# Predict - Many data points in a vector of sparse matrices; no time step
function predict_through_time(m, 
                              xs::Vector)::Matrix{Float32}
    output_size = size(xs[1], 2)
    nb_examples = length(xs)
    preds = Array{Float32}(undef, output_size, nb_examples)
    populatepreds!(preds, m, xs)
    return preds
end
# Predict - Many data points in a vector of sparse matrices; with time steps
function predict_through_time(m, 
                              xs::Vector, 
                              turns::Int)::Matrix{Float32}
    output_size = size(xs[1], 2)
    nb_examples = length(xs)
    preds = Array{Float32}(undef, output_size, nb_examples)
    populatepreds!(preds, m, xs, turns)
    return preds
end

# Wrapper for prediction and loss
function trainingloss(m, xs, ys, mask_mat)
    ys_hat = predict_through_time(m, xs)
    return spectrum_penalized_l2(ys, ys_hat, mask_mat)
end
function trainingloss(m, xs, ys, mask_mat, turns)
    ys_hat = predict_through_time(m, xs, turns)
    return spectrum_penalized_l2(ys, ys_hat, mask_mat)
end

EnzymeRules.inactive(::typeof(reset!), args...) = nothing
EnzymeRules.inactive(::typeof(genbk), args...) = nothing

#====CREATE DATA====#
const K::Int = 400
const N::Int = 64
const RANK::Int = 1
# Create only one mini-batch
const DATASETSIZE::Int = 64
const KNOWNENTRIES::Int = 1500

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

function masktuple2array(fixedmask::Vector{Tuple{Int, Int}})
    k = length(fixedmask)    
    is = [x[1] for x in fixedmask]
    js = [x[2] for x in fixedmask]
    sparsemat = sparse(is, js, ones(k))
    return Matrix(sparsemat)
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
    X = []
    for i in 1:dataset_size
        inputmat = spzeros(Float32, k, M*N)
        entry_count = 1
        for agent in 1:k
            for l in 1:knownentries_per_agent[agent]
                row, col = masks[entry_count]
                flat_index = M * (col - 1) + row
                inputmat[agent, flat_index] = Y[row, col, i]
                entry_count += 1
            end
        end
        push!(X, inputmat)
    end
    return X
end

function datasetgeneration(m, n, rank, dataset_size, knownentries, net_width)
    Y = Array{Float32, 3}(undef, m, n, dataset_size)
    for i in 1:dataset_size
        Y[:, :, i] = creatematrix(m, n, rank, 1131+i)
    end
    fixedmask = sensingmasks(m, n; k=knownentries, seed=9632)
    mask_mat = Float32.(masktuple2array(fixedmask))
    @assert size(mask_mat) == (m, n)
    X = matrixinput_setup(Y, net_width, m, n, dataset_size, knownentries, fixedmask)
    @info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)
    return X, Y, fixedmask, mask_mat
end

sparse_dataX, dataY, fixedmask, mask_mat = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K)
dataX = stack(sparse_dataX)
# (Initially tried with sparse arrays, but that did not help for pullbacks, so focusing on regular arrays for now)

size(dataX), size(dataY)

#====INITIALIZE MODEL====#
activemodel = MatrixRNN(
    matrnn_constructor(N^2, K, N^2), 
    WeightedMeanLayer(K)
) 

#====TEST FWD PASS AND LOSS====#

using BenchmarkTools
#=
# Forward pass, one datum, no time step
@benchmark activemodel($dataX[:,:,1]) 
# Forward pass, one datum, no time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1]) 
# Forward pass, one datum, 2 time steps
@benchmark predict_through_time($activemodel, $dataX[:,:,1], 2) 

# Forward pass, all data points, no time steps
@benchmark predict_through_time($activemodel, $dataX) 
# Forward pass, all data points, 2 time steps
@benchmark predict_through_time($activemodel, $dataX, 2) 

ys_hat = predict_through_time(activemodel, dataX)
@benchmark spectrum_penalized_l2($dataY[:,:,1:2], $ys_hat[:,1:2], $mask_mat)

using InteractiveUtils
@code_warntype predict_through_time(activemodel, dataX) 
@code_warntype predict_through_time(activemodel, dataX, 2) 

# Training loss
@benchmark trainingloss($activemodel, $dataX, $dataY, $mask_mat)
=#
#====BEST GRADIENT COMPUTATION====#

# Enzyme

@btime loss, grads = Flux.withgradient((m,x) -> sum(m(x)), 
                                Duplicated(activemodel), 
                                dataX[:,:,1])

@btime loss, grads = Flux.withgradient((m,x) -> sum(predict_through_time(m, x)), 
                                Duplicated(activemodel), 
                                dataX)

@btime loss, grads = Flux.withgradient($trainingloss, 
                                $Duplicated(activemodel), 
                                $dataX[:,:,1], $dataY[:,:,1], $mask_mat)
# 279.431 ms (291 allocations: 241.41 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, 
                                $Duplicated(activemodel), 
                                $dataX[:,:,1:3], $dataY[:,:,1:3], $mask_mat)
# 551.074 ms (570 allocations: 322.92 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, 
                                $Duplicated(activemodel), 
                                $dataX, $dataY, $mask_mat)
# 23.378 s (9556 allocations: 4.86 GiB)

@btime loss, grads = Flux.withgradient($trainingloss, 
                                $Duplicated(activemodel), 
                                $dataX[:,:,1], $dataY[:,:,1], $mask_mat, 2)
# 855.062 ms (414 allocations: 391.53 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, 
                                $Duplicated(activemodel), 
                                $dataX[:,:,1:3], $dataY[:,:,1:3], $mask_mat, 2)
# 1.700 s (819 allocations: 623.15 MiB)

@btime loss, grads = Flux.withgradient($trainingloss, 
                                $Duplicated(activemodel), 
                                $dataX, $dataY, $mask_mat, 2)


#====TEST OTHER GRADIENT COMPUTATION====#

using DifferentiationInterface
fclosure(m) = trainingloss(m, dataX[:,:,1], dataY[:,:,1], mask_mat)
backend = AutoEnzyme()
@btime DifferentiationInterface.value_and_gradient($fclosure, $backend, $activemodel) 
# 12.429 s (8214 allocations: 2.98 GiB)

using Fluxperimental, Mooncake
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(genbk), Any}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(reset!), Any}
Mooncake.@mooncake_overlay norm(x) = sqrt(sum(abs2, x))
dup_model = Moonduo(activemodel)
fclosure(m) = trainingloss(m, dataX[:,:,1], dataY[:,:,1], mask_mat)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
# 1.281 s (1649063 allocations: 1.30 GiB)
fclosure(m) = trainingloss(m, dataX[:,:,3], dataY[:,:,3], mask_mat)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
# 
fclosure(m) = trainingloss(m, dataX[:,:,1], dataY[:,:,1], mask_mat, 2)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
# 2.157 s (3009 allocations: 2.20 GiB)
fclosure(m) = trainingloss(m, dataX[:,:,3], dataY[:,:,3], mask_mat, 2)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
#