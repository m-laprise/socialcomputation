#= Testing using: 
Julia 1.10.5
Flux 0.16.3
Enzyme 0.13.30
=#

using Flux
using Enzyme
import NNlib
using LinearAlgebra

#====CREATE CUSTOM RNN====#

# CREATE STATEFUL, MATRIX-VALUED RNN LAYER
mutable struct MatrixRnnCell{A<:Matrix{Float32}}
    Whh::A # Recurrent weights
    bh::A  # Recurrent bias
    h::A   # Current state
    const init::A # Store initial state h for reset
end

function matrnn_constructor(n::Int, k::Int)::MatrixRnnCell
    Whh = randn(Float32, k, k) / sqrt(Float32(k))
    bh = randn(Float32, k, n) / sqrt(Float32(k))
    h = randn(Float32, k, n) * 0.01f0
    return MatrixRnnCell(Whh, bh, h, h)
end

function(m::MatrixRnnCell)(state::Matrix{Float32}, 
                           I::AbstractArray{Float32, 2}; 
                           selfreset::Bool=false)::Matrix{Float32}
    if selfreset
        h = m.init
    else
        h = state
    end
    h_new = NNlib.tanh_fast.(m.Whh * h .+ m.bh .+ I) 
    m.h = h_new
    return h_new 
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
(a::WeightedMeanLayer)(X::Array{Float32}) = X' * NNlib.sigmoid_fast(a.weight) / sum(NNlib.sigmoid_fast(a.weight))

# CHAIN RNN AND WEIGHTED MEAN LAYERS
struct MatrixRNN{M<:MatrixRnnCell, D<:WeightedMeanLayer}
    rnn::M
    dec::D
end
state(m::MatrixRNN) = state(m.rnn)
reset!(m::MatrixRNN) = reset!(m.rnn)
(m::MatrixRNN)(x::AbstractArray{Float32,2}; 
               selfreset::Bool = false)::Vector{Float32} = (m.dec ∘ m.rnn)(
                state(m), x; selfreset = selfreset)
            
Flux.@layer MatrixRnnCell trainable=(Whh, bh)
Flux.@layer WeightedMeanLayer
Flux.@layer :expand MatrixRNN trainable=(rnn, dec)
Flux.@non_differentiable reset!(m::MatrixRnnCell)
Flux.@non_differentiable reset!(m::MatrixRNN)
EnzymeRules.inactive(::typeof(reset!), args...) = nothing

#====DEFINE LOSS FUNCTIONS====#
# Helper function for prediction loops
# To avoid the segfault, comment out the timemovement! function here, and its call on line 85 and 97.
function timemovement!(m, x, turns)::Nothing
    for _ in 1:turns
        m(x; selfreset = false)
    end
end
function populatepreds!(preds, m, xs::Array{Float32, 3}, turns)::Nothing
    for i in axes(xs, 3)
        reset!(m)
        example = @view xs[:,:,i]
        timemovement!(m, example, turns)
        preds[:,i] .= m(example; selfreset = false)
    end
end

# Predict - Single datum
function modelpredict(m, 
                      x::Matrix{Float32}, 
                      turns::Int)::Matrix{Float32}
    if m.rnn.h != m.rnn.init
        reset!(m)
    end
    timemovement!(m, x, turns)
    preds = m(x; selfreset = false)
    return reshape(preds, :, 1)
end

# Predict - Many data points in an array
function modelpredict(m, 
                      xs::Array{Float32, 3}, 
                      turns::Int)::Matrix{Float32}
    output_size = size(xs, 2)
    nb_examples = size(xs, 3)
    preds = Array{Float32}(undef, output_size, nb_examples)
    populatepreds!(preds, m, xs, turns)
    return preds
end

# Wrapper for prediction and a trivial loss for MWE
function trainingloss(m, xs, ys, turns)
    ys_hat = modelpredict(m, xs, turns)
    return mean(abs2, vec(ys) .- vec(ys_hat))
end

#====CREATE TRIVIAL DATA====#
const K::Int = 400
const N::Int = 64
const DATASETSIZE::Int = 64
dataY = randn(Float32, N, N, DATASETSIZE) 
dataX = randn(Float32, K, N*N, DATASETSIZE) 

#====INITIALIZE MODEL====#
activemodel = MatrixRNN(
    matrnn_constructor(N^2, K), 
    WeightedMeanLayer(K)) 
# Test loss function, entire dataset 
using BenchmarkTools
@btime trainingloss($activemodel, $dataX, $dataY, 0)

#====GRADIENT COMPUTATION====#
# Training loss, one data point, no time step -- this works.
loss, grads = Flux.withgradient(trainingloss, 
                                Duplicated(activemodel), 
                                dataX[:,:,1], 
                                dataY[:,:,1], 0)

# Training loss, 2 (or more) data points, no time step -- segfault.
loss, grads = Flux.withgradient(trainingloss, 
                                Duplicated(activemodel), 
                                dataX[:,:,1:2], 
                                dataY[:,:,1:2], 0)

loss, grads = Flux.withgradient(trainingloss, 
                                Duplicated(activemodel), 
                                dataX[:,:,1:32], 
                                dataY[:,:,1:32], 0) 
