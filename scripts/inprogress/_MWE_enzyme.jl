if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using CUDA, Adapt
#using Random
using Distributions
using Flux
using Enzyme
using LinearAlgebra
@assert CUDA.functional() 

# CREATE STATEFUL, MATRIX-VALUED RNN LAYER
mutable struct MatrixRnnCell{A<:AbstractArray{Float32,2}}
    Wx_in::A
    Wx_out::A
    bx_in::A
    bx_out::A
    Whh::A
    bh::A
    h::A
    const init::A # store initial state h for reset
end

function matrnn_constructor(n::Int, 
                     k::Int,
                     m::Int)::MatrixRnnCell
    Whh = randn(Float32, k, k) / sqrt(Float32(k))
    Wx_in = randn(Float32, m, n) / sqrt(Float32(m))
    Wx_out = randn(Float32, m, n) / sqrt(Float32(m))
    bh = randn(Float32, k, n) / sqrt(Float32(k))
    bx_in = randn(Float32, k, m) / sqrt(Float32(k))
    bx_out = randn(Float32, k, n) / sqrt(Float32(k))
    h = randn(Float32, k, n) * 0.01f0
    return MatrixRnnCell(
        Wx_in, Wx_out, bx_in, bx_out,
        Whh, bh,
        h, h
    )
end

function(m::MatrixRnnCell)(state::AbstractArray{Float32, 2}, 
                           I::AbstractArray{Float32, 2}; 
                           selfreset::Bool=false)::AbstractArray{Float32}
    if selfreset
        h = m.init::AbstractArray{Float32, 2}
    else
        h = state
    end
    M_in = tanh.(m.Wx_in * I' .+ m.bx_in') 
    newI = (m.Wx_out' * M_in .+ m.bx_out')' 
    h_new = tanh.(m.Whh * h .+ m.bh .+ newI) 
    m.h = h_new
    return h_new 
end

state(m::MatrixRnnCell) = m.h
reset!(m::MatrixRnnCell) = (m.h = m.init)

struct WeightedMeanLayer{V<:AbstractVector{Float32}}
    weight::V
end
function WeightedMeanLayer(num::Int;
                           init = ones)
    weight_init = init(Float32, num) * 5f0
    WeightedMeanLayer(weight_init)
end

(a::WeightedMeanLayer)(X::AbstractArray{Float32}) = X' * sigmoid(a.weight) / sum(sigmoid(a.weight))

struct MatrixRNN{M<:MatrixRnnCell, D<:WeightedMeanLayer}
    rnn::M
    dec::D
end

reset!(m::MatrixRNN) = reset!(m.rnn)
state(m::MatrixRNN) = state(m.rnn)

(m::MatrixRNN)(x::AbstractArray{Float32}; 
               selfreset::Bool = false)::AbstractArray{Float32} = (m.dec ∘ m.rnn)(
                state(m), x; 
                selfreset = selfreset)


Flux.@layer MatrixRnnCell trainable=(Wx_in, Wx_out, bx_in, bx_out, Whh, bh)
Flux.@layer WeightedMeanLayer
Flux.@layer :expand MatrixRNN trainable=(rnn, dec)
Flux.@non_differentiable reset!(m::MatrixRnnCell)
Flux.@non_differentiable reset!(m::MatrixRNN)

Adapt.@adapt_structure MatrixRnnCell
Adapt.@adapt_structure WeightedMeanLayer
Adapt.@adapt_structure MatrixRNN


function power_iter(A::AbstractArray{Float32, 2}; num_iterations::Int = 100)::Float32
    b_k = rand(Float32, size(A, 2))::AbstractArray{Float32, 1}
    b_k /= norm(b_k)
    for _ in 1:num_iterations
        b_k1 = A * b_k
        b_k .= b_k1 / norm(b_k1)
    end
    largest_eig = b_k' * (A * b_k)
    return largest_eig
end

approxspectralnorm(A::AbstractArray{Float32, 2})::Float32 = sqrt(power_iter(A' * A))

function spectrum_penalized_l2(m::MatrixRNN, 
                          xs::AbstractArray{Float32, 3}, 
                          ys::AbstractArray{Float32, 3}, 
                          turns::Int = 0;
                          theta::Float32 = 0.8f0,
                          scaling::Float32 = 0.1f0)::Float32
    l, n, nb_examples = size(ys)

    ys_2d = reshape(ys, l*n, nb_examples) 
    ys_hat = predict_through_time(m, xs, turns) 
    @assert size(ys_hat) == size(ys_2d)

    diff = ys_2d .- ys_hat
    sql2 = sum(diff.^2, dims=1) / scaling #|> device

    ys_hat_3d = reshape(ys_hat, l, n, nb_examples)
    penalties = [approxspectralnorm(ys_hat_3d[:,:,i]) for i in 1:nb_examples] #|> device
    
    errors = theta * (sql2' / l*n) .+ (1f0 - theta) * -penalties / scaling

    return mean(errors)
end

function predict_through_time(m::MatrixRNN, 
                              xs::AbstractArray{Float32, 3}, 
                              turns::Int)::AbstractArray{Float32, 2}
    examples = eachslice(xs; dims=3)
    # For each example, reset the state, recur for `turns` steps, 
    # predict the label, store it
    if turns == 0
        reset!(m)
        preds = stack(m.(examples; selfreset = true))
    elseif turns > 0    
        trial_output = m(examples[1])
        output_size = length(trial_output)
        nb_examples = length(examples)
        preds = Array{Float32}(undef, output_size, nb_examples) 
        @inbounds for (i, example) in enumerate(examples)
            reset!(m)
            repeatedexample = [example for _ in 1:turns+1]
            successive_answers = stack(m.(repeatedexample; selfreset = false))
            pred = successive_answers[:,end]
            preds[:,i] .+= pred
        end
    end
    reset!(m)
    return preds
end

#========#

const K::Int = 400
const N::Int = 64
const DATASETSIZE::Int = 10

dataY = randn(Float32, N, N, DATASETSIZE) |> device
dataX = randn(Float32, K, N*N, DATASETSIZE) |> device

activemodel = MatrixRNN(
    matrnn_constructor(N^2, K, N^2), 
    WeightedMeanLayer(K)
) |> device

activemodel(dataX[:,:,1])
InteractiveUtils.@code_warntype activemodel(dataX[:,:,1])
InteractiveUtils.@code_warntype activemodel.rnn(state(activemodel), dataX[:,:,1])
temp = activemodel.rnn(state(activemodel), dataX[:,:,1])
InteractiveUtils.@code_warntype activemodel.dec(temp)

predict_through_time(activemodel, dataX, 0)
predict_through_time(activemodel, dataX, 2)
InteractiveUtils.@code_warntype predict_through_time(activemodel, dataX, 0)
InteractiveUtils.@code_warntype predict_through_time(activemodel, dataX, 2)

spectrum_penalized_l2(activemodel, dataX, dataY, 0)
spectrum_penalized_l2(activemodel, dataX, dataY, 2)
InteractiveUtils.@code_warntype spectrum_penalized_l2(activemodel, dataX, dataY, 0)
InteractiveUtils.@code_warntype spectrum_penalized_l2(activemodel, dataX, dataY, 2)

grads = autodiff(Enzyme.Reverse, 
    spectrum_penalized_l2, Duplicated(activemodel), 
    Const(dataX), Const(dataY))

grads = autodiff(set_runtime_activity(Enzyme.Reverse), 
    spectrum_penalized_l2, Duplicated(activemodel), 
    Const(dataX), Const(dataY))

loss, grads = Flux.withgradient(spectrum_penalized_l2, Duplicated(activemodel), dataX, dataY)

