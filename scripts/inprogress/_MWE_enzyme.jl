if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using CUDA, Adapt
#using Random
#using Distributions
using Flux
using Enzyme
using LinearAlgebra
@assert CUDA.functional() 

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

function(m::MatrixRnnCell)(state::CuArray{Float32, 2}, 
                           I::CuArray{Float32, 2}; 
                           selfreset::Bool=false)::CuArray{Float32}
    if selfreset
        h = m.init::CuArray{Float32, 2}
    else
        h = state
    end
    @assert size(I, 1) == size(h, 1) && size(I, 2) <= size(h, 2)
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

(a::WeightedMeanLayer)(X::CuArray{Float32}) = X' * sigmoid(a.weight) / sum(sigmoid(a.weight))

struct MatrixRNN{M<:MatrixRnnCell, D<:WeightedMeanLayer}
    rnn::M
    dec::D
end

reset!(m::MatrixRNN) = reset!(m.rnn)
state(m::MatrixRNN) = state(m.rnn)

(m::MatrixRNN)(x::CuArray{Float32}; 
            selfreset::Bool = false)::CuArray{Float32} = (m.dec ∘ m.rnn)(
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

function nnm_penalized_l2(m::MatrixRNN, 
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
    sql2 = device(sum(diff.^2, dims=1) / scaling)

    ys_hat_3d = reshape(ys_hat, l, n, nb_examples)
    nnm = device([sum(svdvals(ys_hat_3d[:,:,i])) for i in 1:nb_examples])
    
    errors = theta * (sql2' / l*n) .+ (1f0 - theta) * nnm / scaling

    return mean(errors)
end

function predict_through_time(m::MatrixRNN, 
                              xs::AbstractArray{Float32, 3}, 
                              turns::Int)
    examples = eachslice(xs; dims=3)
    # For each example, reset the state, recur for `turns` steps, 
    # predict the label, store it
    if turns == 0
        reset!(m)
        preds = stack(m.(examples; selfreset = true))
    elseif turns > 0    
        trial_output = m(examples[1])
        output_size = length(trial_output)
        @assert output_size > 1
        nb_examples = length(examples)
        preds = device(zeros(Float32, output_size, nb_examples))
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



dataY = randn(Float32, K, N^2, DATASETSIZE) |> device
dataX = dataY .+ 0.1f0 * randn(Float32, K, N^2, DATASETSIZE) |> device

activemodel = matnet(
    matrnn_constructor(N^2, K, N^2), 
    WeightedMeanLayer(K)
) |> device

activemodel(X[:,:,1])
InteractiveUtils.@code_warntype activemodel(X[:,:,1])
InteractiveUtils.@code_warntype activemodel.rnn(X[:,:,1])
temp = activemodel.rnn(X[:,:,1])
InteractiveUtils.@code_warntype activemodel.dec(temp)

predict_through_time(activemodel, dataX, 0)
predict_through_time(activemodel, dataX, 2)
InteractiveUtils.@code_warntype predict_through_time(activemodel, dataX, 0)
InteractiveUtils.@code_warntype predict_through_time(activemodel, dataX, 2)

nnm_penalized_l2(activemodel, dataX, dataY, 0)
nnm_penalized_l2(activemodel, dataX, dataY, 2)
InteractiveUtils.@code_warntype nnm_penalized_l2(activemodel, dataX, dataY, 0)
InteractiveUtils.@code_warntype nnm_penalized_l2(activemodel, dataX, dataY, 2)


grads = autodiff(Enzyme.Reverse, 
    myloss, Duplicated(activemodel), 
    Const(x), Const(y))

grads = autodiff(set_runtime_activity(Enzyme.Reverse), 
    myloss, Duplicated(activemodel), 
    Const(x), Const(y))

Enzyme.API.strictAliasing!(false)

grads = autodiff(set_runtime_activity(Enzyme.Reverse), 
    myloss, Duplicated(activemodel), 
    Const(x), Const(y))

