if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using Flux
using Enzyme, Zygote 
import NNlib
import ChainRules
using LinearAlgebra
using Distributions
using Random
using SparseArrays

#====CREATE CUSTOM RNN====#

# CREATE STATEFUL, MATRIX-VALUED RNN LAYER
mutable struct MatrixRnnCell{A<:Matrix{Float32}}
    Wx_in::A
    Wx_out::A
    bx_in::A
    bx_out::A
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

function(m::MatrixRnnCell)(state::Matrix{Float32}, 
                           I::AbstractArray{Float32, 2}; 
                           selfreset::Bool=false)::Matrix{Float32}
    if selfreset
        h = m.init::Matrix{Float32}
    else
        h = state
    end
    M_in = NNlib.tanh_fast.(m.Wx_in * I' .+ m.bx_in') 
    newI = (m.Wx_out' * M_in .+ m.bx_out')' 
    h_new = NNlib.tanh_fast.(m.Whh * h .+ m.bh .+ newI) ::Matrix{Float32}
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
                state(m), x; 
                selfreset = selfreset)
            
Flux.@layer MatrixRnnCell trainable=(Wx_in, Wx_out, bx_in, bx_out, Whh, bh)
Flux.@layer WeightedMeanLayer
Flux.@layer :expand MatrixRNN trainable=(rnn, dec)
Flux.@non_differentiable reset!(m::MatrixRnnCell)
Flux.@non_differentiable reset!(m::MatrixRNN)
EnzymeRules.inactive(::typeof(reset!), args...) = nothing

#====DEFINE LOSS FUNCTIONS====#
function power_iter(A::AbstractArray{Float32, 2}, b_k::Vector{Float32}; num_iterations::Int = 50)::Float32    
    for _ in 1:num_iterations
        b_k1 = A * b_k
        b_k = b_k1 / norm(b_k1)
    end
    largest_eig = b_k' * (A * b_k)
    return largest_eig
end

# Generate random vector for power iteration; must not get differentiated
# (Probably not the best way to do this!)
function genbk(A)::Vector{Float32}
    b_k = rand(Float32, size(A, 1))
    return b_k / norm(b_k)
end
Flux.@non_differentiable genbk(A)
EnzymeRules.inactive(::typeof(genbk), args...) = nothing

approxspectralnorm(A::AbstractArray{Float32, 2}, b_k::Vector{Float32} = genbk(A))::Float32 = sqrt(power_iter(A' * A, b_k))

function spectrum_penalized_l2(ys::Array{Float32, 3}, 
                          ys_hat::Matrix{Float32}, 
                          mask_mat::Matrix{Float32};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.01f0)::Float32
    l, n, nb_examples = size(ys)::Tuple{Int, Int, Int}
    ys2 = reshape(ys, l * n, nb_examples)::Matrix{Float32}
    diff = vec(mask_mat) .* (ys2 .- ys_hat)
    sql2_known = (vec(sum(diff.^2, dims = 1)) / sum(mask_mat))
    penalties = approxspectralnorm.(
            eachslice(
                reshape(ys_hat, l, n, nb_examples), 
                dims=3))
    left = theta * (sql2_known / datascale)
    right = (1f0 - theta) * -(penalties .+ 2f0)
    errors = left .+ right 
    return mean(errors)
end

function predict_through_time(m, 
                              xs::Array{Float32, 3}, 
                              turns::Int)::Matrix{Float32}
    trial_output = m(xs[:,:,1])::Vector{Float32}
    output_size = length(trial_output)::Int
    nb_examples = size(xs, 3)::Int

    preds = Array{Float32}(undef, output_size, nb_examples) # Enzyme version
    # preds = Zygote.Buffer(zeros(Float32, output_size, nb_examples)) # Zygote version
    i = 1
    @inbounds for example in eachslice(xs; dims=3)
        if turns == 0
            preds[:,i] = m(example; selfreset = true)::Vector{Float32}
        elseif turns > 0 
            reset!(m)
            for _ in 1:turns
                m(example; selfreset = false)
            end
            preds[:,i] = m(example; selfreset = false)::Vector{Float32}
        end
        i += 1
    end
    reset!(m)
    return preds
end

function predict_through_time(m, 
                              xs::Vector, 
                              turns::Int)::Matrix{Float32}
    trial_output = m(xs[1])::Vector{Float32}
    output_size = length(trial_output)::Int
    nb_examples = length(xs)::Int

    preds = Array{Float32}(undef, output_size, nb_examples) # Enzyme version
    # preds = Zygote.Buffer(zeros(Float32, output_size, nb_examples)) # Zygote version
        
    i = 1
    @inbounds for example in xs
        if turns == 0
            preds[:,i] = m(example; selfreset = true)::Vector{Float32}
        elseif turns > 0
            reset!(m)
            for _ in 1:turns
                m(example; selfreset = false)
            end
            preds[:,i] = m(example; selfreset = false)::Vector{Float32}
        end
        i += 1
    end
    reset!(m)
    return preds
end

function trainingloss(m, xs, ys, mask_mat, turns)
    ys_hat = predict_through_time(m, xs, turns)
    return spectrum_penalized_l2(ys, ys_hat, mask_mat)
end

#====CREATE DATA====#
const K::Int = 400
const N::Int = 64
const RANK::Int = 1
const DATASETSIZE::Int = 160
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

const sparse_dataX, dataY, fixedmask, mask_mat = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K)
const dense_dataX = stack(sparse_dataX)

size(sparse_dataX), size(sparse_dataX[1]), size(dense_dataX), size(dataY)

#====INITIALIZE MODEL====#
activemodel = MatrixRNN(
    matrnn_constructor(N^2, K, N^2), 
    WeightedMeanLayer(K)
) 

#====TEST FWD PASS AND LOSS====#
#dataYs_hat = predict_through_time(activemodel, sparse_dataX, 0)
#loss = spectrum_penalized_l2(dataY, dataYs_hat, mask_mat)

using BenchmarkTools
# Forward pass, one data point, no time step
@benchmark activemodel($sparse_dataX[1]) 
@benchmark activemodel($dense_dataX[:,:,1]) 
# Forward pass, all data points, no time steps
@benchmark predict_through_time($activemodel, $sparse_dataX, 0) 
@benchmark predict_through_time($activemodel, $dense_dataX, 0) 
# Forward pass, all data points, 2 time steps
@benchmark predict_through_time($activemodel, $sparse_dataX, 2) 
@benchmark predict_through_time($activemodel, $dense_dataX, 2) 

# Training loss
@benchmark trainingloss($activemodel, $sparse_dataX, $dataY, $mask_mat, 0)

#====BEST GRADIENT COMPUTATION====#

# Enzyme
@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), $dense_dataX[:,:,1:32], $dataY[:,:,1:32], $mask_mat, 0) 
# 10.417 s (8300 allocations: 2.79 GiB)

loss, grads = Flux.withgradient(trainingloss, Duplicated(activemodel), dense_dataX[:,:,1:32], dataY[:,:,1:32], mask_mat, 0) 
@btime loss, grads = Flux.withgradient($trainingloss, $Duplicated(activemodel), $dense_dataX[:,:,1:32], $dataY[:,:,1:32], $mask_mat, 0) 

#====TEST OTHER GRADIENT COMPUTATION====#
# Zygote
@btime loss, grads = Flux.withgradient($trainingloss, $activemodel, $dense_dataX[:,:,1:32], $dataY[:,:,1:32], $mask_mat, 0) 
# 21.846 s (60940 allocations: 16.77 GiB)
@btime loss, grads = Flux.withgradient($trainingloss, $activemodel, $sparse_dataX[1:32], $dataY[:,:,1:32], $mask_mat, 0) 
# 19.257 s (64218 allocations: 16.44 GiB)

using DifferentiationInterface
fclosure(m) = trainingloss(m, dense_dataX[:,:,1:32], dataY[:,:,1:32], mask_mat, 0)
backend = AutoZygote()
@btime DifferentiationInterface.value_and_gradient($fclosure, $backend, $activemodel) 
# 22.561 s (61023 allocations: 17.75 GiB)

backend = AutoEnzyme()
@btime DifferentiationInterface.value_and_gradient($fclosure, $backend, $activemodel) 
# 12.429 s (8214 allocations: 2.98 GiB)

using Fluxperimental, Mooncake
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(genbk), Any}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(reset!), Any}
Mooncake.@mooncake_overlay norm(x) = sqrt(sum(abs2, x))
dup_model = Moonduo(activemodel)
fclosure(m) = trainingloss(m, sparse_dataX[1:16], dataY[:,:,1:16], mask_mat, 0)
@btime loss, grads = Flux.withgradient($fclosure, $dup_model)
# 87.678 s (12724 allocations: 18.64 GiB) for half the mini-batch size
