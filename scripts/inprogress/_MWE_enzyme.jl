if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using CUDA, Adapt
using Distributions
using Flux
using Enzyme
using LinearAlgebra
using Random
import NNlib
#@assert CUDA.functional() 

device = Flux.gpu_device()
uni(x) = cu(x; unified = true)

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
    M_in = NNlib.tanh_fast.(m.Wx_in * I' .+ m.bx_in') 
    newI = (m.Wx_out' * M_in .+ m.bx_out')' 
    h_new = NNlib.tanh_fast.(m.Whh * h .+ m.bh .+ newI) 
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

(a::WeightedMeanLayer)(X::AbstractArray{Float32}) = X' * NNlib.sigmoid_fast(a.weight) / sum(NNlib.sigmoid_fast(a.weight))

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

function power_iter(A::AbstractArray{Float32, 2}, b_k::AbstractArray{Float32, 1}; num_iterations::Int = 100)::Float32    
    for _ in 1:num_iterations
        b_k1 = A * b_k
        b_k .= b_k1 / norm(b_k1)
    end
    largest_eig = b_k' * (A * b_k)
    return largest_eig
end

function power_iter(A::CuArray{Float32, 2}, b_k::CuArray{Float32, 1}; num_iterations::Int = 100)::Float32
    for _ in 1:num_iterations
        b_k1 = A * b_k
        b_k .= b_k1 / norm(b_k1)
    end
    largest_eig = b_k' * (A * b_k)
    return largest_eig
end

function genbk(A::AbstractArray{Float32, 2}, n::Int)::AbstractArray{Float32, 1}
    b_k = rand(Float32, n)
    return b_k / norm(b_k)
end

function genbk(A::CuArray{Float32, 2}, n::Int)::CuArray{Float32, 1}
    b_k = CuArray(rand(Float32, n))
    return b_k / norm(b_k)
end

approxspectralnorm(A::AbstractArray{Float32, 2})::Float32 = sqrt(power_iter(A' * A, genbk(A, size(A,1))))
#b_k::CuArray{Float32, 1} = genbk(N)

function spectrum_penalized_l2(ys::AbstractArray{Float32, 3}, 
                          ys_hat::AbstractArray{Float32, 2}, 
                          mask_mat::AbstractArray{Float32, 2};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.01f0)::Float32
    l, n, nb_examples = size(ys)::Tuple{Int, Int, Int}
    ys = reshape(ys, (l*n), nb_examples)::AbstractArray{Float32, 2}
    diff = vec(mask_mat) .* (ys .- ys_hat)
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

function spectrum_penalized_l2(ys::CuArray{Float32, 3}, 
                          ys_hat::CuArray{Float32, 2}, 
                          mask_mat::CuArray{Float32, 2};
                          theta::Float32 = 0.8f0,
                          datascale::Float32 = 0.01f0)::Float32
    l, n, nb_examples = size(ys)::Tuple{Int, Int, Int}
    ys = reshape(ys, (l*n), nb_examples)::CuArray{Float32, 2}
    diff = vec(mask_mat) .* (ys .- ys_hat)::CuArray{Float32, 2}
    sql2_known = (vec(sum(diff.^2, dims = 1)) / sum(mask_mat))::CuVector{Float32}
    penalties = device(
        approxspectralnorm.(
            eachslice(
                reshape(ys_hat, l, n, nb_examples), 
                dims=3)))::CuVector{Float32}
    left = theta * (sql2_known / datascale)::CuVector{Float32}
    right = (1f0 - theta) * -(penalties .+ 2f0) ::CuVector{Float32}
    errors = left .+ right ::CuVector{Float32}
    return mean(errors)
end

MSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = mean(abs.((A .- B).^2), dims=1)
RMSE(A::AbstractArray{Float32}, B::AbstractArray{Float32}) = sqrt.(MSE(A, B))
MSE(A::CuArray{Float32}, B::CuArray{Float32}) = mean(abs.((A .- B).^2), dims=1)
RMSE(A::CuArray{Float32}, B::CuArray{Float32}) = sqrt.(MSE(A, B))

function allentriesRMSE(ys::AbstractArray{Float32, 3}, 
                        ys_hat::AbstractArray{Float32, 2};
                        datascale::Float32 = 0.1f0)::Float32
    l, n, nb_examples = size(ys)
    ys = reshape(ys, l * n, nb_examples) 
    RMSerrors = RMSE(ys, ys_hat) / datascale
    return mean(RMSerrors)
end

function allentriesRMSE(ys::CuArray{Float32, 3}, 
                        ys_hat::CuArray{Float32, 2};
                        datascale::Float32 = 0.1f0)::Float32
    l, n, nb_examples = size(ys)::Tuple{Int, Int, Int}
    ys = reshape(ys, l * n, nb_examples)::CuArray{Float32, 2}
    RMSerrors = RMSE(ys, ys_hat) / datascale ::Float32
    return mean(RMSerrors)
end

function predict_through_time(m::MatrixRNN, 
                              xs::AbstractArray{Float32, 3}, 
                              turns::Int)::AbstractArray{Float32, 2}
    trial_output = m(xs[:,:,1])::AbstractVector{Float32}
    output_size = length(trial_output)::Int
    nb_examples = size(xs, 3)::Int
    if turns == 0
        reset!(m)
        #preds = stack(m.(eachslice(xs; dims=3); selfreset = true))::AbstractArray{Float32, 2}
        preds = Array{Float32}(undef, output_size, nb_examples)
        i = 1
        @inbounds for example in eachslice(xs; dims=3)
            preds[:,i] .= m(example; selfreset = true)::AbstractVector{Float32}
            i += 1
        end
    elseif turns > 0 
        preds = Array{Float32}(undef, output_size, nb_examples)
        i = 1
        @inbounds for example in eachslice(xs; dims=3)
            reset!(m)
            for _ in 1:turns
                m(example; selfreset = false)
            end
            preds[:,i] .= m(example; selfreset = false)::AbstractVector{Float32}
            i += 1
        end
    end
    reset!(m)
    return preds
end

function predict_through_time(m::MatrixRNN, 
                              xs::Vector{Any}, 
                              turns::Int)::AbstractArray{Float32, 2}
    trial_output = m(xs[1])::AbstractVector{Float32}
    output_size = length(trial_output)::Int
    nb_examples = length(xs)::Int
    if turns == 0
        reset!(m)
        preds = Array{Float32}(undef, output_size, nb_examples)
        i = 1
        @inbounds for example in xs
            preds[:,i] .= m(example; selfreset = true)::AbstractVector{Float32}
            i += 1
        end
    elseif turns > 0 
        preds = Array{Float32}(undef, output_size, nb_examples)
        i = 1
        @inbounds for example in xs
            reset!(m)
            for _ in 1:turns
                m(example; selfreset = false)
            end
            preds[:,i] .= m(example; selfreset = false)::AbstractVector{Float32}
            i += 1
        end
    end
    reset!(m)
    return preds
end

function predict_through_time(m::MatrixRNN, 
                              xs::CuArray{Float32, 3}, 
                              turns::Int)::CuArray{Float32, 2}
    trial_output = m(xs[:,:,1])::CuVector{Float32}
    output_size = length(trial_output)::Int
    nb_examples = size(xs, 3)::Int
    if turns == 0
        reset!(m)
        #preds = stack(m.(eachslice(xs; dims=3); selfreset = true))::AbstractArray{Float32, 2}
        preds = CuArray{Float32}(undef, output_size, nb_examples)
        i = 1
        @inbounds for example in eachslice(xs; dims=3)
            preds[:,i] .= m(example; selfreset = true)::CuVector{Float32}
            i += 1
        end
    elseif turns > 0 
        preds = CuArray{Float32}(undef, output_size, nb_examples)
        i = 1
        @inbounds for example in eachslice(xs; dims=3)
            reset!(m)
            for _ in 1:turns
                m(example; selfreset = false)
            end
            preds[:,i] .= m(example; selfreset = false)::CuVector{Float32}
            i += 1
        end
    end
    reset!(m)
    return preds
end

function trainingloss(m, xs, ys, mask_mat, turns)
    ys_hat = predict_through_time(m, xs, turns)
    return spectrum_penalized_l2(ys, ys_hat, mask_mat)
end

function trainingloss(m::MatrixRNN, 
                      xs::AbstractArray{Float32, 3},
                      ys::AbstractArray{Float32, 3},
                      mask_mat::AbstractArray{Float32, 2},
                      turns::Int)::Float32
    ys_hat = predict_through_time(m, xs, turns)::AbstractArray{Float32, 2}
    return spectrum_penalized_l2(ys, ys_hat, mask_mat)
end

function trainingloss(m::MatrixRNN, 
                      xs::CuArray{Float32, 3},
                      ys::CuArray{Float32, 3},
                      mask_mat::CuArray{Float32, 2},
                      turns::Int)::Float32
    ys_hat = predict_through_time(m, xs, turns)::CuArray{Float32, 2}
    return spectrum_penalized_l2(ys, ys_hat, mask_mat)
end

#========#

const K::Int = 400
const N::Int = 64
const RANK::Int = 1
const DATASETSIZE::Int = 160
const MINIBATCH_SIZE::Int = 16
const KNOWNENTRIES::Int = 1500
const TRAIN_PROP::Float64 = 0.8
const VAL_PROP::Float64 = 0.1
const TEST_PROP::Float64 = 0.1

function creatematrix(m, n, r, seed; datatype=Float32)
    rng = Random.MersenneTwister(seed)
    #v = randn(rng, datatype, M, r) ./ sqrt(sqrt(Float32(r)))
    #A = v * v'
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

using SparseArrays
function masktuple2array(fixedmask::Vector{Tuple{Int, Int}})
    k = length(fixedmask)    
    is = [x[1] for x in fixedmask]
    js = [x[2] for x in fixedmask]
    sparsemat = sparse(is, js, ones(k))
    return Matrix(sparsemat)
end

function matinput_setup(Y::AbstractArray{Float32}, 
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
    #X = zeros(Float32, k, M*N, dataset_size)
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
    X = matinput_setup(Y, net_width, m, n, dataset_size, knownentries, fixedmask)
    @info("Memory usage after data generation: ", Base.gc_live_bytes() / 1024^3)
    return X, Y, fixedmask, mask_mat
end
dataX, dataY, fixedmask, mask_mat = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K)

function train_val_test_split(X::AbstractArray, 
                              train_prop::Real, val_prop::Real, test_prop::Real)
    @assert train_prop + val_prop + test_prop == 1.0
    dimsX = length(size(X))
    dataset_size = size(X, dimsX)
    train_nb = Int(train_prop * dataset_size)
    val_nb = Int(val_prop * dataset_size)
    train_idxs = 1:train_nb
    val_idxs = train_nb+1:train_nb+val_nb
    test_idxs = train_nb+val_nb+1:dataset_size
    if X isa Vector
        Xtrain, Xval, Xtest = X[train_idxs], X[val_idxs], X[test_idxs]
    elseif dimsX == 2
        Xtrain, Xval, Xtest = X[:,train_idxs], X[:,val_idxs], X[:,test_idxs]
    elseif dimsX == 3
        Xtrain, Xval, Xtest = X[:,:,train_idxs], X[:,:,val_idxs], X[:,:,test_idxs]
    else
        return @warn("Invalid number of dimensions for X: $dimsX")
    end
    @assert size(Xtrain, dimsX) == train_nb
    @assert size(Xval, dimsX) == val_nb
    @assert size(Xtest, dimsX) == dataset_size - train_nb - val_nb
    return Xtrain, Xval, Xtest
end

dataX, valX, testX = train_val_test_split(dataX, TRAIN_PROP, VAL_PROP, TEST_PROP)
dataY, valY, testY = train_val_test_split(dataY, TRAIN_PROP, VAL_PROP, TEST_PROP)
size(dataX), size(dataY)


# For X, use stack to convert Vector{SparseMatrixCSC} (which is not bitstypes) 
# to a 3D tensor of type Array{Float32, 3} that can be stored contiguously on GPU
dev_dataX, dev_dataY = device(stack(dataX)), device(dataY)
isbitstype(eltype(dev_dataX))
dev_valX, dev_valY = device(stack(valX)), device(valY)
dev_testX, dev_testY = device(stack(testX)), device(testY)
dev_mask_mat = device(mask_mat)

# Trivial data setup
#host_dataY = randn(Float32, N, N, DATASETSIZE) 
#host_dataX = randn(Float32, K, N*N, DATASETSIZE) 
#dataY = device(host_dataY)
#dataX = device(host_dataX) 
activemodel = MatrixRNN(
    matrnn_constructor(N^2, K, N^2), 
    WeightedMeanLayer(K)
) 
if CUDA.functional() 
    activemodel = activemodel |> uni
end

host_dataloader = Flux.DataLoader(
    (data=dataX, label=dataY), 
    batchsize=MINIBATCH_SIZE, 
    shuffle=true)
dev_dataloader = Flux.DataLoader(
    (data=dev_dataX, label=dev_dataY),
    batchsize=MINIBATCH_SIZE,
    shuffle=true)

# State optimizing rule
init_eta = 1e-4
decay = 0.7
EPOCHS::Int = 5
TURNS::Int = 0

using ParameterSchedulers
s = Exp(start = init_eta, decay = decay)
println("Learning rate schedule:")

for (eta, epoch) in zip(s, 1:EPOCHS)
    println(" - Epoch $epoch: eta = $eta")
end
opt = Adam()
# Tree of states
opt_state = Flux.setup(opt, activemodel)

# STORE INITIAL METRICS
train_loss = Float32[]
val_loss = Float32[]
train_rmse = Float32[]
val_rmse = Float32[]
Whh_spectra = []
push!(Whh_spectra, eigvals(activemodel.rnn.Whh |> cpu))

# Compute the initial training and validation loss with forward passes on GPU and store it back to CPU
dataYs_hat = predict_through_time(activemodel, dev_dataX, TURNS)
initloss_train = spectrum_penalized_l2(dev_dataY, dataYs_hat, dev_mask_mat)
initRMSE_train = allentriesRMSE(dev_dataY, dataYs_hat)

valYs_hat = predict_through_time(activemodel, dev_valX, TURNS)
initloss_val = spectrum_penalized_l2(dev_valY, valYs_hat, dev_mask_mat)
initRMSE_val = allentriesRMSE(dev_valY, valYs_hat)
push!(train_loss, initloss_train)
push!(train_rmse, initRMSE_train)
push!(val_loss, initloss_val)
push!(val_rmse, initRMSE_val)

##================##
using BenchmarkTools
@benchmark activemodel(dataX[1]) # CPU 62 ms
@benchmark activemodel(dev_dataX[:,:,1]) # CPU 110 ms
@benchmark predict_through_time(activemodel, dataX, 0) # CPU 1s
@benchmark predict_through_time(activemodel, dataX, 2) # CPU 2.6 s
@benchmark predict_through_time(activemodel, dev_dataX, 0) # CPU 1.6 s 
@benchmark predict_through_time(activemodel, dev_dataX, 2) # CPU 4.2 s

#= test = autodiff(set_runtime_activity(Reverse), 
    (x, y, z) -> sum(predict_through_time(x, y, z)), Duplicated(activemodel), 
    Const(dev_dataX), Const(0)) =#
#loss, grads = Flux.withgradient((x, y, z) -> sum(predict_through_time(x, y, z)), Duplicated(activemodel), dataX, 0)

test = autodiff(Enzyme.Reverse, 
    trainingloss, Active, Duplicated(activemodel), 
    Const(stack(dataX)), Const(dataY), Const(mask_mat), Const(0))

trainingloss(activemodel, dataX, dataY, mask_mat, 0)
loss, grads = Flux.withgradient(trainingloss, Duplicated(activemodel), stack(dataX), dataY, mask_mat, 0)
loss, grads = Flux.withgradient(trainingloss, Duplicated(activemodel), dev_dataX, dev_dataY, dev_mask_mat, 0)
@btime Flux.withgradient(trainingloss, Duplicated($activemodel), $dev_dataX, $dev_dataY, $dev_mask_mat, 0)

# 4-5 s (4109 allocations: 838 MiB)

#= DiffInterface
using DifferentiationInterface
import ReverseDiff, Enzyme, Zygote 
# choose a backend
backend = AutoReverseDiff()
# test
fclosure(m) = trainingloss(m, dev_dataX, dev_dataY, dev_mask_mat, 0)
DifferentiationInterface.value_and_gradient(fclosure, backend, Duplicated(activemodel)) 

# prepare the gradient calculation
#   preparation does not depend on the actual components of the vector x, just on its type and size
prep = DifferentiationInterface.prepare_gradient(f, backend, zero(x))
# pre allocate
grad = similar(x)
# compute
DifferentiationInterface.gradient!(f, grad, prep, backend, x)
y, grad = DifferentiationInterface.value_and_gradient!(f, grad, prep, backend, x)

# REVERSEDIFF 
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile, DiffResults
trainingloss(activemodel, dev_dataX, dev_dataY, mask_mat, 0)
inputs = (activemodel, dev_dataX, dev_dataY, dev_mask_mat, 0)
results = (activemodel, similar(dev_dataX), similar(dev_dataY))
all_results = map(DiffResults.GradientResult, results)
cfg = GradientConfig(inputs)

# pre-record a GradientTape for `f` using inputs of shape 100x100 with Float64 elements
const f_tape = GradientTape(f, inputs)
# compile `f_tape` into a more optimized representation
const compiled_f_tape = compile(f_tape)

ReverseDiff.gradient!(all_results, compiled_f_tape, inputs)
=#

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
for (eta, epoch) in zip(s, 1:EPOCHS)
    reset!(activemodel)
    println("Commencing epoch $epoch (eta = $(round(eta, digits=6)))")
    @info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
    Flux.adjust!(opt_state, eta = eta)
    # Initialize counters for gradient diagnostics
    mb, n, v, e = 1, 0, 0, 0
    # Iterate over minibatches
    for (x, y) in dev_dataloader
        # Pass twice over each minibatch (extra gradient learning)
        for _ in 1:2
            # Forward pass (to compute the loss) and backward pass (to compute the gradients)
            train_loss_value, grads = Flux.withgradient(trainingloss, Duplicated(activemodel), x, y, mask_mat, TURNS)
            GC.gc()
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
            GC.gc()
            if mb == 1 || mb % 5 == 0
                println("Minibatch $(epoch)--$(mb): loss of $(round(train_loss_value, digits=4))")
                #@info("Memory usage: ", Base.gc_live_bytes() / 1024^3)
            end
            mb += 1
        end
    end
    push!(Whh_spectra, eigvals(activemodel.rnn.Whh |> cpu))
    # Print a summary of the gradient diagnostics for the epoch
    diagnose_gradients(n, v, e)
    # Compute training metrics -- this is a very expensive operation because it involves a forward pass over the entire training set
    # so I take a subset of the training set to compute the metrics
    dataYs_hat = predict_through_time(activemodel, dev_dataX, TURNS)
    trainloss = spectrum_penalized_l2(dev_dataY, dataYs_hat, dev_mask_mat)
    trainRMSE = allentriesRMSE(dev_dataY, dataYs_hat)
    push!(train_loss, trainloss)
    push!(train_rmse, trainRMSE)
    # Compute validation metrics
    valYs_hat = predict_through_time(activemodel, dev_valX, TURNS)
    valloss = spectrum_penalized_l2(dev_valY, valYs_hat, mask_mat)
    valRMSE = allentriesRMSE(dev_valY, valYs_hat)
    push!(val_loss, valloss)
    push!(val_rmse, valRMSE)
    println("Epoch $epoch: Train loss: $(train_loss[end]); train RMSE: $(train_rmse[end])")
    println("Epoch $epoch: Val loss: $(val_loss[end]); val RMSE: $(val_rmse[end])")
    # Check if validation loss has increased for 2 epochs in a row; if so, stop training
    if length(val_loss) > 2
        if val_loss[end] > val_loss[end-1] && val_loss[end-1] > val_loss[end-2]
            @warn("Early stopping at epoch $epoch")
            break
        end
    end
end
endtime = time()
# training time in minutes
println("Training time: $(round((endtime - starttime) / 60, digits=2)) minutes")
# Assess on testing set
testYs_hat = predict_through_time(activemodel, dev_testX, TURNS)
testloss = spectrum_penalized_l2(dev_testY, testYs_hat, mask_mat)
testRMSE = allentriesRMSE(dev_testY, testYs_hat)
println("Test RMSE: $(testRMSE)")
println("Test loss: $(testloss)")

#=============#

using CairoMakie
lossname = "Mean nuclear-norm penalized l2 loss (known entries)"
rmsename = "Root mean squared reconstr. error / std (all entries)"
tasklab = "Reconstructing $(N)x$(N) rank-$(RANK) matrices from $(KNOWNENTRIES) of their entries"
taskfilename = "$(N)recon_rank$(RANK)"
modlabel = "Matrix Vanilla"

CairoMakie.activate!()
fig = Figure(size = (820, 450))
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
    "Optimizer: Adam with schedule Exp(start = $(init_eta), decay = $(decay))\n"*
    #"Training time: $(round((endtime - starttime) / 60, digits=2)) minutes; "*
    "Test loss: $(round(testloss,digits=4))."*
    "Test RMSE: $(round(testRMSE,digits=4)).",
    #"Optimizer: AdamW(eta=$eta, beta=$beta, decay=$decay)",
    fontsize = 14,
    padding = (0, 0, 0, 0),
)
fig

save("data/$(taskfilename)_$(modlabel)RNNwidth$(net_width)_$(TURNS)turns_knownentries.png", fig)
