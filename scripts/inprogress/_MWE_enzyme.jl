if Sys.CPU_NAME != "apple-m1"
    import Pkg
    Pkg.activate("../")
end

using CUDA, Adapt
using Distributions
using Flux
using Enzyme
using LinearAlgebra
@assert CUDA.functional() 

device = Flux.gpu

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

function power_iter(A::CuArray{Float32, 2}; num_iterations::Int = 100)::Float32
    b_k = device(rand(Float32, size(A, 2)))::CuArray{Float32, 1}
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
                          scaling::Float32 = 1f0)::Float32
    l, n, nb_examples = size(ys)

    ys_2d = reshape(ys, l*n, nb_examples) 
    ys_hat = predict_through_time(m, xs, turns) 
    #@assert size(ys_hat) == size(ys_2d)
    diff = ys_2d .- ys_hat
    sql2 = sum(diff.^2, dims=1) / scaling 
    ys_hat_3d = reshape(ys_hat, l, n, nb_examples)
    penalties = device(approxspectralnorm.(eachslice(ys_hat_3d, dims=3)))
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
        preds = CuArray{Float32}(undef, output_size, nb_examples) 
        @inbounds for (i, example) in enumerate(examples)
            reset!(m)
            repeatedexample = [example for _ in 1:turns+1]
            successive_answers = device(stack(m.(repeatedexample; selfreset = false)))
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

using BenchmarkTools
@benchmark activemodel(dataX[:,:,1]) # CPU 127 ms
@benchmark predict_through_time(activemodel, dataX, 0) # CPU 1.184 s
@benchmark predict_through_time(activemodel, dataX, 2)
@benchmark spectrum_penalized_l2(activemodel, dataX, dataY, 0) # CPU 1.266 s
@benchmark spectrum_penalized_l2(activemodel, dataX, dataY, 2)

function rnn_kernel!(state, activemodel, example)
    state .= activemodel.rnn(state, example)
    return nothing
end
h = state(activemodel)
@cuda threads=256 blocks=1 rnn_kernel!(h, activemodel, dataX[:,:,1])

function activemodel_kernel!(ans, state, activemodel, example)
    ans = activemodel(example)
    state = activemodel.rnn.h
    return nothing
end

ans = CuArray{Float32}(undef, N*N, 1)
@cuda threads=256 blocks=1 activemodel_kernel!(ans, state(activemodel), activemodel, dataX[:,:,1])

function predict_kernel!(array_ans, activemodel, dataX, turns)
    array_ans .= predict_through_time(activemodel, dataX, turns)
    return nothing
end
array_ans = CuArray{Float32}(undef, N, N, DATASETSIZE)
@cuda threads=256 blocks=1 predict_kernel!(array_ans, activemodel, dataX, 0)
@cuda threads=256 blocks=1 predict_kernel!(array_ans, activemodel, dataX, 2)

function loss_kernel!(ans, activemodel, dataX, dataY, turns)
    ans .= spectrum_penalized_l2(activemodel, dataX, dataY, turns)
    return nothing
end
ans = 0f0
@cuda threads=256 blocks=1 loss_kernel!(ans, activemodel, dataX, dataY, 0)
@cuda threads=256 blocks=1 loss_kernel!(ans, activemodel, dataX, dataY, 2)


activemodel(dataX[:,:,1])
InteractiveUtils.@code_warntype activemodel(dataX[:,:,1])
InteractiveUtils.@code_warntype activemodel.rnn(state(activemodel), dataX[:,:,1])
temp = activemodel.rnn(state(activemodel), dataX[:,:,1])
InteractiveUtils.@code_warntype activemodel.dec(temp)

predict_through_time(activemodel, dataX, 0)
predict_through_time(activemodel, dataX, 2)
InteractiveUtils.@code_warntype predict_through_time(activemodel, dataX, 0)
InteractiveUtils.@code_warntype predict_through_time(activemodel, dataX, 2)

test = autodiff(set_runtime_activity(Reverse), 
    (x, y, z) -> sum(predict_through_time(x, y, z)), Duplicated(activemodel), 
    Const(dataX), Const(0))

spectrum_penalized_l2(activemodel, dataX, dataY, 0)
spectrum_penalized_l2(activemodel, dataX, dataY, 2)
InteractiveUtils.@code_warntype spectrum_penalized_l2(activemodel, dataX, dataY, 0)
InteractiveUtils.@code_warntype spectrum_penalized_l2(activemodel, dataX, dataY, 2)

test = autodiff(Enzyme.Reverse, 
    spectrum_penalized_l2, Duplicated(activemodel), 
    Const(dataX), Const(dataY))

loss, grads = Flux.withgradient(spectrum_penalized_l2, Duplicated(activemodel), dataX, dataY)

#tell enzyme it doesn’t need to differentiate the cuBLAS parallel stream setup
function Enzyme.EnzymeRules.inactive(::typeof(CUDA.CUBLAS.handle))
    return nothing
end

#=
Note:
Putting all the data on the device is not sufficient, you also need to call a CUDA.jl kernel 
so the function itself is offloaded to the device. 
If you are not using @cuda somewhere, it won’t work with current Enzyme.
See the Enzyme tests for an example of GPU kernels working with Enzyme:
    https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl
=#


#=
1-element ExceptionStack:
MethodError: no method matching my_methodinstance(::Type{typeof(Enzyme.Compiler.custom_rule_method_error)}, ::Type{Tuple{UInt64, typeof(Core.kwcall), @NamedTuple{dims::Int64, init::Nothing}, typeof(EnzymeCore.EnzymeRules.reverse), EnzymeCore.EnzymeRules.RevConfigWidth{1, true, true, (false, false, false, true), false}, Const{typeof(GPUArrays._mapreduce)}, Type{Duplicated{CuArray{Float32, 2, CUDA.DeviceMemory}}}, Any, Const{typeof(identity)}, Const{typeof(Base.add_sum)}, Duplicated{CuArray{Float32, 2, CUDA.DeviceMemory}}}}, ::UInt64)

Closest candidates are:
  my_methodinstance(::Core.Compiler.AbstractInterpreter, ::Type, ::Type)
   @ Enzyme ~/.julia/packages/Enzyme/R6sE8/src/utils.jl:266
  my_methodinstance(::Core.Compiler.AbstractInterpreter, ::Type, ::Type, ::Union{Nothing, Base.RefValue{UInt64}})
   @ Enzyme ~/.julia/packages/Enzyme/R6sE8/src/utils.jl:266
  my_methodinstance(::Core.Compiler.AbstractInterpreter, ::Type, ::Type, ::Union{Nothing, Base.RefValue{UInt64}}, ::Union{Nothing, Base.RefValue{UInt64}})
   @ Enzyme ~/.julia/packages/Enzyme/R6sE8/src/utils.jl:266
  ...

Stacktrace:
  [1] enzyme_custom_common_rev(forward::Bool, B::LLVM.IRBuilder, orig::LLVM.CallInst, gutils::Enzyme.Compiler.GradientUtils, normalR::Ptr{Ptr{LLVM.API.LLVMOpaqueValue}}, shadowR::Ptr{Ptr{LLVM.API.LLVMOpaqueValue}}, tape::LLVM.ExtractValueInst)
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/rules/customrules.jl:992
  [2] enzyme_custom_rev(B::LLVM.IRBuilder, orig::LLVM.CallInst, gutils::Enzyme.Compiler.GradientUtils, tape::Union{Nothing, LLVM.Value})
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/rules/customrules.jl:1516
  [3] enzyme_custom_rev_cfunc(B::Ptr{LLVM.API.LLVMOpaqueBuilder}, OrigCI::Ptr{LLVM.API.LLVMOpaqueValue}, gutils::Ptr{Nothing}, tape::Ptr{LLVM.API.LLVMOpaqueValue})
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/rules/llvmrules.jl:48
  [4] EnzymeCreatePrimalAndGradient(logic::Enzyme.Logic, todiff::LLVM.Function, retType::Enzyme.API.CDIFFE_TYPE, constant_args::Vector{Enzyme.API.CDIFFE_TYPE}, TA::Enzyme.TypeAnalysis, returnValue::Bool, dretUsed::Bool, mode::Enzyme.API.CDerivativeMode, runtimeActivity::Bool, width::Int64, additionalArg::Ptr{LLVM.API.LLVMOpaqueType}, forceAnonymousTape::Bool, typeInfo::Enzyme.FnTypeInfo, uncacheable_args::Vector{Bool}, augmented::Ptr{Nothing}, atomicAdd::Bool)
    @ Enzyme.API ~/.julia/packages/Enzyme/R6sE8/src/api.jl:268
  [5] enzyme!(job::GPUCompiler.CompilerJob{Enzyme.Compiler.EnzymeTarget, Enzyme.Compiler.EnzymeCompilerParams}, mod::LLVM.Module, primalf::LLVM.Function, TT::Type, mode::Enzyme.API.CDerivativeMode, width::Int64, parallel::Bool, actualRetType::Type, wrap::Bool, modifiedBetween::Tuple{Vararg{Bool, N}} where N, returnPrimal::Bool, expectedTapeType::Type, loweredArgs::Set{Int64}, boxedArgs::Set{Int64})
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:1668
  [6] codegen(output::Symbol, job::GPUCompiler.CompilerJob{Enzyme.Compiler.EnzymeTarget, Enzyme.Compiler.EnzymeCompilerParams}; libraries::Bool, deferred_codegen::Bool, optimize::Bool, toplevel::Bool, strip::Bool, validate::Bool, only_entry::Bool, parent_job::Nothing)
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:4550
  [7] codegen
    @ ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:3353 [inlined]
  [8] _thunk(job::GPUCompiler.CompilerJob{Enzyme.Compiler.EnzymeTarget, Enzyme.Compiler.EnzymeCompilerParams}, postopt::Bool)
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:5410
  [9] _thunk
    @ ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:5410 [inlined]
 [10] cached_compilation
    @ ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:5462 [inlined]
 [11] thunkbase(mi::Core.MethodInstance, World::UInt64, FA::Type{<:Annotation}, A::Type{<:Annotation}, TT::Type, Mode::Enzyme.API.CDerivativeMode, width::Int64, ModifiedBetween::Tuple{Vararg{Bool, N}} where N, ReturnPrimal::Bool, ShadowInit::Bool, ABI::Type, ErrIfFuncWritten::Bool, RuntimeActivity::Bool, edges::Vector{Any})
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:5573
 [12] thunk_generator(world::UInt64, source::LineNumberNode, FA::Type, A::Type, TT::Type, Mode::Enzyme.API.CDerivativeMode, Width::Int64, ModifiedBetween::Tuple{Vararg{Bool, N}} where N, ReturnPrimal::Bool, ShadowInit::Bool, ABI::Type, ErrIfFuncWritten::Bool, RuntimeActivity::Bool, self::Any, fakeworld::Any, fa::Type, a::Type, tt::Type, mode::Type, width::Type, modifiedbetween::Type, returnprimal::Type, shadowinit::Type, abi::Type, erriffuncwritten::Type, runtimeactivity::Type)
    @ Enzyme.Compiler ~/.julia/packages/Enzyme/R6sE8/src/compiler.jl:5758
 [13] autodiff_thunk(::ReverseModeSplit{true, true, false, 0, true, FFIABI, false, false, false}, ::Type{Const{typeof(spectrum_penalized_l2)}}, ::Type{Active}, ::Type{Duplicated{MatrixRNN{MatrixRnnCell{CuArray{Float32, 2, CUDA.DeviceMemory}}, WeightedMeanLayer{CuArray{Float32, 1, CUDA.DeviceMemory}}}}}, ::Type{Const{CuArray{Float32, 3, CUDA.DeviceMemory}}}, ::Type{Const{CuArray{Float32, 3, CUDA.DeviceMemory}}})
    @ Enzyme ~/.julia/packages/Enzyme/R6sE8/src/Enzyme.jl:975
 [14] _enzyme_withgradient(::Function, ::Duplicated{MatrixRNN{MatrixRnnCell{CuArray{Float32, 2, CUDA.DeviceMemory}}, WeightedMeanLayer{CuArray{Float32, 1, CUDA.DeviceMemory}}}}, ::Vararg{Union{Const, Duplicated}}; zero::Bool)
    @ FluxEnzymeExt ~/.julia/packages/Flux/BkG8S/ext/FluxEnzymeExt/FluxEnzymeExt.jl:73
 [15] withgradient(::Function, ::Duplicated{MatrixRNN{MatrixRnnCell{CuArray{Float32, 2, CUDA.DeviceMemory}}, WeightedMeanLayer{CuArray{Float32, 1, CUDA.DeviceMemory}}}}, ::Vararg{Any}; zero::Bool)
    @ Flux ~/.julia/packages/Flux/BkG8S/src/gradient.jl:171
 [16] withgradient(::Function, ::Duplicated{MatrixRNN{MatrixRnnCell{CuArray{Float32, 2, CUDA.DeviceMemory}}, WeightedMeanLayer{CuArray{Float32, 1, CUDA.DeviceMemory}}}}, ::CuArray{Float32, 3, CUDA.DeviceMemory}, ::Vararg{CuArray{Float32, 3, CUDA.DeviceMemory}})
    @ Flux ~/.julia/packages/Flux/BkG8S/src/gradient.jl:169
 [17] top-level scope
    @ REPL[58]:1
=#