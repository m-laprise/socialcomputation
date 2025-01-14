#= 
This script defines various types of recurrent neural network (RNN) cells using the Flux library in Julia.

The main cell types defined are:
- `rnn_cell_xb`: A mutable struct representing an RNN cell with input weights.
- `rnn_cell_b`: A mutable struct representing an RNN cell without input weights.
- `bfl_cell`: A mutable struct representing an RNN cell with attention gating and additional parameters for 
basal activity and gain.

The script also includes functions to initialize these cells, reset their states, and perform forward passes 
through the cells.
=# 

using Random
using Flux
using Zygote

###### VANILLA RNN CELLS
mutable struct rnn_cell_xb{A,V}
    Wxh::A
    Whh::A
    b::V
    h::V
    init::V # store initial state for reset
end

mutable struct rnn_cell_b{A,V}
    Whh::A
    b::V
    h::V
    init::V # store initial state for reset
end

mutable struct rnn_cell_b_dual{A,V}
    Whh::A
    b1::V
    b2::V
    h::A
    g1::V
    g2::V
    init::A # store initial state for reset
end

mutable struct customgru_cell{A,V}
    Whh::A
    b::V
    bz::V
    g1::V
    g2::V
    h::A
    init::A # store initial state for reset
end

mutable struct bfl_cell{A,V}
    Whh::A
    b::V
    h::A
    basal_u::V
    gain::V
    tauh::V
    init::A # store initial state for reset
end

function rnn_cell(input_size::Int, 
                  net_width::Int;
                  h_init::String = "randn", 
                  Whh_init = nothing,
                  gated::Bool = false,
                  bfl::Bool = false,
                  dual::Bool = false,
                  basal_u::Float32 = Float32(1e-2),
                  gain::Float32 = Float32(0.5),
                  tauh::Float32 = Float32(0.5))
    @assert input_size >= 0
    @assert net_width > 0
    # Initialize recurrent weight matrix
    if isnothing(Whh_init)
        Whh = randn(Float32, net_width, net_width) / sqrt(Float32(net_width))
    else
        Whh = Float32.(Whh_init)
    end
    # Initialize bias
    b_init = randn(Float32, net_width) / sqrt(Float32(net_width))
    # Initialize hidden state
    if h_init == "zero"
        h = zeros(Float32, net_width)
    elseif h_init == "randn"
        h = randn(Float32, net_width) * 0.01f0
        h2 = randn(Float32, net_width) * 0.01f0
    else
        error("Invalid h_init value. Choose from 'zero' or 'randn'.")
    end
    if input_size > 0 && gated == false && dual == false
        Wxh = randn(Float32, net_width, input_size) / sqrt(Float32(net_width))
        return rnn_cell_xb(
            Wxh, Whh,
            b_init,
            h, 
            h
        )
    elseif input_size == 0 && gated == false && dual == false
        return rnn_cell_b(
            Whh,
            b_init,
            h, 
            h
        )
    elseif dual == true
        b2 = randn(Float32, net_width) / sqrt(Float32(net_width))
        g1 = rand(Float32, net_width)
        g2 = rand(Float32, net_width)
        hdual = Array{Float32}(undef, net_width, 2)
        hdual[:, 1] = h
        hdual[:, 2] = h2
        return rnn_cell_b_dual(
            Whh,
            b_init,
            b2,
            hdual,
            g1,
            g2,
            hdual
        )
    elseif gated == true
        bz = randn(Float32, net_width) / sqrt(Float32(net_width))
        g1 = ones(Float32, net_width)
        g2 = ones(Float32, net_width)
        htot = Array{Float32}(undef, net_width, 2)
        htot[:, 1] = h
        htot[:, 2] = zeros(Float32, net_width)
        return customgru_cell(
            Whh,
            b_init,
            bz,
            g1, g2,
            htot,
            htot
        )
    elseif bfl == true
        basal_u = ones(Float32, net_width) * basal_u
        gain = ones(Float32, net_width) * gain
        tauh = ones(Float32, net_width) * tauh
        htot = Array{Float32}(undef, net_width, 2)
        htot[:, 1] = h
        htot[:, 2] = basal_u
        return bfl_cell(
            Whh,
            b_init,
            htot,
            basal_u, 
            gain, 
            tauh,
            htot
        )
    end
end

tanhvf0(x::Vector{Float32}) = tanh.(x)

function(m::rnn_cell_xb)(state, x, I=nothing)
    Wxh, Whh, b, h = m.Wxh, m.Whh, m.b, state
    if isnothing(I)
        bias = b
    else
        @assert I isa Vector || size(I, 2) == 1
        bias = b .+ pad_input(I, length(h))
    end
    h_new = tanhvf0(Wxh * x .+ Whh * h .+ bias)
    m.h = h_new
    return h_new, h_new
end

function(m::rnn_cell_b)(state, I=nothing)
    Whh, b, h = m.Whh, m.b, state
    if isnothing(I)
        bias = b
    else
        @assert I isa Vector || size(I, 2) == 1
        bias = b .+ pad_input(I, length(h))
    end
    h_new = tanhvf0(Whh * h .+ bias)
    m.h = h_new
    return h_new, h_new
end

function(m::rnn_cell_b_dual)(state, I=nothing)
    Whh, b1, b2, g1, g2 = m.Whh, m.b1, m.b2, m.g1, m.g2
    h1 = @view state[:, 1]
    h2 = @view state[:, 2]
    if isnothing(I)
        bias1 = b1 
        bias2 = b2
    else
        @assert I isa Vector || size(I, 2) == 1
        bias1 = b1 .+ pad_input(I, length(h1))
        bias2 = b2 .+ pad_input(I, length(h2))
    end
    h1_new = tanhvf0(g1 .* (Whh * h1) .+ bias1)
    h2_new = tanhvf0(g2 .* (Whh * h2) .+ bias2)
    h_new_buf = Zygote.Buffer(zeros(Float32, size(state)))
    h_new_buf[:, 1] = h1_new
    h_new_buf[:, 2] = h2_new
    h_new = copy(h_new_buf)
    m.h = h_new
    return h_new, h_new
end

function(m::customgru_cell)(state, I=nothing)
    Whh, b, bz, g1, g2 = m.Whh, m.b, m.bz, m.g1, m.g2
    h = @view state[:, 1]
    if isnothing(I)
        bias = b
        Iz = zeros(Float32, length(h))
    else
        @assert I isa Vector || size(I, 2) == 1
        bias = b .+ pad_input(I, length(h))
        Iz = pad_input(I, length(h))
    end
    h_upd = tanhvf0(Whh * h .+ bias)
    tau = sigmoid.((g1 .* (Whh * h)) .+ (g2 .* (Whh * Iz)) .+ bz)
    h_new = ((1f0 .- tau) .* h_upd) .+ (tau .* h)
    state_new_buf = Zygote.Buffer(zeros(Float32, size(state)))
    state_new_buf[:, 1] = h_new
    state_new_buf[:, 2] = tau
    state_new = copy(state_new_buf)
    m.h = state_new
    return state_new, state_new
end

function(m::bfl_cell)(state, I=nothing)
    Whh, b, basal_u, gain = m.Whh, m.b, m.basal_u, m.gain
    ho = @view state[:, 1]
    if isnothing(I)
        bias = b
    else
        @assert I isa Vector || size(I, 2) == 1
        bias = b .+ pad_input(I, length(ho))
    end
    ho2 = ho .* ho
    hu_new = basal_u .+ ((Whh * ho2) .* gain)
    ho_new = tanhvf0((Whh * ho) .* hu_new) .+ bias
    #tauh_sig = sigmoid(tauh)
    #ho_new = ((1f0 .- tauh_sig) .* ho) .+ (tauh_sig .* ho_upd)
    h_new_buf = Zygote.Buffer(zeros(Float32, size(state)))
    h_new_buf[:, 1] = ho_new
    h_new_buf[:, 2] = hu_new
    h_new = copy(h_new_buf)
    m.h = h_new
    return h_new, h_new
end

function pad_input(x::AbstractArray, width::Int)
    x = Float32.(x)
    if length(x) < width
        return vcat(x, zeros(Float32, width - length(x)))
    elseif length(x) > width
        @warn "Network too small: less than one node per input. Input will be truncated for distribution."
        return x[1:width]
    else
        return x
    end
end

Flux.@layer rnn_cell_b trainable=(Whh, b)
Flux.@layer rnn_cell_xb trainable=(Wxh, Whh, b)
Flux.@layer rnn_cell_b_dual trainable=(Whh, b1, b2, g1, g2)
Flux.@layer customgru_cell trainable=(Whh, b, bz, g1, g2)
Flux.@layer bfl_cell trainable=(Whh, b, gain)

state(m::rnn_cell_b) = m.h
state(m::rnn_cell_xb) = m.h
state(m::rnn_cell_b_dual) = m.h
state(m::customgru_cell) = m.h
state(m::bfl_cell) = m.h

Base.show(io::IO, l::rnn_cell_xb) = print(
    io, "rnn_cell_xb(", size(l.Wxh), ", ", size(l.Whh), ")")

Base.show(io::IO, l::rnn_cell_b) = print(
    io, "rnn_cell_b(", size(l.Whh), ")")

Base.show(io::IO, l::rnn_cell_b_dual) = print(
    io, "rnn_cell_b_dual(", size(l.Whh), ")")

Base.show(io::IO, l::customgru_cell) = print(
    io, "customgru_cell(", size(l.Whh), ")")

Base.show(io::IO, l::bfl_cell) = print(
    io, "bfl_cell(", size(l.Whh), ")")

###### RECURRENCE

"""
    Recur(cell)
`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:
    h, y = cell(h, x...)
"""
mutable struct Recur{T}
  cell::T
  init
  state
end

Recur(m, h = state(m)) = Recur(m, h, h)

function (m::Recur)(xs...)
  h, y = m.cell(m.state, xs...)
  m.state = h
  return y
end
state(m::Recur{ <:rnn_cell_b} ) = m.state
state(m::Recur{ <:rnn_cell_xb} ) = m.state
state(m::Recur{ <:rnn_cell_b_dual} ) = m.state
state(m::Recur{ <:customgru_cell} ) = m.state
state(m::Recur{ <:bfl_cell} ) = m.state

Flux.@functor Recur cell, init
Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")
rnn(args...;kwargs...) = Recur(rnn_cell(args...;kwargs...))

"""
    reset!(rnn)
Reset the hidden state of a recurrent layer back to its original value. 
"""
reset!(m::Recur) = (m.state = m.init)


# Split layers
struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)
Flux.@layer Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

# CoB layer
struct BasisChange{M<:AbstractMatrix}
    weight::M
    bias::M
end

function BasisChange((in, out)::Pair{<:Integer, <:Integer};
               init = Flux.glorot_uniform)
    bias_init = init(out, out)
    weight_init = init(out, in)
    BasisChange(weight_init, bias_init)
end

Flux.@layer BasisChange trainable=(weight, bias)
(a::BasisChange)(x::AbstractVecOrMat) = (a.weight * x * a.weight' .+ a.bias)

function Base.show(io::IO, l::BasisChange)
    print(io, "BasisChange(", size(l.weight, 1), " => ", size(l.weight, 2))
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

