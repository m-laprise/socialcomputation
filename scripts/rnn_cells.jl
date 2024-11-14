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

function rnn_cell(input_size::Int, net_width::Int;
                  h_init::String, Whh_init=nothing)
    @assert input_size >= 0
    @assert net_width > 0
    # Initialize hidden state
    if h_init == "zero"
        h = zeros(Float32, net_width)
    elseif h_init == "randn"
        h = randn(Float32, net_width) * 0.01f0
    else
        error("Invalid h_init value. Choose from 'zero' or 'randn'.")
    end
    # Initialize recurrent weight matrix
    if isnothing(Whh_init)
        Whh = randn(Float32, net_width, net_width) / sqrt(Float32(net_width))
    else
        Whh = Float32.(Whh_init)
    end
    if input_size > 0
        return rnn_cell_xb(
            randn(Float32, net_width, input_size) / sqrt(Float32(net_width)),
            Whh,
            randn(Float32, net_width) / sqrt(Float32(net_width)),
            h, h
        )
    else
        return rnn_cell_b(
            Whh,
            randn(Float32, net_width) / sqrt(Float32(net_width)),
            h, h
        )
    end
end

function(m::rnn_cell_xb)(state, x, I)
    @assert I isa Vector || size(I, 2) == 1
    Wxh, Whh, b, h = m.Wxh, m.Whh, m.b, state
    bias = b .+ pad_input(I, length(h))
    h_new = tanh.(Wxh * x .+ Whh * h .+ bias)
    m.h = h_new
    return h_new, h_new
end

function(m::rnn_cell_b)(state, I)
    @assert I isa Vector || size(I, 2) == 1
    Whh, b, h = m.Whh, m.b, state
    bias = b .+ pad_input(I, length(h))
    h_new = tanh.(Whh * h .+ bias)
    m.h = h_new
    return h_new, h_new
end

function pad_input(x, width)
    if length(x) < width
        x = vcat(x, zeros(Float32, width - length(x)))
    elseif length(x) > width
        x = x[1:width]
        @warn "Network too small: less than one node per input. Input will be truncated for distribution."
    end
    return x
end

Flux.@layer rnn_cell_b trainable=(Whh, b)
Flux.@layer rnn_cell_xb trainable=(Wxh, Whh, b)

state(m::rnn_cell_b) = m.h
state(m::rnn_cell_xb) = m.h

Base.show(io::IO, l::rnn_cell_xb) = print(
    io, "rnn_cell_xb(", size(l.Wxh), ", ", size(l.Whh), ")")

Base.show(io::IO, l::rnn_cell_b) = print(
    io, "rnn_cell_b(", size(l.Whh), ")")

###### BFL CELLS

mutable struct bfl_cell{A,V}
    Whh::A
    b::V
    h::V
    u::V
    gain::V
    init_h::V # store initial state for reset
    init_u::V # store initial attention for reset
end

function bfl_cell(net_width::Int;
                  h_init::String, Whh_init = nothing, 
                  basal_u::Float32 = Float32(1e-2),
                  gain::Float32 = Float32(0.5))
    @assert net_width > 0
    # Initialize hidden state
    if h_init == "zero"
        h = zeros(Float32, net_width)
    elseif h_init == "randn"
        h = randn(Float32, net_width) * 0.01f0
    else
        error("Invalid h_init value. Choose from 'zero' or 'randn'.")
    end
    # Initialize recurrent weight matrix
    if isnothing(Whh_init)
        Whh = randn(Float32, net_width, net_width) / sqrt(Float32(net_width))
    else
        Whh = Float32.(Whh_init)
    end
    u = ones(Float32, net_width) * basal_u
    gain = ones(Float32, net_width) * gain
    return bfl_cell(
            Whh,
            randn(Float32, net_width) / sqrt(Float32(net_width)),
            h, u, gain, h, u
        )
end

function(m::bfl_cell)(state, I)
    Whh, b, gain, basal_u = m.Whh, m.b, m.gain, m.init_u
    u, h = m.u, state
    bias = pad_input(I, length(h)) .+ b
    h2 = h .* h
    u_new = basal_u .+ ((Whh * h2) .* gain)
    h_new = tanh.((Whh * h) .* u_new) .+ bias 
    m.u = u_new
    m.h = h_new
    return h_new, h_new
end 

Flux.@layer bfl_cell trainable=(Whh, b, gain)

state(m::bfl_cell) = m.h

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
state(m::Recur{ <:bfl_cell} ) = m.state

Flux.@functor Recur cell, init
Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")
rnn(args...;kwargs...) = Recur(rnn_cell(args...;kwargs...))
bfl(args...;kwargs...) = Recur(bfl_cell(args...;kwargs...))

"""
    reset!(rnn)
Reset the hidden state of a recurrent layer back to its original value. 
"""
reset!(m::Recur) = (m.state = m.init)
