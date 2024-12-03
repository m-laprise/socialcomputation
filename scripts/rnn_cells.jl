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
    else
        error("Invalid h_init value. Choose from 'zero' or 'randn'.")
    end
    if input_size > 0 && gated == false
        Wxh = randn(Float32, net_width, input_size) / sqrt(Float32(net_width))
        return rnn_cell_xb(
            Wxh, Whh,
            b_init,
            h, 
            h
        )
    elseif input_size == 0 && gated == false
        return rnn_cell_b(
            Whh,
            b_init,
            h, 
            h
        )
    elseif gated == true
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

function(m::bfl_cell)(state, I=nothing)
    Whh, b, basal_u, gain, tauh = m.Whh, m.b, m.basal_u, m.gain, m.tauh
    ho = @view state[:, 1]
    if isnothing(I)
        bias = b
    else
        @assert I isa Vector || size(I, 2) == 1
        bias = b .+ pad_input(I, length(ho))
    end
    ho2 = ho .* ho
    hu_new = basal_u .+ ((Whh * ho2) .* gain)
    ho_upd = tanhvf0((Whh * ho) .* hu_new) .+ bias
    tauh_sig = sigmoid(tauh)
    ho_new = ((1f0 .- tauh_sig) .* ho) .+ (tauh_sig .* ho_upd)
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
        padded_x = Vector{Float32}(undef, width)
        padded_x[1:length(x)] = x
        padded_x[length(x)+1:end] .= 0.0f0
        return padded_x
    elseif length(x) > width
        @warn "Network too small: less than one node per input. Input will be truncated for distribution."
        return x[1:width]
    else
        return x
    end
end

Flux.@layer rnn_cell_b trainable=(Whh, b)
Flux.@layer rnn_cell_xb trainable=(Wxh, Whh, b)
Flux.@layer bfl_cell trainable=(Whh, b, gain, tauh)

state(m::rnn_cell_b) = m.h
state(m::rnn_cell_xb) = m.h
state(m::bfl_cell) = m.h

Base.show(io::IO, l::rnn_cell_xb) = print(
    io, "rnn_cell_xb(", size(l.Wxh), ", ", size(l.Whh), ")")

Base.show(io::IO, l::rnn_cell_b) = print(
    io, "rnn_cell_b(", size(l.Whh), ")")

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

"""
    reset!(rnn)
Reset the hidden state of a recurrent layer back to its original value. 
"""
reset!(m::Recur) = (m.state = m.init)
