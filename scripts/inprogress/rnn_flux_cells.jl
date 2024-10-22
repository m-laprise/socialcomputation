using Random
using Flux
using Zygote


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
                  h_init::String)
    @assert input_size >= 0
    @assert net_width > 0
    if h_init == "zero"
        h = zeros(Float32, net_width)
    elseif h_init == "randn"
        h = randn(Float32, net_width) * 0.01f0
    else
        error("Invalid h_init value. Choose from 'zero' or 'randn'.")
    end
    if input_size > 0
        return rnn_cell_xb(
            randn(Float32, net_width, input_size) / sqrt(Float32(net_width)),
            randn(Float32, net_width, net_width) / sqrt(Float32(net_width)),
            zeros(Float32, net_width),
            h, h
        )
    else
        return rnn_cell_b(
            randn(Float32, net_width, net_width) / sqrt(Float32(net_width)),
            zeros(Float32, net_width),
            h, h
        )
    end
end

function(m::rnn_cell_xb)(state, x, I)
    Wxh, Whh, b, h = m.Wxh, m.Whh, m.b, state
    h_new = tanh.(Wxh * x .+ Whh * h .+ b .+ I)
    m.h = h_new
    return h_new, h_new
end

function(m::rnn_cell_b)(state, I)
    Whh, b, h = m.Whh, m.b, state
    #width = length(b)
    #if length(I) < width
    #    I = vcat(I, zeros(Float32, width - length(I)))
    #end
    println(typeof(h), typeof(b), typeof(I))
    h_new = tanh.(Whh * h .+ b .+ I)
    m.h = h_new
    return h_new, h_new
end

function pad_input(x, width)
    x = vec(x)
    if length(x) < width
        x = vcat(x, zeros(Float32, width - length(x)))
    elseif length(x) > width
        x = x[1:width]
        @warn "Network too small: less than one node per input. Input will be truncated for distribution."
    end
    return x
end

Flux.@functor rnn_cell_b
Flux.@functor rnn_cell_xb

state(m::rnn_cell_b) = m.h
state(m::rnn_cell_xb) = m.h

Base.show(io::IO, l::rnn_cell_xb) = print(
    io, "rnn_cell_xb(", size(l.Wxh), ", ", size(l.Whh), ")")

Base.show(io::IO, l::rnn_cell_b) = print(
    io, "rnn_cell_b(", size(l.Whh), ")")
######

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

Flux.@functor Recur cell, init
Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")
rnn(args...;kwargs...) = Recur(rnn_cell(args...;kwargs...))
"""
    reset!(rnn)
Reset the hidden state of a recurrent layer back to its original value. 
"""
reset!(m::Recur) = (m.state = m.init)

##########


# Define binary cross entropy loss function
function __check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y)) 
     size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
        "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
      ))
    end
end

function xlogy(x, y)
    result = x * log(y)
    ifelse(iszero(x), zero(result), result)
end

function mybinarycrossentropy(ŷ, y; agg = mean, eps::Real = eps(float(eltype(ŷ))))
    __check_sizes(ŷ, y)
    agg(@.(-xlogy(y, ŷ + eps) - xlogy(1 - y, 1 - ŷ + eps)))
end

function mylogitbinarycrossentropy(ŷ, y; agg = mean)
    __check_sizes(ŷ, y)
    agg(@.((1 - y) * ŷ - logσ(ŷ)))
end