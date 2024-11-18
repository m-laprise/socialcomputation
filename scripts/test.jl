using Random
using Distributions
using Flux
using Zygote
using JLD2, CodecBzip2
using CairoMakie
using LinearAlgebra

include("genrandommatrix.jl")
include("rnn_cells.jl")
include("customlossfunctions.jl")
include("plot_utils.jl")
#include("train_utils.jl")
include("train_setup.jl")


m_vanilla = Chain(
    rnn(100, 150, 
        Whh_init = nothing, 
        h_init="randn"),
    Dense(150 => 1, sigmoid)
)

m_vanilla(nothing)
