#= 
Defines custom recurrent neural network layers using the Lux library.
=#

#= RNN CELLS =#
struct MatrixVlaCell{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init_params::F1 
    init_states::F2
    init_zeros::F3
end

struct MatrixG1Cell{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init_params::F1 
    init_states::F2
    init_zeros::F3
end

#"Parent type for both types of RNN cells"
#MatrixCell = Union{MatrixVlaCell, MatrixGRUCell}

# LAYER INITIALIZATION
function MatrixVlaCell(k::Int, n2::Int, m::Int; 
                       init_params=glorot_uniform, 
                       init_states=glorot_uniform, 
                       init_zeros=zeros32)
    MatrixVlaCell{typeof(init_params), typeof(init_states), typeof(init_zeros)}(
        k, n2, m, init_params, init_states, init_zeros
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::MatrixVlaCell)
    (Wx_in=l.init_params(rng, l.m, l.n2; gain = 1.1f0),
     Whh=l.init_params(rng, l.k, l.k; gain = 1.1f0),
     Bh=l.init_zeros(l.m, l.k))
end
function Lux.initialstates(rng::AbstractRNG, l::MatrixVlaCell)
    h = l.init_states(rng, l.m, l.k; gain = 1.1f0)
    (H=h,
     Xproj=l.init_zeros(l.m, l.k),
     selfreset=[false],
     turns=[1],
     init=deepcopy(h)) 
end

# FORWARD PASS
function timemovement!(st, ps, turns)
    # Agents consult their compressed input, exchange compressed information, and update their state.
    # Repeat for a given number of time steps.
    @inbounds for _ in 1:turns
        #st.H .= NNlib.tanh_fast.((st.H * ps.Whh .+ ps.Bh .+ st.Xproj))
        st.H .= tanh.(st.H * ps.Whh .+ ps.Bh .+ st.Xproj)
    end
end

function (l::MatrixVlaCell)(X, ps, st)
    if st.selfreset[1]
        reset!(st)
    end
    # Each agent takes a col of the n2 x k input matrix (ie, a sparse (n2 x 1) vector), 
    # and projects it to a m x 1 vector, with m << n2 (agents share a compression stragegy Wx_in in R^{m x n2})
    # To avoid recomputing the projection for each time step, store it as a hidden state
    st.Xproj .= ps.Wx_in * X
    timemovement!(st, ps, st.turns[1])
    return st.H, st
end

#= DECODING LAYER =#
struct DecodingLayer{F1} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init::F1
end

function DecodingLayer(k::Int, n2::Int, m::Int; 
                       init=glorot_uniform)
    DecodingLayer{typeof(init)}(k, n2, m, init)
end

function Lux.initialparameters(rng::AbstractRNG, l::DecodingLayer)
    (Wx_out=l.init(rng, l.n2, l.m*l.k; gain = 1.1f0),
     #β=ones(Float32, l.k)
     )
end

Lux.initialstates(::AbstractRNG, ::DecodingLayer) = NamedTuple()

function (l::DecodingLayer)(cellh, ps, st)
    #return ps.Wx_out * cellh * ps.β, st
    return ps.Wx_out * vec(cellh), st
end

#= CUSTOM CHAINS =#

struct ComposedRNN{L1, L2} <: Lux.AbstractLuxContainerLayer{(:cell, :dec)}
    cell::L1
    dec::L2
end

function (c::ComposedRNN)(x::AbstractMatrix, ps, st::NamedTuple)
    h, st_l1 = c.cell(x, ps.cell, st.cell)
    y, st_l2 = c.dec(h, ps.dec, st.dec)
    # Return the new state which has the same structure as `st`
    return y, (cell = st_l1, dec = st_l2)
end

#= HELPER FUNCTIONS =#

# Replace Lux.apply with Luxapply! to allow for custom state handling
function setup!(st; selfreset, turns)
    st.cell.selfreset .= selfreset
    st.cell.turns .= turns
end

function Luxapply!(st, ps, m::A, x; 
                   selfreset::Bool = false, 
                   turns::Int = 1) where A <: Lux.AbstractLuxContainerLayer
    setup!(st; selfreset = selfreset, turns = turns)
    Lux.apply(m, x, ps, st)[1]
end

# *NOTE*: Careful with this; it grabs it from the globals
state(m::ComposedRNN) = st.cell.H
reset!(st, m::ComposedRNN) = (st.cell.H .= deepcopy(st.cell.init); st.cell.Xproj .= 0f0)
reset!(st, m::MatrixVlaCell) = (st.H .= deepcopy(st.init); st.Xproj .= 0f0)
reset!(st) = (st.H .= deepcopy(st.init))
