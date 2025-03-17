#= 
Defines custom recurrent neural network layers using the Lux library.
=#

#= RNN CELLS =#

"Vanilla matrix-valued RNN cell with shared encoder"
struct MatrixVlaCell{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init_params::F1 
    init_states::F2
    init_zeros::F3
    gain::Float32
end

"Matrix-valued RNN cell with shared encoder and matrix-valued update gate"
struct MatrixGatedCell{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init_params::F1 
    init_states::F2
    init_zeros::F3
    gain::Float32
end

"Matrix-valued RNN cell with agent-specific encoders and matrix-valued update gate"
struct MatrixGatedCell2{V, F1, F2, F3, F4} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    nl::V
    init_params::F1 
    init_states::F2
    init_zeros::F3
    init_ones::F4
    gain::Float32
end

"Parent type for various RNN cells"
MatrixCell = Union{MatrixVlaCell, MatrixGatedCell, MatrixGatedCell2}

# LAYER INITIALIZATION
function MatrixVlaCell(k::Int, n2::Int, m::Int; 
                       init_params=glorot_uniform, 
                       init_states=glorot_uniform, 
                       init_zeros=zeros32, gain)
    MatrixVlaCell{typeof(init_params), typeof(init_states), typeof(init_zeros)}(
        k, n2, m, init_params, init_states, init_zeros, gain
    )
end
function MatrixGatedCell(k::Int, n2::Int, m::Int; 
                         init_params=glorot_uniform, 
                         init_states=glorot_uniform, 
                         init_zeros=zeros32, gain)
    MatrixGatedCell{typeof(init_params), typeof(init_states), typeof(init_zeros)}(
        k, n2, m, init_params, init_states, init_zeros, gain
    )
end

function MatrixGatedCell2(k::Int, n2::Int, m::Int, nl::Vector{<:Real};
                         init_params=glorot_uniform, 
                         init_states=glorot_uniform, 
                         init_zeros=zeros32, 
                         init_ones=ones32,
                         gain=0.1f0)
    MatrixGatedCell2{Vector{<:Real}, typeof(init_params), typeof(init_states), typeof(init_zeros), typeof(init_ones)}(
        k, n2, m, nl, init_params, init_states, init_zeros, init_ones, gain
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::MatrixVlaCell)
    (Wx_in=l.init_params(rng, l.m, l.n2),
     Whh=l.init_params(rng, l.k, l.k),
     Bh=l.init_zeros(l.m, l.k))
end
function Lux.initialparameters(rng::AbstractRNG, l::MatrixGatedCell)
    (Wx_in=l.init_params(rng, l.m, l.n2),
     Whh=l.init_params(rng, l.k, l.k),
     Bh=l.init_zeros(l.m, l.k),
     Wa=l.init_params(rng, l.m, l.k),
     Wah=l.init_params(rng, l.m, l.m),
     Wax=l.init_params(rng, l.m, l.m),
     Ba=l.init_zeros(l.m, l.k))
end

function Lux.initialparameters(rng::AbstractRNG, l::MatrixGatedCell2)
    (Wx_in=NamedTuple(
        (Symbol("in$(i)") => l.init_params(rng, l.m, l.nl[i]; gain = l.gain) for i in eachindex(l.nl))
    ),
     Whh=l.init_params(rng, l.k, l.k),
     Bh=l.init_zeros(l.m, l.k),
     Wah=l.init_params(rng, l.m, l.m; gain = l.gain),
     Wax=l.init_params(rng, l.m, l.m; gain = l.gain),
     Ba=l.init_zeros(l.m, l.k),
     γ=l.init_ones(1), 
     β=l.init_zeros(1))
end

function Lux.initialstates(rng::AbstractRNG, l::MatrixVlaCell)
    h = l.init_states(rng, l.m, l.k)
    (H=h,
     Xproj=l.init_zeros(l.m, l.k),
     selfreset=[false],
     turns=[1],
     init=deepcopy(h)) 
end

function Lux.initialstates(rng::AbstractRNG, l::MatrixGatedCell)
    h = l.init_states(rng, l.m, l.k)
    (H=h,
     A=ones(Float32, l.m, l.k),
     Xproj=l.init_zeros(l.m, l.k),
     selfreset=[false],
     turns=[1],
     init=deepcopy(h)) 
end

function Lux.initialstates(rng::AbstractRNG, l::MatrixGatedCell2)
    h = l.init_states(rng, l.m, l.k; gain = 0.01f0)
    (H=h,
     A=l.init_ones(l.m, l.k),
     Xproj=l.init_zeros(l.m, l.k),
     oldH=deepcopy(h),
     selfreset=[false],
     turns=[1],
     init=deepcopy(h)) 
end

# FORWARD PASS
function mylayernorm!(x::AbstractVector{Float32}, γ::Float32, β::Float32)
    μ = mean(x)
    σ = sqrt(var(x) .+ 1f-5)
    @. x = ((x - μ) / σ) * γ + β
end

function mylayernorm!(x::AbstractMatrix{Float32}, γ::Float32, β::Float32)
    @inbounds for i = axes(x, 1)
        row = view(x, i, :)
        μ = mean(row)
        σ = sqrt(var(row) .+ 1f-5)
        @. row = (row - μ) * (γ/σ) + β
    end
end


function updategate!(A, H, Xproj, Wah, Wax, Ba)
    # Formula: st.A = σ.(ps.Wah * st.H .+ ps.Wax * st.Xproj .+ ps.Ba)
    @. A *= 0f0
    mul!(A, Wah, H)
    mul!(A, Wax, Xproj, 1f0, 1f0)
    @. A += Ba
    @. A = NNlib.sigmoid_fast(A)
end

function updategatedstate!(H, oldH, A, Xproj, Whh, Bh)
    # Formula: st.H = st.A .* tanh.(st.H * ps.Whh .+ ps.Bh .+ st.Xproj) + (1f0 .- st.A) .* st.H
    mul!(H, oldH, Whh)
    @. H = (A * NNlib.tanh_fast(H + Bh + Xproj)) + ((1f0 - A) * oldH)
end

"Dynamics of a Vanilla matrix-valued RNN cell"
function timemovement!(st, ps, turns)
    # Agents consult their compressed input, exchange compressed information, and update their state.
    # Repeat for a given number of time steps.
    @inbounds for _ in 1:turns
        st.H .= NNlib.tanh_fast.(st.H * ps.Whh .+ ps.Bh .+ st.Xproj)
    end
end
"Dynamics of a matrix-valued RNN cell with an update gate"
function gatedtimemovement!(st, ps, turns)
    @inbounds for _ = 1:turns
        updategate!(st.A, st.H, st.Xproj, ps.Wah, ps.Wax, ps.Ba)
        @. st.oldH = deepcopy(st.H)
        updategatedstate!(st.H, st.oldH, st.A, st.Xproj, ps.Whh, ps.Bh)
        mylayernorm!(st.H, ps.γ[1], ps.β[1])
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

function (l::MatrixGatedCell)(X, ps, st)
    if st.selfreset[1]
        reset!(st)
    end
    st.Xproj .= ps.Wx_in * X
    gatedtimemovement!(st, ps, st.turns[1])
    return st.H, st
end

function (l::MatrixGatedCell2)(X, ps, st)
    if st.selfreset[1]
        reset!(st)
    end
    for (agent, col) in enumerate(eachcol(X))
        dense = col[findall(col .!= 0)] # nl x 1
        W_in = ps.Wx_in[Symbol("in$(agent)")] # M x nl
        view(st.Xproj, :, agent) .= W_in * dense
    end
    gatedtimemovement!(st, ps, st.turns[1])
    return st.H, st
end

#= DECODING LAYERS =#
"Vector-to-vector feed-forward decoding layer"
struct N2DecodingLayer{F1} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init::F1
end

"Matrix-to-matrix decoding layer with enforced rank restriction"
struct LRDecodingLayer{F1} <: Lux.AbstractLuxLayer
    k::Int
    n::Int
    m::Int
    r::Int
    init::F1
end

"Matrix-to-factors input-dependent decoding layer"
struct FactorDecodingLayer{F1, F2} <: Lux.AbstractLuxLayer
    k::Int
    n::Int
    m::Int
    r::Int
    init::F1
    init_zeros::F2
end

function N2DecodingLayer(k::Int, n2::Int, m::Int; 
                         init=glorot_uniform)
    N2DecodingLayer{typeof(init)}(k, n2, m, init)
end
function LRDecodingLayer(k::Int, n2::Int, m::Int, r::Int; 
                         init=glorot_uniform)
    LRDecodingLayer{typeof(init)}(k, n2, m, r, init)
end

function FactorDecodingLayer(k::Int, n::Int, m::Int, r::Int; 
                             init=glorot_uniform, init_zeros=zeros32)
    FactorDecodingLayer{typeof(init), typeof(init_zeros)}(k, n, m, r, init, init_zeros)
end

function Lux.initialparameters(rng::AbstractRNG, l::N2DecodingLayer)
    (Wx_out=l.init(rng, l.n2, l.m*l.k),
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::LRDecodingLayer)
    (U=l.init(rng, l.n, l.r), 
     Wu=l.init(rng, l.r, l.m),
     Wv=l.init(rng, l.k, l.r),
     V=l.init(rng, l.r, l.n),
    )
end

#=function Lux.initialparameters(rng::AbstractRNG, l::FactorDecodingLayer)
    wu2=(l.init(rng, l.k, l.r))
    wv2=(l.init(rng, l.r, l.k))
    (Wu1=l.init(rng, l.n, l.m),
     Wu2= wu2 / sum(wu2), 
     Wv1=l.init(rng, l.m, l.n),
     Wv2= wv2 / sum(wv2), 
    )
end=#
function Lux.initialparameters(rng::AbstractRNG, l::FactorDecodingLayer)
    (Ωu1 = l.init(rng, l.n, l.k; gain = 1.5f0),
     Ωu2 = l.init(rng, l.m, l.r; gain = 1.5f0),
     Ωv1 = l.init(rng, l.k, l.n; gain = 1.5f0),
     Ωv2 = l.init(rng, l.r, l.m; gain = 1.5f0),
    )
end

Lux.initialstates(::AbstractRNG, ::N2DecodingLayer) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::LRDecodingLayer) = NamedTuple()
Lux.initialstates(::AbstractRNG, l::FactorDecodingLayer) = NamedTuple(
    (Wu1 = l.init_zeros(l.n, l.m),
     Wu2 = l.init_zeros(l.k, l.r),
     Wv1 = l.init_zeros(l.m, l.n),
     Wv2 = l.init_zeros(l.r, l.k),
    )
)

function (l::N2DecodingLayer)(cellh, ps, st)
    return ps.Wx_out * vec(cellh), st
end

function (l::LRDecodingLayer)(cellh, ps, st)
    return vec(ps.U * ps.Wu * cellh * ps.Wv * ps.V), st
end

function (l::FactorDecodingLayer)(cellh, ps, st, Xproj)
    # Input-dependent decoding weights
    st.Wu1 .= ps.Ωu1 * Xproj'
    st.Wu2 .= Xproj' * ps.Ωu2
    st.Wv1 .= Xproj * ps.Ωv1
    st.Wv2 .= ps.Ωv2 * Xproj
    # Decoding the two factors (up to permutations)
    U = st.Wu1 * cellh * st.Wu2
    V = st.Wv2 * cellh' * st.Wv1
    # Recomposing the rank-r matrix
    return vec(U * V), st
end

#= CUSTOM CHAINS =#
"Chain any RNN cell with any decoding layer"
struct ComposedRNN{L1, L2} <: Lux.AbstractLuxContainerLayer{(:cell, :dec)}
    cell::L1
    dec::L2
end

"Define the forward pass of the chain"
function (c::ComposedRNN)(x::AbstractMatrix, ps, st::NamedTuple)
    h, st_layer1 = c.cell(x, ps.cell, st.cell)
    y, st_layer2 = c.dec(h, ps.dec, st.dec, st_layer1.Xproj)
    return y, (cell = st_layer1, dec = st_layer2)
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
reset!(st, m::MatrixGatedCell) = (st.H .= deepcopy(st.init); st.Xproj .= 0f0; st.A .= 1f0)
reset!(st) = (st.H .= deepcopy(st.init))
