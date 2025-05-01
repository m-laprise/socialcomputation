using Lux, Mooncake
using DifferentiationInterface
using Random
using LinearAlgebra
using BenchmarkTools
using InteractiveUtils

"Matrix-valued RNN cell with agent-specific encoders and matrix-valued update gate"
struct GatedCell{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init::F1 
    init_zeros::F2
    init_ones::F3
end

struct GatedCellv2{F1, F2, F3} <: Lux.AbstractLuxLayer
    k::Int
    n2::Int
    m::Int
    init::F1 
    init_zeros::F2
    init_ones::F3
end

function GatedCell(k::Int, n2::Int, m::Int;
                   init = glorot_uniform, 
                   init_zeros = zeros32, 
                   init_ones = ones32)
    GatedCell{typeof(init), typeof(init_zeros), typeof(init_ones)}(
        k, n2, m, init, init_zeros, init_ones
    )
end
function GatedCellv2(k::Int, n2::Int, m::Int;
                   init = glorot_uniform, 
                   init_zeros = zeros32, 
                   init_ones = ones32)
    GatedCell{typeof(init), typeof(init_zeros), typeof(init_ones)}(
        k, n2, m, init, init_zeros, init_ones
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::GatedCell)
    (Wx_in = l.init(rng, l.m, l.n2),
     Whh = l.init(rng, l.k, l.k),
     Bh = l.init_zeros(l.m, l.k),
     Wah = l.init(rng, l.m, l.m),
     Wax = l.init(rng, l.m, l.m),
     Ba = l.init_zeros(l.m, l.k),
     γ = l.init_ones(1), β = l.init_zeros(1))
end

function Lux.initialstates(rng::AbstractRNG, l::GatedCell)
    h = l.init(rng, l.m, l.k; gain = 0.01f0)
    (H = h,
     A = l.init_ones(l.m, l.k),
     Xproj = l.init_zeros(l.m, l.k),
     oldH = deepcopy(h),
     selfreset = [false],
     steps = [1],
     init = deepcopy(h)) 
end
Lux.initialparameters(rng::AbstractRNG, l::GatedCellv2) = Lux.initialparameters(rng::AbstractRNG, l::GatedCell)
Lux.initialstates(rng::AbstractRNG, l::GatedCellv2) = Lux.initialstates(rng::AbstractRNG, l::GatedCell)

function updategate!(A, H, Xproj, Wah, Wax, Ba)
    # Formula: st.A = σ.(ps.Wah * st.H .+ ps.Wax * st.Xproj .+ ps.Ba)
    @. A *= 0f0
    mul!(A, Wah, H)
    mul!(A, Wax, Xproj, 1f0, 1f0)
    @. A += Ba
    @. A = sigmoid(A)
end

function updategatev2!(A, H, Xproj, Wah, Wax, Ba)
    A .= Wah * H .+ Wax * Xproj .+ Ba
    @. A = sigmoid(A)
end

function updategatedstate!(H, oldH, A, Xproj, Whh, Bh)
    # Formula: st.H = st.A .* tanh.(st.H * ps.Whh .+ ps.Bh .+ st.Xproj) + (1f0 .- st.A) .* st.H
    mul!(H, oldH, Whh)
    @. H = (A * tanh(H + Bh + Xproj)) + ((1f0 - A) * oldH)
end

function updategatedstatev2!(H, oldH, A, Xproj, Whh, Bh)
    H .= oldH * Whh
    @. H = (A * tanh(H + Bh + Xproj)) + ((1f0 - A) * oldH)
end

"Dynamics of a matrix-valued RNN cell with an update gate"
function gatedtimemovement!(st, ps, steps)
    @inbounds for _ = 1:steps
        updategate!(st.A, st.H, st.Xproj, ps.Wah, ps.Wax, ps.Ba)
        @. st.oldH = deepcopy(st.H)
        updategatedstate!(st.H, st.oldH, st.A, st.Xproj, ps.Whh, ps.Bh)
    end
end

function gatedtimemovementv2!(st, ps, steps)
    @inbounds for _ = 1:steps
        updategatev2!(st.A, st.H, st.Xproj, ps.Wah, ps.Wax, ps.Ba)
        @. st.oldH = deepcopy(st.H)
        updategatedstatev2!(st.H, st.oldH, st.A, st.Xproj, ps.Whh, ps.Bh)
    end
end

reset!(st, ::Lux.AbstractLuxLayer) = (st.H .= deepcopy(st.init); st.Xproj .= 0f0; st.A .= 1f0)


function (l::GatedCell)(X, ps, st)
    if st.selfreset[1]
        reset!(st, l)
    end
    mul!(st.Xproj, ps.Wx_in, X)
    gatedtimemovement!(st, ps, st.steps[1])
    return st.H, st
end

function (l::GatedCellv2)(X, ps, st)
    if st.selfreset[1]
        reset!(st, l)
    end
    st.Xproj .= ps.Wx_in * X
    gatedtimemovementv2!(st, ps, st.steps[1])
    return st.H, st
end

function forwardpass!(st, ps, m::A, x; 
                      selfreset::Bool = false, 
                      steps::Int = 1) where A <: Lux.AbstractLuxLayer
    st.selfreset .= selfreset
    st.steps .= steps
    Lux.apply(m, x, ps, st)[1]
end

# Lux examples - setup
const x = rand(Float32, 1000, 100)
const y = rand(Float32, 4, 100)
const steps::Int = 50
model = GatedCell(100, 1000, 4)
modelv2 = GatedCellv2(100, 1000, 4)
rng = Random.MersenneTwister(0)
ps, st = Lux.setup(rng, model)
psv2, stv2 = Lux.setup(rng, modelv2)
#yhat = forwardpass!(st, ps, model, x; steps=steps)
myloss(ps, st, m, x, y, steps) = Lux.MSELoss()(forwardpass!(st, ps, m, x; steps=steps), y)

@btime forwardpass!($st, $ps, $model, $x; steps=steps)
# 501.250 μs (0 allocations: 0 bytes)
@btime forwardpass!($stv2, $psv2, $modelv2, $x; steps=steps)

@btime myloss($ps, $st, $model, $x, $y, $steps)
# 501.625 μs (0 allocations: 0 bytes)
@btime myloss($psv2, $stv2, $modelv2, $x, $y, $steps)

# Mooncake direct
cache = Mooncake.prepare_gradient_cache(myloss, ps, st, model, x, y, steps);
@btime Mooncake.prepare_gradient_cache($myloss, $ps, $st, $model, $x, $y, $steps);
# 6.186 ms (1066 allocations: 21.03 MiB)
val, grad = Mooncake.value_and_gradient!!(cache, myloss, ps, st, model, x, y, steps)
@btime Mooncake.value_and_gradient!!($cache, $myloss, $ps, $st, $model, $x, $y, $steps)
# 2.534 ms (360 allocations: 358.56 KiB)
grad[2]

cachev2 = Mooncake.prepare_gradient_cache(myloss, psv2, stv2, modelv2, x, y, steps);
@btime Mooncake.prepare_gradient_cache($myloss, $psv2, $stv2, $modelv2, $x, $y, $steps);
# 6.186 ms (1066 allocations: 21.03 MiB)
valv2, gradv2 = Mooncake.value_and_gradient!!(cachev2, myloss, psv2, stv2, modelv2, x, y, steps)
@btime Mooncake.value_and_gradient!!($cache, $myloss, $psv2, $stv2, $modelv2, $x, $y, $steps)
# 2.534 ms (360 allocations: 358.56 KiB)
gradv2[2]

rng = Random.MersenneTwister(0)
const X = rand(rng, Float32, 1000, 100, 10)
const Y = rand(rng, Float32, 4, 100, 10)

println("MOONCAKE DIRECT")
ctime = time()
cache = Mooncake.prepare_gradient_cache(myloss, ps, st, model, X[:,:,1], Y[:,:,1], steps);
println("Time for cache: ", round((time() - ctime)*1000, digits = 4), " ms")
gctime = time()
GC.gc(true)
println("Ran GC.gc(true) in ", round((time() - gctime)*1000, digits = 4), " ms")
for i in axes(X, 3)
    start_time = time()
    val, grad = Mooncake.value_and_gradient!!(cache, myloss, ps, st, model, X[:,:,i], Y[:,:,i], steps)
    println("Time for $i: ", round((time() - start_time)*1000, digits = 4), " ms")
end


println("MOONCAKE THROUGH DI")
timing(t) = round((time() - t)*1000, digits = 4)
preptime = time()
MCKbackend = AutoMooncake(;config=nothing)
prep = prepare_gradient(
    myloss, MCKbackend, ps, 
    Cache(st), Constant(model), 
    Constant(X[:,:,1]), Constant(Y[:,:,1]), Constant(steps))
println("Time for prep: ", timing(preptime), " ms")
gctime = time()
GC.gc(true)
println("Ran GC.gc(true) in ", timing(gctime), " ms")
for i in axes(X, 3)
    starttime = time()
    val, grad = value_and_gradient(myloss, prep, MCKbackend, ps, 
        (Cache(st)), (Constant(model)),
        (Constant(X[:,:,i])), (Constant(Y[:,:,i])), (Constant(steps)))
    println("Time for $i: ", timing(starttime), " ms")
end

# DI wrapper


DIval, DIgrad = value_and_gradient(
    myloss, MCKbackend, ps, 
    Cache(st), Constant(model), Constant(x), Constant(y), Constant(steps))

@btime value_and_gradient(
    $myloss, $MCKbackend, $ps, 
    $(Cache(st)), $(Constant(model)),
    $(Constant(x)), $(Constant(y)), $(Constant(steps)))
# 16.662 ms (1880 allocations: 45.71 MiB)
prep = prepare_gradient(
    myloss, MCKbackend, ps, 
    Cache(st), Constant(model), 
    Constant(x), Constant(y), Constant(steps)
)
@btime prepare_gradient(
    $myloss, $MCKbackend, $ps, 
    $(Cache(st)), $(Constant(model)), 
    $(Constant(x)), $(Constant(y)), $(Constant(steps))
)
# 14.104 ms (1499 allocations: 45.30 MiB)
@btime value_and_gradient(
    $myloss, $prep, $MCKbackend, $ps, 
    $(Cache(st)), $(Constant(model)),
    $(Constant(x)), $(Constant(y)), $(Constant(steps)))
# 2.615 ms (374 allocations: 418.11 KiB)

ENZbackend = AutoEnzyme()
ENZval, ENZgrad = value_and_gradient(
    myloss, ENZbackend, ps, 
    Cache(st), Constant(model), Constant(x), Constant(y), Constant(steps))
@btime value_and_gradient(
    $myloss, $ENZbackend, $ps, 
    $(Cache(st)), $(Constant(model)), 
    $(Constant(x)), $(Constant(y)), $(Constant(steps))
)
# 14.105 ms (123060 allocations: 4.25 MiB)

ENZgrads = Enzyme.make_zero(ps)
d_st = Enzyme.make_zero(st)
_, loss = autodiff(
    set_runtime_activity(Enzyme.ReverseWithPrimal), 
    myloss, 
    Duplicated(ps, ENZgrads), 
    Duplicated(st, d_st), Const(model), 
    Const(x), Const(y), Const(steps))
ENZgrads
@btime autodiff(
    set_runtime_activity(Enzyme.ReverseWithPrimal), 
    $myloss, 
    Duplicated($ps, $ENZgrads), 
    Duplicated($st, $d_st), Const($model), 
    Const($x), Const($y), Const(steps))
# 19.248 ms (163420 allocations: 5.32 MiB)