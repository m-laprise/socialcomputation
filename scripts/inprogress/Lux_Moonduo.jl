using Mooncake

"Source: Functors.jl https://github.com/FluxML/Functors.jl/blob/3eefb60ca30091f329e8c3e0bf2c04844e0cb9c7/src/walks.jl#L145C1-L145C12"
@generated function _anymutable(x::T) where {T}
    ismutabletype(T) && return true
    fns = QuoteNode.(filter(n -> fieldtype(T, n) != T, fieldnames(T)))
    subs =  [:(anymutable(getfield(x, $f))) for f in fns]
    return Expr(:(||), subs...)
end

"Inspired from https://github.com/FluxML/Fluxperimental.jl/blob/master/ext/FluxMooncakeExt.jl"

"""
    Moonduo(x, [dx])

This stores both an object `x` and its gradient `dx`,
with `dx` in the format used by Mooncake.jl. This is automatically allocated
when you call `Moonduo(x)`.

This serves the same purpose as Enzyme.jl's `Duplicated` type.
Both of these AD engines prefer that space for the gradient be pre-allocated.
"""
struct Moonduo{X,DX}
  val::X
  dval::DX
end

function Moonduo(args...)
  if length(args)==1
    error("The method `Moonduo(x)` is only available when Mooncake.jl is loaded!")
  else
    error("The only legal methods are `Moonduo(x)` and `Moonduo(x, dx)`.")
  end
end

function (m::Moonduo)(x...)
    m.val(x...)
end

function Moonduo(x)
    dx = Mooncake.zero_tangent(x)
    Moonduo(x, dx)
end

function _moonstrip end
_moonstrip(dx::Mooncake.Tangent) = map(_moonstrip, dx.fields)
_moonstrip(dx::Mooncake.MutableTangent) = map(_moonstrip, dx.fields)
_moonstrip(dx::Mooncake.NoTangent) = nothing
_moonstrip(dx::Union{Tuple, NamedTuple, AbstractArray}) = map(_moonstrip, dx)
_moonstrip(dx::AbstractArray{Mooncake.NoTangent}) = nothing
_moonstrip(dx::AbstractArray{<:Number}) = dx
_moonstrip(dx::Number) = nothing
function _moonstrip(dx)
  @error "not sure what to do with this type, in a gradient from Mooncake" typeof(dx)
  dx
end

_check_mutable(x::Moonduo) = _anymutable(x) || error(
    """`Flux.gradient(f, Moonduo(x), ...)` expects `x` to contain mutable parameter arrays."""
)

function _moongrad(dx)
    dx2 = _moonstrip(dx)  # remove all the weird types
    isnothing(dx2) && return
    return dx2
end

mywithgradient(f, args::Moonduo...; zero::Bool=true) = _moon_withgradient(f, args...; zero)

function _moon_withgradient(f, args::Moonduo...; zero)
    plain = map(x -> x.val, args)
    rule = Mooncake.build_rrule(f, plain...)
  
    for x in args
      _check_mutable(x)
      zero && Mooncake.set_to_zero!!(x.dval)
    end
    coduals = map(x -> Mooncake.CoDual(x.val, x.dval), args)
    val, _ = Mooncake.__value_and_gradient!!(rule, Mooncake.zero_codual(f), coduals...)
  
    grad = map(x -> _moongrad(x.dval), args)
    (; val, grad)
end


"""
EXAMPLE LOOP
"""
function train!(loss, model::Moonduo, data, opt; epochs::Int=1)
  for (i,d) in enumerate(data)
    d_splat = d isa Tuple ? d : (d,)
    rule = Mooncake.build_rrule(f, model.val, d_splat...)  # perhaps not ideal to do this inside the loop?
    Mooncake.set_to_zero!!(model.dval)
    l, _ = Mooncake.__value_and_gradient!!(rule, Mooncake.zero_codual(f), model, map(Mooncake.zero_codual, d_splat)...)
    if !isfinite(l)
      throw(DomainError(lazy"Loss is $l on data item $i, stopping training"))
    end
    # UPDATE MODEL PARAMETERS
    # LOG PROGRESS
  end
end
#===================#


struct Moonduo{X,DX}
    val::X
    dval::DX
end

function Moonduo(x)
    dx = Mooncake.zero_tangent(x)
    Moonduo(x, dx)
end

Luxapply!(st::M, ps::M, m::A, x; kwargs...) where A <: Lux.AbstractLuxContainerLayer where M <: Moonduo = 
    Luxapply!(st.val, ps.val, m, x; kwargs...)

d_ps = Mooncake.zero_tangent(ps)
d_st = Mooncake.zero_tangent(st)

duo_ps = Moonduo(ps)
duo_st = Moonduo(st)

plain_ps, plain_st = map(x -> x.val, (duo_ps, duo_st))

rule = Mooncake.build_rrule(f, plain...)

fargs = (myloss, model, X, Y);
cache = Mooncake.prepare_gradient_cache(fargs...);



function _moon_withgradient(f, args::Moonduo...; zero)
    plain = map(x -> x.val, args)
    rule = Mooncake.build_rrule(f, plain...)
  
    for x in args
      _check_mutable(x)
      zero && Mooncake.set_to_zero!!(x.dval)
    end
    coduals = map(x -> Mooncake.CoDual(x.val, x.dval), args)
    val, _ = Mooncake.__value_and_gradient!!(rule, Mooncake.zero_codual(f), coduals...)
  
    grad = map(x -> _moongrad(x.dval), args)
    (; val, grad)
end

# there is no way to mark some arguments constant.
# Instead of `gradient(loss, Duplicated(model), Const(data))`,
# you can write `gradient(m -> loss(m, data), Moonduo(model))`.


