"""
    BalanceLaw

An abstract type representing a PDE balance law of the form

elements for balance laws of the form

```math
q_{,t} + Σ_{i=1,...d} F_{i,i} = s
```

Subtypes `L` should define the following methods:
- `dimension(::L)` the number of dimensions
- `varmap_aux(::L)`: a tuple of symbols containing the auxiliary variables
- `varmap_state(::L)`: a tuple of symbols containing the state variables
- `varmap_state_for_transform(::L)`: a tuple of symbols containing the state variables which are passed to the `transform!` function.
- `varmap_transform(::L)`: a tuple of symbols containing the transformed variables of which gradients are computed
- `varmap_diffusive(::L)`: a tuple of symbols containing the diffusive variables
- `flux!(::L, flux::Grad, state::State, diffstate::State, auxstate::State, t::Real)`
- `gradtransform!(::L, transformstate::State, state::State, auxstate::State, t::Real)`
- `diffusive!(::L, diffstate::State, ∇transformstate::Grad, auxstate::State, t::Real)`
- `source!(::L, source::State, state::State, auxstate::State, t::Real)`
- `wavespeed(::L, nM, state::State, aux::State, t::Real)`
- `boundarycondition!(::L, stateP::State, diffP::State, auxP::State, normalM, stateM::State, diffM::State, auxM::State, bctype, t)`
- `init_aux!(::L, aux::State, coords, args...)`
- `init_state!(::L, state::State, aux::State, coords, args...)`

"""
abstract type BalanceLaw end # PDE part

has_diffusive(m::BalanceLaw) = num_diffusive(m) > 0

# function stubs
function num_aux end
function num_state end
function num_gradtransform end
function num_diffusive end

function num_state_for_gradtransform end

function dimension end
function varmap_aux end
function varmap_state end
function varmap_gradtransform end
function varmap_diffusive end

function varmap_state_for_gradtransform end

function flux! end
function gradtransform! end
function diffusive! end
function source! end 
function wavespeed end
function boundarycondition! end
function init_aux! end
function init_state! end



# TODO: allow aliases and vector values
struct State{varmap, A<:StaticVector}
  arr::A
end
State{varmap}(arr::A) where {varmap,A<:StaticVector} = State{varmap,A}(arr)

struct GetFieldException <: Exception
  sym::Symbol
end



Base.propertynames(s::State{varmap}) where {varmap} = propertynames(varmap)
@inline function Base.getproperty(s::State{varmap}, sym::Symbol) where {varmap}
  i = getfield(varmap, sym)
  if i isa Integer
    return getfield(s,:arr)[i]
  else
    return getfield(s,:arr)[SVector(i...)]
  end
end
@inline function Base.setproperty!(s::State{varmap}, sym::Symbol, val) where {varmap}
  i = getfield(varmap, sym)
  if i isa Integer
    return getfield(s,:arr)[i] = val
  else
    return getfield(s,:arr)[SVector(i...)] = val
  end
end


struct Grad{varmap, A<:StaticMatrix}
  arr::A
end
Grad{varmap}(arr::A) where {varmap,A<:StaticMatrix} = Grad{varmap,A}(arr)

Base.propertynames(s::Grad{varmap}) where {varmap} = propertynames(varmap)
@inline function Base.getproperty(∇s::Grad{varmap}, sym::Symbol) where {varmap}
  i = getfield(varmap, sym)
  if i isa Integer
    return getfield(∇s,:arr)[:,i]
  else
    return getfield(∇s,:arr)[:,SVector(i...)]
  end
end
@inline function Base.setproperty!(∇s::Grad{varmap}, sym::Symbol, val) where {varmap}
  i = getfield(varmap, sym)
  if i isa Integer
    return getfield(∇s,:arr)[:,i] = val
  else
    return getfield(∇s,:arr)[:,SVector(i...)] = val
  end
end
