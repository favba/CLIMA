module DGmethods

using MPI
using ..Grids
using ..MPIStateArrays
using StaticArrays
using ..SpaceMethods
using DocStringExtensions
using ..Topologies
using GPUifyLoops

export BalanceLaw, DGModel, State, Grad

include("balancelaw.jl")
include("dgmodel.jl")
include("DGBalanceLawDiscretizations_kernels.jl")
include("NumericalFluxes.jl")



end