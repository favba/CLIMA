using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMA
using CLIMA.Atmos
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: vars_state, vars_aux, vars_integrals,
                        integrate_aux!

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!, atmos_boundary_flux_diffusive!, flux_diffusive!
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!

# ---------------------------- Begin Boundary Conditions ----------------- #
"""
  BOMEX_BC <: BoundaryCondition
  Prescribes boundary conditions for Dynamics of Marine Stratocumulus Case
#Fields
$(DocStringExtensions.FIELDS)
"""
struct BOMEX_BC{FT} <: BoundaryCondition
  "Drag coefficient"
  C_drag::FT
  "Latent Heat Flux"
  LHF::FT
  "Sensible Heat Flux"
  SHF::FT
end

"""
    atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                          bc::BOMEX_BC, args...)

For the non-diffussive and gradient terms we just use the `NoFluxBC`
"""
atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                      bc::BOMEX_BC, 
                      args...) = atmos_boundary_state!(nf, NoFluxBC(), args...)

"""
    atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                   bc::BOMEX_BC, atmos::AtmosModel,
                                   F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)

When `bctype == 1` the `NoFluxBC` otherwise the specialized BOMEX BC is used
"""
function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                        bc::BOMEX_BC, 
                                        atmos::AtmosModel, F,
                                        state⁺, diff⁺, aux⁺, 
                                        n⁻,
                                        state⁻, diff⁻, aux⁻,
                                        bctype, t,
                                        state1⁻, diff1⁻, aux1⁻)
  if bctype != 1
    atmos_boundary_flux_diffusive!(nf, NoFluxBC(), atmos, F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)
  else
    # Start with the noflux BC and then build custom flux from there
    atmos_boundary_state!(nf, NoFluxBC(), atmos,
                          state⁺, diff⁺, aux⁺, n⁻,
                          state⁻, diff⁻, aux⁻,
                          bctype, t)

    # ------------------------------------------------------------------------
    # (<var>_FN) First node values (First interior node from bottom wall)
    # ------------------------------------------------------------------------
    u_FN = state1⁻.ρu / state1⁻.ρ
    windspeed_FN = norm(u_FN)

    # ----------------------------------------------------------
    # Extract components of diffusive momentum flux (minus-side)
    # ----------------------------------------------------------
    _, τ⁻ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)

    # ----------------------------------------------------------
    # Boundary momentum fluxes
    # ----------------------------------------------------------
    # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
    # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is
    # replaced by the appropriate flux terms
    C_drag = bc.C_drag
    @inbounds begin
      τ13⁺ = - C_drag * windspeed_FN * u_FN[1]
      τ23⁺ = - C_drag * windspeed_FN * u_FN[2]
      τ21⁺ = τ⁻[2,1]
    end

    # Assign diffusive momentum and moisture fluxes
    # (i.e. ρ𝛕 terms)
    FT = eltype(state⁺)
    τ⁺ = SHermitianCompact{3, FT, 6}(SVector(0   ,
                                             τ21⁺, τ13⁺,
                                             0   , τ23⁺, 0))

    # ----------------------------------------------------------
    # Boundary moisture fluxes
    # ----------------------------------------------------------
    # really ∇q_tot is being used to store d_q_tot
    d_q_tot⁺  = SVector(0, 0, bc.LHF/(LH_v0))

    # ----------------------------------------------------------
    # Boundary energy fluxes
    # ----------------------------------------------------------
    # Assign diffusive enthalpy flux (i.e. ρ(J+D) terms)
    d_h_tot⁺ = SVector(0, 0, bc.LHF + bc.SHF)

    # Set the flux using the now defined plus-side data
    flux_diffusive!(atmos, F, state⁺, τ⁺, d_h_tot⁺)
    flux_diffusive!(atmos.moisture, F, state⁺, d_q_tot⁺)
  end
end
# ------------------------ End Boundary Condition --------------------- # 


"""
  BOMEX Sources
"""
geostrophic_forcing = GeostrophicForcing{FT}(3.76e-5, -10 + 1.8e-3, 0)
struct BOMEX_Geostrophic<: Source
  f_coriolis::FT
  u_geostrophic::FT
  v_geostrophic::FT
end
function atmos_source!(s::BOMEX_Sources, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  z          = altitude(atmos.orientation,aux)
  u_geo      = SVector(s.u_geostrophic * z, s.v_geostrophic, 0)
  ẑ          = vertical_unit_vector(atmos.orientation, aux)
  fkvector   = s.f_coriolis * ẑ
  # Accumulate sources


  # Accumulate sources
  source.ρu -= fkvector × (state.ρu .- state.ρ*u_geo)
end



"""
  Initial Condition for BOMEX LES
"""
function init_bomex!(bl, state, aux, (x,y,z), t)

  FT         = eltype(state)
  # Observed ground quantities
  pg::FT     = 1.015e5
  Tg::FT     = 300.4
  q_totg::FT = 0.02245 # Mixing ratio -> Convert to q_tot 
  u::FT = 0
  v::FT = 0
  w::FT = 0 
  # Prescribed heights for piece-wise profile construction
  zl1::FT = 520
  zl2::FT = 1480
  zl3::FT = 2000
  zl4::FT = 3000
  # Assign piecewise quantities to θ_liq and q_tot 
  θ_liq::FT = 0 
  q_tot::FT = 0 
  q_liq::FT = 0
  q_ice::FT = 0 

  if z <= zl1
    θ_liq = 298.7
    q_tot = 17.0 + (z/zl1)*(16.3-17.0)
  elseif z > zl1 && z <= zl2
    θ_liq= 298.7 + (z-zl1) * (302.4-298.7)/(zl2-zl1)
    q_tot= 16.3 + (z-zl1) * (10.7-16.3)/(zl2-zl1)
  elseif z > zl2 && z <= zl3
    θ_liq= 302.4 + (z-zl2) * (308.2-302.4)/(zl3-zl2)
    q_tot= 10.7 + (z-zl2) * (4.2-10.7)/(zl3-zl2)
  else 
    θ_liq= 308.2 + (z-zl3) * (311.85-308.2)/(zl4-zl3)
    q_tot= 4.2 + (z-zl3) * (3.0-4.2)/(zl4-zl3)
  end
  
  # Set velocity profiles - piecewise profile for u
  zlv::FT = 700
  if z <= zlv
    u = -8.75
  else
    u = -8.75 + (z - zlv) * (-4.61 + 8.75)/(zl4 - zlv)
  end
  
  q_tot *= 1e-3 # Convert to kg/kg
  p = pg * exp(-z/3000) # TODO fix hardcoded maximum height  
  TS = LiquidIcePotTempSHumEquil_no_ρ(θ_liq, q_tot, p)
  T = air_temperature(TS)
  ρ = p / gas_constant_air(TS) / T
  q_pt = PhasePartition(TS)
  U           = ρ * u
  V           = ρ * v
  W           = ρ * w
  e_kin       = FT(1//2) * (u^2 + v^2 + w^2)
  e_pot       = FT(grav) * z
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(U, V, W) 
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot
end

function config_bomex(FT, N, resolution, xmax, ymax, zmax)

  # Reference state
  T_min   = FT(289)
  T_s     = FT(290.4)
  Γ_lapse = FT(grav/cp_d)
  T       = LinearTemperatureProfile(T_min, T_s, Γ_lapse)
  rel_hum = FT(0)
  ref_state = HydrostaticState(T, rel_hum)

  # Sources
  f_coriolis    = FT(1.03e-4)
  u_geostrophic = FT(7.0)
  v_geostrophic = FT(-5.5)
  w_ref         = FT(0)
  u_relaxation  = SVector(u_geostrophic, v_geostrophic, w_ref)
  # Sponge
  c_sponge = 1
  # Rayleigh damping
  zsponge = FT(2500.0)
  rayleigh_sponge = RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)

  # Boundary conditions
  # SGS Filter constants
  C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
  C_drag = FT(0.0011)
  LHF    = FT(115)
  SHF    = FT(15)

  bc = BOMEX_BC{FT}(C_drag, LHF, SHF)
  ics = init_bomex!
  source = (Gravity(),
            rayleigh_sponge,
            Subsidence{FT}(D_subsidence),
            geostrophic_forcing)
  model = AtmosModel{FT}(AtmosLESConfiguration;
                                ref_state=ref_state,
                                turbulence=SmagorinskyLilly{FT}(C_smag),
                                moisture=EquilMoist(5),
                                radiation=radiation,
                                source=source,
                                boundarycondition=bc,
                                init_state=ics)
  config = CLIMA.Atmos_LES_Configuration("BOMEX", N, resolution, xmax, ymax, zmax,
                                         init_bomex!,
                                         solver_type=CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
                                         model=model)
    return config
end

function main()
  CLIMA.init()

  FT = Float64

  # DG polynomial order
  N = 4

  # Domain resolution and size
  Δh = FT(100)
  Δv = FT(40)

  resolution = (Δh, Δh, Δv)

  xmax = 6400
  ymax = 6400
  zmax = 3000

  t0 = FT(0)
  timeend = FT(3600*6)

  driver_config = config_bomex(FT, N, resolution, xmax, ymax, zmax)
  solver_config = CLIMA.setup_solver(t0, timeend, driver_config, forcecpu=true)

  cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(2) do (init=false)
      Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
      nothing
  end
    
  result = CLIMA.invoke!(solver_config;
                        user_callbacks=(cbtmarfilter,),
                        check_euclidean_distance=true)
end

main()
