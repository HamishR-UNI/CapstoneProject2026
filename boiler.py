"""
Biomass boiler: steady-state map plus first-order-plus-dead-time dynamics

Independent of the Excel reference in structure, though it now shares the Excel's property
route. Combustion from atomic balance, flue gas properties from CoolProp on a tabulated
temperature-enthalpy curve, boiler as a single counterflow heat exchanger solved by
effectiveness-NTU.

Property routes
---------------
Flue gas   Each species is called individually from CoolProp and mass-weighted into a
           mixture enthalpy, tabulated once against temperature. This matches the Excel.
           An earlier version of this file used NASA 7-coefficient polynomials; the two
           agree to within 0.1 percent on the dry gas sensible term.
Steam      Inlet and outlet enthalpies come from Tim's IAPWS-95 package, the same source
           the turbine model uses, so the boiler/turbine interface shares one datum
           instead of relying on hardcoded constants.

Neither property package enters the GEKKO model. Both are evaluated at build time to
construct the static map, fitted over 0-20 kg/s fuel in physics-shaped forms (see
stack_temp/duty below); GEKKO only ever sees those closed forms.

Structure is Wiener: the static map is evaluated first, at the INSTANT fuel command, then
its outputs are wrapped in unity-gain linear dynamics. The static map is the steady-state
model, fitted once as polynomials in fuel flow. The dynamics carry no gain, so the steady
state is untouched by the choice of time constant.

Fuel metering is instant: `mf` (delivered fuel) is an algebraic, direct function of the fuel
command with no delay or lag at all. Flue gas flow (`gas`) USED to follow it through the
FD/ID fan's own (faster) speed response, dynamics_params.THETA_FAN_S/TAU_FAN_S -- restructure
phase 6 found that let downstream "available" flue-gas heat reach its full new value within
about 100 s of a fuel step, while the actual combustion energy (Q, stack) had not even started
moving yet. A fan can only move flue gas that combustion has actually produced, so `gas` now
carries the SAME dead time + lag as Q/stack (THETA_FUEL/TAU_Q below) -- the effective
combustion-product-generation state, not the fan's own faster mechanical response, which does
not gate this signal physically. THETA_FAN_S/TAU_FAN_S are no longer wired to anything live.

The dead time and lag on the two heat-exchanger outputs, `Q` (duty to steam) and `stack`
(stack temperature), are NOT the gas-side volume/flush numbers this file used to carry
(TAU_METAL, a furnace/tube-metal thermal-mass figure, is kept only as a printed diagnostic
now -- at constant drum pressure the evaporator water and metal sit at saturation and do not
store heat when firing changes, so it was never really the boiler's slow store). The real
slow physics of a grate boiler is the FUEL BED re-balancing on the grate after a feed-rate
change (see the fuel-bed section below for the model and its sources: Xia et al. 2025,
Bauer et al. 2010, Kortela & Jamsae-Jounela). Heat release after a step is modelled as two
linear ramps -- volatiles from ignition to the ignition front reaching the grate, then char
over the rest of the residence time -- reduced to an equivalent FOPTD (THETA_BED, TAU_BED)
by Smith's two-point method. TAU_Q = TAU_BED + TAU (the fuel bed plus the gas flush it sits
in series with) is the lag actually applied, as ONE state (not cascaded with TAU_BED
separately) for the same reason TAU_METAL used to be combined rather than cascaded: two lags
in series is a second-order response with an inflection, a worse shape to hand a PID than one
first-order lag with a bigger time constant. `Q` and `stack` each get their own independent
copy of the same dead-time chain (THETA_FUEL/N_LAG) and lag (TAU_Q), so they still share one
shape with each other; only `gas` moves ahead of them, on the fan's own much faster response.

Run directly. Displays the step response plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from gekko import GEKKO

import iapws95_gekko as iapws
import dynamics_params as dp
from turbine_model import lag_chain

# ----------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------

# Fuel. Replace with the ultimate analysis once available.
FUEL_X, FUEL_Y, FUEL_Z = 31.0, 45.0, 24.5   # C, H, O atoms per fuel molecule
MOISTURE_FUEL = 0.40                         # wet basis
HHV_DRY       = 20.2e3                       # kJ/kg dry fuel

# Combustion
EXCESS_O2    = 16.7003956279603              # kmol O2 per kmol fuel above stoichiometric
MOISTURE_AIR = 0.007                         # kg water per kg dry air

# Design point, used to size UA
MDOT_FUEL_DES = 10.0                         # kg/s dry fuel
T_STACK_DES   = 140.0                        # degC

# Steam side. Specified as states, not enthalpies; both are solved from IAPWS-95 below.
P_HP_BAR   = 100.0                           # bar, boiler drum / outlet pressure
T_STEAM_C  = 529.37                          # degC, superheater outlet, matches turbine_model.TIN

# T_FW used to do two unrelated jobs. They are different physical quantities and are now
# separate: one is a property state, the other is a heat transfer modelling choice.
T_FW        = 20.0                           # degC, real economiser inlet -> H_FW
T_COLD_SIDE = 105.0                          # degC, lumped eps-NTU sink, heat transfer only
FW_COMPRESSED = True                         # True: pumped to P_HP. False: saturated liquid.

# Pressure range over which the steam enthalpy rise is fitted. The superheater outlet
# temperature is held at T_STEAM_C across the range: a real boiler holds it with the
# attemperator against a metallurgical limit, so temperature is the thing that stays put
# and enthalpy is the thing that moves.
P_HP_MIN, P_HP_MAX = 60.0, 130.0             # bar

# Gas-side heat transfer, UA proportional to mdot**N_EXP.
# 0.6 for cross-flow over tube banks, 0.8 for turbulent flow inside tubes.
N_EXP = 0.6

# --- Dynamic parameters -----------------------------------------------------------
# One volume, one time constant. Furnace volume is no longer a guessed number: it is
# sized off the design-point fired duty (fuel flow x net LHV) using the standard
# biomass furnace sizing convention of a 200 kW/m3 volumetric heat release rate,
# V_GAS_PATH = Q_fired_des / Q_RELEASE_RATE. Computed further down, once Q_PER_FUEL
# exists. See that section for the calculation.
Q_RELEASE_RATE = 200.0                       # kW/m3, biomass furnace sizing convention
T_GAS_MEAN = 700.0                           # degC, mean gas temperature for the density
P_FLUE     = 101325.0                        # Pa, flue gas pressure

# --- Fuel-bed heat-release response -------------------------------------------------
# The boiler's dominant time constant is NOT tube/drum metal: at constant drum pressure
# the evaporator water and metal sit at saturation and do not store heat when firing
# changes (TAU_METAL below is kept only as a printed diagnostic, not applied to any lag).
# The real slow physics of a grate boiler is the fuel bed re-balancing after a feed-rate
# change (Xia et al. 2025: fuel stored on the grate must reach a new balance; Bauer et al.
# 2010: drying/devolatilisation zones as first-order mass balances; Kortela &
# Jamsae-Jounela: drum pressure settling limited by fuel drying rate).
#
# Heat release after a feed step is modelled as two linear ramps: volatiles (fraction
# F_VOLATILE_HEAT) released from ignition (T_IGNITION_S) to the ignition front reaching
# the grate (BED_HEIGHT_M/V_IGNITION_FRONT_MPS later), then char released over the rest of
# the grate residence time (T_CONVERSION_END_S). See dynamics_params.py for every raw
# constant here and its source.
_t_a = dp.T_IGNITION_S
_t_b = _t_a + dp.BED_HEIGHT_M / dp.V_IGNITION_FRONT_MPS
_t_c = dp.T_CONVERSION_END_S
assert dp.F_VOLATILE_HEAT > 0.632, 'closed-form Smith fit below needs f_v > 0.632'
assert _t_c > _t_b, 'char burnout must finish after the ignition front reaches the grate'

# Smith two-point (28.3/63.2 %) fit of the volatile ramp, closed form while f_v > 0.632
# (the char tail beyond t_b is not captured by this reduction -- accepted, see section 10
# of the dynamics change notes: checked against Masa et al.'s measured settling time).
TAU_BED   = 1.5*(0.632 - 0.283)/dp.F_VOLATILE_HEAT * (_t_b - _t_a)              # s
THETA_BED = _t_a + (0.632/dp.F_VOLATILE_HEAT)*(_t_b - _t_a) - TAU_BED           # s

THETA_STEAM_LINE = dp.L_HP_STEAM_M / dp.V_HP_STEAM_MPS    # s, boiler -> turbine 1 main line

# THETA_FUEL (keep the name -- other files read it): feeder + fuel-bed dead time + steam
# line transport. Replaces the old "under the 1 min the supervisor indicated" figure, which
# was a misattribution -- that guidance was not about the boiler.
THETA_FUEL = dp.THETA_FEEDER_S + THETA_BED + THETA_STEAM_LINE      # s
N_LAG      = 10                              # lag chain length approximating the dead time

# Furnace/tube wall thermal mass -- kept as a printed DIAGNOSTIC ONLY, not used in any lag
# (see the note above: it is not the boiler's real store once TAU_BED exists). Formula
# unchanged from when it was the reinstated lag: wall 5 mm, rho.c carbon steel, U 90 W/m2.K
# flue-gas-side film coefficient.
TAU_METAL = 0.005*7850*490/90.0              # s, ~214 s, diagnostic only

# Header pressure. A pressure setpoint is reached by the letdown and vent valves, so this
# is a valve stroke plus a small header emptying time, not a process volume.
TAU_P = 20.0                                 # s

# Step test
STEP_FROM, STEP_TO = 10.0, 8.0               # kg/s dry fuel
STEP_TIME, T_END, DT = 300.0, 4200.0, 10.0   # s

T_REF  = 25.0                                # degC energy reference
HFG_25 = 2442.3                              # kJ/kg latent heat of water at T_REF

# ----------------------------------------------------------------------------------
# Steam side properties, IAPWS-95 (Tim's GEKKO package, numeric solvers)
#
# Used at build time only. The same package the turbine model uses, so the boiler outlet
# and the turbine inlet share one enthalpy datum.
# ----------------------------------------------------------------------------------

P_HP_KPA = P_HP_BAR * 100.0

H_STEAM = iapws.solve_tp_numeric(T_STEAM_C + 273.15, P_HP_KPA, 'vapor').h
if FW_COMPRESSED:
    H_FW = iapws.solve_tp_numeric(T_FW + 273.15, P_HP_KPA, 'liquid').h
else:
    H_FW = iapws.solve_saturation_numeric(
        PropsSI('P', 'T', T_FW + 273.15, 'Q', 0, 'Water') / 1e3).liquid.h

T_SAT_HP = iapws.solve_saturation_numeric(P_HP_KPA).T - 273.15


def _dh_at(P_bar):
    """Steam enthalpy rise across the boiler at pressure P_bar, kJ/kg."""
    h_s = iapws.solve_tp_numeric(T_STEAM_C + 273.15, P_bar * 100.0, 'vapor').h
    if FW_COMPRESSED:
        h_f = iapws.solve_tp_numeric(T_FW + 273.15, P_bar * 100.0, 'liquid').h
    else:
        h_f = H_FW
    return h_s - h_f


# Cubic in pressure, fitted once. Residual is order 1e-6 kJ/kg, so the fit is exact for
# any purpose here, and it keeps the GEKKO side pure algebra.
_Pg = np.linspace(P_HP_MIN, P_HP_MAX, 25)
DH_MAP = np.polyfit(_Pg, np.array([_dh_at(P) for P in _Pg]), 3)
DH_RES = float(np.abs(np.array([_dh_at(P) for P in _Pg]) - np.polyval(DH_MAP, _Pg)).max())


def dh_steam(P_bar):
    """Steam enthalpy rise from the fitted cubic, kJ/kg."""
    return np.polyval(DH_MAP, P_bar)
HFG_IAPWS = (PropsSI('H', 'T', T_REF + 273.15, 'Q', 1, 'Water')
             - PropsSI('H', 'T', T_REF + 273.15, 'Q', 0, 'Water')) / 1e3

# ----------------------------------------------------------------------------------
# Flue gas properties, CoolProp per species on a tabulated temperature-enthalpy curve
#
# Same route as the Excel: each species called individually, mass-weighted into a mixture
# enthalpy. Species are evaluated at P_PROP, low enough that every component is a dilute
# gas. That is the ideal-gas limit, which is what a flue gas at near-atmospheric pressure
# is, and it also keeps water vapour superheated down to T_REF -- at its real partial
# pressure water would be under the dewpoint at 25 degC and CoolProp would return liquid.
#
# Tabulated once and interpolated. PropsSI is far too slow to sit inside the stack
# temperature loop, which is called forty times to build the map.
# ----------------------------------------------------------------------------------

MW = {'N2': 28.0134, 'O2': 31.9988, 'CO2': 44.0095, 'H2O': 18.01528}
CP_FLUID = {'N2': 'Nitrogen', 'O2': 'Oxygen', 'CO2': 'CarbonDioxide', 'H2O': 'Water'}

P_PROP = 1.0e3                               # Pa, ideal gas limit
T_TAB_MAX = 1800.0                           # degC, above the adiabatic flame temperature
T_TAB = np.arange(T_REF, T_TAB_MAX + 1.0, 1.0)


def _tabulate(sp):
    h = np.array([PropsSI('H', 'T', T + 273.15, 'P', P_PROP, CP_FLUID[sp]) for T in T_TAB])
    return (h - h[0]) / 1e3                  # kJ/kg sensible above T_REF


H_TAB = {sp: _tabulate(sp) for sp in MW}


def h_species(sp, T_K):
    """Sensible enthalpy above T_REF for one species, kJ/kg."""
    return float(np.interp(T_K - 273.15, T_TAB, H_TAB[sp]))


# ----------------------------------------------------------------------------------
# Combustion, per kmol of fuel. Fixed air-fuel ratio, so this runs once.
# ----------------------------------------------------------------------------------

M_FUEL = FUEL_X * 12.011 + FUEL_Y * 1.008 + FUEL_Z * 15.999

n_O2_stoich = FUEL_X + FUEL_Y / 4.0 - FUEL_Z / 2.0
n_O2_in = n_O2_stoich + EXCESS_O2
n_N2 = 79.0 / 21.0 * n_O2_in
m_dryair = n_O2_in * MW['O2'] + n_N2 * MW['N2']

n_H2O_air = m_dryair * MOISTURE_AIR / MW['H2O']
n_H2O_fuel = M_FUEL * MOISTURE_FUEL / (1.0 - MOISTURE_FUEL) / MW['H2O']

n_out = {'CO2': FUEL_X, 'O2': EXCESS_O2, 'N2': n_N2,
         'H2O': FUEL_Y / 2.0 + n_H2O_air + n_H2O_fuel}
m_out = {sp: n * MW[sp] for sp, n in n_out.items()}
m_tot = sum(m_out.values())
w = {sp: m / m_tot for sp, m in m_out.items()}

GAS_PER_FUEL = m_tot / M_FUEL
LAMBDA = n_O2_in / n_O2_stoich
LHV_DRY = HHV_DRY - (FUEL_Y / 2.0 * MW['H2O'] / M_FUEL) * HFG_25
Q_PER_FUEL = LHV_DRY - (MOISTURE_FUEL / (1.0 - MOISTURE_FUEL)) * HFG_25


# ----------------------------------------------------------------------------------
# Flue gas dewpoint and the two-region enthalpy curve
#
# The dry-basis curve this used to be treated the flue gas water as vapour at every
# temperature. That is correct down to the dewpoint and wrong below it. Burning 40%
# moisture wood puts ~19.7 mol% water in the gas, which at atmospheric pressure gives
# a dewpoint near 60 degC -- and heatpump_dyn cools the gas to 55 degC, five kelvin
# INTO the wet region. On the dry curve that duty reads ~3.8 MW; with the condensate
# accounted for it is ~10.5 MW. The workbook has always had this right (see
# 'Graphs and Calculations', dewpoint cell and the two-phase enthalpy table); it was
# lost in the translation to Python.
#
# DATUM. Water is referenced to VAPOUR at T_REF, not liquid. That is deliberate: it
# makes h_gas reduce exactly to the old dry-basis expression above the dewpoint, and
# it matches the LHV-as-received basis Q_PER_FUEL is built on, which has already
# written the moisture's latent heat off as lost. The consequence is that recovering
# condensate shows up as heat the LHV datum said was gone, so a boiler efficiency
# quoted on LHV can exceed 1.0 once the condensing duty is counted. That is not an
# error -- it is the well-known artefact of the LHV convention -- but say so wherever
# the number is reported, or quote on HHV instead.
#
# The vapour fraction is set by saturation at the water's own partial pressure. Below
# the dewpoint the gas leaves saturated at T, so the water remaining as vapour is
# fixed by Psat(T)/P_FLUE and everything else has dropped out as liquid.
# ----------------------------------------------------------------------------------

N_H2O_TOT = n_out['H2O']                     # kmol water per kmol fuel
N_DRY = sum(n for sp, n in n_out.items() if sp != 'H2O')
Y_H2O = N_H2O_TOT / (N_H2O_TOT + N_DRY)      # mole fraction, wet basis
W_H2O = w['H2O']                             # mass fraction of total water in the gas

T_DEW = PropsSI('T', 'P', Y_H2O * P_FLUE, 'Q', 1, 'Water') - 273.15    # degC

#  Liquid water on the same CoolProp datum as the vapour table, so their difference at
#  any temperature is the real latent heat. Tabulated for the same reason the species
#  curves are: PropsSI is far too slow to sit in an inner loop.
_T_LIQ = np.arange(0.0, 100.0 + 0.5, 0.5)
_H_LIQ = np.array([PropsSI('H', 'T', T + 273.15, 'Q', 0, 'Water') for T in _T_LIQ]) / 1e3
_H_VAP_REF = PropsSI('H', 'T', T_REF + 273.15, 'P', P_PROP, 'Water') / 1e3


def _water_vapour_fraction(T_C):
    """Fraction of the gas's water still airborne at T_C, 0-1."""
    if T_C >= T_DEW:
        return 1.0
    y_sat = PropsSI('P', 'T', T_C + 273.15, 'Q', 1, 'Water') / P_FLUE
    if y_sat >= 1.0:
        return 1.0
    n_vap = N_DRY * y_sat / (1.0 - y_sat)     # kmol vapour per kmol fuel at saturation
    return min(max(n_vap / N_H2O_TOT, 0.0), 1.0)


def h_gas(T_C):
    """Flue gas enthalpy above T_REF, kJ/kg of wet gas.

    Above T_DEW this is the mass-weighted dry-basis sum it always was. Below it the
    water splits, and the condensed part carries (h_liquid(T) - h_vapour(T_REF)) --
    a large negative number, which is what produces the knee in the curve.
    """
    f_v = _water_vapour_fraction(T_C)
    h = sum(w[sp] * h_species(sp, T_C + 273.15) for sp in w if sp != 'H2O')
    h += W_H2O * f_v * h_species('H2O', T_C + 273.15)
    if f_v < 1.0:
        h_liq = float(np.interp(T_C, _T_LIQ, _H_LIQ)) - _H_VAP_REF
        h += W_H2O * (1.0 - f_v) * h_liq
    return h


def condensate_rate(mdot_gas, T_C):
    """Condensate dropping out of mdot_gas [kg/s] cooled to T_C, kg/s.

    Not used by the energy balance -- h_gas already carries the latent term -- but the
    rig wants the number to show, and a real exchanger would need somewhere to drain it.
    """
    return mdot_gas * W_H2O * (1.0 - _water_vapour_fraction(T_C))


def flame_temperature():
    target = Q_PER_FUEL / GAS_PER_FUEL
    lo, hi = T_REF, 3000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h_gas(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


T_HOT = flame_temperature()

MDOT_GAS_DES = MDOT_FUEL_DES * GAS_PER_FUEL
cp_des = (h_gas(T_HOT) - h_gas(T_STACK_DES)) / (T_HOT - T_STACK_DES)
eps_des = (T_HOT - T_STACK_DES) / (T_HOT - T_COLD_SIDE)
NTU_DES = -np.log(1.0 - eps_des)
UA_DES = NTU_DES * MDOT_GAS_DES * cp_des

# ----------------------------------------------------------------------------------
# Steady-state map, solved by bisection on the implicit stack temperature
# ----------------------------------------------------------------------------------

def steady_state(mdot_fuel, P_bar=None):
    mgas = mdot_fuel * GAS_PER_FUEL
    UA = UA_DES * (mgas / MDOT_GAS_DES) ** N_EXP
    T = T_STACK_DES
    for _ in range(100):
        cp_bar = (h_gas(T_HOT) - h_gas(T)) / (T_HOT - T)
        eps = 1.0 - np.exp(-UA / (mgas * cp_bar))
        T_new = T_HOT - eps * (T_HOT - T_COLD_SIDE)
        if abs(T_new - T) < 1e-10:
            T = T_new
            break
        T = 0.5 * (T + T_new)
    Q = mgas * (h_gas(T_HOT) - h_gas(T))
    P_bar = P_HP_BAR if P_bar is None else P_bar
    dh = dh_steam(P_bar)
    return dict(mdot_gas=mgas, T_stack=T, Q=Q, dh=dh,
                mdot_steam=Q / dh, eta=Q / (mdot_fuel * Q_PER_FUEL))


# Static map, fitted so the dynamic model stays smooth and robust. Domain 0 to 2x the
# design rate. The top is not a plant ceiling: there is NO CAP on fuel to the boiler, and
# the site-demand ranges in plant_rig.py can call for up to ~16 kg/s at their top corner.
#
# Fitted DOWN TO ZERO FUEL since 2026-09-24 (was 3-20 kg/s, 0.3-2.0x design, with 3 kg/s
# as plant_rig's turndown floor; Hamish wants fuel able to reach 0 without extrapolating).
# Plain polynomials in fuel could not do that to the accuracy below -- stack temperature
# flattens onto the cold-side temperature exponentially as fuel -> 0, and a degree-11
# polynomial over 0-20 still missed by 0.005 K. So the fits follow the physics' own shape
# (steady_state above):
#
#   T_stack = T_cold + (T_hot - T_cold) * exp(-NTU),  NTU = UA/(mgas*cp) ~ g(f) * f^-(1-N_EXP)
#             g(f) is smooth and nearly constant (9.04-9.13), fitted as a sextic
#   Q       = f * q(f)          q(f) = G*(h_hot - h(T_stack)), smooth, fitted degree 9
#   eta     = q(f) / Q_PER_FUEL   -- no division by fuel anywhere, so fuel = 0 is exact
#
# Accuracy (see MAP_RES), over the whole 0-20 grid: T_stack ~0.0006 K (was 0.001 K over
# 3-20 -- the bound that matters: the hot-water exchanger amplifies a T_stack error ~50x
# into hot-water flow, see model_checks.py's "HW minimum met"), Q ~0.4 kW of ~100 MW.
# Below ~0.14 kg/s the stack approach (T_stack - T_cold) is under 1e-6 K and not
# resolvable in floating point, so g is fitted from 0.14 kg/s up; evaluated below that it
# changes T_stack by far less than the fit residual (exp(-NTU) < 1e-9 there).
# Fuel = 0 itself has no combustion to solve (0/0 in steady_state), so the grid starts at
# 1e-3 kg/s; the forms above are exact at 0 in the limit (Q = 0, T_stack = T_cold).
MDOT_FUEL_MIN, MDOT_FUEL_MAX = 0.0, 2.0 * MDOT_FUEL_DES
_F_SCALE = MDOT_FUEL_MAX            # fits are in x = f/_F_SCALE, for conditioning
_NTU_EXP = 1.0 - N_EXP              # NTU ~ f^-(1-N_EXP): UA ~ mgas^N_EXP over mgas*cp
_FUEL_EPS = 1e-6                    # kg/s, keeps (f + eps)^-0.4 finite at exactly f = 0
_f = np.concatenate([np.geomspace(1e-3, 0.5, 30), np.linspace(0.5, MDOT_FUEL_MAX, 120)[1:]])
_ss = [steady_state(x) for x in _f]
_T_SS = np.array([s['T_stack'] for s in _ss])
_Q_SS = np.array([s['Q'] for s in _ss])
_ST_SS = np.array([s['mdot_steam'] for s in _ss])
_resolvable = (_T_SS - T_COLD_SIDE) > 1e-6
_STACK_G = np.polyfit(_f[_resolvable]/_F_SCALE,
                      -np.log((_T_SS[_resolvable] - T_COLD_SIDE)/(T_HOT - T_COLD_SIDE))
                      * (_f[_resolvable] + _FUEL_EPS)**_NTU_EXP, 6)
#  Weighted by f: the residual that matters is in Q = f*q, not in q itself.
_DUTY_Q = np.polyfit(_f/_F_SCALE, _Q_SS/_f, 9, w=_f)


def _horner(c, x):
    """Polynomial by Horner's rule on plain floats, numpy arrays or GEKKO expressions
    alike (np.polyval wraps a GEKKO expression in a 0-d object array, which m.exp
    does not accept)."""
    y = c[0]
    for ci in c[1:]:
        y = y*x + ci
    return y


def stack_temp(mdot_fuel, exp=np.exp):
    """Steady-state stack temperature [degC] at a dry fuel rate [kg/s]. exp=m.exp in GEKKO."""
    return T_COLD_SIDE + (T_HOT - T_COLD_SIDE)*exp(
        -_horner(_STACK_G, mdot_fuel/_F_SCALE)*(mdot_fuel + _FUEL_EPS)**(-_NTU_EXP))


def duty(mdot_fuel):
    """Steady-state heat to steam [kW] at a dry fuel rate [kg/s]; exactly 0 at 0."""
    return mdot_fuel*_horner(_DUTY_Q, mdot_fuel/_F_SCALE)


def efficiency(mdot_fuel):
    """Steady-state boiler efficiency on LHV; finite at 0 (its limit, ~0.95)."""
    return _horner(_DUTY_Q, mdot_fuel/_F_SCALE)/Q_PER_FUEL


def steam_from_fuel(mdot_fuel):
    """HP steam flow [kg/s] at a given dry fuel rate [kg/s].

    duty/dh at the fixed drum pressure -- exactly what the dynamic model's
    steam = Q/dh(p) settles to, rather than a separately fitted curve."""
    return float(duty(mdot_fuel)/dh_steam(P_HP_BAR))


MDOT_STEAM_RES = float(np.abs(np.array([steam_from_fuel(x) for x in _f]) - _ST_SS).max())
_ST_SCALE = float(_ST_SS.max())

#  The inverse is FITTED, not iterated: nothing here should look like a solver -- this is
#  a lookup evaluated once per sample, outside the GEKKO model. Same f-proportional shape
#  as the forward map (fuel = steam * p(steam)), so it is exactly 0 at 0 steam.
_FUEL_FROM_STEAM = np.polyfit(_ST_SS/_ST_SCALE, _f/_ST_SS, 9, w=_ST_SS)
FUEL_INV_RES = float(np.abs(_ST_SS*np.polyval(_FUEL_FROM_STEAM, _ST_SS/_ST_SCALE) - _f).max())


def fuel_for_steam(m_steam, lo=None, hi=None):
    """Dry fuel rate [kg/s] that raises m_steam [kg/s]. Clamped to the fit range.

    Returns (fuel, clamped) so the caller can flag a demand the boiler cannot
    serve rather than silently reporting a fuel rate it would never run at.
    """
    lo = MDOT_FUEL_MIN if lo is None else lo
    hi = MDOT_FUEL_MAX if hi is None else hi
    fuel = float(m_steam*np.polyval(_FUEL_FROM_STEAM, m_steam/_ST_SCALE)) if m_steam > 0 else 0.0
    if fuel <= lo:
        return lo, True
    if fuel >= hi:
        return hi, True
    return fuel, False


MAP_RES = {'T_stack': float(np.abs(stack_temp(_f) - _T_SS).max()),
           'Q': float(np.abs(duty(_f) - _Q_SS).max())}

# Gas path volume, sized rather than guessed.
#
#   Q_fired = mdot_fuel_des * Q_PER_FUEL          fired duty at the design point [kW]
#   V       = Q_fired / Q_RELEASE_RATE            200 kW/m3 biomass furnace convention
#
# The volumetric heat release rate is the standard sizing basis for a biomass
# furnace: grate-fired biomass runs an order of magnitude below gas/oil because the
# fuel needs residence time to dry, devolatilise and burn out, and 150-250 kW/m3 is
# the usual band. 200 kW/m3 sits in the middle of it. The number that goes in is the
# FIRED duty (fuel x net LHV as-fired), not the recovered steam duty -- heat release
# is what the furnace volume has to accommodate, before any of it is transferred.
#
# This is the furnace proper. The convective passes downstream add to the flushed
# volume and are not counted here, so TAU is, if anything, slightly short.
Q_FIRED_DES = MDOT_FUEL_DES*Q_PER_FUEL                        # kW, design fired duty
V_GAS_PATH = Q_FIRED_DES/Q_RELEASE_RATE                       # m3, furnace volume

# Single time constant, tau = V / Vdot. Mixture molar mass comes from the composition
# solved above, so the density is the model's own gas, not a generic air figure.
MW_MIX = 1.0/sum(w[sp]/MW[sp] for sp in w)                    # kg/kmol
RHO_GAS = P_FLUE*MW_MIX/(8314.462618*(T_GAS_MEAN + 273.15))   # R in J/(kmol.K)
VDOT_GAS = MDOT_GAS_DES/RHO_GAS                               # m3/s at design
TAU = V_GAS_PATH/VDOT_GAS                                     # s

# The lag applied to Q and stack (see _dead_time_lag): fuel-bed heat release (TAU_BED,
# dominant) plus the gas flush (TAU) it sits in series with.
TAU_Q = TAU_BED + TAU                                         # s

# ----------------------------------------------------------------------------------
# Dynamic model
#
# One builder, used by both the standalone step test below and boiler_rig.py. The physics
# is written once; anything that steps the boiler calls this. Two copies of a lag chain is
# how a model and its test harness quietly stop agreeing.
# ----------------------------------------------------------------------------------

def _dead_time_lag(m, u, y0):
    """Dead time (THETA_FUEL, as an N_LAG chain), then ONE lag on the far side of it, at
    TAU_Q (fuel bed + gas flush). m.delay is unsafe here: its buffer starts zero-filled and
    injects a phantom transient at t=0.

    ONE state, not a cascade: two lags in series is a second-order response with an
    inflection (it starts flat, then curves), which is a worse shape to hand a PID than a
    single first-order lag with a bigger time constant. TAU_BED already dominates the gas
    flush TAU by roughly two orders of magnitude, so folding them into TAU_Q costs nothing
    that matters here.

    Returns (y, states): the delayed/lagged signal and the ODE states behind it, in the
    order they should be reseeded.
    """
    tau_link = THETA_FUEL / N_LAG
    prev, chain = u, []
    for _ in range(N_LAG):
        s = m.Var(value=y0)
        m.Equation(tau_link * s.dt() == prev - s)
        chain.append(s)
        prev = s

    y = m.Var(value=y0)
    m.Equation(TAU_Q * y.dt() == prev - y)

    return y, chain + [y]


def build_dynamics(m, fuel_cmd, p_cmd, fuel0, p0=None):
    """Attach the boiler dynamics to a GEKKO model.

    fuel_cmd, p_cmd  GEKKO Param, Var or MV. The commanded fuel flow and header pressure.
    fuel0, p0        Initial values the states are seeded at.

    Returns a dict of the states and outputs. `states` is the list needing reseeding after
    a failed solve.

    Fuel metering is instant -- there is no cap and no lag on the fuel command itself.
    Flue gas flow follows it through the FD/ID fan's own speed response (THETA_FAN_S/
    TAU_FAN_S), a separate, much faster actuator lag from the fuel-bed dead time below. The
    static map is evaluated at the instant fuel to get the heat exchanger's steady-state
    duty and stack temperature, and the dead time + lag (THETA_FUEL/TAU_Q) is applied AFTER
    the map, on those two outputs independently, representing the fuel bed re-balancing
    (dominant) plus the gas flush, rather than fuel metering or combustion.
    """
    p0 = P_HP_BAR if p0 is None else p0
    #  Horner's rule, degree-agnostic, for DH_MAP (cubic). The static map itself goes
    #  through duty()/stack_temp() above.
    poly = lambda c, x: np.polyval(c, x)

    # Instant fuel. No dynamics here by design: fuel metering is essentially instant.
    mf = m.Intermediate(fuel_cmd)

    #  Restructure phase 6. This USED to follow the FD/ID fan's own fast speed
    #  response (THETA_FAN_S/TAU_FAN_S) off the instant fuel command -- a
    #  dedicated fuel-step test showed that let downstream "available" flue-gas
    #  heat (Q_evap_gas_avail = gas*DH_GAS_HP, a FIXED enthalpy constant times
    #  this mass flow) reach its full new value within about 100 s of a step,
    #  while Q/stack/steam -- the actual combustion energy -- had not even
    #  started moving yet (THETA_FUEL alone is ~305 s) and took another 30+
    #  minutes to settle. A fan can only move flue gas that combustion has
    #  actually produced, so this is now the effective burn-rate/
    #  combustion-product-generation state the restructure plan calls for:
    #  the SAME dead time + lag as Q/stack (_dead_time_lag, THETA_FUEL/TAU_Q),
    #  not the fan's own faster mechanical response, which is not what limits
    #  this signal physically.
    gas_ss = m.Intermediate(mf * GAS_PER_FUEL)
    gas, gas_states = _dead_time_lag(m, gas_ss, fuel0 * GAS_PER_FUEL)

    # Static map at the instant fuel, giving the heat exchanger's steady-state targets.
    Q_ss = m.Intermediate(duty(mf))
    stack_ss = m.Intermediate(stack_temp(mf, exp=m.exp))

    # Dead time + lag AFTER the map, on the heat exchanger's outputs only. Q and stack
    # get independent states (so each is a proper FOPTD of its own steady-state target)
    # but share the same time constants, so they still share one shape with each other.
    Q, Q_states = _dead_time_lag(m, Q_ss, float(duty(fuel0)))
    stack, stack_states = _dead_time_lag(m, stack_ss, float(stack_temp(fuel0)))

    # Header pressure. An actuator, not part of the component: valve dead time plus stroke.
    p_act, p_states = lag_chain(m, p_cmd, theta=dp.THETA_VALVE_S, tau=TAU_P, n_lag=3,
                                y0=p0, name='blr_p')

    dh = m.Intermediate(poly(DH_MAP, p_act))
    steam = m.Intermediate(Q / dh)
    #  eta is the STEADY-STATE efficiency at the current (instant) firing rate, off Q_ss
    #  rather than the lagged Q -- with a many-minute lag on Q, Q/(mf*Q_PER_FUEL) swings
    #  through nonsense values (it can exceed 1) every time fuel steps down while Q is
    #  still working off the old, higher firing rate. eta_inst keeps that old expression
    #  as a diagnostic for anyone who wants to see the transient distortion itself.
    #  eta from efficiency(), not Q_ss/(mf*Q_PER_FUEL): same value, but no division by
    #  fuel, so fuel = 0 does not hand the solver 0/0 (2026-09-24, fuel floor -> 0).
    #  eta_inst still divides by fuel, so it is floored at _FUEL_EPS; it reads huge (not
    #  NaN) while a lagged Q is still decaying after fuel has gone to 0 -- diagnostic only.
    eta = m.Intermediate(efficiency(mf))
    eta_inst = m.Intermediate(Q / ((mf + _FUEL_EPS) * Q_PER_FUEL))

    return dict(mf=mf, p_act=p_act, Q=Q, stack=stack, gas=gas,
                dh=dh, steam=steam, eta=eta, eta_inst=eta_inst,
                states=gas_states + Q_states + stack_states + p_states)


def simulate():
    """Standalone open-loop step test at constant header pressure."""
    t = np.arange(0.0, T_END + DT, DT)
    m = GEKKO(remote=False)
    m.time = t

    u = m.Param(value=np.where(t < STEP_TIME, STEP_FROM, STEP_TO))
    p = m.Param(value=P_HP_BAR)
    d = build_dynamics(m, u, p, STEP_FROM)

    m.options.IMODE = 4
    m.options.NODES = 4
    m.solve(disp=False)

    arr = lambda v: np.array(list(v.VALUE))
    return dict(t=t, u=arr(u), mf=arr(d['mf']), steam=arr(d['steam']),
                stack=arr(d['stack']), gas=arr(d['gas']))


def fit_foptd(t, y, u_step_time, du):
    """Two-point (28.3 / 63.2 percent) fit. Smith's method."""
    y0, yf = y[0], y[-1]
    dy = yf - y0
    if abs(dy) < 1e-12:
        return 0.0, 0.0, 0.0
    frac = (y - y0) / dy
    t28 = np.interp(0.283, frac, t) - u_step_time
    t63 = np.interp(0.632, frac, t) - u_step_time
    tau = 1.5 * (t63 - t28)
    return dy / du, tau, max(t63 - tau, 0.0)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    des = steady_state(MDOT_FUEL_DES)

    print('COMBUSTION  fixed by the constant air-fuel ratio')
    print(f'  excess air ratio        {LAMBDA:9.4f}')
    print(f'  flue gas per kg fuel    {GAS_PER_FUEL:9.4f} kg/kg dry')
    print(f'  LHV dry                 {LHV_DRY:9.1f} kJ/kg')
    print(f'  net heat per kg fuel    {Q_PER_FUEL:9.1f} kJ/kg')
    print(f'  adiabatic flame temp    {T_HOT:9.1f} degC')
    print()
    print('PROPERTY ROUTE')
    print(f'  flue gas   CoolProp per species, mass weighted, tabulated '
          f'{T_REF:.0f} to {T_TAB_MAX:.0f} degC at {P_PROP/1e3:.0f} kPa')
    print(f'  steam      IAPWS-95 via iapws95_gekko, the same package as the turbine model')
    print(f'  h_steam   {H_STEAM:9.2f} kJ/kg  at {T_STEAM_C:.2f} degC, {P_HP_BAR:.0f} bar   '
          f'(T_sat {T_SAT_HP:.1f} degC)')
    print(f'  h_fw      {H_FW:9.2f} kJ/kg  at {T_FW:.1f} degC, '
          f'{"compressed liquid at " + format(P_HP_BAR, ".0f") + " bar" if FW_COMPRESSED else "saturated liquid"}')
    print(f'  delta h   {H_STEAM - H_FW:9.2f} kJ/kg')
    print(f'  h_fg at {T_REF:.0f} degC   {HFG_25:.2f} used, IAPWS gives {HFG_IAPWS:.2f}, '
          f'{(HFG_IAPWS-HFG_25)/HFG_25*100:+.3f} percent')
    print()
    print('DESIGN POINT')
    print(f'  fuel                    {MDOT_FUEL_DES:9.2f} kg/s dry')
    print(f'  flue gas                {des["mdot_gas"]:9.2f} kg/s')
    print(f'  stack temperature       {des["T_stack"]:9.2f} degC')
    print(f'  heat to steam           {des["Q"]/1e3:9.2f} MW')
    print(f'  steam                   {des["mdot_steam"]:9.2f} kg/s')
    print(f'  efficiency on LHV       {des["eta"]:9.4f}')
    print(f'  UA                      {UA_DES:9.0f} kW/K   NTU {NTU_DES:.3f}')
    print(f'  static map fit residual  duty {MAP_RES["Q"]:.2e} kW, '
          f'stack {MAP_RES["T_stack"]:.2e} degC, dh {DH_RES:.2e} kJ/kg')
    print()
    print('DYNAMIC PARAMETERS')
    print(f'  fired duty              {Q_FIRED_DES/1e3:9.2f} MW   fuel x net LHV')
    print(f'  gas path volume         {V_GAS_PATH:9.0f} m3   '
          f'= Q_fired / {Q_RELEASE_RATE:.0f} kW/m3')
    print(f'  flue gas density        {RHO_GAS:9.4f} kg/m3 at {T_GAS_MEAN:.0f} degC, '
          f'MW {MW_MIX:.2f}')
    print(f'  volumetric flow         {VDOT_GAS:9.1f} m3/s  at the design point')
    print(f'  tau (gas flush)         {TAU:9.2f} s')
    print(f'  tau_bed (fuel bed)      {TAU_BED:9.2f} s   dominant -- see fuel-bed section above')
    print(f'  tau_q = tau_bed + tau   {TAU_Q:9.2f} s   applied to Q and stack, not the fuel')
    print(f'  tau_metal               {TAU_METAL:9.1f} s   diagnostic only -- NOT applied '
          f'(metal/water sit at saturation)')
    print(f'  t_a, t_b, t_c           {_t_a:6.0f}, {_t_b:6.0f}, {_t_c:6.0f} s   '
          f'ignition, front-at-grate, conversion end')
    print(f'  theta_bed               {THETA_BED:9.1f} s')
    print(f'  theta_steam_line        {THETA_STEAM_LINE:9.2f} s   boiler -> turbine 1 main line')
    print(f'  theta_fuel = feeder+bed+line {THETA_FUEL:9.1f} s')
    print(f'  header valve theta/tau  {dp.THETA_VALVE_S:.1f} / {TAU_P:.1f} s   actuator, not the component')
    print(f'  gas fan theta/tau       {dp.THETA_FAN_S:.1f} / {dp.TAU_FAN_S:.1f} s   FD/ID fan speed response')
    print(f'  settling, theta_fuel + 3*tau_q  {(THETA_FUEL + 3*TAU_Q)/60:9.2f} min  (95%)')
    print('  Literature (Masa et al. 2011, via Hosovsky 2015) puts a fuel-increase FOPTD on')
    print('  a 1 MW moving-grate hot-water boiler at theta 851 / tau 499 s, 95% at 2030 s;')
    print('  this reduction is the fuel bed itself, not the gas flush -- see section 10 of')
    print('  the dynamics change notes for the full literature comparison.')
    print()

    r = simulate()
    du = STEP_TO - STEP_FROM
    print(f'STEP TEST  fuel {STEP_FROM:.1f} to {STEP_TO:.1f} kg/s at t = {STEP_TIME:.0f} s')
    print(f"  {'channel':10s} {'gain':>14s} {'tau (s)':>10s} {'theta (s)':>10s} {'units'}")
    for lab, y, un in [('steam', r['steam'], 'kg/s per kg/s'),
                       ('flue gas', r['gas'], 'kg/s per kg/s'),
                       ('stack T', r['stack'], 'degC per kg/s')]:
        K, tau, th = fit_foptd(r['t'], y, STEP_TIME, du)
        print(f'  {lab:10s} {K:14.4f} {tau:10.1f} {th:10.1f} {un}')
    print()
    print('  Steam and stack T share their own state each (independent, same tau_q/theta),')
    print('  so they share one shape with each other; flue gas follows the FD/ID fan\'s own')
    print(f'  much faster response (theta {dp.THETA_FAN_S:.1f} s, tau {dp.TAU_FAN_S:.1f} s) instead.')
    print('  Fitted values are the EFFECTIVE FOPTD a controller sees on Q/stack: the dead')
    print(f'  time exceeds the {THETA_FUEL:.0f} s value because the N_LAG chain approximating it is')
    print('  itself higher order. Use the fitted values for PID tuning and the physical')
    print('  ones inside the plant model.')

    # ----------------------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------------------

    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    tm = r['t'] / 60.0

    # Common time axis, long enough for the slowest channel to settle
    dy = r['steam'] - r['steam'][0]
    frac = np.abs(dy / dy[-1])
    i99 = np.argmax(frac >= 0.99)
    XMAX = min(tm[i99] * 1.2 if frac[i99] >= 0.99 else tm[-1], tm[-1])


    def norm(y):
        return (y - y[0]) / (y[-1] - y[0])


    ax[0].plot(tm, r['gas'], lw=1.8, color='tab:blue')
    ax[0].set_ylabel('flue gas  kg/s')
    ax[0].set_title(f'Boiler response to a fuel step, {STEP_FROM:.0f} to {STEP_TO:.0f} kg/s'
                    f'\n\nFlue gas flow to the heat pump   fan tau {dp.TAU_FAN_S:.0f} s', fontsize=10)

    ax[1].plot(tm, r['steam'], lw=1.8, color='tab:red')
    ax[1].set_ylabel('steam  kg/s')
    ax[1].set_title(f'Steam flow to the turbine   tau_q {TAU_Q:.1f} s', fontsize=10)

    ax[2].step(tm, norm(r['u']), where='post', lw=1.4, color='0.4', label='fuel command (step)')
    ax[2].plot(tm, norm(r['stack']), lw=1.8, ls='--', color='tab:green', label='stack temperature')
    ax[2].plot(tm, norm(r['steam']), lw=1.8, color='tab:red', label='steam flow')
    ax[2].axhline(0.632, ls=':', c='k', lw=0.8)
    ax[2].text(XMAX * 0.99, 0.652, '63.2 %', ha='right', fontsize=8)
    ax[2].set_ylabel('normalised response')
    ax[2].set_title('All channels normalised, against the instantaneous command', fontsize=10)
    ax[2].set_ylim(-0.05, 1.12)
    ax[2].legend(loc='lower right', fontsize=8)

    for a in ax:
        a.set_xlim(0, XMAX)
        a.set_xlabel('time  minutes')
        a.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
