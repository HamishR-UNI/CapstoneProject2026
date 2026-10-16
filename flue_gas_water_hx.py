"""
Flue-gas-to-water heat exchanger: heats a fixed-temperature water source off the
boiler's flue gas, sized as a specified-outlet-temperature duty calculation rather
than an effectiveness-NTU block -- there is nothing implicitly coupled here (the flue
gas outlet is a fixed spec, not solved), so no bisection is needed.

Flue gas properties are boiler.py's, not re-derived: same combustion products leaving
the same boiler, so h_gas() and the composition it was built from are imported rather
than re-tabulated.

Structure mirrors boiler.py: a build-time property/fit section, a steady_state()
function for offline/design-point use, and a build_dynamics() function for the live
GEKKO rig.

The residence time is the WATER-side one, T_RES_HX = V_water/Vdot_water, NOT the gas
flush that boiler.py and turbine_model.py use. Those two are mixing volumes where the
flowing fluid is itself the thermal store; here the gas holds about a thousandth of
the stored heat and moves 1250x the volumetric flow, so a gas flush time (~0.1 s) is
a correct residence time and the wrong time constant. See the sizing section below
for the derivation and the assumptions it rests on.

The water side is PLUG FLOW along the tube bank, not a stirred tank, so T_RES_HX is
split by Smith's two-point method into a lag (TAU_HX) and dead time (THETA_HX on the
duty channel, THETA_TWOUT on the T_w_out setpoint channel -- see dynamics_params.
PLUG_TAU_FRAC/PLUG_THETA_FRAC), rather than applied whole as a single mixing-volume
lag. The delivered water flow carries its own dead time (THETA_WATER: valve dead time
plus the pipe run to site, also plug flow) and the valve's own stroke (TAU_WATER).
"""

import numpy as np

import boiler as blr
import dynamics_params as dp
from turbine_model import lag_chain

# ----------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------

T_FG_OUT_SPEC = 90.0                         # degC, specified flue gas outlet, fixed

#  Site RECYCLE, not a fresh cold supply: the water this exchanger heats is
#  return water coming back from site, so it arrives already warm. That single
#  number drives the whole water side -- the exchanger's duty is fixed by the
#  gas side alone, so a warmer inlet means the same duty has to be carried by a
#  much larger flow across a much smaller temperature rise (65 K of rise at
#  20 degC became 15 K at 70 degC and an 85 degC setpoint, so ~4.3x the flow).
T_W_IN = 60.0                                # degC, site recycle return, fixed for now
CP_WATER = 4.19                              # kJ/kg.K, liquid water

T_W_OUT_0 = 85.0                             # degC, the site's desired hot water temperature --
#  Floor is a 10 K minimum approach ABOVE T_W_IN, not the old 60 degC. With a
#  70 degC recycle inlet, a setpoint at or below it is not a small error -- it
#  asks the exchanger to deliver its duty across zero or negative temperature
#  rise, and m_water = Q/(cp*(T_sp - T_w_in)) goes to infinity and then
#  negative. Keep this strictly above T_W_IN if T_W_IN ever changes again.
T_W_OUT_LO, T_W_OUT_HI = 80.0, 95.0          # this is the slider now; flow is derived from it
assert T_W_OUT_LO > T_W_IN + 1e-6, (
    f'T_W_OUT_LO ({T_W_OUT_LO} degC) must sit above the water inlet '
    f'({T_W_IN} degC) -- below it the required flow is negative')

# Equipment assumptions. All four are sizing inputs to be checked against real
# equipment data -- they are the assumptions this exchanger's area, volume and
# time constant all rest on.
#
#   U_HX      gas-side-limited overall coefficient, finned economiser on biomass
#             flue gas. 30-50 W/m2K is the usual band; the gas film sets it almost
#             by itself because the water-side coefficient is ~50x larger.
#   BETA_HX   total (finned) area density of the bundle. The fins are what buy the
#             area, which is exactly why the gas side can run at such a low U and
#             still transfer the duty.
#   FIN_RATIO total finned area / bare tube outside area. 6-12 is the normal band
#             for an economiser; it is what converts the finned area back into a
#             tube length, and hence into a water-side holdup.
#   D_TUBE_*  tube outside/inside diameter, standard economiser tube.
U_HX = 40.0                                  # W/m2K, gas-side-limited finned economiser
BETA_HX = 300.0                              # m2/m3, finned-tube area density
FIN_RATIO = 8.0                              # -, finned area / bare tube area
D_TUBE_O = 0.038                             # m, tube OD
D_TUBE_I = 0.034                             # m, tube ID

RHO_WATER = 983.0                            # kg/m3, liquid water near the mean run temperature

# ----------------------------------------------------------------------------------
# Flue gas enthalpy, fitted as a polynomial over this exchanger's operating range
#
# blr.h_gas() interpolates a CoolProp table with np.interp, which GEKKO cannot call
# symbolically. T_fg_in is a live GEKKO variable in the rig, so it needs a polynomial
# the same way boiler.py's own stack temperature/duty outputs do.
# ----------------------------------------------------------------------------------

H_FG_OUT = blr.h_gas(T_FG_OUT_SPEC)          # plain float: the outlet spec is fixed

# Sweep the same fuel-flow domain boiler.py fits its own maps over, to find the range
# of flue gas inlet temperatures this exchanger will actually see.
_fuel_sweep = np.linspace(0.3 * blr.MDOT_FUEL_DES, 1.05 * blr.MDOT_FUEL_DES, 40)
_stack_sweep = np.array([blr.steady_state(x)['T_stack'] for x in _fuel_sweep])

T_FG_IN_LO = float(_stack_sweep.min()) - 15.0
T_FG_IN_HI = float(_stack_sweep.max()) + 15.0

_T_grid = np.linspace(T_FG_IN_LO, T_FG_IN_HI, 60)
H_GAS_MAP = np.polyfit(_T_grid, np.array([blr.h_gas(T) for T in _T_grid]), 3)
H_GAS_RES = float(np.abs(np.array([blr.h_gas(T) for T in _T_grid])
                         - np.polyval(H_GAS_MAP, _T_grid)).max())

# ----------------------------------------------------------------------------------
# Design point, used to size tau = V / Vdot
# ----------------------------------------------------------------------------------

_DES = blr.steady_state(blr.MDOT_FUEL_DES)
MDOT_GAS_DES = _DES['mdot_gas']
T_FG_IN_DES = _DES['T_stack']

R_UNIV = 8314.462618                         # J/(kmol.K)
_T_MEAN_K = 0.5 * (T_FG_IN_DES + T_FG_OUT_SPEC) + 273.15
RHO_GAS_HX = blr.P_FLUE * blr.MW_MIX / (R_UNIV * _T_MEAN_K)   # kg/m3
VDOT_GAS_DES = MDOT_GAS_DES / RHO_GAS_HX                       # m3/s

_Q_DES = MDOT_GAS_DES * (blr.h_gas(T_FG_IN_DES) - H_FG_OUT)   # kW

# ----------------------------------------------------------------------------------
# Sizing, and the time constant
#
# Area from the design duty:
#
#   LMTD  counterflow, gas in vs water out / gas out vs water in
#   A_HX  = Q / (U * LMTD)                   heat transfer area the duty needs [m2]
#   V_HX  = A_HX / BETA_HX                   bundle envelope volume [m3]
#
# T_RES_HX is the WATER-side residence time, not the gas flush. This is a
# deliberate change from the recipe boiler.py and turbine_model.py use, and the
# reason is that those two are mixing volumes where the flowing fluid IS the
# thermal store, whereas here it is not:
#
#   flue gas in the bundle envelope   ~13 kJ/K
#   water in the tubes               ~15000 kJ/K
#
# The gas carries a thousandth of the stored heat and moves 1250x the volumetric
# flow (109 m3/s against 0.088 m3/s), so V/Vdot on the gas gives ~0.1 s -- a
# correct flush time, and the wrong time constant. What actually lags the outlet
# water temperature is the water having to traverse the tubes.
#
# The tube metal is NOT the slow store either, despite the mass: the water-side
# film coefficient is ~2000 W/m2K against ~45 on the gas, so the metal
# equilibrates with the water in of order 1 s (M.c / (UA_gas + UA_water)).
# It is left out rather than lumped in.
#
# Water holdup comes from the geometry already implied by BETA_HX and FIN_RATIO,
# so it introduces no new free parameter:
#
#   A_bare    = A_HX / FIN_RATIO              bare tube outside area  [m2]
#   L_tube    = A_bare / (pi * D_TUBE_O)      total tube length       [m]
#   V_water   = L_tube * pi/4 * D_TUBE_I^2    internal volume         [m3]
#   T_RES_HX  = V_water / Vdot_water
#
# The water side is PLUG FLOW along the tube length, not a stirred mixing volume --
# a slug of water moves down the tube bank and comes out the far end, it does not
# blend with what is already in there. Smith's two-point fit of a plug-flow/uniform-
# heat-flux ramp reduces one residence time to an FOPTD (dynamics_params.
# PLUG_TAU_FRAC/PLUG_THETA_FRAC): TAU_HX = 0.5235*T_RES_HX, and the remainder shows
# up as dead time (THETA_HX/THETA_TWOUT below), not as more lag.
#
# Sensitivity to the one new assumption, FIN_RATIO, over its plausible band, at
# T_W_IN = 60 degC (recomputed with the plug-flow reduction, not the old T_RES
# figure directly):
#   fin ratio  6 -> V_water 4.17 m3 -> t_res  79.1 s -> tau_hx 41.4 s, theta_twout  8.6 s
#   fin ratio  8 -> V_water 3.13 m3 -> t_res  59.3 s -> tau_hx 31.1 s, theta_twout  6.4 s
#   fin ratio 10 -> V_water 2.50 m3 -> t_res  47.5 s -> tau_hx 24.8 s, theta_twout  5.2 s
#   fin ratio 12 -> V_water 2.08 m3 -> t_res  39.6 s -> tau_hx 20.7 s, theta_twout  4.3 s
# PARK FOR TIM alongside U_HX and BETA_HX -- all three are equipment assumptions.
#
# NOTE this does not apply to the heat pump's condenser zones. Those are phase
# change on both sides: steam production follows duty essentially instantly
# (m_lps = Q/dh), there is no inventory that has to warm through first, and a
# near-zero lag is the physically correct answer there. Sensible heating of a
# liquid stream carries a transport lag; isothermal phase change does not.
# ----------------------------------------------------------------------------------

_DT1 = T_FG_IN_DES - T_W_OUT_0                # gas in vs water out (counterflow)
_DT2 = T_FG_OUT_SPEC - T_W_IN                 # gas out vs water in
LMTD_HX = (_DT1 - _DT2) / np.log(_DT1 / _DT2) if abs(_DT1 - _DT2) > 1e-9 else _DT1

A_HX = (_Q_DES * 1e3) / (U_HX * LMTD_HX)      # m2  (Q in W, U in W/m2K)
V_HX = A_HX / BETA_HX                         # m3, bundle envelope

# The site's desired water temperature is what's commanded; duty is fixed purely
# by the gas side (mdot_fg, T_fg_in and the fixed T_fg_out spec above), so the
# water flow needed to hit that temperature is derived, not set directly --
# these are reference/design-point figures only now, not slider bounds.
M_WATER_0  = _Q_DES / (CP_WATER * (T_W_OUT_0 - T_W_IN))
M_WATER_LO = _Q_DES / (CP_WATER * (T_W_OUT_HI - T_W_IN))    # hotter target -> less flow
M_WATER_HI = _Q_DES / (CP_WATER * (T_W_OUT_LO - T_W_IN))    # cooler target -> more flow

VDOT_WATER_DES = M_WATER_0 / RHO_WATER        # m3/s

A_BARE_HX = A_HX / FIN_RATIO                            # m2, bare tube outside
L_TUBE_HX = A_BARE_HX / (np.pi * D_TUBE_O)              # m, total tube length
V_WATER_HX = L_TUBE_HX * np.pi/4 * D_TUBE_I**2          # m3, water in the tubes

T_RES_HX = V_WATER_HX / VDOT_WATER_DES                  # s, water residence (plug flow)

#  Plug-flow reduction: T_RES_HX splits into a lag (TAU_HX) and dead time. The duty
#  channel additionally carries the flue gas duct transport from the boiler exit
#  (dynamics_params.L_FG_DUCT_M/V_FG_DUCT_MPS); the T_w_out setpoint channel does not,
#  since a setpoint change does not physically travel down that duct.
TAU_HX = dp.PLUG_TAU_FRAC * T_RES_HX                    # s, applied to duty and T_w_out
THETA_HX = dp.L_FG_DUCT_M/dp.V_FG_DUCT_MPS + dp.PLUG_THETA_FRAC*T_RES_HX   # s, duty channel
THETA_TWOUT = dp.PLUG_THETA_FRAC * T_RES_HX             # s, T_w_out setpoint channel

# Kept as a reported diagnostic only. This is what TAU_HX used to be treated as, and
# the ratio between the two is the size of the error that basis carried.
TAU_GAS_FLUSH = V_HX / VDOT_GAS_DES                     # s, gas residence

# ----------------------------------------------------------------------------------
# Water-side valve/pipe dynamics. This is the pipe run BETWEEN the control valve
# and the exchanger, and it is applied to the DELIVERED WATER FLOW in
# build_dynamics() -- a different signal from TAU_HX, which lags the duty. The
# two are not in series on one path: a duty change reaches the flow through
# TAU_HX then TAU_WATER, but a SETPOINT change reaches it through TAU_WATER
# alone, because the setpoint does not touch the duty.
#
# This is PLUG FLOW too (a pipe run, not a mixing volume), so it is dead time
# (THETA_WATER: valve dead time plus the pipe transit, dynamics_params.L_HW_RUN_M/
# V_HW_PIPE_MPS) rather than a mixing-volume lag. TAU_WATER (keeping the name other
# files read) is now just the valve stroke -- the old V_WATER_RUN = 2.0 m3 mixing
# volume this used to rest on was flagged in this file's own comments as "the only
# guessed volume left in this module": it double-counted the same pipe run as BOTH
# a stirred tank (V_WATER_RUN/Vdot, a lag) and, now, plug-flow dead time. Retired.
# ----------------------------------------------------------------------------------

THETA_WATER = dp.THETA_VALVE_S + dp.L_HW_RUN_M/dp.V_HW_PIPE_MPS   # s, valve + pipe transit
TAU_WATER = dp.TAU_VALVE_S                                        # s, valve stroke only

# ----------------------------------------------------------------------------------
# Steady-state duty calculation
# ----------------------------------------------------------------------------------

def steady_state(mdot_fg, T_fg_in, T_w_out_setpoint=T_W_OUT_0, T_fg_out=T_FG_OUT_SPEC, T_w_in=T_W_IN):
    """Duty and required water flow from the water-side energy balance, given
    the site's desired outlet temperature. At steady state the flow exactly
    achieves the setpoint (no valve lag yet to create an error), so T_w_out
    here just echoes the setpoint back -- it earns its place once the valve
    lag is added in build_dynamics().

    dT_hot/dT_cold are reported, not enforced -- same convention as the pinch
    diagnostics in plant_flowsheet.py's flue_gas_hx().
    """
    Q = mdot_fg * (blr.h_gas(T_fg_in) - blr.h_gas(T_fg_out))
    m_water = Q / (CP_WATER * (T_w_out_setpoint - T_w_in))
    T_w_out = T_w_out_setpoint
    dT_hot = T_fg_in - T_w_out
    dT_cold = T_fg_out - T_w_in
    return dict(Q=Q, m_water=m_water, T_w_out=T_w_out, dT_hot=dT_hot, dT_cold=dT_cold,
               T_fg_out=T_fg_out)


# ----------------------------------------------------------------------------------
# Dynamic model
# ----------------------------------------------------------------------------------

def build_dynamics(m, T_fg_in, mdot_fg, T_w_out_setpoint, y0=None):
    """Attach the exchanger to a GEKKO model.

    T_fg_in, mdot_fg    Live GEKKO variables -- the rig's own boiler flue gas outlet.
    T_w_out_setpoint    GEKKO Param/Var/MV, the site's desired hot water temperature
                       (slider-set). The outlet temperature follows this behind
                       THETA_TWOUT/TAU_HX -- the water has to traverse the tubes
                       before the new outlet temperature appears, the same
                       residence time (plug-flow reduced) that lags the duty. The
                       flow needed to carry the duty at that actual temperature is
                       then derived, and passed through the valve dead time/stroke
                       (THETA_WATER/TAU_WATER) before it is reported delivered.

                       NOTE the causality is still inverted relative to the plant:
                       a real installation strokes the VALVE and the outlet
                       temperature is the consequence, whereas here the
                       temperature is commanded and the flow is derived. The lag
                       below makes the temperature response physical rather than
                       instantaneous, but it does not fix the direction. Flagged
                       for the supervisor; fixing it properly means making water
                       flow the MV, which changes the slider's meaning and touches
                       plant_rig/plant_manual/the CSV schema.
    y0                  Initial value for the duty state; defaults to the design duty.

    Returns Q, T_w_out, dT_hot, m_water_act, the combined state list, and the
    three state groups separately (duty_states, twout_states, water_states) so
    callers reseed each with the right value rather than slicing by index.
    """
    poly = lambda c, x: c[0] * x**3 + c[1] * x**2 + c[2] * x + c[3]

    Q_ss = m.Intermediate(mdot_fg * (poly(H_GAS_MAP, T_fg_in) - H_FG_OUT))

    # The exchanger's own water residence time (plug-flow reduced into THETA_HX/TAU_HX),
    # plus the flue gas duct transport from the boiler exit. Everything downstream is
    # algebraic on this one lagged duty, so it can never disagree with itself --
    # same reasoning boiler.py uses for lagging delivered fuel ahead of its map.
    y0 = _Q_DES if y0 is None else y0
    Q, duty_states = lag_chain(m, Q_ss, theta=THETA_HX, tau=TAU_HX, n_lag=6,
                               y0=y0, name='hx')

    #  The outlet temperature follows the setpoint behind the same water residence
    #  time (THETA_TWOUT/TAU_HX, no duct term -- a setpoint change does not
    #  physically travel down the flue gas duct). It used to be pinned algebraically
    #  (T_w_out = setpoint), which made it reach target in zero time -- inconsistent
    #  with the water residence this exchanger actually has, and it left the
    #  cmd/actual pair on the trend identical to the last decimal. While twout_cmd
    #  remains an independent MV this lag is what makes the pair mean anything.
    T_w_out, twout_states = lag_chain(m, T_w_out_setpoint, theta=THETA_TWOUT, tau=TAU_HX,
                                      n_lag=6, y0=T_W_OUT_0, name='hxT')

    #  Energy balance closes on the ACTUAL outlet temperature, not the command:
    #  the flow required right now is the one that carries the live duty across
    #  the temperature rise the exchanger is actually delivering.
    m_water_req = m.Intermediate(Q / (CP_WATER * (T_w_out - T_W_IN)))

    #  ...and REQUIRED is not DELIVERED. The control valve has to stroke (dead time
    #  plus stroke, THETA_WATER/TAU_WATER) and the pipe run between it and the
    #  tubes has to be traversed (folded into THETA_WATER, plug flow) before the
    #  new flow arrives.
    m_water_act, water_states = lag_chain(m, m_water_req, theta=THETA_WATER, tau=TAU_WATER,
                                          n_lag=6,
                                          y0=M_WATER_0, name='hxw')

    dT_hot = m.Intermediate(T_fg_in - T_w_out)

    return dict(Q=Q, T_w_out=T_w_out, dT_hot=dT_hot, m_water_act=m_water_act,
                states=duty_states + twout_states + water_states,
                duty_states=duty_states, twout_states=twout_states,
                water_states=water_states)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    ss = steady_state(MDOT_GAS_DES, T_FG_IN_DES, T_W_OUT_0)

    print('FLUE GAS PROPERTIES  reused from boiler.py, not re-tabulated')
    print(f'  h_gas fit range         {T_FG_IN_LO:9.1f} to {T_FG_IN_HI:6.1f} degC')
    print(f'  h_gas fit residual      {H_GAS_RES:9.2e} kJ/kg')
    print()
    print('DESIGN POINT')
    print(f'  flue gas in             {MDOT_GAS_DES:9.2f} kg/s   at {T_FG_IN_DES:6.1f} degC')
    print(f'  flue gas out (spec)     {T_FG_OUT_SPEC:9.1f} degC')
    print(f'  water in / T_w_out sp   {T_W_IN:9.1f} degC / {T_W_OUT_0:.1f} degC')
    print(f'  duty                    {ss["Q"]/1e3:9.2f} MW')
    print(f'  water flow required     {ss["m_water"]:9.2f} kg/s')
    print(f'  water out               {ss["T_w_out"]:9.2f} degC')
    print(f'  pinch  hot end          {ss["dT_hot"]:9.1f} degC  {"" if ss["dT_hot"] > 0 else "<< TEMPERATURE CROSS"}')
    print(f'  pinch  cold end         {ss["dT_cold"]:9.1f} degC  {"" if ss["dT_cold"] > 0 else "<< TEMPERATURE CROSS"}')
    print()
    print('DYNAMICS')
    print(f'  gas density             {RHO_GAS_HX:9.4f} kg/m3 at {_T_MEAN_K - 273.15:.0f} degC mean, '
          f'MW {blr.MW_MIX:.2f}')
    print(f'  volumetric flow         {VDOT_GAS_DES:9.3f} m3/s at the design point')
    print(f'  LMTD                    {LMTD_HX:9.2f} K     counterflow')
    print(f'  area = Q / (U.LMTD)     {A_HX:9.0f} m2    U {U_HX:.0f} W/m2K')
    print(f'  bundle volume = A/beta  {V_HX:9.2f} m3    beta {BETA_HX:.0f} m2/m3')
    print(f'  bare tube area          {A_BARE_HX:9.0f} m2    fin ratio {FIN_RATIO:.0f}')
    print(f'  tube length             {L_TUBE_HX:9.0f} m     OD {D_TUBE_O*1e3:.0f} mm')
    print(f'  water holdup in tubes   {V_WATER_HX:9.2f} m3    ID {D_TUBE_I*1e3:.0f} mm')
    print(f'  t_res = V_water/Vdot_w  {T_RES_HX:9.1f} s   plug flow, reduced below')
    print(f'  tau_hx (duty, T_w_out)  {TAU_HX:9.1f} s   0.5235 * t_res')
    print(f'  theta_hx (duty channel) {THETA_HX:9.1f} s   duct + 0.1085*t_res')
    print(f'  theta_twout (setpoint)  {THETA_TWOUT:9.1f} s   0.1085 * t_res, no duct term')
    print(f'  (gas flush, diagnostic) {TAU_GAS_FLUSH:9.2f} s   '
          f'{TAU_HX/TAU_GAS_FLUSH:.0f}x smaller -- wrong basis, not used')
    print()
    print('WATER-SIDE ACTUATOR DYNAMICS')
    print(f'  water density           {RHO_WATER:9.1f} kg/m3')
    print(f'  volumetric flow         {VDOT_WATER_DES*1e3:9.3f} L/s at the design point')
    print(f'  theta_water (valve+pipe) {THETA_WATER:9.1f} s   dead time -- plug flow, no mixing volume')
    print(f'  tau_water (valve stroke) {TAU_WATER:9.2f} s')
    print(f'  applied to              {"":9}      delivered water flow, not the duty')
