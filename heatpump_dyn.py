"""
Dynamic flue gas heat pump (primary machine) -- Hammerstein structure in GEKKO.

This is the primary of a two-machine arrangement. Its evaporator receives heat from
two separate sources at once: the flue gas directly, after the existing 90 degC HX
spec (flue_gas_water_hx.py), and the condenser reject from a second, water-sourced
booster machine (heatpump_secondary.py). The two sources don't need to share a
temperature -- they each only have to clear their own pinch above this machine's own
fixed evaporating temperature, so their duties simply add. The condenser side keeps
the original single-machine design's split: LPS raised on one side, hot water on
the other, split by subcooling. The refrigerant is R1233zd(E), not the original
n-Butane -- see the REF constant below for why.

Static map plus unity gain dynamics, so the time constants govern the path between
operating points but cannot move where the model settles.

The map is not a regression of the outputs. Fitting the outputs directly works
badly, because hot water production is a ratio of enthalpy differences and no
polynomial of sensible order follows it. Instead the property calls are fitted (or,
where the underlying temperature is now fixed, computed once) as plain numbers, and
the cycle algebra is carried through exactly.

Condenser duty is factored out rather than fitted. Every output is exactly
proportional to it, because duty only sets refrigerant flow while the state points
are fixed by temperatures and pressures.

The flue-gas-side evaporator duty is a fixed-outlet-spec calculation, the same
convention flue_gas_water_hx.py already uses for its own duty: the gas is assumed to
always give up its full duty down to a fixed floor temperature, not throttled to
match demand. Whether that duty is actually enough for what's commanded is a
feasibility question, reported (dQ_shortfall below), not enforced -- there is no
min()/branching here, consistent with the project's documented aversion to the MPCC
slack variables those introduce (see plant_flowsheet.py's letdown/vent valves).

The model does not simulate condensation -- flue gas cooling is sensible heat only,
the same simplification boiler.py already makes throughout (see its h_gas() comment
on keeping water vapour "superheated" in the property tables). T_FG_OUT_HP_SPEC is
chosen conservatively above the likely dewpoint for a 40%-moisture wood fuel so the
recovered-heat numbers stay defensible as genuine sensible heat.

CoolProp is called once at build time, over a grid, to build the polynomial/bspline
fits below. Nothing outside those fits and a handful of fixed numbers reaches the
solver.

Requires CoolProp and gekko.
"""

import numpy as np
from functools import lru_cache
from CoolProp.CoolProp import PropsSI
from gekko import GEKKO

import boiler as blr
import flue_gas_water_hx as hx
import dynamics_params as dp
from iapws95_gekko import smooth_max, smooth_min
from turbine_model import lag_chain


# ----------------------------------------------------------------------
# Plant specification and cycle assumptions
# ----------------------------------------------------------------------

#  R1233zd(E), not the original n-Butane: sharing the header with turbine 2
#  means this machine's condenser has to reach whatever temperature raises
#  LPS steam at the live header pressure, and n-Butane's ~152 degC critical
#  point can't clear that above roughly 2.3-2.5 bar. R1233zd(E) is a real,
#  commercially available refrigerant purpose-built for high-temperature
#  process heat pumps (Tc = 165.7 degC), with enough margin to cover
#  P_LPS_RANGE below.
REF = "R1233zd(E)"
#  DISPLAY-ONLY constant, added 2026-09-23 for the operator dashboard's LPS-pressure
#  warning (plant_operator.py) -- not used in any cycle equation above or below. Lets the
#  dashboard show live margin-to-critical as the header pressure moves, now that
#  header_closures.py's LPS dump/relief no longer caps it (Hamish's own call, see
#  docs/PLAN_A_addendum_2026-09-23.md): T_cond at P_LPS_RANGE's own 4 bar upper edge is
#  153.6 degC, 12 degC below this -- the same margin the comment above already quotes.
T_CRIT_REF = PropsSI('Tcrit', REF) - 273.15   # degC
#  P_LPS is now a live, shared quantity -- the same LPS header pressure
#  turbine 2 discharges into (plant_rig.py's P_LPS slider), not a private
#  fixed constant (see build_dynamics' P_LPS_bar argument below). P_LPS_DESIGN
#  is only where the module's own design point/standalone report is
#  evaluated; P_LPS_RANGE is the domain the live cycle-point fits below are
#  valid over -- capped at 4 bar, where T_cond (153.6 degC) still keeps a
#  12 degC margin below R1233zd(E)'s critical point. heatpump_dyn.py can't
#  import plant_rig.py (which imports this module), so the range is
#  duplicated here rather than shared -- keep it in step with plant_rig.py's
#  P_LPS_LO/HI.
P_LPS_DESIGN = 3.0     # bar
P_LPS_RANGE = (2.0, 4.0)   # bar
DT_SH_LPS = 5.0        # K
T_HW = 85.0            # C
P_HW = 1.0             # bar
T_MU = 20.0            # C

#  LPS feedwater. Not T_MU: an LPS header runs on returned condensate, not on
#  cold makeup, and charging the condenser for heating water from 20 degC
#  overstated the duty per kg of steam by about 12%. T_MU stays where it
#  belongs -- nothing on this plant actually raises LPS from 20 degC water.
T_LPS_FEED = 90.0      # C

ETA_IS = 0.70

#  Subcooling is a fixed design value now, not a manipulated variable. It is
#  needed here (rather than only at the sizing block below) because the flash
#  economiser's mass split depends on the enthalpy of the liquid leaving the
#  condenser, which is set by it.
DT_SUB_FIXED = 10.0    # K
DT_PINCH_EVAP = 10.0   # K
DT_PINCH_COND = 5.0    # K
DT_SUPERHEAT = 5.0     # K

#  DISPLAY-ONLY, added 2026-09-23 alongside T_CRIT_REF above (see that constant's comment).
#  The LPS header pressure at which T_cond = T_sat(P_LPS) + DT_SH_LPS + DT_PINCH_COND would
#  reach T_CRIT_REF -- i.e. where _cycle_points(P_LPS) itself would raise, since R1233zd(E)
#  cannot condense any hotter. Computed once from water's own saturation curve (valid over a
#  vastly wider range than the refrigerant's), so the dashboard can compare a live LPS
#  pressure against a fixed threshold instead of calling _cycle_points() at a pressure that
#  may already be past where it's defined -- a large enough demand-drop transient can reach
#  that in well under one control interval now that header_closures.py's LPS relief no
#  longer caps it.
P_LPS_CRIT_BAR = PropsSI('P', 'T', T_CRIT_REF - DT_SH_LPS - DT_PINCH_COND + 273.15,
                         'Q', 1, 'Water') / 1e5

TURNDOWN = 0.30

# ----------------------------------------------------------------------
# Flue-gas-side evaporator tap. Fixed-outlet-spec duty calculation, same
# convention as flue_gas_water_hx.py: the gas inlet here is that exchanger's own
# fixed outlet spec (90 degC, imported not retyped), and this machine cools it to
# a further fixed floor. Both ends are constants, so the duty available from the
# gas is a plain number computed once from boiler.h_gas() -- no polynomial fit
# needed (unlike flue_gas_water_hx.py, which needs one because ITS gas inlet is a
# live, boiler-firing-rate-dependent variable; this module's two endpoints are
# both fixed by design).
# ----------------------------------------------------------------------

T_FG_IN_HP = hx.T_FG_OUT_SPEC          # degC, fixed -- the existing HX's own outlet spec

# PLACEHOLDER: conservative stack floor, kept above the likely acid/moisture
# dewpoint for a 40%-moisture wood fuel's flue gas so the recovered heat stays
# genuine sensible heat -- this model does not simulate condensation (see module
# docstring). Revisit once real dewpoint/corrosion data exists.
T_FG_OUT_HP_SPEC = 55.0                # degC

DH_GAS_HP = blr.h_gas(T_FG_IN_HP) - blr.h_gas(T_FG_OUT_HP_SPEC)   # kJ/kg, fixed

# ----------------------------------------------------------------------
# The primary's own evaporating temperature is now a fixed design constant rather
# than derived from one external source's temperature (there are two sources, at
# two different temperatures). It's pinched off the flue gas's COLD end
# (T_FG_OUT_HP_SPEC), not its inlet: a single-phase gas being cooled across a
# glide against a refrigerant boiling at a roughly constant temperature is
# pinched at the cold end, the same reasoning flue_gas_water_hx.py's own
# dT_hot/dT_cold pair checks both ends of its own duty for.
# ----------------------------------------------------------------------

T_EVAP_PRIMARY = T_FG_OUT_HP_SPEC - DT_PINCH_EVAP    # degC, 45.0 at the defaults above


# ----------------------------------------------------------------------
# Dynamic parameters.
#
# THETA_DUTY/TAU_DUTY: identified compressor step response, Qv et al. (2024) field
# water-source heat pump system -- e^(-10s)/(1+20s). Cross-checked (printed only, not
# used) against a charge-based refrigerant loop circulation time,
# REF_CHARGE_KG_PER_KW*ss0['Q_cond']/ss0['m_ref'], which comes out close to THETA_DUTY.
#
# THETA_LPS/THETA_HW/TAU_LPS/TAU_HW (the water/steam-side channels) are derived further
# down, once the design point (ss0) exists.
# ----------------------------------------------------------------------

THETA_DUTY = dp.THETA_COMP_S   # s, Qv et al. 2024
TAU_DUTY = dp.TAU_COMP_S       # s, Qv et al. 2024
N_LAG = 5

#  tau = 20 s is equivalent to a ~60 s full-range ramp (T/3); still not wired in as a
#  rate limit (see the dynamics change notes, section 11 -- out of scope here).
RAMP = 1.0             # fraction of MCR per minute


# ----------------------------------------------------------------------
# Suction (evaporator outlet/compressor inlet) is genuinely fixed --
# T_EVAP_PRIMARY is pinned to the flue gas cold end, not to the LPS header,
# so it never moves with P_LPS.
# ----------------------------------------------------------------------

def _suction_fixed():
    p_evap = PropsSI("P", "T", T_EVAP_PRIMARY+273.15, "Q", 1, REF)
    T1 = T_EVAP_PRIMARY + DT_SUPERHEAT
    h1 = PropsSI("H", "P", p_evap, "T", T1+273.15, REF)/1e3
    s1 = PropsSI("S", "P", p_evap, "T", T1+273.15, REF)/1e3
    return h1, s1


H1, S1 = _suction_fixed()


# ----------------------------------------------------------------------
# Condensing-side cycle points, as a function of the live LPS header
# pressure. Everything here moves together once the header pressure moves:
# it sets the pressure LPS steam is raised at directly, and, through the
# fixed pinches downstream of it, the condensing temperature/pressure (and
# so the compressor's discharge enthalpy and COP) the rest of the cycle
# runs at -- this machine's condenser has to discharge at whatever pressure
# the shared header, now also fed by turbine 2, actually sits at.
# ----------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cycle_points(P_LPS_bar):
    """Condensing-side cycle points for a TWO-STAGE machine with a flash economiser.

    Single-stage compression from 45 degC to a ~143 degC condenser is a pressure
    ratio near 10, at a reduced condensing temperature of 0.95 where R1233zd(E)'s
    latent heat has collapsed to about 92 kJ/kg. Two things go wrong there: the
    compressor is well outside single-stage territory, and throttling the liquid
    all the way back to evaporator pressure flashes over half of it straight to
    vapour, which is pure loss. Splitting the compression at the geometric-mean
    pressure fixes both -- two ratios of about 3.1 each, and the flash gas is
    caught at the intermediate pressure and fed to the upper stage instead of
    being re-compressed from the bottom. COP goes from roughly 1.8 to 2.3 with
    no change to the sources, the sink, or the plumbing.

    Mass split. The economiser vessel receives liquid from the condenser and the
    LP stage's discharge, and delivers saturated vapour to the HP stage and
    saturated liquid to the evaporator. An energy balance on it gives
    f = m_LP/m_HP = (hg_i - h3)/(h2a - hf_i), which is what makes the cycle
    close exactly: f*(h1 - hf_i) + f*(h2a - h1) + (H2 - hg_i) == H2 - h3.

    Everything downstream still works per kilogram of CONDENSER flow, so m_ref
    keeps its old meaning and only the specific work changes -- DW replaces the
    old (H2 - H1).

    Subcooling changes h3, and h3 sets the economiser split f, so the specific work
    depends on the LIVE subcooling. It does so linearly, because h3 = H_F - dh_sub:

        DW(dh_sub) = A + K*dh_sub
        K = (h2a - H1)/(h2a - hf_i)
        A = (H2 - hg_i) + K*(hg_i - H_F)

    (an earlier version froze h3 at DT_SUB_FIXED, which understated the work by
    8-25% once subcooling was allowed to move above 10 K). `DW` below is that same
    expression at DT_SUB_FIXED, kept for the design point and sizing; the live
    machine uses A and K with the dh_sub it is actually running at.
    """
    T_sat = PropsSI("T", "P", P_LPS_bar*1e5, "Q", 1, "water") - 273.15
    T_lps_out = T_sat + DT_SH_LPS
    T_cond = T_lps_out + DT_PINCH_COND
    P_cond = PropsSI("P", "T", T_cond+273.15, "Q", 1, REF)
    H_F = PropsSI("H", "P", P_cond, "Q", 0, REF)/1e3
    DH_LPS = (PropsSI("H", "T", T_lps_out+273.15, "P", P_LPS_bar*1e5, "water")
              - PropsSI("H", "T", T_LPS_FEED+273.15, "P", P_LPS_bar*1e5, "water"))/1e3

    P_evap = PropsSI("P", "T", T_EVAP_PRIMARY+273.15, "Q", 1, REF)
    P_int = (P_evap*P_cond)**0.5

    h2a = H1 + (PropsSI("H", "P", P_int, "S", S1*1e3, REF)/1e3 - H1)/ETA_IS
    hf_i = PropsSI("H", "P", P_int, "Q", 0, REF)/1e3
    hg_i = PropsSI("H", "P", P_int, "Q", 1, REF)/1e3
    s_i = PropsSI("S", "P", P_int, "Q", 1, REF)/1e3
    H2 = hg_i + (PropsSI("H", "P", P_cond, "S", s_i*1e3, REF)/1e3 - hg_i)/ETA_IS

    h3_fixed = PropsSI("H", "T", T_cond-DT_SUB_FIXED+273.15, "P", P_cond, REF)/1e3
    f_lp = (hg_i - h3_fixed)/(h2a - hf_i)
    DW = (H2 - hg_i) + f_lp*(h2a - H1)
    K = (h2a - H1)/(h2a - hf_i)
    A = (H2 - hg_i) + K*(hg_i - H_F)

    return dict(T_sat=T_sat, T_lps_out=T_lps_out, T_cond=T_cond, P_cond=P_cond,
                H_F=H_F, DH_LPS=DH_LPS, H2=H2, DW=DW, DW_A=A, DW_K=K,
                P_int=P_int, f_lp=f_lp, PR_lo=P_int/P_evap, PR_hi=P_cond/P_int)


@lru_cache(maxsize=None)
def dh_subcool(dt_sub, P_LPS_bar):
    """Enthalpy released by subcooling the liquid, which is the hot water
    duty per unit refrigerant. Depends on the header pressure too now, since
    the condensing point it subcools from moves with it. Fitted as a
    difference rather than as h3, because h3 is a large number and the
    difference taken from it is small, so fitting h3 loses most of the
    significant figures."""
    cp = _cycle_points(P_LPS_bar)
    return cp['H_F'] - PropsSI("H", "T", cp['T_cond']-dt_sub+273.15,
                               "P", cp['P_cond'], REF)/1e3


#  Design point -- where the module's own standalone report/demo and ss0
#  (the dynamic model's seed) are evaluated. Not the only pressure the live
#  model can run at any more (see P_LPS_RANGE above); just a representative
#  one, matching plant_rig.py's own P_LPS_0 default.
_design = _cycle_points(P_LPS_DESIGN)
T_SAT_LPS, T_LPS_OUT, T_COND, P_COND = (
    _design['T_sat'], _design['T_lps_out'], _design['T_cond'], _design['P_cond'])
H_F, DH_LPS, H2 = _design['H_F'], _design['DH_LPS'], _design['H2']

@lru_cache(maxsize=None)
def dh_hw(T_hw_C):
    """Enthalpy rise of the hot water stream, makeup to delivery, kJ/kg.

    This is what converts the subcooling duty into a hot water FLOW, so the
    delivery temperature has to be live rather than frozen at T_HW: for a
    fixed Q_sub, asking for hotter water must give less of it. It used to be
    a module constant evaluated once at T_HW, which left m_hw completely
    unresponsive to the plant's own hot water setpoint -- the flue gas
    exchanger's flow moved with the slider and the heat pump's did not.

    The INLET is the site recycle, hx.T_W_IN -- not T_MU. Both this subcool
    zone and the flue gas exchanger deliver into the same hot water header,
    so they have to be heating the same water from the same place. Taking
    this one from 20 degC makeup while the exchanger took its own from the
    recycle spread one duty over a 65 K rise and the other over 25 K, which
    made the heat pump's contribution look about 2.5x smaller than it is.
    T_MU stays where it belongs, on DH_LPS: the LPS side really is raised
    from cold condensate, and that is a different stream.
    """
    return (PropsSI("H", "T", T_hw_C+273.15, "P", P_HW*1e5, "water")
            - PropsSI("H", "T", hx.T_W_IN+273.15, "P", P_HW*1e5, "water"))/1e3


DH_HW = dh_hw(T_HW)        # design-point value, for sizing and standalone runs

#  Setpoint range the live fit below has to cover. Imported from the exchanger
#  rather than retyped, so the heat pump's hot water and the exchanger's hot
#  water are quoted against one and the same site setpoint -- they feed the
#  same header, so a second, silently different range here would let the two
#  halves of that header disagree about what "hot water" means.
T_HW_RANGE = (hx.T_W_OUT_LO, hx.T_W_OUT_HI)


# ----------------------------------------------------------------------
# Fit dh_subcool over (dt_sub, P_LPS), and DH_LPS/H2 over P_LPS alone.
# dt_sub and P_LPS are both live inputs now -- dt_sub controls the LPS/hot
# water split, P_LPS is the shared header pressure turbine 2 also discharges
# into. dh_subcool needs both dimensions since the point it subcools from
# moves with P_LPS; DH_LPS and H2 depend on P_LPS alone. A 2-D bspline
# surface handles dh_subcool, since a hand-rolled tensor-product polynomial
# would need coefficients that are themselves polynomials of P_LPS --
# turbine_model.py already uses the same m.bspline technique for its own
# 2-D (P2, h) -> T_out fit.
# ----------------------------------------------------------------------

DEG_SUBCOOL = 5
DEG_PLPS = 5

#  Upper limit raised 25 -> 45 K. Physical basis: the subcooling zone cools the
#  condensed liquid from T_cond by dt while heating site recycle water (60 degC)
#  toward the hot water setpoint, so it must leave above the recycle inlet plus
#  a pinch. T_cond is 130 degC at 2 bar and 154 degC at 4 bar, so at 45 K the
#  liquid leaves at >= 85 degC: >= 25 K above the recycle water at every header
#  pressure (hard limit with a 10 K pinch is about 60 K at 2 bar, 73 K at 3 bar).
#  dh_subcool is a direct property evaluation, so the table only needs the wider
#  grid; spacing is kept at about 0.57 K, as before.
DT_SUB_RANGE = (2.0, 45.0)

_dtsub_grid = np.linspace(*DT_SUB_RANGE, 76)
_plps_grid = np.linspace(*P_LPS_RANGE, 21)

_dh_grid = np.array([[dh_subcool(dt, p) for p in _plps_grid] for dt in _dtsub_grid])
#  SUBCOOLING ENTHALPY IN THE DYNAMIC MODEL: a 2-D polynomial, not m.bspline
#  (2026-09-25). The dynamic model used m.bspline(dt_sub, P, dh_sub, grid, sf=0), but
#  GEKKO's executable does not build an interpolating spline from that -- it keeps only
#  the boundary knots, i.e. ONE bicubic patch over the whole 2-45 K x 2-4 bar grid --
#  and it came out 0.215% LOW at every operating point (17.415 vs 17.452 kJ/kg at
#  10 K / 3 bar). Found as a 0.17% heat-pump hot-water gap (and ~2 kW of compressor
#  work) between the settled running plant and the steady-state closure
#  (excel_verify/python_side.plant_steady, which uses dh_subcool exactly). Handing
#  GEKKO a scipy-built spline instead (bspline(data=False)) evaluates to 0 in this
#  GEKKO version, so: dh_sub = dt_sub * g(dt_sub, P), with g = dh/dt (the mean liquid
#  heat capacity, smooth) fitted as a polynomial of degree 6 in dt_sub and 7 in P on
#  the normalised grid. Error against dh_subcool: 0.003% at 10 K / 3 bar, <= 0.03% up
#  to ~3.5 bar, 0.046% at the worst corner (3.98 bar, 4 K -- the refrigerant's
#  properties steepen towards 4 bar as the condensing point nears critical).
_DH_DEG = (6, 7)
_dtn = lambda dt: (dt - DT_SUB_RANGE[0])/(DT_SUB_RANGE[1] - DT_SUB_RANGE[0])
_pn = lambda p: (p - P_LPS_RANGE[0])/(P_LPS_RANGE[1] - P_LPS_RANGE[0])
_XX, _YY = np.meshgrid(_dtn(_dtsub_grid), _pn(_plps_grid), indexing='ij')
_A = np.stack([_XX.ravel()**i * _YY.ravel()**j for i in range(_DH_DEG[0] + 1)
               for j in range(_DH_DEG[1] + 1)], axis=1)
_DH_POLY = np.linalg.lstsq(_A, (_dh_grid/_dtsub_grid[:, None]).ravel(), rcond=None)[0] \
    .reshape(_DH_DEG[0] + 1, _DH_DEG[1] + 1)          # _DH_POLY[i, j] * x**i * y**j


def dh_sub_fit(dt_sub, P_LPS_bar):
    """dh_subcool as the dynamic model evaluates it (see _DH_POLY). Nested Horner, so it
    takes plain floats or GEKKO expressions alike."""
    x, y = _dtn(dt_sub), _pn(P_LPS_bar)
    g = 0.0
    for i in range(_DH_DEG[0], -1, -1):
        row = 0.0
        for j in range(_DH_DEG[1], -1, -1):
            row = row*y + _DH_POLY[i, j]
        g = g*x + row
    return dt_sub*g

_plps_y = (_plps_grid - P_LPS_RANGE[0])/(P_LPS_RANGE[1] - P_LPS_RANGE[0])
_plps_cp = [_cycle_points(p) for p in _plps_grid]
_DH_LPS_vals = np.array([cp['DH_LPS'] for cp in _plps_cp])
_H2_vals = np.array([cp['H2'] for cp in _plps_cp])
_HF_vals = np.array([cp['H_F'] for cp in _plps_cp])
_DW_A_vals = np.array([cp['DW_A'] for cp in _plps_cp])
_DW_K_vals = np.array([cp['DW_K'] for cp in _plps_cp])
C_DH_LPS = np.polyfit(_plps_y, _DH_LPS_vals, DEG_PLPS)
C_H2 = np.polyfit(_plps_y, _H2_vals, DEG_PLPS)
C_HF = np.polyfit(_plps_y, _HF_vals, DEG_PLPS)
#  Compressor work = A(P) + K(P)*dh_sub: two P_LPS fits, evaluated with the LIVE dh_sub.
C_DW_A = np.polyfit(_plps_y, _DW_A_vals, DEG_PLPS)
C_DW_K = np.polyfit(_plps_y, _DW_K_vals, DEG_PLPS)

#  Hot water enthalpy rise against the live delivery setpoint. Liquid water's
#  h(T) is very nearly linear over this span, so a cubic is far more than
#  enough -- same normalise-then-Horner treatment as the P_LPS fits above.
DEG_THW = 3
_thw_grid = np.linspace(*T_HW_RANGE, 21)
_thw_y = (_thw_grid - T_HW_RANGE[0])/(T_HW_RANGE[1] - T_HW_RANGE[0])
_DH_HW_vals = np.array([dh_hw(T) for T in _thw_grid])
C_DH_HW = np.polyfit(_thw_y, _DH_HW_vals, DEG_THW)

FIT = {
    "DH_LPS(P_LPS)": np.max(np.abs(np.polyval(C_DH_LPS, _plps_y)-_DH_LPS_vals)
                            /_DH_LPS_vals)*100,
    "H2(P_LPS)": np.max(np.abs(np.polyval(C_H2, _plps_y)-_H2_vals)/_H2_vals)*100,
    "H_F(P_LPS)": np.max(np.abs(np.polyval(C_HF, _plps_y)-_HF_vals)/_HF_vals)*100,
    "DW_A(P_LPS)": np.max(np.abs(np.polyval(C_DW_A, _plps_y)-_DW_A_vals)/np.abs(_DW_A_vals))*100,
    "DW_K(P_LPS)": np.max(np.abs(np.polyval(C_DW_K, _plps_y)-_DW_K_vals)/_DW_K_vals)*100,
    "DH_HW(T_hw)": np.max(np.abs(np.polyval(C_DH_HW, _thw_y)-_DH_HW_vals)
                          /_DH_HW_vals)*100,
}


def horner(m, coef, u):
    """Polynomial as nested multiplication, which keeps the expression graph
    shallow and avoids high powers of the normalised input."""
    e = coef[0]
    for c in coef[1:]:
        e = m.Intermediate(e*u + c)
    return e


# ----------------------------------------------------------------------
# Design point, used to size Q_MCR off what the flue gas alone can actually
# supply -- the same idea flue_gas_water_hx.py uses to size its own M_WATER_0
# off its design-point duty, rather than an arbitrary round number. Sized off
# the gas alone (not the secondary's contribution) so nameplate capacity
# reflects the free/waste heat source, with the secondary as bonus capacity
# above that -- matching the two machines' independent-dispatch relationship.
# ----------------------------------------------------------------------

Q_EVAP_GAS_DES = blr.MDOT_GAS_DES * DH_GAS_HP     # kW

DT_SUB_0 = DT_SUB_FIXED
_H3_DES = H_F - dh_subcool(DT_SUB_0, P_LPS_DESIGN)

#  Evaporator duty as a fraction of condenser duty, from the energy balance
#  rather than from the old single-stage enthalpy difference: the two-stage
#  machine's evaporator flow is not its condenser flow, so (H1 - h3)/(H2 - h3)
#  no longer describes it. Q_evap = Q_cond - W, so the ratio is 1 - DW/(H2-h3).
_RATIO_EVAP_TO_COND = 1.0 - _design['DW']/(H2 - _H3_DES)

Q_MCR = Q_EVAP_GAS_DES / _RATIO_EVAP_TO_COND      # kW, condenser duty at rated capacity


#  Recycle tap. The site return water is a second low-grade source for this
#  evaporator, alongside the flue gas. It arrives at hx.T_W_IN and can only be
#  cooled to the evaporator's own pinch limit, so on a 60 degC return against a
#  45 degC evaporator there are just 5 K to take -- about 1.1 MW at design flow.
#  Small next to the gas, but free, and it is heat that would otherwise have
#  gone to site as hot water the site did not ask for.
#
#  Raising T_EVAP_PRIMARY closes this tap before it closes the gas one: the
#  recycle dies at a 50 degC evaporator, the gas not until 80 degC.
T_RECYCLE_EVAP_OUT = T_EVAP_PRIMARY + DT_PINCH_EVAP       # degC


def q_evap_recycle(m_water):
    """Evaporator duty available from the recycle tap, kW, for m_water kg/s."""
    dT = hx.T_W_IN - T_RECYCLE_EVAP_OUT
    return max(m_water * hx.CP_WATER * dT, 0.0)


def q_evap_total(mdot_fg, m_water, Q_evap_secondary):
    """All three low-grade sources into this evaporator, kW."""
    return (mdot_fg * DH_GAS_HP + q_evap_recycle(m_water) + Q_evap_secondary)


@lru_cache(maxsize=4096)
def _cycle_constants_cached(P_LPS_bar, dt_sub):
    return _cycle_constants(P_LPS_bar, dt_sub)


def cycle_constants(P_LPS_bar, dt_sub=DT_SUB_FIXED):
    """Cached wrapper. The feedforward evaluates these inside a Newton loop, a
    dozen times a sample, and each raw call is several PropsSI lookups. Rounding
    the key turns almost all of them into cache hits.

    Quantised at 1e-4 bar, not 1e-3. The ratio moves about 0.06 per bar, so a
    0.0005 bar rounding error put 3e-5 between the duty the feedforward asked for
    and the duty the cycle actually delivers -- invisible on any plot, but it
    showed up as a conservation residual that would not go to zero, and a
    conservation check that cannot reach zero is a check nobody trusts. At 1e-4
    it is under 3e-6 and the cache still hits on every slider value and on every
    pass of the Newton, which holds the pressure fixed anyway."""
    return _cycle_constants_cached(round(float(P_LPS_bar), 4),
                                   round(float(dt_sub), 4))


def _cycle_constants(P_LPS_bar, dt_sub=DT_SUB_FIXED):
    """Per-kilowatt-of-condenser-duty factors at a given header pressure.

    Everything a feedforward calculation needs to turn evaporator heat into
    delivered LPS and hot water, evaluated once so the arithmetic that follows
    is pure. All four move with P_LPS, which is exactly why nothing may assume
    the design value.

        ratio      Q_evap/Q_cond
        sub_frac   Q_sub/Q_cond, the share that leaves as hot water
        dh_lps     kJ per kg of LPS raised
    """
    cp = _cycle_points(P_LPS_bar)
    dh = dh_subcool(dt_sub, P_LPS_bar)
    h3 = cp['H_F'] - dh
    span = cp['H2'] - h3
    dw = cp['DW_A'] + cp['DW_K']*dh          # specific work at THIS subcooling
    return dict(ratio=1.0 - dw/span, sub_frac=dh/span,
                dh_lps=cp['DH_LPS'], span=span, dw=dw)


def q_cond_from_sources(Q_evap_available, P_LPS_bar=None, dt_sub=DT_SUB_FIXED):
    """Condenser duty this machine delivers given the evaporator heat it is fed.

    The primary is DEPENDENT now: nothing commands its duty. It takes whatever
    low-grade heat arrives -- flue gas, the recycle tap, and the secondary's
    condenser -- and delivers that plus the compressor work it had to spend to
    lift it. The secondary's duty is the manipulated variable; this is the
    function that turns it into a primary output.

    One consequence worth keeping in view: the flue gas is always there when
    the boiler is firing and there is no bypass, so Q_evap_available has a
    floor and therefore so does this. The primary cannot be turned down below
    what the gas alone forces through it.
    """
    if P_LPS_bar is None:
        return Q_evap_available / _RATIO_EVAP_TO_COND
    return Q_evap_available / cycle_constants(P_LPS_bar, dt_sub)['ratio']

#  The COMMANDABLE range is deliberately far wider than Q_MCR. Q_MCR answers
#  "how big a machine can the flue gas run unaided" -- it is a sizing result,
#  not a physical ceiling, because the secondary can supply evaporator heat
#  too and its own source is unlimited. Capping the slider at Q_MCR therefore
#  hid the whole interesting region: the operating points where the commanded
#  duty needs more low-grade heat than the gas plus the secondary can deliver.
#  Those points are now reachable and are FLAGGED (dQ_shortfall > 0) rather
#  than forbidden -- the same expression a controller will later carry as a
#  feasibility constraint.
#
#  NAMEPLATE, not a slider. Nothing commands this machine any more -- see
#  q_cond_from_sources above. Q_PRI_HI is sized to clear what the secondary's
#  own ceiling can drive it to (secondary at Q_SEC_MCR, plus the gas and the
#  recycle tap at design firing) with margin, so the clamp never binds inside
#  the reachable envelope and the rig only ever flags it if something upstream
#  has gone wrong. Q_PRI_LO is what the gas alone forces through it.
Q_PRI_HI = 55000.0                                # kW, nameplate ceiling
Q_PRI_LO = 0.0                                    # kW, no commanded floor
Q_PRI_0 = q_cond_from_sources(Q_EVAP_GAS_DES)


def steady_state(Q, mdot_fg, Q_evap_secondary, dt_sub, P_LPS_bar, T_hw_C=T_HW):
    """Reference steady state, straight from CoolProp / the live cycle points.

    T_hw_C is the hot water DELIVERY setpoint. It sets how much flow the
    subcooling duty turns into (hotter water, less of it) -- it does not
    change the duty itself, which is fixed by dt_sub and the refrigerant flow.
    """
    cp = _cycle_points(P_LPS_bar)
    dh = dh_subcool(dt_sub, P_LPS_bar)
    h3 = cp['H_F'] - dh
    m_ref = Q/(cp['H2']-h3)            # CONDENSER-side (HP stage) flow
    W = m_ref*(cp['DW_A'] + cp['DW_K']*dh)   # both stages, per kg of condenser flow, LIVE subcooling
    Q_evap_demand = Q - W
    Q_evap_gas_avail = mdot_fg * DH_GAS_HP
    Q_evap_available = Q_evap_gas_avail + Q_evap_secondary
    return dict(Q_cond=Q, m_ref=m_ref, W=W, Q_evap_demand=Q_evap_demand,
                Q_evap=Q_evap_demand,
                Q_sub=m_ref*dh, dh_sub=dh, m_hw=m_ref*dh/dh_hw(T_hw_C),
                m_lps=(Q - m_ref*dh)/cp['DH_LPS'], COP=Q/W,
                Q_evap_gas_avail=Q_evap_gas_avail,
                dQ_shortfall=Q_evap_demand - Q_evap_available)


ss0 = steady_state(Q_PRI_0, blr.MDOT_GAS_DES, 0.0, DT_SUB_0, P_LPS_DESIGN)


# ----------------------------------------------------------------------
# TAU_LPS / TAU_HW via Tau = V/Vdot, volume from a design-point duty ->
# heat-transfer-area estimate rather than a bare guessed volume.
#
# Design duty is split the same way the cycle itself splits it: everything
# above the subcooling section (desuperheat + condensing) raises the LPS
# steam, subcooling alone raises the hot water.
#
# LMTD for each section comes straight from the fixed cycle temperatures
# above -- no new CoolProp calls needed. The LPS section treats the
# refrigerant side as isothermal at T_COND (condensing dominates; the same
# simplification this module already applies to the gas-side evaporator
# pinch), water/steam heated from T_MU to T_LPS_OUT. The HW section is the
# refrigerant's own subcooling glide (T_COND down to T_COND-DT_SUB_0)
# against water heated from T_MU to T_HW.
#
# The area -> volume step no longer goes through a guessed holdup depth. It
# uses an area density beta = A/V instead, the same recipe
# flue_gas_water_hx.py now uses:
#
#     V = A / beta            beta ~ 400 m2/m3 for both condenser zones
#
# 400 m2/m3 is the conventional threshold above which a liquid-side heat
# exchanger is classed as COMPACT, and it is the right end of the range for
# these two zones: both are liquid/condensing service on both sides, with no
# gas film to carry, so the surface can be packed far more densely than the
# finned flue-gas economiser (300 m2/m3) and there is no reason to size them
# as a loose shell-and-tube bundle. A depth assumption was doing no physical
# work here -- it was a guessed number dressed as a geometry.
#
# U_LPS_HP/U_HW_HP remain equipment assumptions: typical shell-and-tube
# condensing and liquid-subcooling coefficients. Revisit once real equipment
# data exists.
# ----------------------------------------------------------------------

U_LPS_HP = 1.0          # kW/m2K, condensing/steam-raising duty
U_HW_HP = 0.6           # kW/m2K, liquid subcooling duty
BETA_HP = 400.0         # m2/m3, compact liquid-HX area density (both zones)


def _lmtd(dT1, dT2):
    dT1, dT2 = abs(dT1), abs(dT2)
    if abs(dT1 - dT2) < 1e-9:
        return dT1
    return (dT1 - dT2) / np.log(dT1 / dT2)


Q_LPS_DES = ss0['Q_cond'] - ss0['Q_sub']       # kW, desuperheat + condensing
Q_HW_DES = ss0['Q_sub']                        # kW, subcooling

LMTD_LPS = _lmtd(T_COND - T_LPS_OUT, T_COND - T_MU)
LMTD_HW = _lmtd(T_COND - T_HW, (T_COND - DT_SUB_0) - T_MU)

A_LPS = Q_LPS_DES / (U_LPS_HP * LMTD_LPS)      # m2
A_HW = Q_HW_DES / (U_HW_HP * LMTD_HW)          # m2

V_LPS = A_LPS / BETA_HP                        # m3
V_HW = A_HW / BETA_HP                          # m3

RHO_HW_HP = PropsSI("D", "T", T_HW+273.15, "P", P_HW*1e5, "water")          # kg/m3
VDOT_HW_DES = ss0['m_hw'] / RHO_HW_HP          # m3/s

#  Shell/tube metal thermal mass -- no real equipment drawing exists (theoretical plant),
#  so this is a first-pass estimate in the same style as boiler.py's tube-metal figure:
#  tau_metal = t_wall*rho.c/U, reusing the U_LPS_HP/U_HW_HP already defined above and a
#  2 mm stainless tube wall -- thin, because BETA_HP=400 m2/m3 already commits this to
#  compact tube construction, not a heavy shell. rho, c: 8000 kg/m3, 500 J/kg.K, typical
#  stainless steel.
T_WALL_HP = 0.002                              # m, compact tube wall
RHO_METAL_HP, C_METAL_HP = 8000.0, 500.0       # kg/m3, J/kg.K, stainless steel
TAU_LPS_METAL = T_WALL_HP*RHO_METAL_HP*C_METAL_HP/(U_LPS_HP*1e3)   # s (U_LPS_HP is kW/m2K)
TAU_HW_METAL = T_WALL_HP*RHO_METAL_HP*C_METAL_HP/(U_HW_HP*1e3)     # s

#  LPS channel. NOT the old steam-side V_LPS/VDOT_LPS_DES term (vapour density ~1.63
#  kg/m3, giving 0.26 s) -- boiling water at constant header pressure is isothermal and
#  follows duty directly, so that residence time was never a real store. What actually
#  lags here is the shell/tube metal above, plus the refrigerant's own vapour inventory
#  adjusting to a header pressure change: dM/dt = V*(drho/dP)*(dP/dt), and dP/dt relates
#  to duty through the saturation line's dT/dP and the latent heat h_fg.
_dP = 2000.0    # Pa, +/- 2 kPa central difference
P_c = PropsSI('P', 'T', T_COND+273.15, 'Q', 1, REF)
drho_dP = (PropsSI('D', 'P', P_c+_dP, 'Q', 1, REF)
          - PropsSI('D', 'P', P_c-_dP, 'Q', 1, REF)) / (2*_dP)        # kg/m3 per Pa
dT_dP = (PropsSI('T', 'P', P_c+_dP, 'Q', 1, REF)
        - PropsSI('T', 'P', P_c-_dP, 'Q', 1, REF)) / (2*_dP)          # K per Pa
h_fg = PropsSI('H', 'P', P_c, 'Q', 1, REF) - PropsSI('H', 'P', P_c, 'Q', 0, REF)   # J/kg

V_vap = 0.5 * V_LPS                            # m3, refrigerant side, half the zone volume
TAU_LPS_REF = V_vap*drho_dP*h_fg / (U_LPS_HP*1e3*A_LPS*dT_dP)   # s (U_LPS_HP is kW/m2K)
TAU_LPS = TAU_LPS_METAL + TAU_LPS_REF           # s
THETA_LPS = dp.L_LPS_PIPE_M/dp.V_LPS_STEAM_MPS + dp.THETA_VALVE_S   # s, pipe + valve dead time

#  HW channel. Plug-flow water side (T_RES_HW, half the zone volume, Smith-reduced into
#  a lag + dead time -- same reasoning flue_gas_water_hx.py applies to its own tube
#  bank), plus the shell/tube metal above.
T_RES_HW = 0.5 * V_HW / VDOT_HW_DES             # s, water side, half the zone volume
TAU_HW = dp.PLUG_TAU_FRAC*T_RES_HW + TAU_HW_METAL   # s
THETA_HW = (dp.PLUG_THETA_FRAC*T_RES_HW + dp.L_HW_RUN_M/dp.V_HW_PIPE_MPS
           + dp.THETA_VALVE_S)                  # s, water residence + delivery pipe + valve


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def build_dynamics(m, mdot_fg, Q_evap_secondary, Q_pri_cmd, dt_sub, P_LPS_bar,
                   T_hw_cmd=None, y0=None, driven=True):
    """Attach the primary heat pump to a GEKKO model.

    mdot_fg           Live GEKKO variable -- the boiler's own flue gas flow (the
                       same one already passed into flue_gas_water_hx.py), used
                       here only for the reported gas-side available duty.
    Q_evap_secondary  Live GEKKO variable -- heatpump_secondary.py's condenser
                       duty, handed straight in by the integration layer.
    Q_pri_cmd         GEKKO Param/Var/MV, the condenser duty commanded.
    dt_sub            GEKKO Param/Var/MV, subcooling -- controls the LPS/hot
                       water split.
    P_LPS_bar         Live GEKKO variable, bar -- the shared LPS header
                       pressure (plant_rig.py's own plps_act), the same one
                       turbine 2 discharges into. Sets the condensing side of
                       this machine's own cycle -- see _cycle_points.
    T_hw_cmd          GEKKO Param/Var/MV, degC -- the site's hot water delivery
                       setpoint, the SAME one the flue gas exchanger is given
                       (both feed the one hot water header). Sets how much
                       flow this machine's subcooling duty becomes: hotter
                       water, less of it. Pass None to freeze it at the design
                       T_HW, which is what the standalone run below does.
    y0                Reference steady_state() dict for seeding; defaults to ss0.

    Returns m_lps, m_hw, W_comp, Q_cond, COP, Q_evap_gas_avail, dQ_shortfall and
    the states needing reseeding after a failed solve -- split out per channel
    (duty_states/lps_states/hw_states) rather than one combined list, since
    each channel's own state count depends on its own theta/tau (0 or more
    states), so a caller cannot safely assume a fixed split by position.
    """
    y0 = ss0 if y0 is None else y0

    #  Clamped, not the raw header pressure: this machine's own cycle fit
    #  (P_LPS_RANGE, above) only covers what R1233zd(E) can actually condense
    #  against. Turbine 2's own back pressure can range further than that
    #  (see plant_rig.py) since it shares this header -- a real machine
    #  pushed against more backpressure than it can condense against would
    #  just fail to add flow past that point, which this clamp represents.
    #  Smooth, not m.min3/max3 (an MPCC slack formulation) -- consistent
    #  with this project's own documented aversion to those, and the same
    #  smooth-clamp technique iapws95_gekko.py already uses. A real Var, not
    #  an Intermediate -- m.bspline below needs one on the input side too
    #  (same requirement it has for the output, dh_sub).
    P_LPS_hp = m.Var(value=P_LPS_DESIGN, name='hpp_plps_clamped')
    m.Equation(P_LPS_hp == smooth_max(
        m, smooth_min(m, P_LPS_bar, P_LPS_RANGE[1], 0.05), P_LPS_RANGE[0], 0.05))
    plps_y = m.Intermediate((P_LPS_hp - P_LPS_RANGE[0])
                            /(P_LPS_RANGE[1] - P_LPS_RANGE[0]))
    DH_LPS_live = horner(m, C_DH_LPS, plps_y)
    H2_live = horner(m, C_H2, plps_y)
    HF_live = horner(m, C_HF, plps_y)
    DW_A_live = horner(m, C_DW_A, plps_y)
    DW_K_live = horner(m, C_DW_K, plps_y)

    #  Polynomial, not m.bspline -- see _DH_POLY for why (0.215% bias from the spline).
    dh_sub = m.Intermediate(dh_sub_fit(dt_sub, P_LPS_hp), name='hpp_dhsub')
    h3 = m.Intermediate(HF_live - dh_sub)
    #  Specific compressor work follows the live subcooling (see _cycle_points).
    DW_live = m.Intermediate(DW_A_live + DW_K_live*dh_sub)

    #  DRIVEN MODE. Q_pri_cmd is the low-grade heat ARRIVING at the evaporator,
    #  not a commanded condenser duty, and the duty that results is worked out
    #  here rather than by the caller. That placement is the whole point: the
    #  conversion is Q_cond = Q_evap/(1 - DW/(H2 - h3)), and all three of those
    #  move with the LPS header pressure. Doing it outside, against the module
    #  constant _RATIO_EVAP_TO_COND, silently pins the machine to P_LPS_DESIGN
    #  -- correct at 3 bar and wrong everywhere else, which shows up as a
    #  residual that never settles and a duty that jumps the wrong way when the
    #  header pressure moves. Everything the primary does is a function of the
    #  live header pressure, so its source-to-duty conversion has to be too.
    #
    #  driven=False restores the old behaviour, where Q_pri_cmd is a commanded
    #  condenser duty. Nothing in the plant uses it now; it is kept so a test
    #  can still drive this machine directly.
    if driven:
        ratio_live = m.Intermediate(1.0 - DW_live/(H2_live - h3))
        Q_pri_target = m.Intermediate(Q_pri_cmd/ratio_live)
    else:
        Q_pri_target = Q_pri_cmd

    # Duty ramp. The cycle keeps up with this, so the refrigerant side is taken
    # algebraically off the delivered duty and the energy balance closes at
    # every instant rather than only at rest.
    Q_cond, duty_states = lag_chain(m, Q_pri_target, theta=THETA_DUTY, tau=TAU_DUTY,
                                    y0=y0['Q_cond'], n_lag=N_LAG, name='hpp_duty')

    m_ref = m.Intermediate(Q_cond/(H2_live-h3))
    W_comp = m.Intermediate(m_ref*DW_live)
    COP = m.Intermediate(Q_cond/W_comp)
    Q_sub = m.Intermediate(m_ref*dh_sub)
    Q_evap_demand = m.Intermediate(Q_cond - W_comp)

    Q_evap_gas_avail = m.Intermediate(mdot_fg * DH_GAS_HP)
    #  In driven mode this is a pure transient residual: the evaporator demand
    #  chases the supply through TAU_DUTY and matches it exactly at rest, at any
    #  header pressure. Non-zero here means the machine is still settling, not
    #  that it is short of heat.
    dQ_shortfall = m.Intermediate(Q_evap_demand - Q_evap_gas_avail - Q_evap_secondary)

    #  Hot water enthalpy rise, live off the delivery setpoint rather than
    #  frozen at the design T_HW -- this is what makes m_hw actually respond
    #  to the hot water slider (see dh_hw's docstring).
    if T_hw_cmd is None:
        DH_HW_live = DH_HW
    else:
        thw_y = m.Intermediate((T_hw_cmd - T_HW_RANGE[0])
                               /(T_HW_RANGE[1] - T_HW_RANGE[0]))
        DH_HW_live = horner(m, C_DH_HW, thw_y)

    #  TAU_LPS/TAU_HW already carry the shell/tube metal contribution combined in
    #  (TAU_LPS_METAL/TAU_HW_METAL, folded in above) rather than cascaded as a second
    #  state -- see the comment there.
    m_lps, lps_states = lag_chain(m, m.Intermediate((Q_cond - Q_sub)/DH_LPS_live),
                                  theta=THETA_LPS, tau=TAU_LPS, n_lag=N_LAG,
                                  y0=y0['m_lps'], name='hpp_lps')
    m_hw, hw_states = lag_chain(m, m.Intermediate(Q_sub/DH_HW_live),
                                theta=THETA_HW, tau=TAU_HW, n_lag=N_LAG,
                                y0=y0['m_hw'], name='hpp_hw')

    #  Q_evap_demand is returned, not just used internally: it is the number
    #  the shortfall balance is actually built from (Q_cond - W_comp, NOT
    #  Q_cond), and without it on display the dQ_shortfall figure cannot be
    #  checked by eye against the gas and secondary duties beside it.
    return dict(m_lps=m_lps, m_hw=m_hw, W_comp=W_comp, Q_cond=Q_cond, COP=COP,
               Q_evap_demand=Q_evap_demand,
               Q_evap_gas_avail=Q_evap_gas_avail, dQ_shortfall=dQ_shortfall,
               states=duty_states + lps_states + hw_states,
               duty_states=duty_states, lps_states=lps_states, hw_states=hw_states)


# ----------------------------------------------------------------------
# Standalone report / step test
# ----------------------------------------------------------------------

#  T_END gives the last two steps (duty, then LPS header pressure) their own dead time
#  plus 5*tau to settle -- not a fixed run length that would leave the steady-state
#  closure check below reporting a settling-time gap as if it were an error. TAU_HW
#  already carries TAU_HW_METAL combined in (see its definition above), so this does
#  NOT add TAU_HW_METAL again.
T_STEP_Q, T_STEP_QSEC, T_STEP_SUB, T_STEP_PLPS = 300.0, 1400.0, 2500.0, 3600.0
T_END = T_STEP_PLPS + THETA_DUTY + THETA_HW + 5.0*(TAU_DUTY + TAU_HW)
#  DT was 2.0 s before the duty/LPS/HW dead-time chains existed (N_LAG=5 states each,
#  none of them present when theta was 0 for all three). Adding them roughly quadruples
#  the state count over this run's ~1900 points at the old DT, which one all-at-once
#  IMODE=4 solve (this standalone demo, unlike the batched live rig) ran out of memory
#  factoring. 8.0 s keeps this demo's own point count down to a size that solves; the
#  live plant (plant_rig.py) steps in small batches and is unaffected.
DT = 8.0

MDOT_FG_REP = blr.MDOT_GAS_DES        # representative flue gas flow, fixed here
Q_SEC_REP_0, Q_SEC_REP_1 = 0.0, 400.0  # representative secondary contribution

Q_CMD_0, Q_CMD_1 = Q_PRI_0, Q_MCR
DT_SUB_STEP_0, DT_SUB_STEP_1 = 10.0, 18.0
#  A drop in the shared LPS header pressure -- e.g. turbine 2 pulled down
#  toward the true low LPS pressure -- rather than a step up, since that's
#  the direction that used to be off limits (see plant_rig.py's own
#  docstring on why P_LPS's range was kept high).
PLPS_STEP_0, PLPS_STEP_1 = P_LPS_DESIGN, P_LPS_RANGE[0]


def main():
    m = GEKKO(remote=False)
    m.time = np.arange(0.0, T_END+DT, DT)
    n_t = len(m.time)

    def profile(t0, v0, v1):
        a = np.full(n_t, v0)
        a[m.time >= t0] = v1
        return a

    Q_cmd = m.Param(value=profile(T_STEP_Q, Q_CMD_0, Q_CMD_1))
    mdot_fg = m.Param(value=MDOT_FG_REP)
    Q_sec = m.Param(value=profile(T_STEP_QSEC, Q_SEC_REP_0, Q_SEC_REP_1))
    dt_sub = m.Param(value=profile(T_STEP_SUB, DT_SUB_STEP_0, DT_SUB_STEP_1))
    p_lps = m.Param(value=profile(T_STEP_PLPS, PLPS_STEP_0, PLPS_STEP_1))

    d = build_dynamics(m, mdot_fg, Q_sec, Q_cmd, dt_sub, p_lps)

    m.options.IMODE = 4
    m.options.NODES = 4
    m.options.SOLVER = 3
    m.solve(disp=False)

    t = np.array(m.time)
    res = {k: np.array(list(v.value)) for k, v in
           dict(Q_cond=d['Q_cond'], W=d['W_comp'], COP=d['COP'],
                m_lps=d['m_lps'], m_hw=d['m_hw'],
                Q_gas=d['Q_evap_gas_avail'], shortfall=d['dQ_shortfall']).items()}

    print("PROPERTY FITS  worst error over the envelope")
    for k, v in FIT.items():
        print(f"   {k:16}{v:9.5f} %")
    print(f"\nFIXED CYCLE POINTS  (design point, P_LPS = {P_LPS_DESIGN:.1f} bar)")
    print(f"   T_evap_primary   {T_EVAP_PRIMARY:9.2f} degC")
    print(f"   T_cond           {T_COND:9.2f} degC")
    print(f"   h1 / h2          {H1:9.2f} / {H2:9.2f} kJ/kg")
    print(f"   Q_MCR            {Q_MCR:9.1f} kW  (from Q_evap_gas_des "
          f"{Q_EVAP_GAS_DES:.1f} kW at design fuel rate)")

    print("\nDYNAMICS")
    print(f"   theta_duty / tau_duty      {THETA_DUTY:6.1f} / {TAU_DUTY:6.1f} s   Qv et al. 2024")
    _loop_s = dp.REF_CHARGE_KG_PER_KW * ss0['Q_cond'] / ss0['m_ref']
    print(f"   charge-based loop time     {_loop_s:9.1f} s   cross-check only, not used "
          f"(vs theta_duty {THETA_DUTY:.1f} s)")
    print(f"   theta_lps / tau_lps        {THETA_LPS:6.1f} / {TAU_LPS:6.1f} s   "
          f"(tau_lps_metal {TAU_LPS_METAL:.2f}, tau_lps_ref {TAU_LPS_REF:.2f})")
    print(f"   theta_hw / tau_hw          {THETA_HW:6.1f} / {TAU_HW:6.1f} s   "
          f"(t_res_hw {T_RES_HW:.1f} s, tau_hw_metal {TAU_HW_METAL:.2f})")

    print("\nSTEADY STATE CLOSURE  end of run against the reference steady_state()")
    ss = steady_state(Q_CMD_1, MDOT_FG_REP, Q_SEC_REP_1, DT_SUB_STEP_1, PLPS_STEP_1)
    for key, lab in [("W", "compressor (kW)"), ("m_lps", "LPS flow (kg/s)"),
                     ("m_hw", "HW flow (kg/s)"), ("COP", "COP")]:
        a, b = res[key][-1], ss[key]
        print(f"   {lab:20}{a:12.4f}{b:12.4f}   {100*(a-b)/b:+8.4f} %")

    print("\nFEASIBILITY  gas-available and shortfall at each operating point")
    marks = [("start", 0), ("after duty step", int(np.searchsorted(t, 1350))),
             ("after secondary step", int(np.searchsorted(t, 2450))),
             ("after subcool step", int(np.searchsorted(t, 3550))),
             ("after P_LPS step", len(t)-1)]
    print(f"   {'':20}{'Q_cond':>9}{'Q_gas avail':>13}{'shortfall':>11}"
          f"{'COP':>7}{'LPS':>8}{'HW':>8}")
    for lab, i in marks:
        print(f"   {lab:20}{res['Q_cond'][i]:9.0f}{res['Q_gas'][i]:13.1f}"
              f"{res['shortfall'][i]:11.1f}{res['COP'][i]:7.3f}"
              f"{res['m_lps'][i]:8.3f}{res['m_hw'][i]:8.3f}")

    fig, ax = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    panels = [
        (ax[0, 0], "Q_cond", "condenser duty (kW)"),
        (ax[0, 1], "W", "compressor power (kW)"),
        (ax[1, 0], "m_lps", "LPS produced (kg/s)"),
        (ax[1, 1], "m_hw", "hot water produced (kg/s)"),
        (ax[2, 0], "COP", "COP"),
        (ax[2, 1], "shortfall", "evaporator shortfall (kW)"),
    ]
    for a, key, lab in panels:
        a.plot(t, res[key], lw=1.6)
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
        for s in (T_STEP_Q, T_STEP_QSEC, T_STEP_SUB, T_STEP_PLPS):
            a.axvline(s, color="0.85", lw=0.8, zorder=0)
    ax[2, 0].set_xlabel("time (s)")
    ax[2, 1].set_xlabel("time (s)")
    fig.suptitle("Primary heat pump dynamic response -- duty step at 300 s, "
                 "secondary step at 1400 s, subcooling step at 2500 s, "
                 "LPS header pressure step at 3600 s",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("heatpump_dyn.png", dpi=130)
    print("\nsaved heatpump_dyn.png")


if __name__ == "__main__":
    #  Backend forced to Agg (and pyplot imported) only for a standalone run --
    #  this module is also imported into plant_rig.py/plant_manual.py, which
    #  need the interactive backend intact for their own GUI window, so this
    #  must not run at module import time.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()
