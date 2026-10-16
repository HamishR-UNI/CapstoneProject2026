"""
Dynamic turbine stage for GEKKO.

One stage, one time constant. The stage is a volume that the steam has to fill
before what comes out reflects what went in, so

    tau = V / Vdot        V    steam chest, inlet pipe and blading path  [m3]
                          Vdot inlet volumetric flow at the design point [m3/s]

Everything downstream of that single lag is algebraic. The lag is applied to the
OUTLET ENTHALPY, and both outputs are derived from it:

    h_out  lagged state
    dh_act = h_in - h_out          ->  power = mdot * dh_act
    T_out  = T(P2, h_out)          ->  receiving header superheat

so power and exhaust temperature can never disagree about how much energy left
the stage. An earlier version lagged dh and T separately, which let the two
outputs imply different enthalpies during a transient.

Valve stroke and any transport BEFORE the stage are NOT part of tau or theta here --
they belong to the actuator and the caller's own pipe run, and are applied by the caller.
theta (below) is only the stage's OWN transit dead time -- steam physically travelling the
steam chest and blading path before it reaches the outlet -- which is small enough (a
tenth of a second, see THETA_STAGE) that it is folded in here rather than left to callers
who would otherwise all have to reinvent the same few lines.

Requires: gekko, scipy, numpy, matplotlib, iapws95_gekko.py
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dynamics_params as dp
from iapws95_gekko import (State, saturation_state, solve_ph_numeric,
                           solve_ps_numeric, solve_saturation_numeric,
                           solve_tp_numeric, state_ph, state_ph_smooth,
                           state_ps, state_ps_smooth, state_tp)

# ----------------------------------------------------------------------
#  Operating point -- Turbine 1, HP -> MPS, from the Excel model
# ----------------------------------------------------------------------

#  Standalone default only. In the combined plant the boiler sets this -- see
#  plant_rig.py, where Turbine is constructed with the boiler's steam flow and this
#  number is never read. 40.395 kg/s is what boiler.py makes at its 10 kg/s dry fuel
#  design point on the Channiwala heating value; the Excel figure was 39.450.
MDOT = 40.395        # kg/s   steam mass flow
TIN  = 529.370       # C      inlet temperature
PIN  = 100.0         # bar    inlet pressure
POUT = 40.0          # bar    outlet pressure
ETA  = 0.70720       # -      isentropic efficiency

# ----------------------------------------------------------------------
#  Dynamics -- one volume, one time constant
#
#  PLACEHOLDER volume. Replace with the real steam chest plus inlet pipe
#  volume before quoting any result. At the design point 39.45 kg/s of steam
#  at 100 bar / 529 C is 28.99 kg/m3, so 1.361 m3/s: every 1.36 m3 of stage
#  volume buys one second of tau.
# ----------------------------------------------------------------------

#  ONE volume for the whole stage: inlet pipe, steam chest and blading path treated as a
#  single well-mixed volume.
#
#  Set from the turbine-governor literature rather than guessed. The charging time of the
#  inlet piping and steam chest, TCH in the IEEEG1 family of models, is consistently given
#  as 0.2 to 0.3 s, and that is the same quantity as V/Vdot here. At the design point
#  1.394 m3/s of inlet steam, 0.25 s implies 0.35 m3.
#
#  Cross-check from geometry: 40.4 kg/s at 100 bar and 529 C needs a 211 mm bore at 40 m/s,
#  and a 10 m run of that pipe holds 0.348 m3. The two routes agree.
V_STAGE = 0.35       # m3     inlet pipe + steam chest + blading path


def stage_tau(volume=V_STAGE, mdot=MDOT, T1_C=TIN, P1_bar=PIN):
    """tau = V / Vdot at the given inlet state, seconds."""
    rho = solve_tp_numeric(T1_C + 273.15, 100.0*P1_bar, 'vapor').rho
    return volume*rho/mdot


#  CASING/ROTOR THERMAL MASS. TAU above is a fluid residence time -- how long
#  steam takes to flush the stage -- and says nothing about the metal around
#  it. No number here has ever answered that question for this component; it
#  is a real gap, not a checked-and-dismissed one the way flue_gas_water_hx.py
#  dismisses its own tube metal (there the water-side film coefficient is high
#  enough that the metal settles in ~1 s, so it is genuinely negligible -- see
#  that module's docstring). A steam turbine casing has no such let-off: it is
#  a thick pressure casting, not a thin tube.
#
#  No real casing drawing exists for this project (theoretical plant, nothing
#  to check against), so this is a first-pass estimate from the same style of
#  reasoning as V_STAGE above, tau = t_wall * rho * c / h:
#
#    t_wall   thin-wall pressure vessel, t = P*D/(2*sigma_allow), at the
#             211 mm bore already used for V_STAGE: 100 bar * 0.211 m /
#             (2*70 MPa) = 15 mm, sigma_allow=70 MPa a mid-range creep-limited
#             design stress for a Cr-Mo casing steel around 530 degC.
#             Real casings run several times thicker than the pressure-only
#             minimum (flanges, nozzle chests, stiffness, thermal-cycling
#             margin) -- x4 gives 60 mm, the number used below.
#    rho, c   7850 kg/m3, 490 J/kg.K -- same cast/forged steel as the boiler's
#             own tube-metal figure (boiler.py), for one consistent material
#             assumption across the plant's pressure equipment.
#    h        500 W/m2.K steam-side film coefficient -- dense superheated
#             steam at speed (28.99 kg/m3, ~40 m/s, see V_STAGE above) sits
#             well above a dilute gas (boiler.py's flue-gas U=90) and below a
#             liquid (flue_gas_water_hx.py's ~2000), so a few hundred W/m2.K
#             is the reasonable middle ground.
#
#  tau_metal = 0.060*7850*490/500 = 461.6 s ~ 7.7 min, in the same few-minutes
#  band the boiler's own furnace/tube-metal figure lands in.
#
#  NOT APPLIED to h_out (or to anything else): a grid-synchronised turbine's power
#  response is genuinely fast -- speed is fixed by the grid, and the only store between
#  valve and blades is the steam chest (see TAU/THETA_STAGE below, and IEEE PES-TR1 2013 /
#  the 1991 IEEE working group paper on steam turbine dynamic models). Casing/rotor
#  thermal mass is real, but what it actually limits is the LOADING RATE the metal can
#  tolerate without excess thermal stress on a fast transient -- that is a ramp-rate
#  constraint a controller enforces, not a lag this component should carry on its own
#  enthalpy state. Kept here as a printed diagnostic for that future controller work.
T_WALL_CASING = 4 * (100.0*1e5 * 0.211) / (2 * 70e6)   # m, x4 over pressure-only minimum
RHO_CASING, C_CASING = 7850.0, 490.0                    # kg/m3, J/kg.K
H_STEAM_CASING = 500.0                                  # W/m2.K
TAU_CASING = T_WALL_CASING * RHO_CASING * C_CASING / H_STEAM_CASING   # s, diagnostic only


TAU = stage_tau()

#  Stage transit dead time: steam physically travelling the steam chest + blading path
#  before it reaches the outlet, on top of (not instead of) the charging lag TAU above.
#  Sourced as a layout assumption -- see dynamics_params.py for L_STAGE_PATH_M/
#  V_STAGE_STEAM_MPS -- and applied inside build_dynamics as a proper lag_chain, not
#  folded into TAU, since it is a transport delay rather than a mixing-volume lag.
THETA_STAGE = dp.L_STAGE_PATH_M / dp.V_STAGE_STEAM_MPS   # s


def _numeric(v):
    """Pull a float out of a float, Param, Var or MV; None if there isn't one.

    Before a GEKKO model's first solve(), a Var/Param's .value (a GK_Value)
    wraps a bare scalar rather than a per-time-point list -- val[0] raises
    TypeError from inside GK_Value.__getitem__ itself. Unwrap one layer (the
    GK_Value's own .value) to reach that scalar directly instead of falling
    through to the "no number here" case, which used to silently return None
    for any live, not-yet-solved GEKKO object.
    """
    if isinstance(v, (int, float)):
        return float(v)
    val = getattr(v, 'value', None)
    if val is None:
        return None
    raw = getattr(val, 'value', val)
    try:
        return float(raw[0])
    except (TypeError, IndexError):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


def lag_chain(m, u, theta=0.0, tau=0.0, n_lag=6, y0=None, name='lag'):
    """
    Dead time as a chain of n_lag small lags, optionally followed by one
    first order lag. Returns (output, [states]).

    m.delay() is unsafe here -- its buffer is zero filled, which injects a
    phantom transient at the start of a run. Pade gives an inverse response
    that is not physical for a transport delay.

    Used for the inlet pipe delay and the valve stroke, and re-exported so
    the rig does not need its own copy.
    """
    y0 = _numeric(u) if y0 is None else y0
    y, states = u, []
    if theta and n_lag:
        step_tau = float(theta)/n_lag
        for i in range(n_lag):
            nxt = m.Var(value=y0, name=f'{name}_d{i}')
            m.Equation(step_tau*nxt.dt() == y - nxt)
            states.append(nxt)
            y = nxt
    if tau:
        nxt = m.Var(value=y0, name=f'{name}_lag')
        m.Equation(tau*nxt.dt() == y - nxt)
        states.append(nxt)
        y = nxt
    return y, states


def _seed_outlet_state(P_kPa, target, prop):
    """Numeric outlet state at P_kPa matching a target entropy (prop='s') or
    enthalpy (prop='h'), covering the wet-steam region a single-phase vapor
    solve cannot reach (a low back pressure can put a real expansion's outlet
    inside the saturation dome, e.g. turbine 2 down near a true 2-3 bar LPS).

    Falls back to an exact liquid/vapor solve when the target sits outside
    the dome. Inside it, no root-find is needed -- entropy and enthalpy are
    both exactly linear in quality at fixed P (the lever rule), so the
    mixture state comes straight from the two coexisting saturation states.
    """
    sat = solve_saturation_numeric(P_kPa)
    lo, hi = ((sat.liquid.s, sat.vapor.s) if prop == 's'
              else (sat.liquid.h, sat.vapor.h))
    if target <= lo:
        return (solve_ps_numeric(P_kPa, target, 'liquid') if prop == 's'
                else solve_ph_numeric(P_kPa, target, 'liquid'))
    if target >= hi:
        return (solve_ps_numeric(P_kPa, target, 'vapor') if prop == 's'
                else solve_ph_numeric(P_kPa, target, 'vapor'))
    x = (target - lo) / (hi - lo)
    liquid, vapor = sat.liquid, sat.vapor
    rho = 1.0/((1.0-x)/liquid.rho + x/vapor.rho)
    h = (1.0-x)*liquid.h + x*vapor.h
    s = (1.0-x)*liquid.s + x*vapor.s
    return State(sat.T, P_kPa, rho, h, s, h - P_kPa/rho, h - sat.T*s,
                float('nan'), float('nan'), 0.0, quality=x, phase='two-phase')


def _seed(T1_C, P1_bar, P2_bar, eta_is):
    """
    Inlet, ideal outlet and actual outlet in plain Python, so GEKKO starts
    on the intended steam branch. GEKKO holds variables at their starting
    value on the first time point, so a bad guess makes t = 0 wrong.
    """
    T1, P1, P2 = _numeric(T1_C), _numeric(P1_bar), _numeric(P2_bar)
    eta = _numeric(eta_is)
    if None in (T1, P1, P2, eta):
        T1, P1, P2, eta = TIN, PIN, POUT, ETA
    inlet = solve_tp_numeric(T1 + 273.15, 100.0*P1, 'vapor')
    ideal = _seed_outlet_state(100.0*P2, inlet.s, 's')
    actual = _seed_outlet_state(100.0*P2, inlet.h - eta*(inlet.h - ideal.h), 'h')
    return inlet, ideal, actual


# ----------------------------------------------------------------------
#  Fixed-inlet outlet fit -- precomputed once per (T1_C, P1_bar), the same
#  idea as boiler.py's DH_MAP/MAP: solve the numeric IAPWS-95 EOS offline on
#  a grid, then let GEKKO only ever evaluate the resulting closed-form fit
#  instead of carrying two live symbolic EOS solves in every model solve.
#
#  Both fits only need to cover the range a caller can actually command.
#  2026-09-25: widened from 18-62 to 8-62 bar for the MPS target's new 10 bar
#  floor (plant_rig.P_MPS_LO). At 8 bar the ISENTROPIC outlet of the 100 bar /
#  529 C inlet is just inside the dome, so the fit uses the dome-aware helpers;
#  the actual outlet (eta 0.707) stays about 200 kJ/kg superheated. Degree 11 on
#  49 points: dh_is within 0.007 kJ/kg and T2s within 0.001 K off-grid.
# ----------------------------------------------------------------------

P2_FIT_LO, P2_FIT_HI = 8.0, 62.0      # bar

_P2_FIT_DEGREE = 11                   # isentropic-outlet polynomial degree
_P2_FIT_GRID_N = 49                   # 1-D fitting grid points
_BSPLINE_GRID_N = 11                  # 2-D (P2, h) grid points per axis
_TOUT_FIT_DEGREE = 7                  # T_out(P2, h) polynomial, per axis


@dataclass
class FixedInletFit:
    """Precomputed outlet-property maps for a fixed-inlet Turbine.

    Built by _fixed_inlet_outlet_fit(), which callers should use instead of
    constructing this directly -- it's the cache key.
    """
    inlet: object                # State, the fixed inlet
    dh_is_poly: object           # np.ndarray, isentropic dh vs normalized P2
    t2s_poly: object             # np.ndarray, isentropic T_out vs normalized P2
    dh_is_residual_kJkg: float
    t2s_residual_C: float
    p2_grid_bar: object          # np.ndarray, ascending
    h_grid_kJkg: object          # np.ndarray, ascending
    Tout_grid_C: object          # np.ndarray, shape (p2_grid_bar.size, h_grid_kJkg.size)
    tout_poly: object = None     # _polyfit2d coefficients of Tout_grid_C, normalised axes


def _normalize_p2(p2_bar):
    """Map [P2_FIT_LO, P2_FIT_HI] to [-1, 1]. A high-degree polyfit is poorly
    conditioned over the raw 44 bar-wide domain; normalizing fixes that.
    Works for a float or a GEKKO expression."""
    mid = 0.5*(P2_FIT_LO + P2_FIT_HI)
    half = 0.5*(P2_FIT_HI - P2_FIT_LO)
    return (p2_bar - mid)/half


def _poly1(coeffs, x):
    """Horner evaluation of np.polyfit-style coefficients (highest degree
    first). Works for a float or a GEKKO expression."""
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result*x + c
    return result


@lru_cache(maxsize=8)
def _fixed_inlet_outlet_fit(T1_C, P1_bar):
    """Fit the fixed-inlet turbine outlet properties over P2_FIT_LO..HI.

    Called once per distinct (T1_C, P1_bar) -- at Turbine.__init__ time,
    never from inside a GEKKO solve. Keyed on the actual inlet passed in,
    not on this module's TIN/PIN, since fixed_inlet accepts any numeric
    inlet.
    """
    inlet = solve_tp_numeric(T1_C + 273.15, 100.0*P1_bar, 'vapor')

    p2_grid_bar = np.linspace(P2_FIT_LO, P2_FIT_HI, _P2_FIT_GRID_N)
    x_norm = _normalize_p2(p2_grid_bar)
    h2s_vals = np.array([_h_numeric_any_phase(100.0*p2, inlet.s) for p2 in p2_grid_bar])
    dh_is_vals = inlet.h - h2s_vals
    t2s_vals = np.array([_T_numeric_any_phase(100.0*p2, h) - 273.15
                         for p2, h in zip(p2_grid_bar, h2s_vals)])

    dh_is_poly = np.polyfit(x_norm, dh_is_vals, _P2_FIT_DEGREE)
    t2s_poly = np.polyfit(x_norm, t2s_vals, _P2_FIT_DEGREE)
    dh_is_residual = float(np.abs(dh_is_vals - np.polyval(dh_is_poly, x_norm)).max())
    t2s_residual = float(np.abs(t2s_vals - np.polyval(t2s_poly, x_norm)).max())

    #  h_out is a first-order lag of h_ss = h_in - eta_is*dh_is(P2), so it can
    #  never leave the range h_ss sweeps for eta_is in [0,1] -- safe for any
    #  actual (fixed) eta_is this Turbine is built with.
    #  The grid is rectangular, so its low-h edge must stay superheated at EVERY P2 on it
    #  (the vapour solve below cannot bracket a wet point): clipped to 40 kJ/kg above the
    #  highest hg on the grid. Only an eta_is near 1 at the lowest P2 would reach that.
    hg_max = max(solve_saturation_numeric(100.0*p2).vapor.h for p2 in p2_grid_bar)
    h_lo = max(inlet.h - 1.15*float(dh_is_vals.max()), hg_max + 40.0)
    h_hi = inlet.h + 0.02*abs(inlet.h)
    h_grid_kJkg = np.linspace(h_lo, h_hi, _BSPLINE_GRID_N)

    Tout_grid_C = np.array([
        [solve_ph_numeric(100.0*p2, h, 'vapor').T - 273.15 for h in h_grid_kJkg]
        for p2 in p2_grid_bar])
    tout_poly = _polyfit2d(_norm(p2_grid_bar, P2_FIT_LO, P2_FIT_HI),
                           _norm(h_grid_kJkg, h_lo, h_hi), Tout_grid_C, _TOUT_FIT_DEGREE)

    return FixedInletFit(inlet, dh_is_poly, t2s_poly, dh_is_residual,
                         t2s_residual, p2_grid_bar, h_grid_kJkg, Tout_grid_C, tout_poly)


# ----------------------------------------------------------------------------------
# Moving-inlet surrogate  (turbine 2)
#
# Turbine 1's inlet is pinned by the HP header, so its outlet states are pure
# functions of P2 and can be fitted in one dimension (above). Turbine 2 cannot
# do that: its inlet is turbine 1's exhaust, so inlet pressure AND inlet
# temperature are both live. The fallback is a full symbolic IAPWS-95 solve
# every step -- measured on this model, that one component is about 740 ms and
# 740 intermediates, roughly nine tenths of the entire plant model.
#
# EVERYTHING HERE IS PLAIN POLYNOMIAL ALGEBRA, deliberately. The obvious
# implementation would use m.bspline surfaces like the fixed-inlet path does,
# and it does not work: APMonitor segfaults when one bspline's output feeds
# another bspline's input, and copy variables in between do not help. That
# rules bsplines out here twice over -- the surrogate needs h(P,s) evaluated
# at a fitted s_in, AND turbine 2's inlet temperature is itself turbine 1's
# bspline output. Polynomials are just equations, so they chain freely.
#
# The fits exploit two facts, both verified by sweeping the reachable
# operating band:
#
#   * the ISENTROPIC outlet is always INSIDE the dome. Inside, h is exactly
#     linear in s at fixed pressure, so the lever rule is not an approximation
#     -- it is exact, needing only hf, hg, sf, sg as 1-D polynomials in P2.
#     T2s is likewise exactly Tsat(P2).
#   * the ACTUAL outlet is always OUTSIDE it, by at least 38 kJ/kg. So
#     T_out(P2, h) is a smooth two-phase-free surface.
#
# Neither fact is assumed silently. Both are exposed as live diagnostics --
# .x_ideal and .dsh_act below -- on the same "reported, not enforced"
# convention the rest of this plant model uses for pinch and evaporator
# feasibility. If x_ideal reaches 1 or dsh_act reaches 0, the operating point
# has left the region this surrogate is valid over and the exact path
# (surrogate=False) should be used instead.
#
# T_out is fitted against SUPERHEAT ABOVE SATURATION rather than absolute
# enthalpy. An absolute-h grid has to start above hg at the highest P2, which
# puts its lower edge inside the range h_out visits at the LOWEST P2 -- the
# fit then extrapolates, at an error that densifying the grid does not touch.
# In the superheat coordinate the saturation line IS the lower edge.
# ----------------------------------------------------------------------------------

#  2026-09-25: P1 floor 22 -> 8 bar for the MPS target's new 10 bar floor. Two things
#  change below that the original box never met:
#   * the inlet box is no longer all vapour -- at 58 bar, 240 C is liquid -- so the
#     inlet fits sample only T1 >= Tsat(P1) + MOVING_T1_SUPERHEAT_MIN (least squares
#     over that band, not the full rectangle);
#   * at low MPS and normal LPS the ISENTROPIC outlet is superheated (10 bar -> 4 bar
#     ends 0.15 kJ/kg/K past sg), where the lever rule is not exact. Extended past sg
#     the lever line is hg + u*Tsat (u = s_in - sg); the true h is higher by
#     integral (T - Tsat) ds, which is O(u^2) (dh/ds = T is continuous across sg).
#     That correction is fitted as u^2 * poly(P2, u) and added through a smooth
#     positive part of u, so the wet side is unchanged and still exact.
MOVING_P1_LO, MOVING_P1_HI = 8.0, 58.0       # bar, turbine 2 inlet (MPS header)
MOVING_T1_LO, MOVING_T1_HI = 200.0, 500.0    # degC, turbine 2 inlet (normalisation box)
MOVING_T1_SUPERHEAT_MIN = 30.0               # K, inlet fit band starts this far above Tsat
MOVING_P2_LO, MOVING_P2_HI = 1.6, 4.6        # bar, turbine 2 exhaust (LPS header)
MOVING_DSH_HI = 320.0                        # kJ/kg, top of the superheat axis
MOVING_U_HI = 0.6                            # kJ/kg/K, isentropic entropy past sg covered
_MOVING_U_EPS = 1e-3                         # kJ/kg/K, smoothing of the positive part
_MOVING_SAT_DEGREE = 8                       # 1-D fits in P2
_MOVING_2D_DEGREE = 6                        # per axis on the 2-D fits
_MOVING_INLET_DEGREE = 7                     # per axis on the inlet (ln P1, T1) fits
_MOVING_GRID_N = 26                          # sample points per axis


@dataclass
class MovingInletFit:
    """Polynomial coefficients for a live-inlet turbine stage.

    Built by _moving_inlet_fit(); do not construct directly. Every entry is
    plain algebra -- see the module comment above for why no B-splines.
    """
    hf: np.ndarray
    hg: np.ndarray
    sf: np.ndarray
    sg: np.ndarray
    tsat: np.ndarray
    h_in: np.ndarray
    s_in: np.ndarray
    t_out: np.ndarray
    sup_h: np.ndarray            # superheated isentropic outlet: h correction / u^2
    sup_t: np.ndarray            # and T2s - Tsat, / u
    box: dict
    residuals: dict


def _T_numeric_any_phase(P_kPa, h):
    """T(P, h), valid on BOTH sides of the saturation line.

    solve_ph_numeric has to be told a phase and cannot bracket a two-phase
    point at all. Inside the dome the answer is just the saturation
    temperature, so it is returned directly rather than solved.
    """
    sat = solve_saturation_numeric(P_kPa)
    if sat.liquid.h < h < sat.vapor.h:
        return sat.T
    return solve_ph_numeric(P_kPa, h, 'vapor' if h >= sat.vapor.h else 'liquid').T


def _h_numeric_any_phase(P_kPa, s):
    """h(P, s), likewise valid across the dome -- lever rule inside it.

    Exact, not an approximation: at fixed pressure h really is linear in s
    between the saturated liquid and vapour states.
    """
    sat = solve_saturation_numeric(P_kPa)
    if sat.liquid.s < s < sat.vapor.s:
        x = (s - sat.liquid.s)/(sat.vapor.s - sat.liquid.s)
        return sat.liquid.h + x*(sat.vapor.h - sat.liquid.h)
    return solve_ps_numeric(P_kPa, s, 'vapor' if s >= sat.vapor.s else 'liquid').h


def _norm(v, lo, hi):
    """Map [lo, hi] to [-1, 1]. A polyfit on raw bar/degC is badly
    conditioned; on [-1, 1] it is not."""
    return (2.0*v - (hi + lo))/(hi - lo)


def _polyfit2d(x, y, z, deg):
    """Least-squares tensor-product polynomial over a rectangular grid.

    x, y are the normalised axes; z is z[i, j] at (x[i], y[j]). Returns
    coefficients c[a, b] for sum_ab c[a,b] * x**a * y**b.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    basis = np.array([[xx**a * yy**b for a in range(deg + 1) for b in range(deg + 1)]
                      for xx, yy in zip(X.ravel(), Y.ravel())])
    coeffs, *_ = np.linalg.lstsq(basis, z.ravel(), rcond=None)
    return coeffs.reshape(deg + 1, deg + 1)


def _polyval2d(c, x, y):
    """Evaluate _polyfit2d coefficients. Works on numpy arrays for the
    residual check and on GEKKO expressions inside a model."""
    total = 0.0
    for a in range(c.shape[0]):
        xa = x**a if a else 1.0
        for b in range(c.shape[1]):
            if c[a, b] != 0.0:
                total = total + c[a, b]*xa*(y**b if b else 1.0)
    return total


def _polyfit2d_pts(x, y, z, deg):
    """_polyfit2d for scattered points (x, y, z all 1-D, same length) -- used where
    the valid region is not a rectangle. Same coefficient layout as _polyfit2d."""
    basis = np.array([[xx**a * yy**b for a in range(deg + 1) for b in range(deg + 1)]
                      for xx, yy in zip(x, y)])
    coeffs, *_ = np.linalg.lstsq(basis, z, rcond=None)
    return coeffs.reshape(deg + 1, deg + 1)


def _smooth_pos(d, eps=_MOVING_U_EPS):
    """max(d, 0), smoothed over +-eps. Works for a float or a GEKKO expression
    (GEKKO overloads ** on its expressions)."""
    return 0.5*(d + (d*d + eps*eps)**0.5)


@lru_cache(maxsize=8)
def _moving_inlet_fit(p1_lo, p1_hi, t1_lo, t1_hi, p2_lo, p2_hi, dsh_hi, n, u_hi=MOVING_U_HI):
    """Fit every surface once, at Turbine construction time.

    Cached on the box, so building several stages over the same envelope --
    or rebuilding the rig after a failed solve -- costs one set of EOS
    sweeps rather than several.
    """
    res = {}

    #  --- saturation properties, 1-D in P2 -------------------------------
    p2 = np.linspace(p2_lo, p2_hi, 4*n)
    p2n = _norm(p2, p2_lo, p2_hi)
    sats = [solve_saturation_numeric(100.0*p) for p in p2]
    out = {}
    for key, vals in (('hf', [s.liquid.h for s in sats]),
                      ('hg', [s.vapor.h for s in sats]),
                      ('sf', [s.liquid.s for s in sats]),
                      ('sg', [s.vapor.s for s in sats]),
                      ('tsat', [s.T - 273.15 for s in sats])):
        vals = np.array(vals)
        c = np.polyfit(p2n, vals, _MOVING_SAT_DEGREE)
        out[key] = c
        res[key] = float(np.abs(np.polyval(c, p2n) - vals).max())

    #  --- inlet state, 2-D in (P1, T1), over the superheated band only ------
    pts = []
    for p in np.linspace(p1_lo, p1_hi, n):
        t_min = max(t1_lo, solve_saturation_numeric(100.0*p).T - 273.15
                    + MOVING_T1_SUPERHEAT_MIN)
        for t in np.linspace(t_min, t1_hi, n):
            st = solve_tp_numeric(t + 273.15, 100.0*p, 'vapor')
            pts.append((p, t, st.h, st.s))
    pts = np.array(pts)
    #  ln(P1) axis: s goes as -R ln P, so on a linear P axis over 8-58 bar the s_in fit was
    #  0.0012 kJ/kg/K off (0.5 kJ/kg on h2s); on ln P it is 0.00002 (0.01 kJ/kg).
    p1n = _norm(np.log(pts[:, 0]), np.log(p1_lo), np.log(p1_hi))
    t1n = _norm(pts[:, 1], t1_lo, t1_hi)
    for key, col in (('h_in', 2), ('s_in', 3)):
        c = _polyfit2d_pts(p1n, t1n, pts[:, col], _MOVING_INLET_DEGREE)
        out[key] = c
        res[key] = float(np.abs(_polyval2d(c, p1n, t1n) - pts[:, col]).max())

    #  --- superheated isentropic outlet, 2-D in (P2, u = s - sg) -----------
    #  sup_h = (h(P2, sg+u) - hg - u*Tsat_K)/u^2,  sup_t = (T(P2, sg+u) - Tsat)/u
    u = np.linspace(0.005, u_hi, n)
    p2b = np.linspace(p2_lo, p2_hi, n)
    gh, gt = np.zeros((n, n)), np.zeros((n, n))
    for i, p in enumerate(p2b):
        sat = solve_saturation_numeric(100.0*p)
        for j, uu in enumerate(u):
            st = solve_ps_numeric(100.0*p, sat.vapor.s + uu, 'vapor')
            gh[i, j] = (st.h - sat.vapor.h - uu*sat.T)/uu**2
            gt[i, j] = (st.T - sat.T)/uu
    p2bn, un = _norm(p2b, p2_lo, p2_hi), _norm(u, 0.0, u_hi)
    for key, grid, scale in (('sup_h', gh, u[None, :]**2), ('sup_t', gt, u[None, :])):
        c = _polyfit2d(p2bn, un, grid, _MOVING_2D_DEGREE)
        out[key] = c
        #  residual in the quantity actually used (kJ/kg, K), not in the scaled one
        res[key] = float(np.abs((_polyval2d(c, p2bn[:, None], un[None, :]) - grid)*scale).max())

    #  --- outlet temperature, 2-D in (P2, superheat above hg) ------------
    dsh = np.linspace(1.0, dsh_hi, n)
    p2b = np.linspace(p2_lo, p2_hi, n)
    p2bn, dshn = _norm(p2b, p2_lo, p2_hi), _norm(dsh, 1.0, dsh_hi)
    tg = np.array([[solve_ph_numeric(100.0*p,
                                     solve_saturation_numeric(100.0*p).vapor.h + d,
                                     'vapor').T - 273.15 for d in dsh]
                   for p in p2b])
    c = _polyfit2d(p2bn, dshn, tg, _MOVING_2D_DEGREE)
    out['t_out'] = c
    res['t_out'] = float(np.abs(_polyval2d(c, p2bn[:, None], dshn[None, :]) - tg).max())

    box = dict(P1=(p1_lo, p1_hi), T1=(t1_lo, t1_hi), P2=(p2_lo, p2_hi),
               dsh=(1.0, dsh_hi), u=(0.0, u_hi), n=n)
    return MovingInletFit(out['hf'], out['hg'], out['sf'], out['sg'], out['tsat'],
                          out['h_in'], out['s_in'], out['t_out'], out['sup_h'],
                          out['sup_t'], box, res)


def surrogate_error(p1_bar, T1_C, p2_bar, eta_is=ETA):
    """Measure the surrogate against the exact EOS at one operating point.

    Returns absolute errors in dh_is (kJ/kg), T2s and T_out (degC), plus the
    two validity diagnostics. plant_rig.py calls this at its base point, so
    the surrogate is never trusted without having been compared against the
    path it replaces.
    """
    from gekko import GEKKO
    inlet = solve_tp_numeric(T1_C + 273.15, 100.0*p1_bar, 'vapor')
    sat = solve_saturation_numeric(100.0*p2_bar)
    x = (inlet.s - sat.liquid.s)/(sat.vapor.s - sat.liquid.s)
    h2s = _h_numeric_any_phase(100.0*p2_bar, inlet.s)
    h_out = inlet.h - eta_is*(inlet.h - h2s)
    dsh = h_out - sat.vapor.h
    exact = dict(dh_is=inlet.h - h2s, T2s=_T_numeric_any_phase(100.0*p2_bar, h2s) - 273.15,
                 T_out=solve_ph_numeric(100.0*p2_bar, h_out, 'vapor').T - 273.15)

    #  Which solvers the local APM binary carries varies between machines, so
    #  probe rather than assume -- but build a FRESH model for each attempt.
    #  Retrying a failed solve on the same model object silently returns
    #  wrong numbers: the failure leaves the Params holding defaults rather
    #  than the values they were given, so the next solve answers a different
    #  question. (This is the same hazard plant_rig._reseed() exists for.)
    for choice in (3, 2, 1):
        m = GEKKO(remote=False)
        m.options.IMODE = 1
        m.options.SOLVER = choice
        t = Turbine(m, 1.0, m.Param(T1_C), m.Param(p1_bar), m.Param(p2_bar),
                    eta_is, surrogate=True, name='chk')
        try:
            m.solve(disp=False)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('no solver converged on the surrogate check point')
    got = dict(dh_is=t.dh_is.value[0], T2s=t.T2s_C.value[0],
               T_out=t.T_out_C.value[0])
    err = {k: abs(got[k] - exact[k]) for k in exact}
    err['x_ideal'] = x            # >1 is fine since 2026-09-25 (superheat correction)
    err['u_is'] = max(inlet.s - sat.vapor.s, 0.0)   # must stay below MOVING_U_HI
    err['dsh_act'] = dsh          # must stay above 0 for the T_out fit
    return err


class Turbine:
    """
    One turbine stage.

        m         GEKKO model
        mdot      kg/s   steam mass flow      float, Param, Var or MV
        T1_C      C      inlet temperature    float, Param, Var or MV
        P1_bar    bar    inlet pressure       float, Param, Var or MV
        P2_bar    bar    outlet pressure      float, Param, Var or MV
        eta_is    -      isentropic efficiency
        tau       s      stage time constant, V/Vdot.  0 = steady state
        theta     s      stage transit dead time (see THETA_STAGE above) --
                         a lag_chain dead-time chain BEFORE the tau lag, on
                         the steady-state target h_ss, not combined with tau
                         (it is a transport delay, not a mixing-volume lag).
                         0 = not modelled
        n_lag_theta -    dead-time chain length for theta.  Default 3
                         (theta is well under 1 s at the default THETA_STAGE)
        fixed_inlet      True treats the inlet state as constants. Valid only
                         when T1_C and P1_bar are numbers, which they are for
                         a stage fed by a pressure-controlled header. Both
                         outlet states are then read from a precomputed fit
                         over P2_FIT_LO..HI (see _fixed_inlet_outlet_fit)
                         instead of solving IAPWS-95 live every step.
        surrogate        For a stage whose inlet MOVES (turbine 2), read every
                         property off the precomputed B-spline surfaces
                         instead of solving IAPWS-95 symbolically. ~50x
                         cheaper. Valid inside the MOVING_* box only; outside
                         it the splines extrapolate. Mutually exclusive with
                         fixed_inlet. Set False to fall back to the exact
                         path -- which is what surrogate_error() compares
                         against, and what to use for validation runs and for
                         any operating point outside the box.
        name             unique suffix if more than one turbine

    Outputs
        .power_kW  .dh_act  .T_out_C     what the stage is doing now
        .power_ss  .dh_ss                what it will settle at
        .dh_is  .T2s_C                   isentropic (ideal) values
        .states                          the lagging state, for seeding
    """

    def __init__(self, m, mdot, T1_C, P1_bar, P2_bar, eta_is,
                 tau=0.0, theta=0.0, n_lag_theta=3, fixed_inlet=False,
                 surrogate=False, name=''):
        if fixed_inlet and surrogate:
            raise ValueError('fixed_inlet and surrogate are alternative '
                             'property routes -- pick one')
        self.m = m
        self.mdot, self.eta_is = mdot, eta_is
        self._T1_C, self._P1, self._P2 = T1_C, P1_bar, P2_bar
        self.tau, self.theta = tau, theta
        self.states = []
        self.quality = None         # set below for the wet-capable (exact) path
        self.surrogate = surrogate
        self.fit = None
        #  Validity diagnostics, populated only on the surrogate path. None
        #  on the exact path, which needs no domain to stay inside.
        self.x_ideal = self.dsh_act = self.u_is = None

        inlet0, ideal0, actual0 = _seed(T1_C, P1_bar, P2_bar, eta_is)

        if surrogate:
            fit = _moving_inlet_fit(MOVING_P1_LO, MOVING_P1_HI, MOVING_T1_LO,
                                    MOVING_T1_HI, MOVING_P2_LO, MOVING_P2_HI,
                                    MOVING_DSH_HI, _MOVING_GRID_N)
            self.fit = fit
            p1_lo, p1_hi = fit.box['P1']
            ln_p1 = (np.log(float(P1_bar)) if isinstance(P1_bar, (int, float))
                     else m.log(P1_bar))
            p1n = m.Intermediate(_norm(ln_p1, np.log(p1_lo), np.log(p1_hi)))
            t1n = m.Intermediate(_norm(T1_C, *fit.box['T1']))
            p2n = m.Intermediate(_norm(P2_bar, *fit.box['P2']))

            h_in = m.Intermediate(_polyval2d(fit.h_in, p1n, t1n))
            s_in = m.Intermediate(_polyval2d(fit.s_in, p1n, t1n))

            hf = m.Intermediate(_poly1(fit.hf, p2n))
            hg = m.Intermediate(_poly1(fit.hg, p2n))
            sf_ = m.Intermediate(_poly1(fit.sf, p2n))
            sg = m.Intermediate(_poly1(fit.sg, p2n))
            self._hg = hg

            #  Isentropic outlet by the lever rule. Exact, not an
            #  approximation -- h really is linear in s inside the dome.
            #  Past sg (x_ideal > 1, low MPS) the lever line is continued
            #  and the O(u^2) superheat correction added -- see the note
            #  at MOVING_P1_LO. u_is is published: it must stay below
            #  box['u'][1] (MOVING_U_HI) for the correction fit to apply.
            self.x_ideal = m.Intermediate((s_in - sf_)/(sg - sf_))
            self.u_is = m.Intermediate(_smooth_pos(s_in - sg))
            un = _norm(self.u_is, *fit.box['u'])
            tsat = m.Intermediate(_poly1(fit.tsat, p2n))
            h2s = m.Intermediate(hf + self.x_ideal*(hg - hf)
                                 + self.u_is**2*_polyval2d(fit.sup_h, p2n, un))
            self.dh_is = m.Intermediate(h_in - h2s)
            #  Inside the dome the isentropic outlet temperature IS the
            #  saturation temperature; past sg it rises by u*sup_t.
            self.T2s_C = m.Intermediate(tsat + self.u_is*_polyval2d(fit.sup_t, p2n, un))
        elif fixed_inlet:
            if None in (_numeric(T1_C), _numeric(P1_bar)):
                raise TypeError('fixed_inlet needs a numeric T1_C and P1_bar')
            #  Both outlet states are then pure functions of P2_bar (and, for
            #  the actual outlet, the lag state h_out below) -- fit once
            #  instead of carrying two live IAPWS-95 solves in the model.
            outlet_fit = _fixed_inlet_outlet_fit(round(_numeric(T1_C), 6),
                                                 round(_numeric(P1_bar), 6))
            h_in, s_in = outlet_fit.inlet.h, outlet_fit.inlet.s
            p2_norm = _normalize_p2(P2_bar)
            self.dh_is = m.Intermediate(_poly1(outlet_fit.dh_is_poly, p2_norm))
            self.T2s_C = m.Intermediate(_poly1(outlet_fit.t2s_poly, p2_norm))
        else:
            P2 = m.Intermediate(100.0*P2_bar)          # bar -> kPa
            inlet = state_tp(m, m.Intermediate(T1_C + 273.15),
                             m.Intermediate(100.0*P1_bar), phase='vapor',
                             name=f'in{name}', initial_T=inlet0.T,
                             initial_P=inlet0.P)
            h_in, s_in = inlet.h, inlet.s
            #  The isentropic (ideal) and actual outlets are both evaluated
            #  at this same back pressure P2, so the liquid/vapor coexistence
            #  state at P2 -- two full IAPWS-95 Helmholtz evaluations plus
            #  the pressure/Gibbs equations -- is identical for both. Solve
            #  it once here and hand it to both flashes below instead of
            #  letting each independently rebuild the same saturation state
            #  (this is the dominant cost of the non-fixed-inlet path, e.g.
            #  turbine 2 in plant_rig.py).
            sat_P2 = saturation_state(m, P2, name=f'satP2{name}',
                                      initial_P=ideal0.P)
            #  Smooth (dome-aware), not vapor-only: a live back pressure can
            #  put the isentropic outlet inside the saturation dome (e.g.
            #  turbine 2 at a true 2-3 bar LPS), which a vapor-only solve
            #  cannot represent -- see _seed_outlet_state's docstring.
            ideal = state_ps_smooth(m, P2, s_in, name=f'is{name}',
                                    initial_s=inlet0.s, initial_P=ideal0.P,
                                    sat=sat_P2)
            self.dh_is = m.Intermediate(h_in - ideal.h)
            self.T2s_C = m.Intermediate(ideal.T - 273.15)

        self.dh_ss = m.Intermediate(eta_is*self.dh_is)
        self.power_ss = m.Intermediate(mdot*self.dh_ss)

        #  The one state. Lagging the outlet enthalpy rather than dh and T
        #  separately is what keeps the two outputs telling the same story.
        h_ss = m.Intermediate(h_in - self.dh_ss)

        #  Stage transit dead time (THETA_STAGE), BEFORE the tau lag below -- a transport
        #  delay, not a mixing-volume lag, so it is a dead-time chain on h_ss rather than
        #  folded into tau. Same lag_chain the rest of the plant uses for dead time.
        if theta:
            h_ss_d, theta_states = lag_chain(m, h_ss, theta=theta, tau=0.0,
                                             n_lag=n_lag_theta, y0=actual0.h,
                                             name=f'hss{name}')
            self.states.extend(theta_states)
        else:
            h_ss_d = h_ss

        if surrogate:
            if tau:
                h_out = m.Var(value=actual0.h, name=f'hout{name}')
                m.Equation(tau*h_out.dt() == h_ss_d - h_out)
                self.states.append(h_out)
            else:
                h_out = h_ss_d
            #  Superheat above saturation is the fit's second axis -- see the
            #  module comment. Published for the same reason as x_ideal: at
            #  dsh <= 0 the actual outlet has entered the dome and this fit no
            #  longer applies.
            self.dsh_act = m.Intermediate(h_out - self._hg)
            dshn = m.Intermediate(_norm(self.dsh_act, *fit.box['dsh']))
            self.T_out_C = m.Intermediate(_polyval2d(fit.t_out, p2n, dshn))
        elif fixed_inlet:
            h_out = m.Var(value=actual0.h, name=f'hout{name}')
            if tau:
                m.Equation(tau*h_out.dt() == h_ss_d - h_out)
            else:
                m.Equation(h_out == h_ss_d)
            self.states.append(h_out)

            #  T_out(P2, h) as a tensor polynomial over the same grid (2026-09-25). This was
            #  an m.bspline, but with data=True, sf=0 GEKKO's executable does not build an
            #  interpolating spline -- it keeps only the boundary knots (one bicubic patch
            #  over the whole grid). That was 0.0017 K over 18-62 bar; the grid now spans
            #  8-62 bar, and the polynomial holds 0.0001 K off-grid across the whole band
            #  (same fix as heatpump_dyn._DH_POLY).
            #  Kept a Var with a starting value, not an Intermediate: turbine 2 is
            #  seeded from this (_seed reads .value), and an Intermediate has none.
            g = outlet_fit
            T_out_var = m.Var(value=actual0.T - 273.15, name=f'tout{name}')
            m.Equation(T_out_var == _polyval2d(
                g.tout_poly, _normalize_p2(P2_bar),
                (2.0*h_out - (g.h_grid_kJkg[-1] + g.h_grid_kJkg[0]))
                / (g.h_grid_kJkg[-1] - g.h_grid_kJkg[0])))
            self.T_out_C = T_out_var
        else:
            if tau:
                h_out = m.Var(value=actual0.h, name=f'hout{name}')
                m.Equation(tau*h_out.dt() == h_ss_d - h_out)
                self.states.append(h_out)
            else:
                h_out = h_ss_d
            #  Reuse the same P2 saturation state computed above for the
            #  ideal outlet -- see the comment there.
            actual = state_ph_smooth(m, P2, h_out, name=f'out{name}',
                                     initial_h=actual0.h, initial_P=actual0.P,
                                     sat=sat_P2)
            self.T_out_C = m.Intermediate(actual.T - 273.15)
            self.quality = getattr(actual, 'quality', None)

        self.dh_act = m.Intermediate(h_in - h_out)
        self.power_kW = m.Intermediate(mdot*self.dh_act)

    def initialise(self, remote=False):
        """Set the starting state so a run begins already settled."""
        from gekko import GEKKO
        args = [_numeric(v) for v in (self.mdot, self._T1_C, self._P1,
                                      self._P2, self.eta_is)]
        if None in args:
            raise TypeError(
                'initialise() could not read a number from every input. '
                'Set .states by hand instead, from a plant-wide steady-state '
                'solve or from measurements.')
        m0 = GEKKO(remote=remote)
        m0.options.IMODE = 1
        ref = Turbine(m0, *args)
        m0.solve(disp=False)
        for y in self.states:
            y.value = ref.dh_act.value[0]
        return self


def steady_numeric(mdot=MDOT, p2_bar=POUT, T1_C=TIN, P1_bar=PIN, eta=ETA):
    """Steady state at one operating point, without building a GEKKO model.

    Identical arithmetic to the Turbine block -- isentropic expansion to the
    outlet pressure, the isentropic drop scaled by eta, the outlet state taken
    from (P2, h2) -- evaluated directly against the same IAPWS-95 routines. The
    dome-aware helpers are used on both legs, so a wet outlet is handled by the
    lever rule rather than falling over.

    This exists because spinning up a GEKKO model to evaluate one algebraic
    expansion costs about eight hundred milliseconds, which is fine for seeding
    a rig once and ruinous for a sweep: a 500-point envelope check spent 99% of
    its time in solver startup. Here it is under a millisecond. steady() below
    is kept as the reference the fast path is checked against.
    """
    p1_kPa, p2_kPa = P1_bar*100.0, p2_bar*100.0
    inlet = solve_tp_numeric(T1_C + 273.15, p1_kPa, 'vapor')
    h2s = _h_numeric_any_phase(p2_kPa, inlet.s)
    dh_is = inlet.h - h2s
    dh_act = eta*dh_is
    h2 = inlet.h - dh_act
    return {'power': mdot*dh_act, 'T_out': _T_numeric_any_phase(p2_kPa, h2) - 273.15,
            'dh_act': dh_act, 'dh_is': dh_is,
            'T2s': _T_numeric_any_phase(p2_kPa, h2s) - 273.15}


def steady(mdot=MDOT, p2_bar=POUT, T1_C=TIN, P1_bar=PIN, eta=ETA):
    """Steady state through a one-shot GEKKO solve. The reference path."""
    from gekko import GEKKO
    m0 = GEKKO(remote=False)
    m0.options.IMODE = 1
    t0 = Turbine(m0, mdot, T1_C, P1_bar, p2_bar, eta)
    m0.solve(disp=False)
    return {'power': t0.power_kW.value[0], 'T_out': t0.T_out_C.value[0],
            'dh_act': t0.dh_act.value[0], 'dh_is': t0.dh_is.value[0],
            'T2s': t0.T2s_C.value[0]}


def demo():
    """Steady state, then a step in back pressure."""
    import numpy as np
    import matplotlib.pyplot as plt
    from gekko import GEKKO

    ss = steady()
    print('=' * 58)
    print('  STEADY STATE')
    print('=' * 58)
    for lbl, v, u in [('inlet T', TIN, 'C'), ('inlet P', PIN, 'bar'),
                      ('outlet P', POUT, 'bar'), ('mass flow', MDOT, 'kg/s'),
                      ('eta_is', ETA, '-')]:
        print(f'  {lbl:22s}{v:12.3f}  {u}')
    print('  ' + '-'*54)
    for lbl, key, u in [('T2s isentropic', 'T2s', 'C'), ('T_out actual', 'T_out', 'C'),
                        ('dh_is', 'dh_is', 'kJ/kg'), ('dh_act', 'dh_act', 'kJ/kg'),
                        ('power', 'power', 'kW')]:
        print(f'  {lbl:22s}{ss[key]:12.3f}  {u}')
    print('  ' + '-'*54)
    print(f'  {"stage volume":22s}{V_STAGE:12.3f}  m3')
    print(f'  {"tau = V/Vdot":22s}{TAU:12.3f}  s')

    t_end = 10.0*TAU
    m = GEKKO(remote=False)
    m.options.IMODE = 4
    m.options.NODES = 4
    m.time = np.linspace(0.0, t_end, 401)

    p2 = m.Param(value=np.where(m.time < 0.2*t_end, POUT, 0.85*POUT))
    td = Turbine(m, MDOT, TIN, PIN, p2, ETA, tau=TAU, fixed_inlet=True, name='t1')
    h_in = solve_tp_numeric(TIN + 273.15, 100.0*PIN, 'vapor').h
    td.states[0].value = h_in - ss['dh_act']
    m.solve(disp=False)

    print('\n' + '=' * 58)
    print(f'  STEP  P2 {POUT:.0f} -> {0.85*POUT:.0f} bar at t = {0.2*t_end:.1f} s')
    print('=' * 58)
    print(f'  {"t [s]":>8s}{"dh_act":>11s}{"power":>11s}{"T_out":>10s}')
    tm = np.array(m.time)
    for frac in (0.0, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0):
        k = int(abs(tm - frac*t_end).argmin())
        print(f'  {tm[k]:8.2f}{td.dh_act.value[k]:11.2f}'
              f'{td.power_kW.value[k]:11.1f}{td.T_out_C.value[k]:10.2f}')

    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for a, series, colour, ylab in [
            (ax[0], td.dh_act, 'g', 'dh_act [kJ/kg]'),
            (ax[1], td.power_kW, 'r', 'power [kW]'),
            (ax[2], td.T_out_C, 'b', 'T_out [C]')]:
        a.plot(tm, series.value, colour + '-', lw=2)
        a.set_ylabel(ylab)
        a.grid(alpha=0.3)
    ax[0].set_title(f'One stage, one time constant  tau = V/Vdot = {TAU:.2f} s',
                    fontsize=10, loc='left')
    ax[2].set_xlabel('time [s]')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo()
