"""
Dynamic flue gas heat pump (secondary machine) -- water-sourced booster feeding
the primary's evaporator.

This machine's evaporator draws off a plain water source at the same temperature
as the boiler's own feedwater (blr.T_FW), an effectively unlimited, fixed-temperature
reservoir -- the same assumption the boiler already makes about its own feedwater
supply. Its condenser rejects that lifted heat as a pure duty (kW), handed straight
to heatpump_dyn.py's evaporator alongside the flue gas: no water-side stream is
modelled on the condenser side, since the heat goes to the next machine, not to an
external customer.

Both the source and sink temperatures are fixed design constants (unlike the primary,
which has a live flue-gas-side input), so the whole cycle collapses to fixed numbers
computed once with CoolProp -- no polynomial fits needed anywhere in this module.

Dispatched independently of the primary: it draws off an unlimited source, so it
isn't capacity-linked to anything else in the plant. Its only relationship to the
primary is that its condenser duty is one of the two terms feeding the primary's
evaporator-availability check (see heatpump_dyn.py's dQ_shortfall) -- reported, not
enforced.

Requires CoolProp and gekko.
"""

import numpy as np
from CoolProp.CoolProp import PropsSI
from gekko import GEKKO

import boiler as blr
from turbine_model import lag_chain
from heatpump_dyn import (REF, ETA_IS, DT_PINCH_EVAP, DT_PINCH_COND, DT_SUPERHEAT,
                          T_EVAP_PRIMARY, THETA_DUTY, TAU_DUTY, N_LAG, TURNDOWN)

# ----------------------------------------------------------------------
# Plant specification and cycle assumptions
# ----------------------------------------------------------------------

T_SRC_SEC = blr.T_FW                              # degC, same source as the boiler feedwater

# Must reject a few degrees above whatever the primary's evaporator is running at,
# reusing the same condenser pinch heatpump_dyn.py uses for its own LPS condenser
# rather than inventing a second constant for it.
T_COND_SEC = T_EVAP_PRIMARY + DT_PINCH_COND        # degC

T_EVAP_SEC = T_SRC_SEC - DT_PINCH_EVAP - DT_SUPERHEAT   # degC

# This machine is now the plant's single heat-pump manipulated variable. The
# primary is dependent on it (heatpump_dyn.q_cond_from_sources), so this slider
# is what moves LPS between the boiler and the heat pump -- the plant's one
# fuel-versus-electricity lever. Its source is an unlimited 20 degC water
# supply, which is exactly what makes it a lever: it is the only energy input
# on the plant that does not scale with firing rate, so turning it up really
# does substitute for burning more biomass rather than just following it.
#
# 20 MW drives the primary to about 53 MW, which is roughly 19 kg/s of LPS --
# a bit over half a typical header. Sized as a demand range to explore, not as
# a machine anyone has quoted.
Q_SEC_MCR = 20000.0                               # kW
#  No turndown floor. A real machine below minimum stable load would cycle off
#  and this model has no on/off state, but zero has to be reachable: the whole
#  point of the lever is being able to take the heat pump out of the LPS mix
#  entirely when electricity is dear. See the note on COP_sec below for how the
#  0/0 problem that used to force a floor is avoided.
Q_SEC_LO, Q_SEC_HI = 0.0, Q_SEC_MCR

Q_SEC_0 = 5000.0


def _cycle_points():
    """Compressor inlet/discharge/liquid enthalpy at the fixed source and sink --
    plain numbers, since neither temperature moves."""
    p_evap = PropsSI("P", "T", T_EVAP_SEC+273.15, "Q", 1, REF)
    p_cond = PropsSI("P", "T", T_COND_SEC+273.15, "Q", 1, REF)
    T1 = T_EVAP_SEC + DT_SUPERHEAT
    h1 = PropsSI("H", "P", p_evap, "T", T1+273.15, REF)/1e3
    s1 = PropsSI("S", "P", p_evap, "T", T1+273.15, REF)/1e3
    h2s = PropsSI("H", "P", p_cond, "S", s1*1e3, REF)/1e3
    h2 = h1 + (h2s-h1)/ETA_IS
    h3 = PropsSI("H", "P", p_cond, "Q", 0, REF)/1e3      # saturated liquid, no subcool needed
    return h1, h2, h3


H1_SEC, H2_SEC, H3_SEC = _cycle_points()
COP_SEC = (H2_SEC - H3_SEC) / (H2_SEC - H1_SEC)


#  Specific work per kilogram of condenser duty. Both temperatures are fixed
#  design constants, so this is a constant and W is strictly proportional to Q.
#  That is what makes zero duty safe: W goes to zero linearly with Q instead of
#  being recovered from a Q/COP division that is 0/0 there. COP itself is only
#  ever a reported number, so it is the one thing that needs guarding.
DW_SEC = (H2_SEC - H1_SEC) / (H2_SEC - H3_SEC)     # kW work per kW condenser


def steady_state(Q):
    """Reference steady state, straight from the fixed cycle points.

    Work is taken from DW_SEC rather than from Q/COP so that Q = 0 is a legal
    operating point -- off, drawing nothing. COP is reported as the cycle's own
    constant value in that case, since a ratio of two zeros carries no
    information and NaN propagates into every plot that touches it.
    """
    W = Q * DW_SEC
    return dict(Q_cond_sec=Q, m_ref=Q/(H2_SEC - H3_SEC), W_comp_sec=W,
                COP_sec=COP_SEC)


ss0 = steady_state(Q_SEC_0)


def build_dynamics(m, Q_sec_cmd, y0=None):
    """Attach the secondary heat pump to a GEKKO model.

    Q_sec_cmd   GEKKO Param/Var/MV, the condenser duty commanded.
    y0          Reference steady_state() dict for seeding; defaults to ss0.

    Returns Q_cond_sec, W_comp_sec, COP_sec and the state list needing reseeding
    after a failed solve.
    """
    y0 = ss0 if y0 is None else y0

    Q_cond_sec, states = lag_chain(m, Q_sec_cmd, theta=THETA_DUTY, tau=TAU_DUTY,
                                   n_lag=N_LAG, y0=y0['Q_cond_sec'], name='hps_duty')

    m_ref = m.Intermediate(Q_cond_sec/(H2_SEC-H3_SEC))
    W_comp_sec = m.Intermediate(Q_cond_sec*DW_SEC)
    #  Constant by construction (both temperatures are fixed), written as an
    #  Intermediate so the returned dict keeps the same shape as before. NOT
    #  Q/W -- that is 0/0 at the bottom of the range, which is now reachable.
    COP_sec = m.Intermediate(COP_SEC + 0.0*Q_cond_sec)

    return dict(Q_cond_sec=Q_cond_sec, W_comp_sec=W_comp_sec, COP_sec=COP_sec,
               states=states)


# ----------------------------------------------------------------------
# Standalone report / step test
# ----------------------------------------------------------------------

T_END = 1200.0
DT = 2.0
T_STEP = 300.0


def main():
    m = GEKKO(remote=False)
    m.time = np.arange(0.0, T_END+DT, DT)
    n_t = len(m.time)

    a = np.full(n_t, Q_SEC_LO + 0.3*(Q_SEC_HI-Q_SEC_LO))
    a[m.time >= T_STEP] = Q_SEC_LO + 0.8*(Q_SEC_HI-Q_SEC_LO)
    Q_cmd = m.Param(value=a)

    d = build_dynamics(m, Q_cmd)

    m.options.IMODE = 4
    m.options.NODES = 4
    m.options.SOLVER = 3
    m.solve(disp=False)

    print("FIXED CYCLE POINTS")
    print(f"   T_src / T_evap        {T_SRC_SEC:9.2f} / {T_EVAP_SEC:6.2f} degC")
    print(f"   T_cond                {T_COND_SEC:9.2f} degC")
    print(f"   h1 / h2 / h3          {H1_SEC:9.2f} / {H2_SEC:6.2f} / {H3_SEC:6.2f} kJ/kg")
    print(f"   COP                   {COP_SEC:9.3f}")
    print(f"   Q_SEC_MCR             {Q_SEC_MCR:9.1f} kW")

    W = np.array(list(d['W_comp_sec'].value))
    Q = np.array(list(d['Q_cond_sec'].value))
    COP = np.array(list(d['COP_sec'].value))
    cmd = np.array(list(Q_cmd.value))

    print("\nSTEADY STATE CLOSURE  end of run against the reference steady_state()")
    ss = steady_state(cmd[-1])
    for key, arr, lab in [("W_comp_sec", W, "compressor (kW)"),
                          ("COP_sec", COP, "COP")]:
        a_, b = arr[-1], ss[key]
        print(f"   {lab:20}{a_:12.4f}{b:12.4f}   {100*(a_-b)/b:+8.4f} %")

    print("\nENERGY BALANCE  max |Q_cond - W - Q_evap| over the run "
          f"(Q_evap = Q_cond - W by construction, so this is exactly zero)")
    Q_evap = Q - W
    print(f"   {np.max(np.abs(Q - W - Q_evap)):.2e} kW")

    print("\nOPERATING POINTS")
    print(f"   {'':16}{'Q_cond (kW)':>13}{'W (kW)':>9}{'COP':>7}")
    for lab, i in [("start", 0), ("after step", len(cmd)-1)]:
        print(f"   {lab:16}{Q[i]:13.1f}{W[i]:9.1f}{COP[i]:7.3f}")


if __name__ == "__main__":
    main()
