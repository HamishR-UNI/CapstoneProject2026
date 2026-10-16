"""Real-time optimiser (RTO) over the Feedforward+PID layer.

Added 2026-09-27 (plan: docs/09_economics_rto_mpc_plan.md, Phase 2).

WHAT IT DECIDES. With the site's pressure targets and hot-water temperature fixed, the MPS and
LPS headers must each balance at steady state: two equations on the plant's four handles (fuel,
turbine 2 split, HP1 subcooling, HP2 duty). So only two are free. The RTO picks those two --
HP2 duty and a subcooling FLOOR -- to maximise the steady-state margin (economics.margin's
terms) at the current electricity price and site demands. The feedforward then computes the
fuel and split that go with them, and the PID trims. Pressures and HW temperature are site inputs
and are never moved here.

MODEL. excel_verify/python_side.plant_steady: the same static closure the feedforward uses,
matching the dynamic plant's own steady state to ~0.003% (so this RTO has an almost perfect
model -- a real one would not; noted for the report).

SEARCH. plant_steady has corners (clamps, the turbine 2 cut-off), so no gradients: a 9 x 6 grid
over (HP2 duty, subcooling floor), then a bounded Nelder-Mead refine from the best grid point.
~80 evaluations at ~90 ms each, ~7 s per optimisation.

DEMANDS ARE CONSTRAINTS. A candidate that leaves any demand short at steady state is rejected
(the site's demands are requirements, not something to trade for money). If no candidate meets
every demand, the one with the smallest total shortfall is taken.

WHEN IT RUNS. OperatorRig calls it at the start of a control interval when the electricity price
steps to a new half-hour, when any site input changes, or 5 simulated minutes after the last run;
otherwise the last answer stands. Synchronous, so runs are deterministic (the dashboard steps the
rig in a worker thread, so the window does not freeze during the few seconds).

MOVE LIMITS AND A MINIMUM GAIN (added after the first live run, 2026-09-27). The steady-state
optimum is nearly bang-bang (HP2 0 or ~11 MW, subcooling 2 or 45 K) and flips with small price
changes. Applied in one jump, the plant paid for it dynamically: cutting HP2 5 -> 0 MW left LPS
short by up to 6.6 kg/s for ~15 min (HP2 acts in ~20 s, the boiler takes ~15 min to replace
that LPS), and a subcooling drop 45 -> 2 K left hot water short, so the HW override over-fired
the boiler (margin 1,876 -> 368 NZ$/h). So, as industrial RTOs do: each run may move HP2 by at
most MOVE_Q and the subcooling floor by at most MOVE_DT from where they are (each move is still
applied as an instant step), and it only moves at all if the new point is worth at least
MIN_GAIN more than staying put. The RTO still ignores the transient itself -- that is the MPC's
job, and the comparison shows the difference.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'excel_verify'))
import python_side as ps    # noqa: E402

import economics as econ   # noqa: E402
import plant_rig as pr     # noqa: E402

Q_BOUNDS = (pr.Q_SEC_LO, pr.Q_SEC_HI)          # kW, HP2 duty
DT_BOUNDS = (pr.DT_SUB_LO, pr.DT_SUB_HI)       # K, subcooling floor
SHORT_TOL = 1e-3                               # kg/s: below this a demand counts as met
#  4000 kW / 8 K were tried first: LPS still ran 3-6 kg/s short for ~15 min after the first
#  moves (a cut in HP2 or a rise in subcooling takes LPS away in ~20 s; the boiler needs ~15 min
#  to replace it). Halved: slower to follow the price, smaller transient shortfalls.
MOVE_Q = 2000.0                                # kW, largest HP2 move per RTO run
MOVE_DT = 4.0                                  # K, largest subcooling-floor move per run
MIN_GAIN = 20.0                                # NZ$/h, smallest gain worth moving for
GRID_N = 5                                     # grid points per variable inside the move box


def steady_margin(c, prices, t):
    """Steady-state NZ$/h terms at site inputs c (python_side keys) plus q_sec and dt_floor,
    with the electricity price in force at simulation time t. Same terms as economics.margin."""
    r = ps.plant_steady(c)
    pe = prices.electricity
    fuel_cost = r['fuel']*econ.GJ_PER_KG_FUEL*3600.0*prices.biomass_per_gj
    pn = r['power_net']
    elec = pn/1000.0*(pe.export(t) if pn >= 0.0 else pe.import_(t))
    mps_short = max(-r['mps_gap'], 0.0)
    lps_short = max(-r['lps_gap'], 0.0)
    hw_short = r['hw_short']
    revenue = 3.6*((c['d_mps'] - mps_short)*prices.mps_steam_per_t
                   + (c['d_lps'] - lps_short)*prices.lps_steam_per_t
                   + (c['hw_min'] - hw_short)*prices.hot_water_per_t)
    return dict(margin_h=revenue + elec - fuel_cost, fuel=r['fuel'], dt_sub=r['dt_sub'],
                power_net=pn, shortfall=mps_short + lps_short + hw_short, raw=r)


class RTO:
    def __init__(self, prices, interval_s=300.0, move_limits=True):
        self.prices = prices
        self.interval_s = interval_s
        self.move_limits = move_limits
        self.last = None            # dict(q_sec, dt_floor, margin_h, ...)
        self._key = None
        self._t_last = -np.inf
        self.runs = 0
        self.history = []           # every decision, for the report

    def _key_for(self, site, t):
        return (self.prices.electricity.slot(t), self.prices.electricity.mode,
                round(self.prices.electricity.mean, 6), round(self.prices.biomass_per_gj, 6),
                tuple(round(site[k], 6) for k in ps.DRIVERS if k != 'q_sec'))

    def due(self, site, t):
        return (self.last is None or self._key_for(site, t) != self._key
                or t - self._t_last >= self.interval_s - 1e-9)

    def decide(self, site, t, q_now, dt_now=None):
        """HP2 duty and subcooling floor for site inputs `site` (d_mps, d_lps, hw_min, p_mps,
        p_lps, t_hw) at simulation time t. q_now/dt_now: where the plant is now (used for the
        first run; later runs move from the RTO's own last answer). Re-optimises only when
        due(); otherwise returns the standing answer."""
        if self.due(site, t):
            cur = ((self.last['q_sec'], self.last['dt_floor']) if self.last is not None
                   else (q_now, pr.DT_SUB_0 if dt_now is None else dt_now))
            self.last = self.optimise(site, t, current=cur if self.move_limits else None)
            self._key = self._key_for(site, t)
            self._t_last = t
            self.runs += 1
            self.history.append(self.last)
        return self.last

    def optimise(self, site, t, current=None):
        """Best (HP2 duty, subcooling floor). With `current` = (q, dt): searched only within the
        move limits around it, and returned unchanged unless the gain is at least MIN_GAIN."""
        t0 = time.perf_counter()
        base = {k: site[k] for k in ps.DRIVERS if k != 'q_sec'}
        evals = []
        if current is None:
            qb, db = Q_BOUNDS, DT_BOUNDS
        else:
            qb = (max(Q_BOUNDS[0], current[0] - MOVE_Q), min(Q_BOUNDS[1], current[0] + MOVE_Q))
            db = (max(DT_BOUNDS[0], current[1] - MOVE_DT), min(DT_BOUNDS[1], current[1] + MOVE_DT))

        def ev(q, dt):
            q = float(np.clip(q, *qb))
            dt = float(np.clip(dt, *db))
            m = steady_margin(dict(base, q_sec=q, dt_floor=dt), self.prices, t)
            evals.append((q, dt, m))
            return m

        def score(m):
            #  feasible (all demands met) always beats infeasible; then margin
            return (m['shortfall'] <= SHORT_TOL, m['margin_h'] if m['shortfall'] <= SHORT_TOL
                    else -m['shortfall'])

        stay = ev(*current) if current is not None else None
        n = GRID_N if current is not None else 9
        for q in np.linspace(*qb, n):
            for dt in np.linspace(*db, n if current is not None else 6):
                ev(q, dt)

        q0, dt0, _ = max(evals, key=lambda e: score(e[2]))
        if score(max((e[2] for e in evals), key=score))[0] and qb[1] > qb[0] and db[1] > db[0]:
            span_q, span_dt = qb[1] - qb[0], db[1] - db[0]

            def f(x):
                m = ev(qb[0] + x[0]*span_q, db[0] + x[1]*span_dt)
                return -m['margin_h'] if m['shortfall'] <= SHORT_TOL else 1e9 + m['shortfall']

            x0 = [(q0 - qb[0])/span_q, (dt0 - db[0])/span_dt]
            minimize(f, x0, method='Nelder-Mead', bounds=[(0, 1), (0, 1)],
                     options=dict(maxfev=25, xatol=1e-3, fatol=0.5,
                                  initial_simplex=[x0, [min(x0[0] + .1, 1), x0[1]],
                                                   [x0[0], min(x0[1] + .1, 1)]]))
        q, dt, m = max(evals, key=lambda e: score(e[2]))
        moved = True
        if stay is not None and score(stay)[0] and score(m)[0] \
                and m['margin_h'] - stay['margin_h'] < MIN_GAIN:
            q, dt, m, moved = current[0], current[1], stay, False
        return dict(q_sec=q, dt_floor=dt, margin_h=m['margin_h'], fuel=m['fuel'],
                    dt_sub=m['dt_sub'], power_net=m['power_net'], shortfall=m['shortfall'],
                    feasible=m['shortfall'] <= SHORT_TOL, moved=moved, evals=len(evals),
                    secs=time.perf_counter() - t0, t=t,
                    price=self.prices.electricity.export(t))
