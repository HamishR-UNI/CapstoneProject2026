"""Model predictive control of the plant (tracking and economic versions).

Added 2026-09-27 (plan: docs/09_economics_rto_mpc_plan.md, Phase 3).

CONTROL MODEL. The MPC does not carry the full rig (turbine surrogates, IAPWS fits, every lag):
as a simulator that already costs ~1.3 s per 10 s interval, and the MPC has to look ~25 min
ahead (boiler dead time 305 s + lag 601 s). Instead it carries a reduced model built from the
SAME pieces and constants the rig uses, simplified where the effect is small:

    fuel command -> dispatch (5 s) -> dead-time chain (304.6 s) -> lag (601 s) = burned fuel fe
    steam = boiler.duty(fe)/dh (same fit as the rig);  flue gas = fe * GAS_PER_FUEL
    exchanger hot water = fit of flue_gas_water_hx.steady_state over fe (at the HW setpoint)
    HP2 command -> compressor lag (10 s dead + 20 s) -> link pipe (20 s) -> HP1 evaporator
    HP1: condenser duty = evaporator heat / ratio(subcooling) -> 10 s dead + 20 s lag;
         LPS and hot water out with their own lags; work = duty x (1 - ratio)
         (ratio, sub_frac, dh_lps: heatpump_dyn.cycle_constants at the LPS target, fitted in dt)
    turbine power = steam x specific work at the MPS/LPS targets (turbine_model.steady_numeric)
    headers: the rig's own mass balances and header_closures.flows (site delivery law, letdown,
             dump, relief) with a smooth abs, and the rig's own capacitances.

Simplifications (stated for the report): the static boiler map is applied after the lag (the
rig applies it before; the map is near-linear); the exchanger's own ~s lags, the split valve's
2 s stroke and the turbine stage lags are dropped; turbine specific work is held at the target
pressures; the turbine 2 cut-off below 1% is not modelled.

FEEDBACK. An observer runs the same model in numpy alongside the plant, driven by the commands
the plant was actually given (every sample), so the boiler's 5-minute dead-time chain always
holds the real fuel history. Each interval the MPC starts from the observer's states and the
MEASURED header pressures, and output biases (steam, HP1 LPS, hot water, net power: measured
minus observer) are held constant over the horizon -- the standard offset-free correction.

VERSIONS.
    tracking ('mpc')         hold the pressure targets and the HW minimum; HP2 duty and
                             subcooling are pulled towards the RTO's targets but may be used
                             dynamically (e.g. HP1 covering LPS while the boiler catches up).
    economic ('economic_mpc') maximise the margin over the horizon (economics.margin's terms,
                             priced with the forecast from the looping price) with the same
                             pressure and HW terms as soft constraints.
Fuel cost is charged on the BURNED fuel (fe), not the command, so the cost lands when the
steam it buys arrives: without that, a finite horizon sees fuel cost now and the benefit after
the horizon and under-fires at the end of every prediction. A terminal term asks the fuel
command to be at rest (equal to fe) at the horizon end.

If a solve fails or takes longer than FAIL_SECS, OperatorRig falls back to Feedforward+PID for
that interval and counts it.
"""
import math
import time

import numpy as np

import boiler as blr
import dynamics_params as dp
import economics as econ
import flue_gas_water_hx as hx
import header_closures as headers
import heatpump_dyn as hpp
import heatpump_secondary as hps
import plant_rig as pr
import turbine_model as tmod

HORIZON_S = (0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 660, 900, 1200, 1500)
FAIL_SECS = 8.0

#  Chain structure (theta, tau, n) per path -- shared by the GEKKO model and the observer.
PATHS = dict(
    fuel=(dp.THETA_DISPATCH_S + blr.THETA_FUEL, blr.TAU_Q, 6),
    hp2=(hps.THETA_DUTY if hasattr(hps, 'THETA_DUTY') else dp.THETA_COMP_S,
         hps.TAU_DUTY if hasattr(hps, 'TAU_DUTY') else dp.TAU_COMP_S, 3),
    link=(pr.THETA_LINK, 0.0, 3),
    duty=(hpp.THETA_DUTY, hpp.TAU_DUTY, 3),
    lps=(hpp.THETA_LPS, hpp.TAU_LPS, 2),
    hw=(hpp.THETA_HW, hpp.TAU_HW, 3),
    split=(dp.THETA_DISPATCH_S, dp.TAU_VALVE_S, 2),
)

DH_STEAM = float(blr.dh_steam(blr.P_HP_BAR))
DT_LO, DT_HI = pr.DT_SUB_LO, pr.DT_SUB_HI
F_SCALE = 20.0


def _taus(theta, tau, n):
    return [theta/n]*n + ([tau] if tau else [])


# ---------------------------------------------------------------------------------------------
#  Static fits at the site's targets
# ---------------------------------------------------------------------------------------------
class Fits:
    """Everything that depends only on the site targets (MPS/LPS pressure, HW temperature)."""

    def __init__(self, p_mps, p_lps, t_hw):
        self.key = (round(p_mps, 4), round(p_lps, 4), round(t_hw, 4))
        pl_hp = float(np.clip(p_lps, *hpp.P_LPS_RANGE))
        dts = np.linspace(DT_LO, DT_HI, 44)
        x = (dts - DT_LO)/(DT_HI - DT_LO)
        cc = [hpp.cycle_constants(pl_hp, float(d)) for d in dts]
        self.c_inv_ratio = np.polyfit(x, [1.0/c['ratio'] for c in cc], 6)
        self.c_work = np.polyfit(x, [1.0 - c['ratio'] for c in cc], 6)
        self.c_lps = np.polyfit(x, [(1.0 - c['sub_frac'])/c['dh_lps'] for c in cc], 6)
        self.c_sub = np.polyfit(x, [c['sub_frac'] for c in cc], 6)
        self.dh_hw = float(hpp.dh_hw(t_hw))
        fs = np.linspace(0.0, F_SCALE, 41)
        mhx = [hx.steady_state(f*blr.GAS_PER_FUEL, float(blr.stack_temp(f)), t_hw)['m_water']
               if f > 0 else 0.0 for f in fs]
        self.c_mhx = np.polyfit(fs/F_SCALE, mhx, 6)
        self.q_rec_per_kg = hx.CP_WATER*max(hx.T_W_IN - hpp.T_RECYCLE_EVAP_OUT, 0.0)
        t1 = tmod.steady_numeric(1.0, p_mps, pr.T_HP, pr.P_HP, tmod.ETA)
        self.dh1 = t1['dh_act']
        self.dh2 = tmod.steady_numeric(1.0, p_lps, t1['T_out'], p_mps, pr.ETA2)['dh_act']
        self.w_sec = 1.0/hps.COP_SEC
        self.p_mps, self.p_lps, self.t_hw = p_mps, p_lps, t_hw


def _poly(c, x):
    y = c[0]
    for k in c[1:]:
        y = y*x + k
    return y


# ---------------------------------------------------------------------------------------------
#  Observer (numpy copy of the control model, driven by the applied commands)
# ---------------------------------------------------------------------------------------------
class Observer:
    def __init__(self, row, fits):
        f0 = row['mfuel_act']
        self.x = {}
        self._set('fuel', f0)
        self._set('hp2', row['Q_cond_sec'])
        self._set('link', row['Q_cond_sec'])
        self._set('split', row['frac_t2_act'])
        self.fits = fits
        qc = row['Q_cond_pri']
        self._set('duty', qc)
        self._set('lps', row['m_lps_hp'])
        self._set('hw', row['m_hw_hp'])

    def _set(self, name, v):
        self.x[name] = [float(v)]*len(_taus(*PATHS[name]))

    def _advance(self, name, u, dt):
        xs = self.x[name]
        inp = u
        for i, tau in enumerate(_taus(*PATHS[name])):
            a = 1.0 - math.exp(-dt/tau) if tau > 0 else 1.0
            xs[i] += (inp - xs[i])*a
            inp = xs[i]
        return xs[-1]

    def out(self, name):
        return self.x[name][-1]

    def static(self, dt_sub):
        """Steady-state targets of the HP1 chains at the current observer inputs."""
        f = self.fits
        fe = self.out('fuel')
        xdt = (min(max(dt_sub, DT_LO), DT_HI) - DT_LO)/(DT_HI - DT_LO)
        m_hx = max(_poly(f.c_mhx, fe/F_SCALE), 0.0)
        q_evap = fe*blr.GAS_PER_FUEL*hpp.DH_GAS_HP + m_hx*f.q_rec_per_kg + self.out('link')
        qc_ss = q_evap*_poly(f.c_inv_ratio, xdt)
        return dict(fe=fe, m_hx=m_hx, qc_ss=qc_ss, xdt=xdt)

    def update(self, row, dt=pr.DT):
        """Advance by one plant sample using the commands that sample was run with."""
        self._advance('fuel', row['mfuel_cmd'], dt)
        self._advance('hp2', row['qsec_cmd'], dt)
        self._advance('link', self.out('hp2'), dt)
        self._advance('split', row['frac_t2_cmd'], dt)
        s = self.static(row['dtsub_cmd'])
        self._advance('duty', s['qc_ss'], dt)
        qc = self.out('duty')
        self._advance('lps', qc*_poly(self.fits.c_lps, s['xdt']), dt)
        self._advance('hw', qc*_poly(self.fits.c_sub, s['xdt'])/self.fits.dh_hw, dt)

    def predictions(self, dt_sub, q_sec):
        f = self.fits
        s = self.static(dt_sub)
        fe = s['fe']
        steam = float(blr.duty(fe))/DH_STEAM
        w1 = self.out('duty')*_poly(f.c_work, s['xdt'])
        split = self.out('split')
        pn = steam*f.dh1 + steam*split*f.dh2 - w1 - self.out('hp2')*f.w_sec
        return dict(steam=steam, m_lps_hp=self.out('lps'), m_hw_total=s['m_hx'] + self.out('hw'),
                    power_net=pn)


# ---------------------------------------------------------------------------------------------
#  The controller
# ---------------------------------------------------------------------------------------------
class MPC:
    #  Objective weights (per hour of horizon, see the time weights below). Pressure errors are
    #  scaled by 1 bar (MPS) and 0.1 bar (LPS); money in thousands of NZ$/h.
    W_MPS = 5.0
    W_LPS = 5.0
    W_HW = 2.0            # per (kg/s)^2 of hot-water shortfall
    W_HW_LIN = 5.0        # per kg/s of hot-water shortfall
    W_Q_TRACK = 2.0       # tracking: per (5 MW)^2 away from the RTO's HP2 duty
    W_DT_TRACK = 2.0      # tracking: per (10 K)^2 away from the RTO's subcooling
    W_TERM = 20.0         # terminal: fuel command at rest (= burned fuel), per (kg/s)^2
    DCOST = dict(fuel=0.5, split=5.0, dt=0.002, q=2e-9)

    def __init__(self, rig, version='tracking'):
        self.rig = rig
        self.version = version
        self.fits = None
        self.m = None
        self.observer = None
        self.solves = self.failures = 0
        self.solve_secs = []
        self.last = None
        self.demand_forecast = None     # optional callable t -> dict(d_mps, d_lps, hw_min)

    # ---- observer: called by OperatorRig after every step, in every mode ----
    def observe(self, rows):
        fits = self._fits(rows[-1]['pmps_cmd'], rows[-1]['plps_cmd'], rows[-1]['twout_cmd'])
        if self.observer is None:
            self.observer = Observer(rows[0], fits)
        self.observer.fits = fits
        for r in rows:
            self.observer.update(r)

    def _fits(self, pm, pl, tw):
        key = (round(pm, 4), round(pl, 4), round(tw, 4))
        if self.fits is None or self.fits.key != key:
            self.fits = Fits(pm, pl, tw)
            self.m = None          # rebuild the GEKKO model around the new fits
        return self.fits

    # ---- GEKKO model ----
    def _build(self):
        from gekko import GEKKO
        f = self.fits
        m = GEKKO(remote=False)
        m.time = np.array(HORIZON_S, dtype=float)
        m.options.IMODE = 6
        m.options.NODES = 2
        m.options.SOLVER = 3
        m.options.MAX_ITER = 300
        m.options.TIME_SHIFT = 0
        m.options.CV_TYPE = 2
        sabs = lambda v: (v*v + 1e-6)**0.5

        P = {}
        for name, val in (('d_mps', pr.D_MPS_0), ('d_lps', pr.D_LPS_0), ('hw_min', pr.HW_MIN_0),
                          ('price_exp', 120.0), ('price_imp', 180.0), ('biomass', 12.0),
                          ('p_mps_price', 45.0), ('p_lps_price', 40.0), ('p_hw_price', 2.0),
                          ('b_steam', 0.0), ('b_lps', 0.0), ('b_hw', 0.0), ('b_pn', 0.0),
                          ('q_ref', pr.Q_SEC_0), ('dt_ref', pr.DT_SUB_0),
                          ('econ_on', 0.0), ('track_on', 1.0)):
            P[name] = m.Param(value=val, name=f'p_{name}')
        wt = np.diff(np.r_[HORIZON_S, HORIZON_S[-1]])/3600.0   # hours each point stands for
        wt[0] = 0.0
        P['wt'] = m.Param(value=wt, name='p_wt')
        term = np.zeros(len(HORIZON_S)); term[-1] = 1.0
        P['term'] = m.Param(value=term, name='p_term')

        U = {}
        for name, lo, hi, v in (('fuel', pr.MFUEL_LO, pr.MFUEL_HI, 7.8), ('split', 0.0, 1.0, 0.68),
                                ('dt', DT_LO, DT_HI, 10.0), ('q', pr.Q_SEC_LO, pr.Q_SEC_HI, 5000.0)):
            u = m.MV(value=v, lb=lo, ub=hi, name=f'u_{name}')
            u.STATUS = 1
            u.DCOST = self.DCOST[name]
            U[name] = u

        S = {}

        def chain(name, u):
            theta, tau, n = PATHS[name]
            y, states = tmod.lag_chain(m, u, theta=theta, tau=tau, n_lag=n, y0=1.0, name=f'c_{name}')
            S[name] = states
            return y

        fe = chain('fuel', U['fuel'])
        q2 = chain('hp2', U['q'])
        q2_hp1 = chain('link', q2)
        split = chain('split', U['split'])
        xdt = m.Intermediate((U['dt'] - DT_LO)/(DT_HI - DT_LO))
        m_hx = m.Intermediate(_poly(f.c_mhx, fe/F_SCALE))
        q_evap = m.Intermediate(fe*blr.GAS_PER_FUEL*hpp.DH_GAS_HP + m_hx*f.q_rec_per_kg + q2_hp1)
        qc = chain('duty', m.Intermediate(q_evap*_poly(f.c_inv_ratio, xdt)))
        m_lps_hp = m.Intermediate(chain('lps', m.Intermediate(qc*_poly(f.c_lps, xdt))) + P['b_lps'])
        m_hw_hp = chain('hw', m.Intermediate(qc*_poly(f.c_sub, xdt)/f.dh_hw))
        m_hw_tot = m.Intermediate(m_hx + m_hw_hp + P['b_hw'])
        steam = m.Intermediate(blr.duty(fe)/DH_STEAM + P['b_steam'])

        pm = m.Var(value=40.0, name='pm')
        pl = m.Var(value=3.0, name='pl')
        S['pm'], S['pl'] = [pm], [pl]
        fl = headers.flows(pm, pl, P['d_mps'], P['d_lps'], absolute=sabs,
                           target_mps=f.p_mps, target_lps=f.p_lps)
        m.Equation(self.rig.C_MPS*pm.dt() == steam*(1 - split) - fl['mps'] - fl['letdown'])
        m.Equation(self.rig.C_LPS*pl.dt() == steam*split + m_lps_hp + fl['letdown']
                   - fl['lps'] - fl['dump'])

        w1 = m.Intermediate(qc*_poly(f.c_work, xdt))
        pn = m.Intermediate((steam*f.dh1 + steam*split*f.dh2 - w1 - q2*f.w_sec + P['b_pn'])/1000.0)
        e_exp = m.Var(value=8.0, lb=0.0, name='e_exp')
        e_imp = m.Var(value=0.0, lb=0.0, name='e_imp')
        m.Equation(pn == e_exp - e_imp)
        sh = m.Var(value=0.0, lb=0.0, name='hw_slack')
        m.Equation(m_hw_tot + sh >= P['hw_min'])

        smin = lambda a, b: 0.5*(a + b - ((a - b)**2 + 1e-4)**0.5)
        revenue = 3.6*(smin(fl['mps'], P['d_mps'])*P['p_mps_price']
                       + smin(fl['lps'], P['d_lps'])*P['p_lps_price']
                       + (P['hw_min'] - sh)*P['p_hw_price'])
        fuel_cost = fe*econ.GJ_PER_KG_FUEL*3600.0*P['biomass']
        elec = e_exp*P['price_exp'] - e_imp*P['price_imp']
        margin = m.Intermediate((revenue + elec - fuel_cost)/1000.0)   # k NZ$/h

        m.Minimize(P['wt']*(self.W_MPS*(pm - f.p_mps)**2 + self.W_LPS*((pl - f.p_lps)/0.1)**2
                            + self.W_HW*sh**2 + self.W_HW_LIN*sh
                            - P['econ_on']*margin
                            + P['track_on']*(self.W_Q_TRACK*((U['q'] - P['q_ref'])/5000.0)**2
                                             + self.W_DT_TRACK*((U['dt'] - P['dt_ref'])/10.0)**2)))
        m.Minimize(P['term']*self.W_TERM*(U['fuel'] - fe)**2)

        self.m, self.P, self.U, self.S = m, P, U, S
        self.out = dict(pm=pm, pl=pl, steam=steam, m_lps_hp=m_lps_hp, m_hw_tot=m_hw_tot,
                        pn=pn, margin=margin, fe=fe, split=split)

    # ---- one control move ----
    def solve(self, row, site, prices, t, rto_decision=None):
        """Commands for the next interval: dict(fuel, split, dt, q). Raises on failure."""
        f = self._fits(site['p_mps'], site['p_lps'], site['t_hw'])
        ob = self.observer
        ob.fits = f
        if self.m is None:
            self._build()
        m, P, U, S = self.m, self.P, self.U, self.S

        #  initial states: observer chains, measured pressures
        for name in ('fuel', 'hp2', 'link', 'split', 'duty', 'lps', 'hw'):
            for v, x in zip(S[name], ob.x[name]):
                v.value = x
        S['pm'][0].value = row['pmps_act']
        S['pl'][0].value = row['plps_act']
        #  biases: measured minus observer
        pred = ob.predictions(row['dtsub_cmd'], row['qsec_cmd'])
        P['b_steam'].value = row['steam'] - pred['steam']
        P['b_lps'].value = row['m_lps_hp'] - pred['m_lps_hp']
        P['b_hw'].value = row['m_hw_total'] - pred['m_hw_total']
        P['b_pn'].value = (row['power'] - row['W_hp_total']) - pred['power_net']
        #  disturbances and prices. Demands are held at their current values over the horizon
        #  (the controllers get no advance knowledge), unless a demand_forecast(t) -> dict
        #  is set -- the dedicated "forecast demand step" scenario.
        if self.demand_forecast is None:
            P['d_mps'].value, P['d_lps'].value, P['hw_min'].value = site['d_mps'], site['d_lps'], site['hw_min']
        else:
            fc = [self.demand_forecast(t + dt) for dt in HORIZON_S]
            P['d_mps'].value = [x['d_mps'] for x in fc]
            P['d_lps'].value = [x['d_lps'] for x in fc]
            P['hw_min'].value = [x['hw_min'] for x in fc]
        pe = prices.electricity
        P['price_exp'].value = pe.forecast(t, HORIZON_S)
        P['price_imp'].value = [p + pe.import_adder for p in pe.forecast(t, HORIZON_S)]
        P['biomass'].value = prices.biomass_per_gj
        P['p_mps_price'].value = prices.mps_steam_per_t
        P['p_lps_price'].value = prices.lps_steam_per_t
        P['p_hw_price'].value = prices.hot_water_per_t
        economic = self.version == 'economic'
        P['econ_on'].value = 1.0 if economic else 0.0
        P['track_on'].value = 0.0 if economic else 1.0
        if rto_decision is not None:
            P['q_ref'].value = rto_decision['q_sec']
            P['dt_ref'].value = max(rto_decision['dt_floor'], rto_decision.get('dt_sub', 0.0))
        #  MVs start from what is applied now
        U['fuel'].value = row['mfuel_cmd']
        U['split'].value = row['frac_t2_cmd']
        U['dt'].value = row['dtsub_cmd']
        U['q'].value = row['qsec_cmd']

        t0 = time.perf_counter()
        #  IPOPT first; on a failure retry once with APOPT (different algorithm, often
        #  succeeds where the other stalls) before OperatorRig falls back to FF+PID.
        ok = False
        for solver in (3, 1):
            m.options.SOLVER = solver
            try:
                m.solve(disp=False)
                ok = m.options.APPSTATUS == 1
            except Exception:
                ok = False
            if ok:
                break
        m.options.SOLVER = 3
        secs = time.perf_counter() - t0
        self.solves += 1
        self.solve_secs.append(secs)
        if not ok:
            self.failures += 1
            raise RuntimeError('MPC solve did not converge')
        u = dict(fuel=float(U['fuel'].NEWVAL), split=float(U['split'].NEWVAL),
                 dt=float(U['dt'].NEWVAL), q=float(U['q'].NEWVAL))
        self.last = dict(u, secs=secs, pm_pred=list(self.out['pm'].value),
                         pl_pred=list(self.out['pl'].value), t=t,
                         margin_pred=list(self.out['margin'].value))
        return u

    def cleanup(self):
        if self.m is not None:
            try:
                self.m.cleanup()
            except Exception:
                pass
