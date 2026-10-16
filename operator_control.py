"""Operator interface to the plant: three named regulation-layer modes.

No plant equations changed -- this module only decides which existing
plant_rig.py function supplies each MV each step. The modes are:

    no_control       explicit operator commands (fuel, split, subcooling); no
                     _invert() call, no PI anywhere. The open-loop step-test /
                     PID-tuning harness and the no-regulation baseline row.
    feedforward      fuel and subcooling come from PlantRig._invert() every
                     step; turbine-2 split from live measured steam (see
                     step()); trims pinned at zero.
    feedforward_pid  _invert() sets the target every step; PI loops on measured
                     header pressure / hot-water delivery trim the residual on
                     top (see step()).

A fourth mode, `pid` (the same PI loops around a FIXED initial operating point,
no _invert() call, subcooling never moved), was removed 2026-09-25: it could not
settle onto the static closure whenever the hot-water minimum bound (subcooling
stuck at its initial value), and Feedforward+PID is the one controlled mode taken
forward. docs/07_regulation_models_report.md still describes the four-mode version.

Only the MV source branches by mode (see step() below); plant stepping and
every diagnostic field stay identical in shape across all modes so logging/
plotting never has to know which mode produced them.

The background MPS-pressure PI and the live-steam turbine-2 allocation rule are
bundled with the fuel PID and only run in `feedforward_pid` (for a clean
no-control baseline). `no_control` uses the explicit split command; `feedforward`
uses the same live-steam split (but no MPS PI trim on it). The LPS-surplus
subcooling protection is also `feedforward_pid` only.

Legacy mode names 'manual' (-> feedforward), 'auto' and 'auto_ff' (->
feedforward_pid) are still accepted so that scripts outside this task's scope
(excel_verify/, the older validation/analysis scripts) keep working unchanged,
but nothing in this module or plant_operator.py writes those names anymore.
"""
import math
import numpy as np
import plant_rig as pr
import header_closures as headers
import economics as econ
from pressure_feedback import PI, Tuning, MPS_TUNING
from scheduled_lps import RESPONSIVE_TUNING

#  PRESSURE LOOPS IN FLOW UNITS (2026-09-25). Both pressure PIs now act on the pressure
#  error converted into the steam-flow imbalance it represents: how much more or less
#  steam is leaving the header (site draw + letdown / vent) than would at the target,
#  from header_closures.flows(). The site draw is referenced to the target (its SITE
#  DELIVERY LAW), so every header self-regulates around any target, and in flow units
#  the loop gain no longer depends on the target, the demand, or which side of the
#  vent/letdown onset the pressure is on. This replaces the fixed-bar tunings (MPS
#  Kc=-0.0031/bar Ti=8.3 s; LPS Kc=0.85/bar Ti=607 s with a x4.36 error scaling below
#  3 bar), which had been tuned for the old 40/3-bar-only self-regulation and
#  limit-cycled at other targets once that was gone.
#
#  LPS: fuel (kg/s) per kg/s of LPS imbalance. First set to the old validated tuning's
#  effective gain at its design point (0.85 kg/s fuel per bar over the vent's 8 kg/s per
#  bar = 0.106, Ti 607 s); that recovered slowly after large disturbances (LPS still
#  0.02-0.19 bar off target an hour later). Retuned 2026-09-25 to SIMC for the fuel ->
#  LPS-imbalance process in flow units -- static gain K ~5 (kg/s steam per kg/s fuel),
#  boiler lag tau ~600 s, delay theta ~315 s (fuel bed + steam line 305 s, dispatch
#  5 s, half the 10 s control interval) -- with tau_c = theta: Kc = tau/(K*2*theta)
#  ~0.19, Ti = tau = 600 s. Confirmed against 7 other (Kc, Ti) pairs on three
#  disturbances (MPS demand + HP2 step, hot-water minimum step up/down, LPS demand
#  step): time back inside +/-0.02 bar 2189 s vs 3496 s and 2097 s vs 3057 s against
#  the old setting, and the best integrated error on the hot-water case. Shorter Ti
#  overshot; higher Kc was faster on some cases but worse on the hot-water case.
LPS_FLOW_TUNING = Tuning(0.19, 600.0)
LPS_FLOW_TUNING_OLD = Tuning(RESPONSIVE_TUNING.kc/headers.DEFAULT.dump_gain, RESPONSIVE_TUNING.ti)
#  MPS: turbine-2 flow moved (kg/s) per kg/s of MPS imbalance, scheduled every interval
#  from the header's own time constant tau = C_MPS / (self-regulation slope): SIMC for a
#  first-order process in flow units (static gain 1), Kc = tau/(tau_c + theta),
#  Ti = tau, with theta = 7.3 s (5 s dispatch delay, 2.3 s valve) plus half the control
#  interval, and tau_c = MPS_TAUC_RATIO*theta. At the default 10 s interval,
#  tau_c + theta = 3*12.3 ~ 37 s (was a fixed 36 s); computed from the actual interval
#  since 2026-09-25 so a longer batch (e.g. --batch 20 for speed) stays tuned.
MPS_THETA_FIXED_S = 7.3
MPS_TAUC_RATIO = 2.0
MPS_TD_S = 0.0     # MPS derivative time; the PID tuning test (pid_tuning_test.py) decides

EXTRA_FIELDS = ('operator_mode','fuel_basis','fuel_pi_request','fuel_hw_request',
 'fuel_mps_request','fuel_selector','allocation_base','allocation_trim',
 'lps_error','lps_control_error','lps_p','lps_i','hw_error','mps_error',
 'service_ok','control_dt','subcooling_calc','hw_binds','dt_clamped','hw_override_latched',
 'mps_guard_error','mps_guard_eligible','mps_control_error','mps_tau')+econ.ECON_FIELDS

MODES = ('no_control', 'feedforward', 'feedforward_pid', 'feedforward_pid_rto', 'mpc', 'economic_mpc')
MPC_MODES = {'mpc': 'tracking', 'economic_mpc': 'economic'}
_LEGACY_ALIASES = {'manual': 'feedforward', 'auto': 'feedforward_pid', 'auto_ff': 'feedforward_pid'}

def _mode(mode):
    mode = _LEGACY_ALIASES.get(mode, mode)
    if mode not in MODES:raise ValueError(mode)
    return mode

class OperatorRig(pr.PlantRig):
    LPS_PROTECT_ON_BAR=3.6      # protection starts at max(this, LPS target + 0.02)
    LPS_PROTECT_BAND_BAR=.25    # and is fully applied this much higher
    LPS_PROTECT_MAX_BAR=4.4     # ... but never later than this (header stop is 4.5)
    LPS_PROTECT_LEAD_S=60.0     # looks this far ahead along the pressure trend
    LPS_PROTECT_TAU_UP_S=30.0   # added subcooling rises with this time constant
    LPS_PROTECT_TAU_DOWN_S=120.0  # and falls back with this one

    #  GEKKO solver used for the running plant: 1 = APOPT, about 16% faster per solve
    #  than IPOPT (3, PlantRig's own first choice) with identical results to 5 decimals
    #  on the checks. None keeps whatever PlantRig picked.
    SOLVER=1

    def __init__(self,*args,mode='feedforward',guards=True,absorb_lps_surplus=True,solver='default',
                 prices=None,**kwargs):
        self.operator_mode=_mode(mode)
        #  Economics (2026-09-27): every sample row gets the price, the NZ$/h margin terms and
        #  the cumulative NZ$ since the last reset (economics.margin). Scoring only -- nothing
        #  here feeds back into control in these modes.
        self.prices=prices if prices is not None else econ.Prices()
        self.margin_cum=0.
        from rto import RTO   # imported here: rto pulls in excel_verify/python_side
        self.rto=RTO(self.prices);self.rto_decision=None;self.dt_floor=None
        self.absorb_lps_surplus=absorb_lps_surplus
        self._prot_inc=0.;self._p_prev=None
        self.guards=guards
        self._hw_latched=False
        self.lps_pi=PI(LPS_FLOW_TUNING)
        self.mps_pi=PI(MPS_TUNING)   # re-tuned every interval in step() (flow units)
        #  Per-instance copies of the MPS tuning knobs, so a test can vary them.
        self.mps_tauc_ratio=MPS_TAUC_RATIO;self.mps_td=MPS_TD_S
        self._pm_prev=self._pl_prev=None   # last measurements, for the D terms
        self._pm_ref=self._pm_ref_target=None   # MPS reference trajectory, see step()
        self._pm_ref_hist=[]                    # (time, MPS target) changes, for its delay
        # Provisional minimum-service overrides; flow error for HW, bar for MPS.
        self.hw_pi=PI(Tuning(.05,600.))
        self.mps_guard=PI(Tuning(.5,600.))
        self._mps_guard_on=False   # guard selected last interval (its hold-in, see step())
        self._transfer=True
        super().__init__(*args,**kwargs)
        chosen=self.SOLVER if solver=='default' else solver
        if chosen is not None:self.m.options.SOLVER=chosen
        self.fuel_bias=self.fuel_base
        self._decorate_initial()
        self._init_mpc()

    def _init_mpc(self):
        #  MPC (2026-09-27, mpc.py). Its observer runs in EVERY mode, so the boiler's dead-time
        #  history is already correct whenever an MPC mode is switched on.
        from mpc import MPC
        if getattr(self,'mpc',None) is not None:self.mpc.cleanup()
        self.mpc=MPC(self);self.mpc.observe([self.last])
        self.mpc_fallbacks=0;self.mpc_overruns=0

    def _decorate_initial(self):
        r=self.last
        selector={'no_control':'explicit','feedforward':'feedforward'}.get(self.operator_mode,'LPS')
        r.update(operator_mode=self.operator_mode,fuel_basis=self.fuel_bias,
            fuel_pi_request=r['mfuel_cmd'],fuel_hw_request=r['mfuel_cmd'],
            fuel_mps_request=r['mfuel_cmd'],fuel_selector=selector,
            allocation_base=r['frac_t2_cmd'],allocation_trim=0.,lps_error=0.,
            lps_control_error=0.,lps_p=0.,lps_i=0.,hw_error=r['hw_min']-r['m_hw_total'],
            mps_error=0.,service_ok=self._service(r),
            control_dt=self.batch*pr.DT,subcooling_calc=self.dtsub_cmd_val,
            hw_binds=False,dt_clamped=False,hw_override_latched=self._hw_latched,
            mps_guard_error=0.,mps_guard_eligible=False,mps_control_error=0.,mps_tau=0.,
            dt_floor=float('nan'),rto_runs=0,mpc_solve_s=float('nan'),mpc_fallback=False)
        r.update(econ.margin(r,self.prices),margin_cum=self.margin_cum)

    def _add_economics(self,rows):
        for row in rows:
            e=econ.margin(row,self.prices)
            self.margin_cum+=e['margin_h']*pr.DT/3600.
            row.update(e,margin_cum=self.margin_cum)

    @staticmethod
    def _service(r):
        return r['ok'] and max(r['mps_shortfall'],r['lps_shortfall'],r['hw_short'])<=.05

    def _absorb_lps_surplus(self,dl,pl,dt,r):
        """More subcooling while LPS is being oversupplied right now.

        The plant's inverse serves MPS first and keeps subcooling at its default
        while hot water is in surplus. But the boiler is slow (steam follows a
        fuel cut over ~10+ minutes) while HP2 acts in a couple of minutes, so
        after an HP2 increase LPS is oversupplied for a long while even though
        the steady-state fuel is right. Subcooling moves primary duty from LPS to
        hot water, and extra hot water is not a problem, so raise it just far
        enough that measured turbine LPS + heat pump LPS (at the primary's
        current duty) equals demand, up to its stop. Never lowers the inverse's
        subcooling (which carries the hot water minimum) and never changes fuel.

        Only ever called in `feedforward_pid` (see the module docstring's
        decision 2) -- it is a feedback correction layered on _invert()'s own
        subcooling target, so it has no meaning without both a live target and
        live feedback.
        """
        base=r['m_lps']+r['m_letdown']
        p_hp=float(np.clip(r['plps_act'],*pr.hpp.P_LPS_RANGE))  # heat pump condenses at the live header pressure
        def surplus(x):
            c=pr.hpp.cycle_constants(p_hp,x)
            return base+r['Q_cond_pri']*(1-c['sub_frac'])/c['dh_lps']-dl
        if surplus(dt)<=1e-6:return dt
        lo,hi=dt,pr.DT_SUB_HI
        if surplus(hi)>=0:return hi
        for _ in range(30):
            mid=(lo+hi)/2
            if surplus(mid)>0:lo=mid
            else:hi=mid
        return (lo+hi)/2

    def set_mode(self,mode):
        mode=_mode(mode)
        if self.halted:raise RuntimeError('Reset the stopped model first')
        if mode!=self.operator_mode:
            #  Leaving feedforward_pid: the LPS-surplus protection's own lead/lag
            #  state would otherwise read a huge spurious dp/dt on return (its
            #  "previous pressure" could be arbitrarily stale), so it resets
            #  rather than carries a memory across a mode it isn't running in.
            pid_modes=('feedforward_pid','feedforward_pid_rto')
            if self.operator_mode in pid_modes and mode not in pid_modes:
                self._prot_inc=0.;self._p_prev=None
            self.operator_mode=mode;self._transfer=True

    def step(self,d_mps,d_lps,hw_min,pmps_cmd,plps_cmd,twout_cmd,qsec_cmd=None,
             mfuel_cmd=None,frac_t2_cmd=None,dtsub_cmd=None):
        """Advance one batch. mfuel_cmd/frac_t2_cmd/dtsub_cmd are read only in
        `no_control` (explicit commands); every other mode computes its own MVs
        and ignores them. Omitted explicit commands retain the previous value,
        same convention as PlantRig.step's own explicit path.
        """
        if self.halted:raise RuntimeError('Reset the stopped model first')
        values=(d_mps,d_lps,hw_min,pmps_cmd,plps_cmd,twout_cmd)
        if not all(math.isfinite(x) for x in values):raise ValueError('Inputs must be finite')
        if qsec_cmd is None:qsec_cmd=self.qsec_cmd_val
        if not math.isfinite(qsec_cmd):raise ValueError('HP2 duty must be finite')
        dm=float(np.clip(d_mps,pr.D_MPS_LO,pr.D_MPS_HI))
        dl=float(np.clip(d_lps,pr.D_LPS_LO,pr.D_LPS_HI))
        hw=float(np.clip(hw_min,pr.HW_MIN_LO,pr.HW_MIN_HI))
        pm=float(np.clip(pmps_cmd,pr.P_MPS_LO,pr.P_MPS_HI))
        pl=float(np.clip(plps_cmd,pr.P_LPS_LO,pr.P_LPS_HI))
        q=float(np.clip(qsec_cmd,pr.Q_SEC_LO,pr.Q_SEC_HI))
        tw=float(np.clip(twout_cmd,pr.TWOUT_LO,pr.TWOUT_HI))
        r=self.last
        saved=(self.lps_pi.i,self.hw_pi.i,self.mps_guard.i,self.mps_pi.i)
        mode=self.operator_mode
        if mode in MPC_MODES:
            return self._step_mpc(dm,dl,hw,pm,pl,tw,q)
        #  RTO (2026-09-27): in feedforward_pid_rto the optimiser's HP2 duty replaces the
        #  slider and its subcooling floor goes to the inverse; everything else is exactly
        #  feedforward_pid. See rto.py for why it sets these two and not fuel or split.
        self.rto_decision=None
        if mode=='feedforward_pid_rto':
            self.rto_decision=self.rto.decide(dict(d_mps=dm,d_lps=dl,hw_min=hw,p_mps=pm,p_lps=pl,t_hw=tw),
                                              self.t,q_now=self.qsec_cmd_val,dt_now=r['dtsub_cmd'])
            q=float(self.rto_decision['q_sec']);self.dt_floor=float(self.rto_decision['dt_floor'])
        else:
            self.dt_floor=None
        self.hw_min_val,self.plps_cmd_val,self.twout_cmd_val,self.qsec_cmd_val=hw,pl,tw,q

        ff_family=mode in ('feedforward','feedforward_pid','feedforward_pid_rto')
        pid_family=mode in ('feedforward_pid','feedforward_pid_rto')

        #  Pressure-aware effective demand. At the target pressures the headers only
        #  sit in equilibrium if supply also covers what the letdown and dump take
        #  there (header_closures: letdown above 40 bar, dump above 3 bar). This is
        #  plain flow bookkeeping for the live turbine-2 allocation and the inverse.
        cl=headers.flows(pm,pl,dm,dl)
        dm_eff=dm+cl['letdown']
        dl_eff=max(dl+cl['dump']-cl['letdown'],0.)

        if ff_family:
            #  if mode == 'feedforward': u = plant._invert(d_mps, d_lps)          # trims = 0
            #  if mode == 'feedforward_pid': u = pid_layer(plant._invert(...), measurements)
            basis,_,dt_sub,_,_=self._invert(dm_eff,dl_eff)  # split: see live-steam allocation below
            hw_binds,dt_clamped=self.hw_binds,self.dt_clamped
        else:
            #  no_control: u = explicit_slider_values (fuel/frac/dtsub set below)
            basis,dt_sub=self.fuel_base,self.dt_sub_base
            hw_binds=dt_clamped=False

        if ff_family or pid_family:
            #  Turbine-2 split from the steam the boiler is making NOW, not the
            #  steady steam at the new fuel: the boiler lags a fuel change by ~15
            #  min, and a split sized for the future steam pushes the not-yet-gone
            #  surplus into the MPS header. Used by `feedforward` too since
            #  2026-09-24 -- it was taking _invert()'s own split (sized
            #  for the future steam), and after a demand drop plus an MPS target
            #  raise to 55 bar that split went to 0 while the boiler was still
            #  making ~37 kg/s; the surplus overran the saturated letdown and hit
            #  the MPS ceiling in seconds. Measured steam is a measured
            #  disturbance, not header-pressure feedback, so `feedforward` stays
            #  feedforward.
            base=float(np.clip(1-dm_eff/max(r['steam'],1e-9),pr.FRAC_T2_LO,pr.FRAC_T2_HI))
        else:  # no_control
            base=self.frac_t2_cmd_val if frac_t2_cmd is None else float(frac_t2_cmd)

        #  LPS pressure protection -- feedforward_pid only (decision 2 above).
        if pid_family and self.absorb_lps_surplus:
            #  Starts just above the target (or 3.6 bar, whichever is higher), so a high
            #  target leaves no room for a bump, and looks a little ahead along the
            #  current pressure trend: after a big HP2 step the header can gain ~0.5 bar
            #  in ~90 s, faster than the smoothed protection could otherwise react.
            dt_s=self.batch*pr.DT
            dpdt=0. if self._p_prev is None else max((r['plps_act']-self._p_prev)/dt_s,0.)
            self._p_prev=r['plps_act']
            p_eff=r['plps_act']+self.LPS_PROTECT_LEAD_S*dpdt
            p_on=max(self.LPS_PROTECT_ON_BAR,pl+.02)
            p_full=min(p_on+self.LPS_PROTECT_BAND_BAR,self.LPS_PROTECT_MAX_BAR)
            s=float(np.clip((p_eff-p_on)/(p_full-p_on),0,1))
            inc=s*(self._absorb_lps_surplus(dl_eff,pl,dt_sub,r)-dt_sub) if s>0 else 0.
            #  Smoothing of the added subcooling (fast up, slow down): without it the
            #  protection hunts around its threshold (pressure and subcooling chase each
            #  other through the header lag), which shows as a wiggle.
            tau=self.LPS_PROTECT_TAU_UP_S if inc>self._prot_inc else self.LPS_PROTECT_TAU_DOWN_S
            self._prot_inc+=(inc-self._prot_inc)*(1-math.exp(-dt_s/tau))
            dt_sub+=self._prot_inc

        el=pl-r['plps_act'];eh=hw-r['m_hw_total'];em=pm-r['pmps_act']
        #  MPS GUARD (minimum-service override on fuel), reworked 2026-09-25. Its job: fuel
        #  is normally set by the LPS loop alone, but the boiler's steam feeds both headers,
        #  so MPS can run short with nothing asking for more fuel (e.g. HP2 high keeps LPS
        #  full, the LPS loop holds or cuts fuel, and the MPS PI has already moved turbine 2's
        #  split to 0 -- all steam on MPS -- and MPS is still short). It used to measure MPS
        #  against a FIXED 40 bar (the full-service pressure when targets were fixed at
        #  40/3 bar); once the MPS target became a slider, any target below 40 read as a
        #  permanent shortfall and ratcheted fuel up without limit. Now:
        #    - error is against the MPS TARGET, not 40 bar;
        #    - it is eligible only when the MPS PI has run out of room (turbine 2's split
        #      commanded at its lower stop last interval) and MPS is below target, or it
        #      already won last interval and MPS is still below target (hold-in, so it does
        #      not drop out the moment extra steam lifts the split off 0 and cycle).
        eg=em
        split_at_stop=r['frac_t2_cmd']<=pr.FRAC_T2_LO+1e-9
        guard_eligible=bool(self.guards and eg>.01 and (split_at_stop or self._mps_guard_on))
        #  Pressure errors in flow units (see PRESSURE LOOPS IN FLOW UNITS at the top):
        #  steam leaving the header at the target minus at the measured pressure, so
        #  positive = pressure low = header short of steam, same sign as the bar errors.
        pm_meas,pl_meas=r['pmps_act'],r['plps_act']
        def out_m(p):
            f=headers.flows(p,pl_meas,dm,dl,target_mps=pm,target_lps=pl)
            return f['mps']+f['letdown']
        def out_l(p):
            f=headers.flows(pm_meas,p,dm,dl,target_mps=pm,target_lps=pl)
            return f['lps']+f['dump']
        #  MPS REFERENCE TRAJECTORY (2026-09-25). After a target step the header moves to
        #  the new target by itself: the site law draws up to 3x demand above target (less
        #  below), and the feedforward split keeps supplying the demand, so
        #  C*dP/dt = out_m(target) - out_m(P) -- a fall from 40 to 10 bar takes ~7 s with
        #  no overshoot. The PI used to see that whole transient as error (-24 kg/s on the
        #  first interval), push turbine 2 flow up on top of it, and with a 10 s interval
        #  undershoot the target by ~2 bar -- through the 8.5 bar domain floor at a 10 bar
        #  target (and 22.5 at the old 25 bar minimum). Now the PI's error is taken against
        #  the pressure that same header equation predicts (p_ref, integrated here over
        #  the interval just run, at the target that was in force during it), so it acts
        #  only on what the header does NOT do by itself. p_ref converges to the target,
        #  so at rest the error is unchanged: target vs measured.
        #  The site draw follows a target change at once (the delivery law uses the target
        #  directly); the SUPPLY the feedforward sends to MPS follows it only after the
        #  dispatch/valve/stage dead time, MPS_THETA_FIXED_S -- without that delay the
        #  reference ran ~7 s ahead on a 10 -> 55 bar step and the PI chased it into a
        #  9 bar overshoot.
        ctrl_dt=self.batch*pr.DT
        if self._pm_ref is None:
            self._pm_ref=pm_meas
            self._pm_ref_hist=[(self.t-1e9,pm)]
        elif self._pm_ref_target is not None:
            n_sub=max(int(ctrl_dt/.1),1);h_sub=ctrl_dt/n_sub;pt=self._pm_ref_target
            t_start=self.t-ctrl_dt
            def out_ref(p,target):
                f=headers.flows(p,pl_meas,dm,dl,target_mps=target,target_lps=pl)
                return f['mps']+f['letdown']
            def target_at(t):
                return next(v for tt,v in reversed(self._pm_ref_hist) if tt<=t)
            for k in range(n_sub):
                ts=t_start+k*h_sub
                ps=target_at(ts-MPS_THETA_FIXED_S)
                self._pm_ref+=h_sub*(out_ref(ps,ps)-out_ref(self._pm_ref,pt))/self.C_MPS
        if not self._pm_ref_hist or self._pm_ref_hist[-1][1]!=pm:
            self._pm_ref_hist.append((self.t,pm))
            self._pm_ref_hist=self._pm_ref_hist[-20:]
        self._pm_ref_target=pm
        em_flow=out_m(self._pm_ref)-out_m(pm_meas)
        ec=out_l(pl)-out_l(pl_meas)
        #  MPS header time constant at the target, from its local self-regulation slope
        #  (a +/-0.25 bar secant, so a letdown onset right at the target counts half).
        slope_m=(out_m(pm+.25)-out_m(pm-.25))/.5
        tau_m=self.C_MPS/max(slope_m,1e-3)
        #  Rates of change of the measured pressures in FLOW units (same local slopes),
        #  for the derivative terms; None on the first interval after a (re)start.
        slope_l=(out_l(pl+.02)-out_l(pl-.02))/.04
        dpm=None if self._pm_prev is None else slope_m*(pm_meas-self._pm_prev)/ctrl_dt
        dpl=None if self._pl_prev is None else slope_l*(pl_meas-self._pl_prev)/ctrl_dt
        self._pm_prev,self._pl_prev=pm_meas,pl_meas
        lf=hf=mf=split=0.
        l=h=g=dict(p=0.,i=0.,d=0.,error=0.)
        if mode=='no_control':
            fuel=self.mfuel_cmd_val if mfuel_cmd is None else float(mfuel_cmd)
            dt_sub=self.dtsub_cmd_val if dtsub_cmd is None else float(dtsub_cmd)
            basis=fuel  # no baseline/trim split in this mode; diagnostics report the command itself
            selector='explicit'
        elif mode=='feedforward':
            fuel=basis
            selector='feedforward'
        else:  # feedforward_pid: the PID layer on top of the inverse
            #  MPS PI output is turbine-2 FLOW (kg/s); the split trim is that over the
            #  steam the boiler is making now.
            theta_m=MPS_THETA_FIXED_S+ctrl_dt/2
            lam_m=(1+self.mps_tauc_ratio)*theta_m
            self.mps_pi.tuning=Tuning(-tau_m/lam_m,tau_m,td=self.mps_td)
            steam_now=max(r['steam'],1e-6)
            split_flow,m=self.mps_pi.output(em_flow,dmeas_dt=dpm,dt=ctrl_dt)
            split=split_flow/steam_now
            track=r['mfuel_cmd']-basis if self._transfer else None
            lf,l=self.lps_pi.output(ec,track=track,dmeas_dt=dpl,dt=ctrl_dt)
            hf,h=self.hw_pi.output(eh,track=track)
            mf,g=self.mps_guard.output(eg,track=track)  # see MPS GUARD above
            reqs={'LPS':basis+lf}
            hw_latched=self._hw_latched or eh>.05
            if self.guards and hw_latched:reqs['HW']=basis+hf
            if guard_eligible:reqs['MPS']=basis+mf
            selector=max(reqs,key=reqs.get)
            fuel=reqs[selector]
            # Mode transfer holds the previous absolute fuel command for one interval.
            if self._transfer:fuel=r['mfuel_cmd']

        start=self.t
        rows=super().step(dm,dl,hw,pm,pl,tw,qsec_cmd=q,mfuel_cmd=fuel if mode=='no_control' else basis,
            frac_t2_cmd=base,dtsub_cmd=dt_sub,
            fuel_trim=0. if mode in ('no_control','feedforward') else fuel-basis,
            frac_t2_trim=split if pid_family else 0.)
        last=rows[-1]
        if pid_family and all(v['ok'] for v in rows):
            dt=self.t-start;applied=last['mfuel_cmd']-basis
            self.mps_pi.finish(em_flow,dt,split_flow,last['frac_t2_trim_applied']*steam_now,m['p'],True)
            for key,loop,e,terms,request in [('LPS',self.lps_pi,ec,l,lf),('HW',self.hw_pi,eh,h,hf),('MPS',self.mps_guard,eg,g,mf)]:
                # External-reset tracking stops unselected override windup.
                loop.finish(e,dt,request,applied,terms['p'],key==selector)
                if key=='HW' and key!=selector:
                    # Retain a learned minimum without locking fuel at its old
                    # value when HW is in surplus. External reset plus error
                    # integration allows the unselected HW request to fall.
                    loop.i += loop.tuning.kc/loop.tuning.ti*e*dt
            self._hw_latched=hw_latched
            self._mps_guard_on=selector=='MPS'
            self._transfer=False
        elif pid_family:
            self.lps_pi.i,self.hw_pi.i,self.mps_guard.i,self.mps_pi.i=saved
        else:
            #  no_control / feedforward: the PID loops are disabled entirely (see
            #  the module docstring), not just unselected -- no output(), no
            #  finish(), no windup tracking while inactive.
            self._transfer=False
            self._mps_guard_on=False
        for row in rows:
            row.update(operator_mode=mode,fuel_basis=basis,
                fuel_pi_request=basis+lf,fuel_hw_request=basis+hf,fuel_mps_request=basis+mf,
                fuel_selector=selector,allocation_base=base,allocation_trim=split,
                lps_error=el,lps_control_error=ec,lps_p=l['p'],lps_i=l['i'],hw_error=eh,
                mps_error=em,service_ok=self._service(row),
                control_dt=self.batch*pr.DT,subcooling_calc=self.dtsub_cmd_val,
                hw_binds=hw_binds,dt_clamped=dt_clamped,
                hw_override_latched=self._hw_latched,
                mps_guard_error=eg,mps_guard_eligible=guard_eligible and pid_family,
                mps_control_error=em_flow,mps_tau=tau_m,
                dt_floor=self.dt_floor if self.dt_floor is not None else float('nan'),
                rto_runs=self.rto.runs,mpc_solve_s=float('nan'),mpc_fallback=False)
        self._add_economics(rows)
        if rows[-1]['ok']:self.mpc.observe(rows)
        return rows

    def _step_mpc(self,dm,dl,hw,pm,pl,tw,q_slider):
        """One interval under MPC: the MPC sets fuel, split, subcooling and HP2 duty directly
        (explicit commands, no PID trims). Tracking MPC follows the RTO's HP2/subcooling
        targets; economic MPC maximises margin itself. On a failed solve this interval runs
        as Feedforward+PID instead (counted in mpc_fallbacks)."""
        mode=self.operator_mode;r=self.last
        site=dict(d_mps=dm,d_lps=dl,hw_min=hw,p_mps=pm,p_lps=pl,t_hw=tw)
        self.mpc.version=MPC_MODES[mode]
        self.rto_decision=None;self.dt_floor=None
        if mode=='mpc':
            self.rto_decision=self.rto.decide(site,self.t,q_now=r['qsec_cmd'],dt_now=r['dtsub_cmd'])
        try:
            u=self.mpc.solve(r,site,self.prices,self.t,self.rto_decision)
        except Exception as e:
            self.mpc_fallbacks+=1
            self.operator_mode='feedforward_pid';self._transfer=True
            try:
                rows=self.step(dm,dl,hw,pm,pl,tw,qsec_cmd=q_slider)
            finally:
                self.operator_mode=mode
            for row in rows:row.update(operator_mode=mode,mpc_fallback=True,fuel_selector='MPC fallback')
            return rows
        secs=self.mpc.last['secs']
        from mpc import FAIL_SECS
        if secs>FAIL_SECS:self.mpc_overruns+=1
        self.hw_min_val,self.plps_cmd_val,self.twout_cmd_val=hw,pl,tw
        rows=pr.PlantRig.step(self,dm,dl,hw,pm,pl,tw,qsec_cmd=u['q'],mfuel_cmd=u['fuel'],
                              frac_t2_cmd=u['split'],dtsub_cmd=u['dt'],fuel_trim=0.,frac_t2_trim=0.)
        self._transfer=True   # a later switch to a PID mode starts bumplessly from here
        for row in rows:
            row.update(operator_mode=mode,fuel_basis=u['fuel'],fuel_pi_request=u['fuel'],
                fuel_hw_request=u['fuel'],fuel_mps_request=u['fuel'],fuel_selector='MPC',
                allocation_base=u['split'],allocation_trim=0.,lps_error=pl-row['plps_act'],
                lps_control_error=0.,lps_p=0.,lps_i=0.,hw_error=hw-row['m_hw_total'],
                mps_error=pm-row['pmps_act'],service_ok=self._service(row),
                control_dt=self.batch*pr.DT,subcooling_calc=u['dt'],hw_binds=False,dt_clamped=False,
                hw_override_latched=False,mps_guard_error=0.,mps_guard_eligible=False,
                mps_control_error=0.,mps_tau=0.,
                dt_floor=self.rto_decision['dt_floor'] if self.rto_decision else float('nan'),
                rto_runs=self.rto.runs,mpc_solve_s=secs,mpc_fallback=False)
        self._add_economics(rows)
        if rows[-1]['ok']:self.mpc.observe(rows)
        return rows

    def reset(self):
        super().reset()
        self.margin_cum=0.
        from rto import RTO
        self.rto=RTO(self.prices);self.rto_decision=None;self.dt_floor=None
        for loop in (self.lps_pi,self.mps_pi,self.hw_pi,self.mps_guard):loop.i=0.
        self.fuel_bias=self.fuel_base
        self._transfer=True;self._hw_latched=False;self._prot_inc=0.;self._p_prev=None
        self._mps_guard_on=False;self._pm_prev=self._pl_prev=None
        self._pm_ref=self._pm_ref_target=None;self._pm_ref_hist=[]
        self.lps_pi.d=self.mps_pi.d=0.
        self._decorate_initial()
        self._init_mpc()
        return self.last
