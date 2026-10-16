"""Shared pressure PI trims around the unchanged PlantRig feedforward inverse.

Control runs once per solve batch with zero-order-held trims. Default 10 s;
validation also checks a 2 s update. Positive error = setpoint - measurement.
Kc includes the process sign. Derivative is deliberately zero. Parameters are
provisional from header_revision_results/provisional_pi.csv, not plant tuning.
"""
from dataclasses import dataclass
import math
import plant_rig as pr


@dataclass(frozen=True)
class Tuning:
    kc: float
    ti: float
    #  Derivative time [s] (2026-09-25). 0 = PI, the original behaviour. When > 0 the
    #  loop is a PID with the D term on the MEASUREMENT (a setpoint step causes no
    #  derivative kick) and a first-order filter of time constant td/n_filter.
    td: float = 0.0
    n_filter: float = 10.0


MPS_TUNING = Tuning(-0.0031312183279458877, 8.348853749639126)
LPS_TUNING = Tuning(0.29095280578411875, 604.3542592417803)
DIAGNOSTIC_FIELDS = ('control_mode', 'control_dt', 'mps_error', 'lps_error',
    'mps_p', 'mps_i', 'mps_d', 'lps_p', 'lps_i', 'lps_d',
    'mps_aw', 'lps_aw', 'lps_control_error')


class PI:
    """Integral state is in output units, allowing either controller sign.

    Output uses the current integral; finish() integrates the error over the
    completed interval. Saturation tracks the actually applied trim and blocks
    integration only when Ki*error pushes farther into the active limit.

    With tuning.td > 0 it is a PID: output() also takes the measurement's rate
    of change (dmeas_dt, in error units per second) and adds a filtered
    D = -Kc*Td*dmeas/dt. D on the measurement, not the error, so setpoint changes
    do not kick it. The D state is part of the output, so bumpless tracking and
    anti-windup (which back-calculate the integral from the applied output)
    already account for it.
    """
    def __init__(self, tuning):
        if not math.isfinite(tuning.kc) or not math.isfinite(tuning.ti) or tuning.ti <= 0:
            raise ValueError('PI tuning must be finite with Ti > 0')
        if not math.isfinite(tuning.td) or tuning.td < 0:
            raise ValueError('Derivative time must be finite and >= 0')
        self.tuning = tuning
        self.i = 0.0
        self.d = 0.0

    def output(self, error, track=None, dmeas_dt=None, dt=None):
        p = self.tuning.kc * error
        td = self.tuning.td
        if td > 0 and dmeas_dt is not None and dt:
            tf = td/self.tuning.n_filter
            a = tf/(tf + dt)
            self.d = a*self.d + (1 - a)*(-self.tuning.kc*td*dmeas_dt)
        elif td <= 0:
            self.d = 0.0
        if track is not None:
            self.i = track - p - self.d
        return p + self.i + self.d, dict(p=p + self.d, i=self.i, d=self.d, error=error)

    def finish(self, error, dt, requested, applied, p, automatic):
        ki_error = self.tuning.kc / self.tuning.ti * error
        saturated = abs(requested - applied) > 1e-12
        if not automatic or saturated:
            self.i = applied - p
        if automatic:
            farther = saturated and (requested - applied) * ki_error > 0
            if not farther:
                self.i += ki_error * dt
        return saturated


class PressureRig(pr.PlantRig):
    """Same plant, with ff / auto / hold control modes and logged PI terms.

    Switching to auto tracks the last applied trim for one control interval.
    Hold keeps trim offsets, so feedforward still responds to changing demand.
    FF deliberately removes trims immediately. Reset preserves selected mode,
    clears controller memory and restores the same physical base point.
    """
    def __init__(self, *args, mode='ff', **kwargs):
        if mode not in ('ff', 'auto', 'hold'):
            raise ValueError('mode must be ff, auto or hold')
        self.control_mode = mode
        self.mps_pi = PI(MPS_TUNING)
        self.lps_pi = PI(LPS_TUNING)
        self._transfer = mode == 'auto'
        self._held = dict(fuel=0.0, split=0.0)
        super().__init__(*args, **kwargs)
        self._initial_diagnostics()

    def _initial_diagnostics(self):
        self.last.update(control_mode=self.control_mode, control_dt=self.batch*pr.DT,
            mps_error=self.last['pmps_cmd']-self.last['pmps_act'],
            lps_error=self.last['plps_cmd']-self.last['plps_act'],
            lps_control_error=self.last['plps_cmd']-self.last['plps_act'],
            mps_p=0.0, mps_i=0.0, mps_d=0.0, lps_p=0.0, lps_i=0.0, lps_d=0.0,
            mps_aw=False, lps_aw=False)

    def set_mode(self, mode):
        if mode not in ('ff', 'auto', 'hold'):
            raise ValueError('mode must be ff, auto or hold')
        if mode == self.control_mode:
            return
        if self.halted:
            raise RuntimeError('Reset the stopped plant before changing control mode')
        self._held = dict(fuel=self.last['fuel_trim_applied'],
                          split=self.last['frac_t2_trim_applied'])
        self.control_mode = mode
        self._transfer = mode == 'auto'

    def _lps_control_error(self, error, measurement):
        """Identity for the existing PI; an experimental subclass may map it."""
        return error

    def step(self, d_mps, d_lps, hw_min, pmps_cmd, plps_cmd, twout_cmd,
             qsec_cmd=None, **kwargs):
        if self.halted:
            raise RuntimeError('Reset the stopped plant before continuing')
        if 'fuel_trim' in kwargs or 'frac_t2_trim' in kwargs:
            raise ValueError('PressureRig owns the trims; use PlantRig for external trim experiments')
        e_m = min(max(pmps_cmd,pr.P_MPS_LO),pr.P_MPS_HI)-self.last['pmps_act']
        e_l = min(max(plps_cmd,pr.P_LPS_LO),pr.P_LPS_HI)-self.last['plps_act']
        if not all(math.isfinite(v) for v in (e_m,e_l)):
            raise ValueError('Pressure setpoints and measurements must be finite')
        e_l_physical = e_l
        e_l = self._lps_control_error(e_l, self.last['plps_act'])
        auto = self.control_mode == 'auto'
        if auto:
            track_m = self._held['split'] if self._transfer else None
            track_l = self._held['fuel'] if self._transfer else None
        else:
            track_m = self._held['split'] if self.control_mode == 'hold' else 0.0
            track_l = self._held['fuel'] if self.control_mode == 'hold' else 0.0
        split, m = self.mps_pi.output(e_m, track=track_m)
        fuel, l = self.lps_pi.output(e_l, track=track_l)
        if track_m is not None: split = track_m
        if track_l is not None: fuel = track_l
        self._transfer = False
        start = self.t
        rows = super().step(d_mps,d_lps,hw_min,pmps_cmd,plps_cmd,twout_cmd,
                            qsec_cmd=qsec_cmd,fuel_trim=fuel,frac_t2_trim=split,**kwargs)
        last = rows[-1]
        aw_m = abs(split-last['frac_t2_trim_applied'])>1e-12
        aw_l = abs(fuel-last['fuel_trim_applied'])>1e-12
        # No integral advancement or adaptation from any invalid batch.
        if all(r['ok'] for r in rows):
            self.mps_pi.finish(e_m,self.t-start,split,last['frac_t2_trim_applied'],m['p'],auto)
            self.lps_pi.finish(e_l,self.t-start,fuel,last['fuel_trim_applied'],l['p'],auto)
        for row in rows:
            row.update(control_mode=self.control_mode,control_dt=self.batch*pr.DT,
                mps_error=e_m,lps_error=e_l_physical,lps_control_error=e_l,mps_p=m['p'],mps_i=m['i'],mps_d=0.0,
                lps_p=l['p'],lps_i=l['i'],lps_d=0.0,mps_aw=aw_m,lps_aw=aw_l)
        return rows

    def reset(self):
        result = super().reset()
        self.mps_pi.i = self.lps_pi.i = 0.0
        self._held = dict(fuel=0.0,split=0.0)
        self._transfer = self.control_mode == 'auto'
        self._initial_diagnostics()
        return result
