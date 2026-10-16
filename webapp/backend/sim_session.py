"""SimSession: the simulation-driving logic of plant_operator.OperatorApp, without matplotlib.

One SimSession owns one OperatorRig (one GEKKO model), the electricity/biomass prices, the demand
schedule and the row history. It is synchronous and thread-safe: step_once() is what the desktop
app's synchronous tick() does, and every state-changing method takes the same lock, so a command
arriving mid-solve waits for that solve (the desktop's _finish_solve()).

Ported, not redesigned. Where a method mirrors an OperatorApp method, the name is in the docstring.
The physics and controller code (operator_control.py, plant_rig.py, ...) is imported unchanged.
"""
import csv
import io
import math
import threading
import time

import numpy as np

import ui_config as ui  # also puts Models/ on sys.path
import economics as econ
import plant_rig as cfg
from operator_control import OperatorRig

#  Start-up economics settings, copied from the constants at the top of plant_operator.py.
PRICE_MODE_DEFAULT = 'fixed'
PRICE_DAY_LENGTH_S = 6 * 3600.0
PRICE_START_HOUR = 6.0
DEMAND_SCHEDULE_DEFAULT = False
DEMAND_INTERVAL_S = 3600.0
DEMAND_STEP_FRAC = 0.10
DEMAND_SEED = 582


def derive(row):
    """Fields OperatorApp.redraw() adds to every row (they also end up in the CSV)."""
    if 'm_lps_produced' not in row:
        row['m_lps_produced'] = row['m_lps'] + row['m_lps_hp'] + row['m_letdown']
        row['net_export'] = row['power'] - row['W_hp_total']
    return row


def clean(v):
    """JSON-safe value: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def nzd(v):
    return f"{'-' if v < 0 else ''}NZ${abs(v):,.0f}"


class SimSession:
    def __init__(self, batch=None):
        prices = econ.Prices(electricity=econ.ElectricityPrice(
            mode=PRICE_MODE_DEFAULT, day_length_s=PRICE_DAY_LENGTH_S, start_hour=PRICE_START_HOUR))
        self.demand = econ.DemandSchedule(interval_s=DEMAND_INTERVAL_S, step_frac=DEMAND_STEP_FRAC,
                                          seed=DEMAND_SEED)
        self.demand_on = DEMAND_SCHEDULE_DEFAULT
        self.rig = OperatorRig(verbose=False, prices=prices,
                               **({} if batch is None else {'batch': batch}))
        self.lock = threading.RLock()
        self.running = False
        self.closed = False
        self.last_step_ms = 0.0
        self.last_error = None
        self.epoch = 0          # bumped by reset(); lets a client detect a stale delta
        #  Slider defaults (OperatorApp specs' valinit) and the live slider values.
        rig = self.rig
        self.defaults = dict(
            qsec_cmd=float(round(cfg.Q_SEC_0)), pmps_cmd=cfg.P_MPS_0, mfuel_cmd=rig.fuel_base,
            plps_cmd=cfg.P_LPS_0, frac_t2_cmd=rig.frac_t2_base, d_mps=cfg.D_MPS_0,
            d_lps=cfg.D_LPS_0, hw_min=cfg.HW_MIN_0, twout_cmd=cfg.TWOUT_0,
            dtsub_cmd=rig.dt_sub_base)
        self.values = dict(self.defaults)
        self.rows = [derive(dict(rig.last))]
        if self.demand_on:
            self._apply_demand_schedule()

    # ------------------------------------------------------------------ inputs
    @property
    def mode(self):
        return self.rig.operator_mode

    def visible(self, key):
        """Which sliders show in the current mode (OperatorApp._apply_mode_visibility)."""
        no_control = self.mode == 'no_control'
        if key in ui.EXPLICIT_KEYS:
            return no_control
        if key in ui.PRESSURE_KEYS:
            return not no_control
        return True

    def active(self, key):
        """Which sliders accept input: visible, not driven by a controller, not by the schedule."""
        if not self.visible(key):
            return False
        if key == 'qsec_cmd' and self.mode in ui.HP2_CONTROLLED:
            return False
        if key in ui.DEMAND_KEYS and self.demand_on:
            return False
        return True

    def set_value(self, key, value):
        """A slider drag or typed value (OperatorApp._submit_textbox): clip to the slider's range,
        round whole-number sliders. Writes to an inactive slider are ignored, as on the desktop.

        Deliberately takes no lock: a drag must not queue behind an in-flight solve. A single dict
        write is atomic, and step_once() copies the values at the start of each batch, so a value
        written mid-solve is simply used by the next batch (as with the desktop's worker thread)."""
        if key not in ui.SLIDER_LIMITS:
            raise KeyError(key)
        v = float(value)
        if not math.isfinite(v) or not self.active(key):
            return False
        lo, hi = ui.SLIDER_LIMITS[key]
        v = min(max(v, lo), hi)
        if ui.SLIDER_WHOLE[key]:
            v = float(round(v))
        self.values[key] = v
        return True

    def set_values(self, values):
        return {k: self.set_value(k, v) for k, v in values.items() if k in ui.SLIDER_LIMITS}

    # ------------------------------------------------------------------ run control
    def start(self):
        with self.lock:
            if not self.rig.halted:
                self.running = True

    def pause(self):
        self.running = False

    def reset(self):
        """OperatorApp.reset: back to the common initial state, keeping the selected mode."""
        with self.lock:
            self.running = False
            self.rig.reset()
            self.values = dict(self.defaults)
            if self.demand_on:
                self._apply_demand_schedule()
            self.rows = [derive(dict(self.rig.last))]
            self.last_error = None
            self.epoch += 1

    def set_mode(self, mode):
        """OperatorApp.change_mode (the slider side effects of a mode switch)."""
        if mode not in ui.MODE_IDS:
            raise ValueError(mode)
        with self.lock:
            if self.rig.halted or mode == self.rig.operator_mode:
                return
            was_no_control = self.rig.operator_mode == 'no_control'
            self.rig.set_mode(mode)
            if mode == 'no_control':
                #  Pressure targets go inert in no_control: pin them at nominal.
                self.values['pmps_cmd'] = self.defaults['pmps_cmd']
                self.values['plps_cmd'] = self.defaults['plps_cmd']
            elif was_no_control:
                #  Leaving no_control: sync the explicit sliders to what the plant is doing now.
                r = self.rig.last
                for key in ui.EXPLICIT_KEYS:
                    self.values[key] = r[key]

    # ------------------------------------------------------------------ economics / demand
    def set_price_mode(self, mode):
        if mode not in ('fixed', 'daily'):
            raise ValueError(mode)
        with self.lock:
            self.rig.prices.electricity.mode = mode

    def set_price_mean(self, value):
        v = float(value)
        if math.isfinite(v) and v >= 0:
            with self.lock:
                self.rig.prices.electricity.mean = v

    def set_biomass_price(self, value):
        v = float(value)
        if v not in econ.BIOMASS_PRICE_OPTIONS:
            raise ValueError(value)
        with self.lock:
            self.rig.prices.biomass_per_gj = v

    def set_demand_mode(self, on):
        """OperatorApp._change_demand_mode / _apply_demand_mode."""
        with self.lock:
            self.demand_on = bool(on)
            if self.demand_on:
                self._apply_demand_schedule()

    def _apply_demand_schedule(self):
        """Move the demand values to the schedule's level at the plant's current time."""
        for key, v in self.demand.levels(self.rig.t).items():
            self.values[key] = v

    # ------------------------------------------------------------------ stepping
    def step_once(self):
        """One batch: OperatorApp.tick() in its synchronous (threaded=False) form.

        Returns (start_index, new_rows) or None if nothing was stepped (paused, halted or failed).
        """
        with self.lock:
            if not self.running:
                return None
            if self.rig.halted:
                self.running = False
                return None
            if self.demand_on:
                self._apply_demand_schedule()
            u = dict(self.values)
            t0 = time.perf_counter()
            try:
                new = self.rig.step(**u)
            except Exception as e:   # desktop _collect: print the error and stop running
                self.running = False
                self.last_error = f'Step failed: {e}'
                return None
            secs = time.perf_counter() - t0
            start = len(self.rows)
            for row in new:
                derive(row)
            self.rows.extend(new)
            self.last_step_ms = secs * 1000 / self.rig.batch
            if self.rig.operator_mode in ui.HP2_CONTROLLED:
                #  Show the controller's HP2 duty on the (greyed) slider.
                lo, hi = ui.SLIDER_LIMITS['qsec_cmd']
                self.values['qsec_cmd'] = float(min(max(round(new[-1]['qsec_cmd']), lo), hi))
            if self.rig.halted:
                self.running = False
            return start, new

    # ------------------------------------------------------------------ views
    def status(self):
        """The two-line status block of OperatorApp.redraw()."""
        rig, rows = self.rig, self.rows
        r = rows[-1]
        stop = ('STOPPED: ' + rig.stop_reason) if rig.halted else ('Running' if self.running else 'Paused')
        if r['service_ok']:
            service = 'Demand met'
        else:
            service = (f"SHORT  MPS {r['mps_shortfall']:.2f} / LPS {r['lps_shortfall']:.2f}"
                       f" / HW {r['hw_short']:.2f} kg/s")
        if r.get('hw_override_latched'):
            service += '  [HW override armed]'
        mode = rig.operator_mode.replace('_', ' ').upper()
        pe = rig.prices.electricity
        price_txt = (f"NZ${r['price_export']:.0f}/MWh ({pe.clock_text(r['t'])})"
                     if pe.mode == 'daily' else f"NZ${r['price_export']:.0f}/MWh")
        extra = ''
        if self.demand_on:
            extra += f" | demand change in {(self.demand.next_change(r['t']) - r['t']) / 60:.0f} min"
        if rig.operator_mode in ui.MPC_MODES and rig.mpc.solve_secs:
            extra += f" | MPC solve {rig.mpc.solve_secs[-1]:.1f} s, {rig.mpc_fallbacks} fallbacks"
        elif rig.operator_mode == 'feedforward_pid_rto' and rig.rto_decision:
            extra += f" | RTO subcooling floor {rig.rto_decision['dt_floor']:.1f} K"
        line1 = f"{stop} | {mode} | t={r['t']:.0f} s | {service}"
        line2 = (f"Electricity {price_txt} | net export {r['power'] - r['W_hp_total']:.0f} kW"
                 f" | margin {nzd(r['margin_h'])}/h | since reset {nzd(r['margin_cum'])}{extra}")
        return dict(line1=line1, line2=line2, ok=not (rig.halted or not r['service_ok']))

    def state(self):
        """Everything the client shows apart from the time series."""
        rig = self.rig
        pe = rig.prices.electricity
        return dict(
            running=self.running, halted=rig.halted, stop_reason=rig.stop_reason,
            mode=rig.operator_mode, t=self.rows[-1]['t'], n_rows=len(self.rows), epoch=self.epoch,
            sliders={k: dict(value=self.values[k], visible=self.visible(k), active=self.active(k))
                     for k in ui.SLIDER_KEYS},
            demand_on=self.demand_on, price_mode=pe.mode, price_mean=pe.mean,
            biomass_price=rig.prices.biomass_per_gj,
            status=self.status(), step_ms=self.last_step_ms, error=self.last_error)

    def last_row(self):
        r = self.rows[-1]
        return {k: clean(r.get(k)) for k in ui.SCHEMATIC_KEYS}

    def columns(self, rows):
        """Column-oriented series data for the plotted keys (what the trend panels draw)."""
        keys = ['t'] + [k for k in ui.series_keys() if k != 't']
        return {k: [clean(r.get(k)) for r in rows] for k in keys}

    def snapshot(self):
        """Full resync: the rolling window of history plus current state."""
        with self.lock:
            t_last = self.rows[-1]['t']
            lo = max(0.0, t_last - ui.WINDOW_S)
            i0 = 0
            for i in range(len(self.rows) - 1, -1, -1):   # newest rows are the window
                if self.rows[i]['t'] < lo:
                    i0 = i + 1
                    break
            return dict(type='snapshot', start=i0, cols=self.columns(self.rows[i0:]),
                        last=self.last_row(), state=self.state())

    def delta(self, start, new):
        with self.lock:
            return dict(type='rows', start=start, cols=self.columns(new),
                        last=self.last_row(), state=self.state())

    # ------------------------------------------------------------------ export / close
    def csv_text(self):
        """OperatorApp.save_csv, to a string."""
        with self.lock:
            f = io.StringIO()
            f.write('# Operator control; no_control/feedforward/feedforward_pid; actual mode recorded per row\n')
            f.write('# See docs/07_regulation_models_report.md for what each mode drives biomass/split/subcooling from\n')
            f.write('# LPS tuning ' + str(self.rig.lps_pi.tuning) + '; guards ' + str(self.rig.guards) + '\n')
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
            return f.getvalue()

    def close(self):
        with self.lock:
            self.running = False
            self.closed = True
            self.rig.m.cleanup()
