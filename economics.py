"""Plant economics: prices, a changing site-demand schedule, and the margin (NZ$/h).

Added 2026-09-27 (plan: docs/09_economics_rto_mpc_plan.md). Pure Python, no GEKKO -- the running
rig (operator_control.OperatorRig) calls margin() on every sample, and the RTO/MPC read the same
prices, so every controller is scored by the one calculation.

PRICES
    Electricity is the only price that can change. ElectricityPrice.mode is 'fixed' (one number)
    or 'daily' (a 48 x half-hour profile repeated forever). The day clock comes from simulation
    time t and wraps with a modulo, so a run of any length keeps cycling:
        day_seconds(t) = (t * 86400/day_length_s + start_hour*3600) mod 86400
    The built-in profile is normalised to average exactly 1.0, so 'fixed' and 'daily' have the
    same mean and a comparison of the two isolates the effect of the price VARYING. Import price
    = export price + a fixed network/retail adder, so the two move together. Biomass, steam and
    hot-water prices are fixed contract prices.

    Values are round-about NZ trends, not precise: 2025 wholesale quarters NZ$131-223/MWh with an
    October average near NZ$30 (Electricity Authority); biomass chip NZ$9-15/GJ (Bioenergy
    Association TNSB74); spot prices cheap overnight, peaks 07:00-11:00 and 17:00-21:00 (evening
    highest), a midday solar dip. The built-in shape changes every clock hour (every 900 simulated
    s at the default 6 h day) and crosses HP2's ~NZ$77/MWh break-even twice a day.

SITE DEMAND SCHEDULE
    DemandSchedule steps MPS demand, LPS demand and the hot-water minimum to a new level every
    interval_s of simulation time -- an instant step, like any setpoint; the plant's own dynamics
    delay the response. Each step is a bounded, seeded random walk pulled gently back towards
    the base point, so every controller sees the identical sequence and a long run stays near
    the normal operating region.

MARGIN
    margin = product revenue + electricity - fuel, in NZ$/h. Revenue is earned on DELIVERED
    product only, capped at the demand (the site delivery law can draw more than demand above
    the target pressure) and, for hot water, at the minimum (surplus earns nothing). A shorted
    demand simply earns no revenue; there is no extra penalty. Shortfall and vent are reported by
    the rig as their own quantities.
"""
import csv
import math
from dataclasses import dataclass, field

import numpy as np

import boiler as blr
import plant_rig as pr

# ---------------------------------------------------------------------------------------------
#  Electricity price
# ---------------------------------------------------------------------------------------------
HALF_HOUR_S = 1800.0
DAY_S = 86400.0

#  One multiplier per clock hour (00:00-01:00 first) -- the illustrative NZ weekday shape, before
#  normalising. A new price every hour (2026-09-27): the first version had 9 flat blocks, so the
#  price sat still for up to 5 clock hours (4500 simulated s at the 6 h day) and a run saw only a
#  handful of changes. Normalised, at mean 120 the overnight hours and the 12:00-15:00 dip fall
#  below HP2's ~NZ$77/MWh break-even.
_SHAPE_HOURLY = (0.50, 0.45, 0.42, 0.40, 0.42, 0.55, 0.80, 1.10, 1.30, 1.15, 0.95, 0.75,
                 0.60, 0.52, 0.55, 0.70, 1.00, 1.50, 1.90, 1.75, 1.40, 1.15, 0.90, 0.70)


def _default_shape():
    raw = [_SHAPE_HOURLY[k//2] for k in range(48)]    # each hour fills its two half-hour slots
    mean = sum(raw)/len(raw)
    return tuple(v/mean for v in raw)


DEFAULT_SHAPE = _default_shape()      # 48 half-hour multipliers, average exactly 1.0

DAY_LENGTH_OPTIONS_S = (24*3600.0, 6*3600.0, 2*3600.0)   # simulated seconds per price day


def load_emi_csv(path, column=None):
    """48 half-hourly prices (NZ$/MWh) from a CSV, e.g. one day downloaded from the Electricity
    Authority's EMI wholesale price report. Takes the first numeric column (or `column`) and the
    first 48 numeric rows. Returned as ABSOLUTE prices, used as-is (not normalised)."""
    vals = []
    with open(path, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            keys = [column] if column else list(row)
            for k in keys:
                try:
                    vals.append(float(row[k]))
                    break
                except (TypeError, ValueError):
                    continue
            if len(vals) == 48:
                break
    if len(vals) != 48:
        raise ValueError(f'{path}: found {len(vals)} half-hourly prices, need 48')
    return tuple(vals)


@dataclass
class ElectricityPrice:
    """Export (sell) and import (buy) electricity price, fixed or on a looping daily profile.

    mode          'fixed' or 'daily'
    mean          NZ$/MWh export price: the fixed value, or the daily profile's average
    import_adder  NZ$/MWh added to the export price when the plant buys
    day_length_s  simulated seconds per price day (24 h real, or 6 h / 2 h compressed)
    start_hour    time of day at t = 0
    smooth_s      optional cosine blend across each half-hour boundary, in DAY-CLOCK seconds
                  (0 = market-like steps). For the MPC, which cannot act on a price jump.
    profile       48 half-hour values: multipliers on `mean` (absolute=False) or NZ$/MWh
                  (absolute=True, e.g. from load_emi_csv)
    """
    mode: str = 'daily'
    mean: float = 120.0
    import_adder: float = 60.0
    day_length_s: float = 6*3600.0
    start_hour: float = 6.0
    smooth_s: float = 0.0
    profile: tuple = DEFAULT_SHAPE
    absolute: bool = False

    def __post_init__(self):
        if self.mode not in ('fixed', 'daily'):
            raise ValueError(f"electricity price mode must be 'fixed' or 'daily', not {self.mode!r}")
        if len(self.profile) != 48:
            raise ValueError('profile needs 48 half-hourly values')

    # ---- day clock ----
    def day_seconds(self, t):
        """Time of day (0-86400 s) at simulation time t. Wraps forever."""
        return (t*DAY_S/self.day_length_s + self.start_hour*3600.0) % DAY_S

    def _slot_value(self, k):
        v = self.profile[k % 48]
        return v if self.absolute else self.mean*v

    def export(self, t):
        """NZ$/MWh the plant is paid for exported electricity at simulation time t."""
        if self.mode == 'fixed':
            return self.mean
        s = self.day_seconds(t)
        k = int(s // HALF_HOUR_S)
        cur = self._slot_value(k)
        if self.smooth_s <= 0.0:
            return cur
        into, left, half = s - k*HALF_HOUR_S, (k + 1)*HALF_HOUR_S - s, 0.5*self.smooth_s
        if into < half:            # blend in from the previous slot: midpoint at the boundary
            a = 0.5 + 0.5*math.sin(math.pi*into/self.smooth_s)
            return self._slot_value(k - 1) + a*(cur - self._slot_value(k - 1))
        if left < half:            # blend out towards the next slot
            a = 0.5 + 0.5*math.sin(math.pi*left/self.smooth_s)
            return self._slot_value(k + 1) + a*(cur - self._slot_value(k + 1))
        return cur

    def import_(self, t):
        """NZ$/MWh the plant pays for imported electricity at simulation time t."""
        return self.export(t) + self.import_adder

    def forecast(self, t, times):
        """Export prices at the future simulation times t + each of `times` (a perfect forecast:
        the loop is deterministic). Used by the economic MPC."""
        return [self.export(t + dt) for dt in times]

    def slot(self, t):
        """Index of the half-hour price period in force at t (changes when the price steps).
        Always 0 in fixed mode."""
        if self.mode == 'fixed':
            return 0
        return int(self.day_seconds(t) // HALF_HOUR_S)

    def sim_seconds_per_step(self):
        """Simulated seconds between price steps (7.5 min at the default 6 h day)."""
        return HALF_HOUR_S*self.day_length_s/DAY_S

    def clock_text(self, t):
        s = self.day_seconds(t)
        return f'{int(s // 3600):02d}:{int(s % 3600 // 60):02d}'


# ---------------------------------------------------------------------------------------------
#  All prices
# ---------------------------------------------------------------------------------------------
@dataclass
class Prices:
    """Every price the margin uses. Only `electricity` can change with time."""
    electricity: ElectricityPrice = field(default_factory=ElectricityPrice)
    biomass_per_gj: float = 12.0        # NZ$/GJ, net as-fired energy
    mps_steam_per_t: float = 45.0       # NZ$/t delivered
    lps_steam_per_t: float = 40.0       # NZ$/t delivered
    hot_water_per_t: float = 2.0        # NZ$/t delivered, up to the minimum only


#  Biomass price choices, NZ$/GJ (2026-09-28). 12 is the base case and 15 the top of the cited
#  NZ$9-15/GJ chip range (Bioenergy Association TNSB74); 18 is a sensitivity case (a tighter
#  wood-fuel market). The price decides HP2's break-even: ~NZ$78, 98 and 117/MWh respectively.
BIOMASS_PRICE_OPTIONS = (12.0, 15.0, 18.0)


#  Fuel is kg/s DRY; boiler.Q_PER_FUEL (kJ per dry kg) is the net as-fired heat release, i.e.
#  after evaporating the fuel's 40% moisture, which is the basis NZ$/GJ wood prices are quoted on.
GJ_PER_KG_FUEL = blr.Q_PER_FUEL/1e6


def margin(sample, prices, t=None):
    """NZ$/h terms for one plant sample (a rig row). t defaults to sample['t'].

    Uses: mfuel_act (fuel actually fed), power (gross turbine kW), W_hp_total (heat-pump kW),
    m_mps_delivered / m_lps_delivered (site draw), d_mps / d_lps, m_hw_total, hw_min.
    """
    t = sample['t'] if t is None else t
    pe = prices.electricity
    p_exp, p_imp = pe.export(t), pe.import_(t)
    fuel_cost = sample['mfuel_act']*GJ_PER_KG_FUEL*3600.0*prices.biomass_per_gj
    power_net = sample['power'] - sample['W_hp_total']                 # kW
    elec = power_net/1000.0*(p_exp if power_net >= 0.0 else p_imp)     # NZ$/h, + earned
    mps = min(max(sample['m_mps_delivered'], 0.0), sample['d_mps'])
    lps = min(max(sample['m_lps_delivered'], 0.0), sample['d_lps'])
    hw = min(max(sample['m_hw_total'], 0.0), sample['hw_min'])
    revenue = 3.6*(mps*prices.mps_steam_per_t + lps*prices.lps_steam_per_t
                   + hw*prices.hot_water_per_t)                        # kg/s -> t/h
    return dict(price_export=p_exp, price_import=p_imp, power_net=power_net,
                fuel_cost_h=fuel_cost, elec_h=elec, revenue_h=revenue,
                margin_h=revenue + elec - fuel_cost)


ECON_FIELDS = ('price_export', 'price_import', 'power_net', 'fuel_cost_h', 'elec_h',
               'revenue_h', 'margin_h', 'margin_cum')


# ---------------------------------------------------------------------------------------------
#  Site demand schedule
# ---------------------------------------------------------------------------------------------
DEMAND_DRIVERS = ('d_mps', 'd_lps', 'hw_min')


@dataclass
class DemandSchedule:
    """MPS demand, LPS demand and hot-water minimum stepping to a new level every interval_s.

    Each step: new = old + pull*(base - old) + u, u uniform in +/- step_frac*(slider range),
    clipped to the slider limits. The first period is the base point. Seeded: the same seed gives
    the same sequence in every run and for every controller.
    """
    interval_s: float = 3600.0
    step_frac: float = 0.10
    pull: float = 0.2
    seed: int = 582
    base: dict = field(default_factory=lambda: dict(d_mps=pr.D_MPS_0, d_lps=pr.D_LPS_0,
                                                    hw_min=pr.HW_MIN_0))
    limits: dict = field(default_factory=lambda: dict(d_mps=(pr.D_MPS_LO, pr.D_MPS_HI),
                                                      d_lps=(pr.D_LPS_LO, pr.D_LPS_HI),
                                                      hw_min=(pr.HW_MIN_LO, pr.HW_MIN_HI)))

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._levels = [dict(self.base)]

    def _extend(self, k):
        while len(self._levels) <= k:
            old = self._levels[-1]
            new = {}
            for name in DEMAND_DRIVERS:        # fixed order: the sequence depends only on seed
                lo, hi = self.limits[name]
                u = self._rng.uniform(-1.0, 1.0)*self.step_frac*(hi - lo)
                v = old[name] + self.pull*(self.base[name] - old[name]) + u
                new[name] = float(min(max(v, lo), hi))
            self._levels.append(new)

    def period(self, t):
        return int(max(t, 0.0)//self.interval_s)

    def levels(self, t):
        """dict(d_mps, d_lps, hw_min) in force at simulation time t. Keeps changing for any t."""
        k = self.period(t)
        self._extend(k)
        return dict(self._levels[k])

    def next_change(self, t):
        return (self.period(t) + 1)*self.interval_s
