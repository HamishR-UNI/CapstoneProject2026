"""One-way header closures and explicit pressure-limited site service.

All capacities and service curves below are engineering assumptions, not vendor
ratings. The nominal point has zero spill/dump and full service. Pressure floors
are existing model-domain limits, not certified equipment operating limits.

LPS's own dump/relief was removed on 2026-09-23, then RESTORED the same day, gentler
than the original, once removing it turned out to break more than intended (see
docs/PLAN_A_addendum_2026-09-23.md for the full path there):

  - With the vent gone entirely, even a genuinely UNDISTURBED plant (no_control,
    nothing touched) drifted to the pressure stop in under 15 minutes, on nothing
    more than the always-present ~1e-4 kg/s mismatch between the static demand
    inverse and the dynamic model's true equilibrium -- a mismatch this small was
    always there, just always invisible, because the (removed) vent absorbed it
    instantly.
  - The plant has no OTHER negative feedback on LPS pressure once at or above
    nominal: the site delivery-availability law saturates at exactly the demand
    value there and stops responding to further pressure rise, and the primary heat
    pump's own duty is mildly self-REINFORCING (not correcting) in the 3-4 bar band
    right around nominal (see plant_operator.py's dashboard warning threshold and
    the addendum for the numbers) -- it only starts pulling the other way above ~4
    bar, by which point pressure can already be moving too fast for that lagged
    correction to catch it.
  - Removing it also cost `pid`/`feedforward_pid` their own margin on large
    disturbances they previously handled fine (`demand_step_test.py`'s own
    "reported" flagship case), not just `no_control` -- the vent was protecting
    every mode, not papering over the uncontrolled one specifically.

So the vent is back. It starts at exactly `lps_nominal`, instantly, the same as the
original always did -- a first attempt moved that onset out to 3.5 bar to give
`no_control` more visible room, and that was a real mistake, found from a fresh
screenshot showing `feedforward` (which has no pressure feedback of its own, and
never did) drifting a genuine 0.5 bar on nothing but the same tiny residual, which
it had never done before today. Since the vent runs identically in all four modes
(no branching), any gap in its onset shows up in every mode with no other way to
correct it -- not just the one mode that was actually being redesigned around it.
Onset is restored to instant; only the ceiling (raised to `lps_ceiling` below) and
the relief stage's capacity (sized for that higher ceiling) differ from the
original. It is still plant equipment, not a control action -- see `flows()` below
and plant_operator.py's own note on this -- and runs identically, unconditionally,
in all four regulation modes, the same as it always has.
"""
from dataclasses import dataclass

import heatpump_dyn as hpp


@dataclass(frozen=True)
class HeaderConfig:
    mps_nominal: float = 40.0       # letdown opens above this (flows() below)
    lps_nominal: float = 3.0
    #  SITE DELIVERY LAW, changed 2026-09-25 (twice).
    #  Originally the site got its full demand only at or above *_nominal (40 / 3 bar),
    #  ramping to zero at 25 / 2 bar. The static closure (PlantRig._invert,
    #  excel_verify/python_side.plant_steady and the Excel workbook) always assumes full
    #  delivery AT THE TARGET pressure, so any target below nominal described a state the
    #  dynamic plant could not be in (a 30 bar MPS target "serving 12 kg/s" where the
    #  dynamic site could only take 4 kg/s).
    #  First fix tried: full service across the whole slider range, flat draw above it.
    #  That matched the static closure but took away the headers' only self-regulation
    #  between the slider minimum and the letdown/vent onsets, and the MPS and LPS PI
    #  tunings turned out to depend on it (both were tuned at 40 / 3 bar, where that ramp
    #  plus the letdown/vent made each header self-regulating): at a 30 bar MPS target the
    #  MPS loop limit-cycled 24.5-39.7 bar with a ~150 s period, and at a 2.5 bar LPS
    #  target LPS cycled 2.0-3.6 bar.
    #  Now: consumers behave like fixed valves sized to draw exactly their demand at the
    #  header's TARGET pressure -- draw = demand * (P - sink) / (P_target - sink), sink =
    #  atmospheric. At the target the site takes exactly its demand (so the static closure
    #  and Excel are unchanged and still describe the dynamic plant's steady state, at ANY
    #  target), and around it the header self-regulates in both directions: a little more
    #  draw above target, a little less below. Capped at site_draw_cap x demand; with 3x no
    #  pressure inside the domain limits leaves a header without self-regulation (the cap
    #  is only reached above the letdown/vent onsets). Below target, the draw falling short
    #  of demand shows as a site shortfall, as before.
    #  mps_nominal stays at 40: it is also the letdown's opening pressure.
    site_draw_cap: float = 3.0
    #  Added 2026-09-24, same day and same shape as the mps_ceiling fix below -- these are
    #  domain-guard-ONLY floors. (They used to be separate from the delivery law's own
    #  zero points, mps_floor/lps_floor; those were removed 2026-09-25 with the new site
    #  law above, which has no floor of its own.) The guard
    #  in in_domain() was checking pm/pl against those SAME fields, which are exactly the
    #  sliders' own minimums (plant_rig.P_MPS_LO=25.0, P_LPS_LO=2.0) -- identical problem
    #  to mps_ceiling's, just on the low side: pinning a pressure target at the slider's
    #  own floor asks the plant to sit exactly on the domain guard's edge, with no room
    #  for undershoot on the way there (found the same way -- reported not stopping on
    #  the high side any more, then: "doesn't stop when LPS is set to 2 and MPS is
    #  minimum as well" on the low side).
    #  mps_domain_floor=22.5 stays inside turbine_model.MOVING_P1_LO (22.0 bar, same
    #  surrogate as mps_ceiling's own justification) with the same 0.5 bar cushion used
    #  there. LPS has far less room to work with: MOVING_P2_LO is 1.6 bar and the slider's
    #  own floor is already only 2.0, a 0.4 bar band total (vs MPS's 3.0). Split that
    #  band, not full cushion: lps_domain_floor=1.8, giving the slider's pl=2.0 minimum
    #  0.2 bar of undershoot room while keeping 0.2 bar clear of the turbine surrogate's
    #  own hard floor.
    #  2026-09-25: 22.5 -> 8.5 with the MPS target floor (plant_rig.P_MPS_LO 25 -> 10)
    #  and turbine_model.MOVING_P1_LO (22 -> 8 bar); same 0.5 bar cushion above the fit,
    #  1.5 bar of undershoot room below the slider's minimum.
    mps_domain_floor: float = 8.5
    lps_domain_floor: float = 1.8
    #  Raised 2026-09-24 from 55.0 -- exactly plant_rig.P_MPS_HI, the pressure-target
    #  slider's own max -- to 57.5. Asking the plant to hold pressure AT the domain
    #  ceiling left no room for step-response overshoot on the way there, so a run in
    #  FEEDFORWARD tripped this guard at t=173s with the slider simply pinned at 55 bar
    #  (screenshot: MPS pressure oscillating right at 55, LPS shortfall a tiny 0.01 kg/s
    #  -- not a large disturbance, just no headroom). 57.5 stays inside
    #  turbine_model.MOVING_P1_HI (58.0 bar, turbine 2's own inlet-fit ceiling for the
    #  MPS header) with margin to spare, so the surrogate is never asked to extrapolate.
    #  Slider max (plant_rig.P_MPS_HI) deliberately left at 55 -- Hamish asked for the
    #  ceiling to move, not the slider.
    #  Raised again 2026-09-24 to 100 bar (Hamish: only stop MPS at 100 bar). That is the
    #  boiler's own fixed outlet pressure (plant_rig.P_HP); well before it turbine 2's inlet
    #  surrogate is extrapolating (MOVING_P1_HI 58 bar) and turbine 1's pressure drop, hence
    #  its power, heads to zero/negative -- results up there are not physical, accepted
    #  knowingly ("it's a simulation, if it gets too high I fix it later").
    mps_ceiling: float = 100.0
    #  Raised 2026-09-23 from a conservative 4.5 bar margin to heatpump_dyn.P_LPS_CRIT_BAR
    #  (~5.54 bar) -- the actual point R1233zd(E) can no longer condense at all, not just a
    #  margin below it (see 2026-09-23 note above). LPS TARGETS still stay 2-4 bar
    #  (plant_rig.P_LPS_LO/HI); turbine 2's own fit is separately valid to 4.6 bar
    #  (turbine_model.MOVING_P2_HI), inside this new ceiling. Above the old 4.5 bar mark,
    #  results already reflect the surrogates' own extrapolation rather than a re-verified
    #  fit -- the dashboard's own warning (plant_operator.py) flags that separately from
    #  this hard stop, which exists only to keep the model from computing on a refrigerant
    #  state that is not physically real.
    #  Raised 2026-09-24 to 10 bar (Hamish: only stop if LPS goes above 10 bar), and the
    #  dashboard's LPS warning removed. Above P_LPS_CRIT_BAR (~5.54 bar) the primary heat
    #  pump could not really condense; the model does not fail there because
    #  heatpump_dyn.build_dynamics clamps the pressure its cycle fits see to P_LPS_RANGE
    #  (<= 4 bar), so above ~4 bar the heat pump simply behaves as if at 4 bar. Turbine 2's
    #  outlet surrogate is extrapolating above 4.6 bar. Accepted knowingly, as for MPS.
    lps_ceiling: float = 10.0
    letdown_gain: float = 1.0
    letdown_capacity: float = 15.0  # previous slope * nominal-to-upper pressure span
    #  VENT, restored 2026-09-23 -- see the module docstring for why. Two stages, same
    #  shape and same onset as the original (dump_start == lps_nominal, instant -- a
    #  gap here was tried and found to make feedforward/pid/feedforward_pid drift on
    #  nothing too, since none of them correct this specific residual either). Only
    #  difference from the original: relief_gain/capacity reduced from 100/45 to 60/60
    #  -- weaker per bar, but with more total capacity, since it now has to arrest a
    #  disturbance before lps_ceiling, which is over a bar higher than the original 4.5.
    dump_start: float = 3.0  # == lps_nominal above; kept a separate field, not a reference to
                             # it, since dataclass field defaults can't refer to sibling fields
    dump_gain: float = 8.0
    dump_capacity: float = 16.0
    #  Relief valve reshaped 2026-09-24. The old stage was a straight 60 kg/s per bar
    #  from 4.0 bar, clipped at 60 kg/s: two hard corners, so LPS pressure hit 4.0,
    #  flattened abruptly at ~4.4 and the trend showed a sharp corner -- and the 4.5 bar
    #  dashboard warning could never show. (Briefly disabled outright the same day; with
    #  no relief a large LPS surplus ran to the lps_ceiling stop within ~10 s.)
    #  Now a gradually lifting valve: lift follows a smoothstep from cracking at
    #  relief_start to full lift at relief_full, flow = relief_capacity * lift. Zero flow
    #  below relief_start (no leak at nominal) and zero slope at both ends, so no corner in
    #  the trend. A ~30 kg/s surplus now settles mid-band (~4.5 bar) instead of being
    #  pinned at 4.4. The gentle dump above is unchanged.
    #  No flat maximum either (same day): past full lift the flow keeps rising in
    #  proportion to LPS pressure, as choked flow through a fixed open area does
    #  (flow ~ upstream absolute pressure), so relief keeps growing the higher LPS goes
    #  -- 60 kg/s at relief_full, ~115 kg/s at 10 bar. The pressure factor is held at
    #  lps_ceiling only so the flow has a finite bound (max_vent()); the run stops there.
    #  relief_full widened 5.2 -> 6.5 bar (2026-09-24, same day) once the valve got its
    #  own FOPTD lift response in plant_rig: the steeper curve (~80 kg/s per bar near 4.5
    #  bar) rang against the valve lag and the small LPS header. Gentler curve = less
    #  stiff = better damped, at the cost of settling higher for the same surplus.
    relief_start: float = 4.0
    relief_full: float = 6.5
    relief_capacity: float = 60.0   # kg/s at full lift, AT relief_full; scales with P above
    sink_pressure: float = 1.01325 # atmospheric receiver, absolute bar
    forward_pressure_span: float = 1.0 # full forward permissive by +1 bar


DEFAULT = HeaderConfig()
FIELDS = ('m_letdown', 'm_vent', 'm_mps_delivered', 'm_lps_delivered',
          'mps_shortfall', 'lps_shortfall', 'm_mps_supply', 'm_lps_supply',
          'net_mps', 'net_lps', 'domain_ok', 'terminal', 'stop_reason')


def clipped(x, low, high, absolute=abs):
    """Exact bounded piecewise linear expression; no binary or MPCC variables."""
    positive = lambda z: (z + absolute(z)) / 2
    return low + positive(x-low) - positive(x-high)


def relief(pl, config=DEFAULT, absolute=abs):
    """LPS relief valve flow at pressure pl, STEADY-STATE (valve fully settled).

    Lift: smoothstep 3s^2 - 2s^3 over relief_start..relief_full (see HeaderConfig);
    the clip's corners at s=0 and s=1 are where the smoothstep's slope is already
    zero, so the flow curve has no corner. Times the choked-flow pressure factor
    P/relief_full. plant_rig puts the valve's own FOPTD response on this (the valve
    takes time to lift); flows() below includes it undelayed for static uses.
    """
    c = config
    clip = lambda x, lo, hi: clipped(x, lo, hi, absolute)
    s = clip((pl-c.relief_start)/(c.relief_full-c.relief_start), 0, 1)
    return c.relief_capacity*s*s*(3-2*s)*clip(pl, 0, c.lps_ceiling)/c.relief_full


def flows(pm, pl, demand_mps, demand_lps, config=DEFAULT, absolute=abs,
          include_relief=True, target_mps=None, target_lps=None):
    """include_relief=False leaves the relief valve out of 'dump', for a caller that
    models the valve's dynamics itself (plant_rig). Relief is 0 below relief_start, and
    the sink permissive below is 1 there, so leaving it out is exact, not approximate.

    target_mps/target_lps: the header pressure targets the site draw is referenced to
    (see HeaderConfig's SITE DELIVERY LAW). Omitted, the pressure itself is taken as the
    target, i.e. the site takes exactly its demand -- what every static caller wants."""
    c = config
    clip = lambda x, lo, hi: clipped(x, lo, hi, absolute)
    letdown = clip(c.letdown_gain*(pm-c.mps_nominal), 0, c.letdown_capacity)
    letdown *= clip((pm-pl)/c.forward_pressure_span, 0, 1)
    #  Gentle vent, restored 2026-09-23 -- see module docstring. Starts at dump_start
    #  (looser than lps_nominal), not the instant pl exceeds nominal.
    dump = clip(c.dump_gain*(pl-c.dump_start), 0, c.dump_capacity)
    if include_relief:
        dump = dump + relief(pl, c, absolute)
    dump *= clip((pl-c.sink_pressure)/c.forward_pressure_span, 0, 1)
    # Site draw: fixed consumer valves sized for demand at the target pressure (see
    # HeaderConfig). Assumed site behaviour, not an extra steam source or PID.
    tm = pm if target_mps is None else target_mps
    tl = pl if target_lps is None else target_lps
    ps = c.sink_pressure
    mps = demand_mps*clip((pm-ps)/(tm-ps), 0, c.site_draw_cap)
    lps = demand_lps*clip((pl-ps)/(tl-ps), 0, c.site_draw_cap)
    return dict(letdown=letdown, dump=dump, mps=mps, lps=lps)


def max_vent(config=DEFAULT):
    """Upper bound on flows()['dump']: full dump plus full-lift relief at lps_ceiling."""
    c = config
    return c.dump_capacity + c.relief_capacity*c.lps_ceiling/c.relief_full


def in_domain(pm, pl, config=DEFAULT):
    c = config
    #  LPS ceiling raised, not removed, 2026-09-23 -- see HeaderConfig.lps_ceiling's own
    #  comment. Tried removing it outright first: with the dump/relief above also gone, an
    #  unresponsive plant (no_control) drove LPS past 100 bar in two minutes before the GEKKO
    #  solver itself failed, which is the model breaking, not "showing where it goes." Stopping
    #  instead at heatpump_dyn.P_LPS_CRIT_BAR keeps the header well clear of the old 4.5 bar
    #  margin (so it genuinely reads as unresponsive to a demand step, not artificially capped)
    #  while stopping at the one point in this model that isn't an engineering assumption: the
    #  refrigerant physically cannot condense any hotter than that. mps_ceiling was raised
    #  2026-09-24 (see its own field comment above); the two floor checks below were changed the
    #  same day, from mps_floor/lps_floor to the dedicated mps_domain_floor/lps_domain_floor (see
    #  those fields' own comment) -- mps_floor/lps_floor themselves are still exactly what they
    #  always were: the delivery law's own zero point in flows() above, untouched by this guard.
    return c.mps_domain_floor <= pm <= c.mps_ceiling and c.lps_domain_floor <= pl <= c.lps_ceiling
