"""Demand-driven biomass CHP plant, stepped with GEKKO IMODE=4.

The boiler feeds turbine 1 and the MPS header; turbine 2 takes a controlled
fraction of that steam to LPS. Both header pressures are accumulation states:
C_MPS*dP_MPS/dt = steam - delivered_MPS - m_to_t2 - m_letdown
C_LPS*dP_LPS/dt = m_to_t2 + m_lps_hp + m_letdown - delivered_LPS - m_vent.
The spill/dump closures are bounded and one-way. Site delivery decreases below
nominal pressure using an explicit assumed availability curve; unmet demand is
reported. Runs stop on solver failure or pressure-domain exit. See
header_revision_report.md for assumptions and revised validation.
Turbines have no swallowing-capacity relation and HP pressure remains fixed.

Pressure inputs are setpoints. P_LPS also enters the static demand inverse;
P_MPS does not directly command actual pressure. No pressure controller is here.
The inverse computes fuel, turbine-2 fraction and primary subcooling from site
steam draws and the hot-water minimum. Optional fuel_trim/frac_t2_trim add
command-side corrections before clipping and the existing dispatch dynamics.
Explicit fuel/fraction commands remain available for open-loop experiments.

HP2 duty Q_sec is the independent economic fuel/electricity lever. HP1 uses
available flue-gas, recycle and HP2 heat. Hot-water temperature follows local
specified dynamics; the HX derives water flow from temperature and duty, so
this is not a valve-to-temperature feedback model.

Sample diagnostics expose untrimmed commands, requested/applied trims, final
commands and clipping flags. In explicit mode, 'feedforward' labels the
untrimmed explicit baseline. Existing inverse feasibility flags are separate.
Fuel's upper bound is the boiler surrogate validity limit, not plant capacity.
See header_revision_report.md for revised controller-facing dynamics.
The signed-flow results in pid_identification_report.md are historical only.
"""

import numpy as np
from gekko import GEKKO

import boiler as blr
import turbine_model as tmod
from turbine_model import Turbine, lag_chain
import flue_gas_water_hx as hx
import heatpump_secondary as hps
import heatpump_dyn as hpp
import dynamics_params as dp
import header_closures as headers

HEADER_MODEL_REVISION = 'one_way_pressure_limited_v1'

# ----------------------------------------------------------------------------------
#  Operating point
# ----------------------------------------------------------------------------------

#  kg/s dry fuel -- Newton-solve INITIAL GUESS for the demand-consistent base
#  fuel rate PlantRig.__init__ actually seeds the model at (see fuel0 there),
#  not the base rate itself. It only matters for how fast that solve
#  converges, never for where it lands.
MFUEL_0 = 10.0
P_MPS_0 = tmod.POUT             # bar, back pressure
P_HP = blr.P_HP_BAR             # bar, boiler outlet = turbine inlet, fixed
T_HP = blr.T_STEAM_C            # degC, superheater outlet = turbine inlet, fixed
ETA = tmod.ETA

#  NO CEILING ON THE FUEL. There is no plant limit represented here -- if the
#  site demands imply 20 kg/s, the model runs at 20 kg/s and says so. The only
#  bound is how far boiler.py's surrogate has actually been fitted, which is now
#  2x the design rate, and hitting it means the fit has run out rather than the
#  boiler has. MFUEL_HI is not a plant ceiling and should not be read as one.
#  MFUEL_LO was a 3.0 kg/s turndown floor (30% of design) until 2026-09-24; now 0,
#  at Hamish's request, with boiler.py refitted down to 0 so this is not
#  extrapolation. A real grate boiler cannot fire that low -- the model no longer
#  represents that limit. The demand inverse still never goes below the fuel that
#  serves the MPS draw (D_MPS_LO = 2 kg/s -> ~0.4 kg/s); fuel reaches 0 only by
#  explicit command (no_control) or a PID trim.
MFUEL_LO, MFUEL_HI = 0.0, blr.MDOT_FUEL_MAX
#  P_MPS_LO 25 -> 10 bar on 2026-09-25 at Hamish's request (the model stopped when the
#  target was dropped). Needed the turbine fits widened to match: turbine 1's outlet fit
#  now runs 8-62 bar and turbine 2's inlet fit 8-58 bar, with a superheat correction for
#  low-MPS expansions that no longer end inside the dome (turbine_model, MOVING_P1_LO).
P_MPS_LO, P_MPS_HI = 10.0, 55.0 # bar

#  Turbine 2, MPS -> LPS. ETA2 and the stage volume below (V_STAGE2) are
#  PLACEHOLDERS -- turbine 2's real isentropic efficiency and geometry are not
#  yet known, so this reuses turbine 1's numbers, same as V_STAGE does for
#  turbine 1 in turbine_model.py. Replace both once real design data exists.
#  -   fraction of MPS admitted to turbine 2. Same status as MFUEL_0 above --
#  a placeholder used nowhere once __init__ has solved frac0 from the base
#  demands; kept only as a documented default.
FRAC_T2_0 = 0.50
ETA2 = ETA                       # PLACEHOLDER, see note above

#  Property route for turbine 2. True uses the polynomial surrogate in
#  turbine_model.py (about 30x cheaper, and the dominant cost in this model);
#  False restores the full symbolic IAPWS-95 expansion. Set False for
#  validation runs, for any operating point outside the MOVING_* fit box, or
#  if turbine 2's exhaust is ever expected to land inside the vapour dome --
#  the surrogate publishes turb2.x_ideal and turb2.dsh_act so that can be
#  watched rather than assumed.
T2_SURROGATE = True

FRAC_T2_LO, FRAC_T2_HI = 0.0, 1.0
#  TURBINE 2 CUT-OFF (2026-09-25). A split below this is treated as turbine 2 OFF (split
#  0): a turbine is not run at a sliver of a percent of its flow. Applied to every split
#  the plant is commanded (PlantRig.step) and every split the static closure reports
#  (_frac_t2_from_fuel here, excel_verify/python_side.plant_steady, and the Excel
#  workbook's 9_Solve!C60), so all of them agree. It also stops a numerically-zero
#  split (e.g. Python 5.5e-7 vs Excel exactly 0) reading as a 100% power2 difference in
#  the Excel comparison. The steam turbine 2 would have taken stays on the MPS header.
FRAC_T2_OFF = 0.01


def turbine2_cutoff(frac):
    """Split with turbine 2 turned off below FRAC_T2_OFF (see above)."""
    return 0.0 if frac < FRAC_T2_OFF else frac

#  Turbine 2's EOS path (state_ps_smooth/state_ph_smooth, see turbine_model.py)
#  is dome-aware, so a low back pressure that puts the exhaust into the
#  two-phase region is no longer a hard floor the way it used to be -- P_LPS
#  now shares its header with the primary heat pump's own LPS output
#  (hpp.build_dynamics' P_LPS_bar). Kept to 2-4 bar, matching
#  heatpump_dyn.P_LPS_RANGE exactly, so the heat pump is always actually
#  responding to this slider rather than sitting clamped at its own ceiling
#  for most of it -- turbine 2 itself could go wider now, but there's no
#  reason to let this MV roam somewhere the heat pump can't follow it.
P_LPS_0 = 3.0                    # bar, turbine 2 back pressure
P_LPS_LO, P_LPS_HI = 2.0, 4.0    # bar

#  Flue gas HX hot water setpoint -- the plant MV is now the site's desired
#  temperature, not the flow (see flue_gas_water_hx.py); range and default
#  come from that module, not redefined here. M_WATER_0 is flue_gas_water_hx's
#  OWN design-point flow, at ITS design fuel/duty -- not fuel0's. It is kept
#  here only as a documented reference value; __init__ seeds the model's water
#  states from self.hx_ss['m_water'] instead, the flow fuel0/TWOUT_0 actually
#  imply, so the seed agrees with the demand-consistent base point everything
#  else is built on rather than with a different, unrelated operating point.
TWOUT_0 = hx.T_W_OUT_0
TWOUT_LO, TWOUT_HI = hx.T_W_OUT_LO, hx.T_W_OUT_HI
M_WATER_0 = hx.M_WATER_0

#  Two independently dispatched heat pumps, downstream of the HX on the flue gas
#  exhaust -- ranges and defaults come from their own modules, not redefined here.
#  See heatpump_dyn.py's module docstring for why the secondary and primary carry
#  separate duty commands rather than one.
Q_SEC_0 = hps.Q_SEC_0
Q_SEC_LO, Q_SEC_HI = hps.Q_SEC_LO, hps.Q_SEC_HI

Q_PRI_0 = hpp.Q_PRI_0
Q_PRI_LO, Q_PRI_HI = hpp.Q_PRI_LO, hpp.Q_PRI_HI

#  SUBCOOLING IS A DEGREE OF FREEDOM AGAIN, not a fixed design value.
#  It is the handle that lets the LPS header be matched without venting: it sets
#  how the primary heat pump's condenser duty divides between raising LPS and
#  making hot water, and over its range that split moves a long way --
#
#      dt_sub  2 K   ->  96% of the duty to LPS,  4% to hot water
#      dt_sub 25 K   ->  70% to LPS,             30% to hot water
#
#  So when the hot water minimum bites, the first thing to reach for is not more
#  fuel: it is moving the heat pump's output toward hot water, which raises hot
#  water AND drops the heat pump's LPS so the turbine picks up the difference.
#  Only when this is exhausted does firing rate have to rise, and only then is
#  there surplus steam with nowhere to go.
DT_SUB_0 = hpp.DT_SUB_0
DT_SUB_LO, DT_SUB_HI = hpp.DT_SUB_RANGE

# ----------------------------------------------------------------------------------
#  Site demands.  These are DISTURBANCES, not plant handles: the site takes what it
#  takes and the plant has to follow. They replace fuel and frac_T2 as the things a
#  person sets, because those two are now computed from these (see _invert below).
#
#  The base pair puts the plant at 8 kg/s of fuel, mid-range of the boiler's fitted
#  3-20 kg/s, so there is headroom in both directions. The ranges are deliberately wider than
#  the plant can serve -- the top corner needs about 16 kg/s of fuel and the bottom
#  corner about 3.7 -- because a demand the plant cannot meet is a legitimate thing
#  to look at, and the rig flags it rather than forbidding it (the same philosophy
#  the duty sliders already follow).
#
#  Hot water is a MINIMUM, not a setpoint. It costs no degree of freedom unless it
#  binds, which is what lets the two steam demands be met exactly without the three
#  requirements over-determining the plant. At the base point there is plenty of
#  slack; it starts to bind below roughly 6.5 kg/s of fuel, which is the interesting
#  case -- the site wants little steam but still wants its hot water, so the boiler
#  has to keep firing anyway.
# ----------------------------------------------------------------------------------

D_MPS_0 = 12.0                   # kg/s, site MPS draw
D_MPS_LO, D_MPS_HI = 2.0, 30.0

D_LPS_0 = 34.0                   # kg/s, site LPS draw
D_LPS_LO, D_LPS_HI = 5.0, 60.0

HW_MIN_0 = 45.0                  # kg/s, site hot water MINIMUM
HW_MIN_LO, HW_MIN_HI = 0.0, 120.0

#  False restores the old behaviour: fuel and frac_T2 are set directly and the
#  demands become passive readouts. Keep it available -- the open-loop step test
#  and the no-regulation baseline both need to drive fuel by hand, and neither
#  works if the inverse is always in the way.
DEMAND_DRIVEN = True

#  By design, not incidentally: the primary is the plant's main heat-upgrade unit
#  and the secondary is only a trim/booster source, so its capacity should exceed
#  the secondary's -- guard it here rather than let a future constant change in
#  either module silently invert that relationship.
assert Q_PRI_HI > Q_SEC_HI, (
    f'primary duty range (up to {Q_PRI_HI:.0f} kW) should exceed the secondary\'s '
    f'(up to {Q_SEC_HI:.0f} kW) -- check heatpump_dyn.Q_MCR / heatpump_secondary.Q_SEC_MCR sizing')

# ----------------------------------------------------------------------------------
#  Dynamics.  Each component keeps its own time constant, from its own volume.
# ----------------------------------------------------------------------------------

TAU_BOILER = blr.TAU_Q          # s, fuel bed + gas flush -- see boiler.py
THETA_FUEL = blr.THETA_FUEL     # s, feeder + fuel-bed dead time + steam line
TAU_STAGE = tmod.TAU            # s, V/Vdot on the steam chest
THETA_STAGE = tmod.THETA_STAGE  # s, steam chest + blading transit dead time
#  No longer wired to a back-pressure valve lag (restructure phase 2/3 replaced
#  that with the real header states, pmps_act/plps_act) -- kept as a documented
#  reference value only, still printed in the manual dashboards' CSV header.
TAU_VALVE = dp.TAU_VALVE_S      # s, valve stroke reference (unused directly here)

#  PLACEHOLDER volume, same status as turbine_model.V_STAGE for turbine 1.
#  TAU_STAGE2 itself can't be a module constant like TAU_STAGE -- turbine 2's
#  inlet state (turbine 1's exhaust) is only known once PlantRig has solved
#  turbine 1's steady state, so it's computed in PlantRig.__init__.
V_STAGE2 = tmod.V_STAGE          # m3, PLACEHOLDER turbine 2 stage volume
TAU_ADMIT = dp.TAU_VALVE_S       # s, admission valve stroke (frac_T2)

#  HP2 (secondary) condenser -> HP1 (primary) evaporator intermediate water loop. Plug-flow
#  pipe, so a dead time (no mixing-volume tau -- no unsourced holdup assumption to invent).
#  DT_LINK_K and V_LINK_MPS are recorded for the pipe-sizing basis and printed (see
#  _self_check); the loop flow they imply, hps.Q_SEC_0/(hx.CP_WATER*DT_LINK_K), is a printed
#  diagnostic only.
LINK_DYNAMICS = True
THETA_LINK = dp.L_LINK_M / dp.V_LINK_MPS   # s, plug-flow pipe HP2 -> HP1
TAU_LINK = 0.0                             # s, no unsourced mixing volume

#  ----------------------------------------------------------------------------------
#  MPS/LPS header accumulation (restructure phase 2/3). P_MPS/P_LPS used to be
#  valve-lagged copies of an operator command -- imposed, not emergent -- so a site
#  demand step never touched header pressure at all. They are now real states
#  integrating the steam mass balance (C*dP/dt = in - out), so a demand disturbance
#  shows up as a genuine pressure disturbance and pmps_cmd/plps_cmd become
#  SETPOINTS rather than direct actuator commands (the actuator, once phase 16's
#  PID exists, is the fuel/frac_T2 trim that moves flow to hold the setpoint -- see
#  plant_rig_pid.py). Until then nothing forces P_MPS/P_LPS toward their setpoints;
#  that is deliberate, not an oversight, per the restructure plan's phase 3.
#
#  Capacitance C = V*(drho/dP)|T, the PROPERTY-BASED route (not the coarser V*rho/P
#  approximation), evaluated once per instance at the real demand-consistent base
#  point in PlantRig.__init__ (see C_MPS/C_LPS there) via CoolProp. Header VOLUMES
#  below are an ENGINEERING ASSUMPTION -- no real plant geometry is available --
#  sized as modest distribution-header-plus-piping volumes; LPS is bigger because
#  its lower density needs more volume to hold a comparable mass.
V_MPS_M3 = 15.0                 # m3, ASSUMED -- MPS header + connected piping
V_LPS_M3 = 40.0                 # m3, ASSUMED -- LPS header + connected piping

# One-way operating spill/dump plus pressure-limited site delivery.
# Capacities/curves are explicit assumptions in header_closures.py.
# Legacy signed-flow plant is saved in header_revision_results/legacy/.
K_LETDOWN_MPS_LPS = headers.DEFAULT.letdown_gain
K_VENT_LPS = headers.DEFAULT.dump_gain
HEADER_DIAGNOSTIC_FIELDS = headers.FIELDS

DT = 1.0                        # s   sample, and the trend resolution
#  Turbine 2's live IAPWS-95 solve (fixed_inlet=False) is compute-bound, not
#  subprocess/parse overhead like the rest of this model -- measured around
#  2.3 s/solve at BATCH=1, rising only to ~2.8 s/solve at BATCH=4 (0.7 s/sample).
#  Batching trades trend smoothness for real throughput here: each GUI tick
#  still blocks for roughly the same wall time either way, so a higher BATCH
#  gets proportionally more sim-time per freeze for about the same per-tick
#  wall time. Interactive response is genuinely slower than the
#  turbine-1-only rig; there is no cheap fix while turbine 2 needs a live
#  property solve at every step.
BATCH = 10                      # -   samples per solve, see note above
#  Collocation nodes per 1 s interval. Lowered 4 -> 3 on 2026-09-25 for speed: ~9% less
#  wall time per solve, and over a 900 s Feedforward+PID run with LPS demand, MPS
#  target and HW minimum steps the largest differences from NODES=4 were 0.006 bar
#  MPS, 0.002 bar LPS, 0.002 kg/s fuel, 7 kW power. NODES=2 was ~18% faster but moved
#  MPS by 0.2 bar and power by 170 kW -- do not go below 3.
NODES = 3                       # -


TRIM_DIAGNOSTIC_FIELDS = (
    'fuel_feedforward', 'fuel_trim_requested', 'fuel_trim_applied',
    'fuel_final', 'fuel_saturated', 'frac_t2_feedforward',
    'frac_t2_trim_requested', 'frac_t2_trim_applied', 'frac_t2_final',
    'frac_t2_saturated',
)


class PlantRig:
    """Boiler, Turbine 1 and Turbine 2 as one plant you can step."""

    def __init__(self, verbose=True, batch=BATCH):
        self.batch = max(1, int(batch))
        self.halted = False
        self.stop_reason = ""

        #  DEMAND-CONSISTENT BASE POINT. The plant is DEMAND_DRIVEN, so the
        #  fuel/frac_T2/dt_sub the model actually starts at have to be the
        #  same triple _invert would compute for the base sliders
        #  (D_MPS_0, D_LPS_0, HW_MIN_0, Q_SEC_0, TWOUT_0, P_LPS_0) -- not the
        #  separately hand-picked MFUEL_0/FRAC_T2_0/DT_SUB_0 placeholders.
        #  Solved here with _solve_demands/_frac_t2_from_fuel BEFORE the GEKKO
        #  model exists (that solve is plain NumPy, no GEKKO object needed),
        #  so every steady state and seed below is built on the point the
        #  plant will actually be sitting at when the window opens paused --
        #  otherwise the first Run press yanks fuel/frac_T2 to wherever the
        #  demand inverse actually wanted them, which reads as the plant not
        #  starting "at setpoint".
        p_lps_base = float(np.clip(P_LPS_0, *hpp.P_LPS_RANGE))
        fuel0, dt_sub0, _, _, _ = self._solve_demands(
            D_MPS_0, D_LPS_0, HW_MIN_0, Q_SEC_0, TWOUT_0, p_lps_base,
            hpp.dh_hw(TWOUT_0), fuel_guess=MFUEL_0)
        frac0, _ = self._frac_t2_from_fuel(fuel0, D_MPS_0)
        self.fuel_base, self.frac_t2_base, self.dt_sub_base = fuel0, frac0, dt_sub0

        #  Independent steady states, for seeding and for the self-check.
        self.blr_ss = blr.steady_state(fuel0, P_HP)
        self.trb_ss = tmod.steady(self.blr_ss['mdot_steam'], P_MPS_0, T_HP, P_HP, ETA)
        #  Turbine 2's inlet is turbine 1's exhaust, so its independent steady
        #  state is chained off trb_ss rather than off the fixed HP interface.
        self.trb2_ss = tmod.steady(frac0*self.blr_ss['mdot_steam'], P_LPS_0,
                                   self.trb_ss['T_out'], P_MPS_0, ETA2)
        tau_stage2 = tmod.stage_tau(V_STAGE2, frac0*self.blr_ss['mdot_steam'],
                                    self.trb_ss['T_out'], P_MPS_0)
        #  Flue gas HX independent steady state, chained off the boiler's own
        #  flue gas outlet -- same role as blr_ss/trb_ss above.
        self.hx_ss = hx.steady_state(self.blr_ss['mdot_gas'], self.blr_ss['T_stack'],
                                     TWOUT_0)

        m = GEKKO(remote=False)
        m.time = [DT*i for i in range(self.batch + 1)]
        m.options.IMODE = 4
        m.options.NODES = NODES
        self.m = m

        self.mfuel_cmd = m.Param(value=fuel0, name='mfuel_cmd')
        #  Pressure setpoints do not directly impose actual pressure.
        #  P_LPS also enters the static inverse; feedback is not implemented.
        self.pmps_cmd = m.Param(value=P_MPS_0, name='pmps_cmd')
        self.frac_t2_cmd = m.Param(value=frac0, name='fract2_cmd')
        self.plps_cmd = m.Param(value=P_LPS_0, name='plps_cmd')
        #  Site demands as live GEKKO Params, not just the plain Python floats
        #  d_mps_val/d_lps_val below -- the header mass balances need them
        #  inside the pressure-limited delivery law, as well as in the inverse.
        self.d_mps_param = m.Param(value=D_MPS_0, name='d_mps')
        self.d_lps_param = m.Param(value=D_LPS_0, name='d_lps')

        #  DISPATCH DELAY. mfuel_cmd/frac_t2_cmd are not operator-set MVs like
        #  pmps_cmd/plps_cmd above -- they are recomputed from the site demands
        #  by _invert on every solve, and until now the result was applied to
        #  the boiler/admission valve the instant it was computed. A real
        #  supervisory calc-and-dispatch cycle is not instantaneous either, so
        #  a short, dead-time-only delay (THETA_DISPATCH_S, see
        #  dynamics_params.py) sits between the freshly solved target and
        #  everything physical downstream of it. Short relative to that
        #  downstream: seconds, against a feeder dead time of ~60 s and a fuel
        #  bed lag of ~600 s -- it moves the target a few samples later, not
        #  the shape of the transient that follows.
        self.mfuel_disp, fuel_disp_states = lag_chain(
            m, self.mfuel_cmd, theta=dp.THETA_DISPATCH_S, tau=0.0, n_lag=3,
            y0=fuel0, name='mfueldisp')
        self.frac_t2_disp, frac_disp_states = lag_chain(
            m, self.frac_t2_cmd, theta=dp.THETA_DISPATCH_S, tau=0.0, n_lag=3,
            y0=frac0, name='fract2disp')

        #  Boiler. P_HP is fixed, so its pressure path is inert but harmless.
        p_hp_const = m.Param(value=P_HP, name='p_hp')
        d = blr.build_dynamics(m, self.mfuel_disp, p_hp_const, fuel0, P_HP)
        self.mf, self.Q = d['mf'], d['Q']
        self.stack, self.gas, self.steam = d['stack'], d['gas'], d['steam']
        self.eta_b = d['eta']

        #  MPS header pressure. A real state now (see the module docstring's
        #  header-accumulation section), not a valve-lagged copy of pmps_cmd --
        #  its governing mass-balance Equation is added further down, once
        #  every flow that feeds it (self.steam, self.m_to_t2, the letdown) is
        #  available. Declared here because turbine 1/2 need it as an input.
        #  Property-based capacitance, evaluated at THIS instance's real
        #  demand-consistent base point (self.trb_ss['T_out']), not a module
        #  constant guessed before the base point is known.
        self.C_MPS = V_MPS_M3 * self._drho_dp(self.trb_ss['T_out'], P_MPS_0)
        self.pmps_act = m.Var(value=P_MPS_0, name='pmps')

        #  Turbine 1, fed the boiler's steam. Inlet state is fixed by the HP header, so
        #  the constant-inlet path applies and the EOS is only solved for the outlet.
        #  theta=THETA_STAGE is the stage's own transit dead time (steam chest + blading
        #  path); casing/rotor thermal mass (tmod.TAU_CASING) is NOT applied here -- see
        #  that constant's comment.
        self.turb = Turbine(m, self.steam, T_HP, P_HP, self.pmps_act, ETA,
                            tau=TAU_STAGE, theta=THETA_STAGE,
                            fixed_inlet=True, name='t1')

        #  Admission fraction follows the dispatch delay above, then its own
        #  valve, same pattern as the back pressure valve above.
        self.frac_t2_act, admit_states = lag_chain(
            m, self.frac_t2_disp, theta=dp.THETA_VALVE_S, tau=TAU_ADMIT, n_lag=3,
            y0=frac0, name='fract2')
        #  LPS header pressure -- same status as pmps_act above: a real state,
        #  declared here because turbine 2 and the primary heat pump need it as
        #  an input, governed by an Equation added further down.
        self.C_LPS = V_LPS_M3 * self._drho_dp(self.trb2_ss['T_out'], P_LPS_0)
        self.plps_act = m.Var(value=P_LPS_0, name='plps')

        #  Turbine 2, fed a fraction of turbine 1's exhaust. Its inlet (turbine 1's
        #  T_out/P_MPS) is a live variable, not a fixed number, so it cannot use
        #  turbine 1's fixed-inlet fit. Left to solve IAPWS-95 symbolically it is
        #  about nine tenths of this entire model, so it runs on the polynomial
        #  surrogate instead -- see turbine_model.py's moving-inlet section.
        #  T2_SURROGATE=False restores the exact path; the two are compared at
        #  the base point by _surrogate_check() below.
        self.m_to_t2 = m.Intermediate(self.frac_t2_act * self.steam)
        self.turb2 = Turbine(m, self.m_to_t2, self.turb.T_out_C, self.pmps_act,
                             self.plps_act, ETA2, tau=tau_stage2, theta=THETA_STAGE,
                             fixed_inlet=False, surrogate=T2_SURROGATE,
                             name='t2')

        #  Flue gas HX, fed the boiler's own live flue gas outlet. The plant MV is
        #  the site's desired hot water temperature -- T_w_out is set at this
        #  value (no valve lag on this actuator), so the flow needed to hit it
        #  is calculated live from the current duty, with no transient of its own.
        self.twout_cmd = m.Param(value=TWOUT_0, name='twout_cmd')
        hxd = hx.build_dynamics(m, self.stack, self.gas, self.twout_cmd,
                                y0=self.hx_ss['Q'])
        self.Q_hx, self.T_w_out, self.dT_hot = hxd['Q'], hxd['T_w_out'], hxd['dT_hot']
        self.mwater_act = hxd['m_water_act']

        #  Secondary heat pump. Water-sourced (an unlimited reservoir, same
        #  assumption as the boiler's own feedwater), so its steady state does
        #  not depend on anything else in the plant -- unlike every other
        #  component here, it has no upstream chaining.
        self.qsec_cmd = m.Param(value=Q_SEC_0, name='qsec_cmd')
        self.hps_ss = hps.steady_state(Q_SEC_0)
        hpsd = hps.build_dynamics(m, self.qsec_cmd, y0=self.hps_ss)
        self.Q_cond_sec = hpsd['Q_cond_sec']
        self.W_hp_sec, self.COP_sec = hpsd['W_comp_sec'], hpsd['COP_sec']

        #  Transport lag on the intermediate water loop carrying HP2's condenser duty to
        #  HP1's evaporator -- a real pipe run between the two machines, not an instant
        #  handoff. W_hp_sec and every other HP2 report stay on the undelayed Q_cond_sec;
        #  only what reaches HP1's evaporator is delayed.
        if LINK_DYNAMICS:
            self.Q_sec_at_hp1, link_states = lag_chain(
                m, self.Q_cond_sec, theta=THETA_LINK, tau=TAU_LINK, n_lag=6,
                y0=self.hps_ss['Q_cond_sec'], name='hp_link')
        else:
            self.Q_sec_at_hp1, link_states = self.Q_cond_sec, []

        #  Primary heat pump. Evaporator fed by two live sources: the boiler's
        #  own flue gas (self.gas, the same variable the HX already uses) and
        #  the secondary's condenser duty, AS IT ARRIVES at the primary's
        #  evaporator (self.Q_sec_at_hp1, delayed by the link above).
        #  Recycle tap. The site return water is a second low-grade source into
        #  the primary's evaporator, live off the exchanger's own delivered flow
        #  -- a warmer or larger recycle is more heat here, the same way a
        #  higher firing rate is more flue gas.
        self.q_evap_rec = m.Intermediate(
            self.mwater_act * hx.CP_WATER * (hx.T_W_IN - hpp.T_RECYCLE_EVAP_OUT))

        #  Everything into the primary's evaporator that is not the flue gas.
        #  build_dynamics does its own gas arithmetic, so the tap goes in
        #  alongside the secondary's condenser rather than being added twice.
        self.q_evap_other = m.Intermediate(self.Q_sec_at_hp1 + self.q_evap_rec)

        #  The primary is DEPENDENT. Nothing commands it: it takes the low-grade
        #  heat that arrives and delivers that plus the work it spent lifting it.
        #  The secondary's duty is the only heat-pump handle, and this is how it
        #  reaches the LPS header. Note the consequence -- the flue gas is always
        #  there when the boiler fires and there is no bypass, so this has a floor
        #  it cannot be turned down below.
        #  The evaporator's total low-grade supply. Converting it to a condenser
        #  duty happens inside build_dynamics, where the live cycle points are,
        #  because that conversion moves with P_LPS -- see the driven-mode note
        #  there.
        self.qpri_drv = m.Intermediate(self.gas*hpp.DH_GAS_HP + self.q_evap_other)
        self.dtsub_cmd = m.Param(value=dt_sub0, name='dtsub_cmd')
        #  min(): P_LPS_0 is kept inside hpp.P_LPS_RANGE by design now (see the
        #  P_LPS_LO/HI comment above), so this is just belt and braces against
        #  the two ranges drifting apart again in future, not live clamping.
        #  twout_cmd goes to BOTH the exchanger and this machine: they feed the
        #  one hot water header, so they deliver against one setpoint. Without
        #  it here the slider moved the exchanger's flow and left the heat
        #  pump's flat, which is not what the steady-state model does.
        #  Base point for seeding, evaluated the same way the live model computes
        #  it rather than from a commanded number, so the seed and the equations
        #  agree at t = 0. hx_ss['m_water'] -- the flow fuel0/TWOUT_0 actually
        #  require -- not M_WATER_0, a fixed design-point flow from a different
        #  operating point: seeding this evaporator source off M_WATER_0 while
        #  everything upstream is seeded off the demand-consistent fuel0 left
        #  the primary heat pump settled somewhere the site's own demands never
        #  asked for, invisible only because it never got compared against them.
        q_rec_0 = max(self.hx_ss['m_water']*hx.CP_WATER*(hx.T_W_IN - hpp.T_RECYCLE_EVAP_OUT), 0.0)
        q_evap_0 = (self.blr_ss['mdot_gas']*hpp.DH_GAS_HP
                    + self.hps_ss['Q_cond_sec'] + q_rec_0)
        self.q_pri_0 = hpp.q_cond_from_sources(q_evap_0)
        self.hpp_ss = hpp.steady_state(self.q_pri_0, self.blr_ss['mdot_gas'],
                                       self.hps_ss['Q_cond_sec'] + q_rec_0, dt_sub0,
                                       min(P_LPS_0, hpp.P_LPS_RANGE[1]),
                                       TWOUT_0)
        hppd = hpp.build_dynamics(m, self.gas, self.q_evap_other, self.qpri_drv,
                                  self.dtsub_cmd, self.plps_act,
                                  T_hw_cmd=self.twout_cmd, y0=self.hpp_ss)
        self.m_lps_hp, self.m_hw_hp = hppd['m_lps'], hppd['m_hw']
        self.W_hp_pri, self.COP_pri = hppd['W_comp'], hppd['COP']
        self.Q_cond_pri = hppd['Q_cond']
        self.Q_evap_gas_avail = hppd['Q_evap_gas_avail']
        self.Q_evap_demand = hppd['Q_evap_demand']
        self.dQ_shortfall = hppd['dQ_shortfall']

        # Actual service, not requested demand, leaves each accumulator.
        #  Site draw referenced to the live pressure targets (header_closures' SITE
        #  DELIVERY LAW): exactly the demand at the target, more above it, less below.
        flow = headers.flows(self.pmps_act, self.plps_act,
                             self.d_mps_param, self.d_lps_param, absolute=m.abs,
                             include_relief=False,
                             target_mps=self.pmps_cmd, target_lps=self.plps_cmd)
        self.m_letdown = m.Intermediate(flow['letdown'])
        #  LPS relief valve with its own FOPTD response (added 2026-09-24): the flow
        #  the valve WOULD pass once settled at the current LPS pressure
        #  (headers.relief) reaches the header through a dead time and a first-order
        #  lag -- the valve takes time to lift -- so a sudden LPS surplus pushes the
        #  pressure up before the relief catches it. Starts closed (0) at the base
        #  point, which sits below relief_start.
        self.m_relief_ss = m.Intermediate(headers.relief(self.plps_act, absolute=m.abs))
        self.m_relief, relief_states = lag_chain(
            m, self.m_relief_ss, theta=dp.THETA_RELIEF_S, tau=dp.TAU_RELIEF_S,
            n_lag=3, y0=0.0, name='relief')
        self.m_vent = m.Intermediate(flow['dump'] + self.m_relief)
        self.m_mps_delivered = m.Intermediate(flow['mps'])
        self.m_lps_delivered = m.Intermediate(flow['lps'])
        self.m_mps_supply = m.Intermediate(self.steam-self.m_to_t2-self.m_letdown)
        self.m_lps_supply = m.Intermediate(self.m_to_t2+self.m_lps_hp+self.m_letdown)
        self.net_mps = m.Intermediate(self.m_mps_supply-self.m_mps_delivered)
        self.net_lps = m.Intermediate(self.m_lps_supply-self.m_lps_delivered-self.m_vent)
        m.Equation(self.C_MPS*self.pmps_act.dt() == self.net_mps)
        m.Equation(self.C_LPS*self.plps_act.dt() == self.net_lps)
        self.m_lps_total = self.m_lps_delivered
        self.m_hw_total = m.Intermediate(self.mwater_act + self.m_hw_hp)
        self.W_hp_total = m.Intermediate(self.W_hp_sec + self.W_hp_pri)

        self.m_mps_site = self.m_mps_delivered

        self._seeds = ([(v, self._scalar(v.VALUE)) for v in d['states']]
                       + [(v, fuel0) for v in fuel_disp_states]
                       + [(v, frac0) for v in frac_disp_states]
                       #  Header pressure states -- see the module docstring's
                       #  header-accumulation section. Real integrator states
                       #  now, so they need reseeding on a solver retry just
                       #  like every other state here, not the lag-chain
                       #  outputs valve_states/lps_valve_states used to be.
                       + [(self.pmps_act, P_MPS_0), (self.plps_act, P_LPS_0)]
                       + [(v, 0.0) for v in relief_states]  # relief valve closed at base
                       #  Every one of turb.states, not just the old single h_out state --
                       #  THETA_STAGE now puts a dead-time chain ahead of it, so this list
                       #  has more than one entry. Same generic per-state pattern turb2
                       #  already used below, which handles either count correctly.
                       + [(v, self._scalar(v.VALUE)) for v in self.turb.states]
                       + [(v, frac0) for v in admit_states]
                       + [(v, self._scalar(v.VALUE)) for v in self.turb2.states]
                       + [(v, self.hx_ss['Q']) for v in hxd['duty_states']]
                       + [(v, TWOUT_0) for v in hxd['twout_states']]
                       #  hx_ss['m_water'], not M_WATER_0 -- see the q_rec_0
                       #  comment above, same reasoning.
                       + [(v, self.hx_ss['m_water']) for v in hxd['water_states']]
                       + [(v, self.hps_ss['Q_cond_sec']) for v in hpsd['states']]
                       + [(v, self.hps_ss['Q_cond_sec']) for v in link_states]
                       + [(v, self.hpp_ss['Q_cond']) for v in hppd['duty_states']]
                       + [(v, self.hpp_ss['m_lps']) for v in hppd['lps_states']]
                       + [(v, self.hpp_ss['m_hw']) for v in hppd['hw_states']])

        self.solver = self._pick_solver()

        self.t = 0.0
        self.mfuel_cmd_val, self.pmps_cmd_val = fuel0, P_MPS_0
        self.frac_t2_cmd_val, self.plps_cmd_val = frac0, P_LPS_0
        self.twout_cmd_val = TWOUT_0
        self.qsec_cmd_val = Q_SEC_0
        self.dtsub_cmd_val = dt_sub0
        self.d_mps_val, self.d_lps_val = D_MPS_0, D_LPS_0
        self.hw_min_val = HW_MIN_0
        self.fuel_clamped = self.frac_clamped = self.hw_binds = False
        self.dt_clamped = False
        self.last = None
        self.last = self._sample(0, ok=True)
        self.check = self._self_check(verbose)
        self.surrogate_check = self._surrogate_check(verbose)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _scalar(value):
        try:
            return float(list(value)[-1])
        except TypeError:
            return float(value)

    @staticmethod
    def _drho_dp(T_C, P_bar, dP_bar=0.5):
        """drho/dP [kg/m3 per bar] at fixed T, centred finite difference via
        CoolProp -- the property-based header capacitance route (restructure
        phase 2), evaluated once per instance at its own real base point."""
        from CoolProp.CoolProp import PropsSI
        T_K = T_C + 273.15
        r_lo = PropsSI('D', 'T', T_K, 'P', (P_bar - dP_bar)*1e5, 'Water')
        r_hi = PropsSI('D', 'T', T_K, 'P', (P_bar + dP_bar)*1e5, 'Water')
        return (r_hi - r_lo) / (2*dP_bar)

    def _pick_solver(self):
        for choice in (3, 2, 1):            # IPOPT, BPOPT, APOPT
            try:
                self.m.options.SOLVER = choice
                self.m.options.TIME_SHIFT = 0
                self._reseed()
                self.m.solve(disp=False)
                #  Only two Params change between steps from here on -- the
                #  model structure itself never does, so skip re-serializing
                #  it to .apm on every later solve(). Undocumented GEKKO
                #  internal state; if a future GEKKO removes/renames it this
                #  just falls back to today's (slower but correct) behaviour.
                try:
                    self.m._model = 'provided'
                except Exception:
                    pass
                self.m.options.TIME_SHIFT = self.batch
                return choice
            except Exception:
                continue
        raise RuntimeError('no solver converged on the combined plant base point')

    def _reseed(self):
        for var, value in self._seeds:
            var.value = value

    def _sample(self, k, ok):
        val = lambda v: float(v.value[k])
        power1, power2 = val(self.turb.power_kW), val(self.turb2.power_kW)
        return dict(
            **self._trim_diagnostics(),
            m_letdown=val(self.m_letdown), m_vent=val(self.m_vent),
            m_relief=val(self.m_relief), m_relief_ss=val(self.m_relief_ss),
            m_mps_delivered=val(self.m_mps_delivered),
            m_lps_delivered=val(self.m_lps_delivered),
            mps_shortfall=max(self.d_mps_val-val(self.m_mps_delivered), 0.0),
            lps_shortfall=max(self.d_lps_val-val(self.m_lps_delivered), 0.0),
            m_mps_supply=val(self.m_mps_supply), m_lps_supply=val(self.m_lps_supply),
            net_mps=val(self.net_mps), net_lps=val(self.net_lps),
            domain_ok=headers.in_domain(val(self.pmps_act), val(self.plps_act)),
            terminal=False, stop_reason='',
            t=self.t,
            mfuel_cmd=self.mfuel_cmd_val, pmps_cmd=self.pmps_cmd_val,
            frac_t2_cmd=self.frac_t2_cmd_val, plps_cmd=self.plps_cmd_val,
            twout_cmd=self.twout_cmd_val, mwater_act=val(self.mwater_act),
            qsec_cmd=self.qsec_cmd_val, dtsub_cmd=self.dtsub_cmd_val,
            d_mps=self.d_mps_val, d_lps=self.d_lps_val, hw_min=self.hw_min_val,
            fuel_clamped=self.fuel_clamped, frac_clamped=self.frac_clamped,
            hw_binds=self.hw_binds, dt_clamped=self.dt_clamped,
            mfuel_act=val(self.mf), pmps_act=val(self.pmps_act),
            frac_t2_act=val(self.frac_t2_act), plps_act=val(self.plps_act),
            steam=val(self.steam), m_lps=val(self.m_to_t2),
            power1=power1, power2=power2, power=power1 + power2,
            T_out1=val(self.turb.T_out_C), T_out2=val(self.turb2.T_out_C),
            stack=val(self.stack),
            gas=val(self.gas), duty=val(self.Q), eta_b=val(self.eta_b),
            dh_act=val(self.turb.dh_act), dh_act2=val(self.turb2.dh_act),
            Q_hx=val(self.Q_hx), twout=val(self.T_w_out),
            dT_hot_hx=val(self.dT_hot),
            Q_cond_sec=val(self.Q_cond_sec), W_hp_sec=val(self.W_hp_sec),
            COP_sec=val(self.COP_sec),
            m_lps_hp=val(self.m_lps_hp), m_hw_hp=val(self.m_hw_hp),
            W_hp_pri=val(self.W_hp_pri), COP_pri=val(self.COP_pri),
            Q_cond_pri=val(self.Q_cond_pri),
            Q_evap_gas_avail=val(self.Q_evap_gas_avail),
            Q_evap_demand=val(self.Q_evap_demand),
            dQ_shortfall=val(self.dQ_shortfall),
            m_lps_total=val(self.m_lps_total), m_hw_total=val(self.m_hw_total),
            W_hp_total=val(self.W_hp_total),
            m_mps_site=val(self.m_mps_site),
            q_evap_rec=val(self.q_evap_rec),
            #  Actual delivered minus requested: zero or a service deficit.
            #  Surplus supply is stored, let down or dumped, not site delivery.
            dm_mps=val(self.m_mps_site) - self.d_mps_val,
            dm_lps=val(self.m_lps_total) - self.d_lps_val,
            #  Hot water is an inequality: only the deficit side means anything,
            #  and a surplus is free.
            dm_hw=val(self.m_hw_total) - self.hw_min_val,
            hw_short=max(self.hw_min_val - val(self.m_hw_total), 0.0),
            ok=ok)

    def _surrogate_check(self, verbose):
        """Measure turbine 2's surrogate against the exact path it replaces.

        Runs at the base point only -- enough to catch a fit built over the
        wrong box, a stale cache, or an operating point that has drifted out
        of the fitted region. A surrogate that has never been compared with
        what it stands in for is not a surrogate, it is a guess.
        """
        if not T2_SURROGATE:
            return None
        err = tmod.surrogate_error(P_MPS_0, self.trb_ss['T_out'], P_LPS_0, ETA2)
        #  u_is < MOVING_U_HI means the isentropic outlet is inside the dome
        #  (lever rule, exact) or within the fitted superheat correction past it;
        #  dsh_act > 0 means the actual outlet is superheated, where the T_out
        #  fit applies. Both are properties of the operating point, not of the fit.
        in_domain = err['u_is'] < tmod.MOVING_U_HI and err['dsh_act'] > 0.0
        if verbose:
            print(f'  turbine 2 surrogate vs exact IAPWS-95 at the base point:')
            print(f'     dh_is {err["dh_is"]:.2e} kJ/kg   '
                  f'T2s {err["T2s"]:.2e} C   T_out {err["T_out"]:.2e} C')
            print(f'     isentropic quality {err["x_ideal"]:.4f}   '
                  f'actual superheat {err["dsh_act"]:.1f} kJ/kg (>0)'
                  f'{"" if in_domain else "   << OUT OF DOMAIN"}')
            if not in_domain:
                print('     WARNING set T2_SURROGATE=False for this operating point')
        err['in_domain'] = in_domain
        return err

    def _self_check(self, verbose):
        """
        Confirm the connected plant sits on the two independent steady solves.

        This is the first thing the combination can check that neither rig could:
        that the boiler's output state is the state the turbine expands from.
        """
        got = self.last
        rows = [('steam kg/s', self.blr_ss['mdot_steam'], got['steam']),
                ('stack degC', self.blr_ss['T_stack'], got['stack']),
                ('power1 kW', self.trb_ss['power'], got['power1']),
                ('T_out1 degC', self.trb_ss['T_out'], got['T_out1']),
                ('power2 kW', self.trb2_ss['power'], got['power2']),
                ('T_out2 degC', self.trb2_ss['T_out'], got['T_out2']),
                ('Q_hx kW', self.hx_ss['Q'], got['Q_hx']),
                ('T_w_out degC', self.hx_ss['T_w_out'], got['twout']),
                ('Q_cond_sec kW', self.hps_ss['Q_cond_sec'], got['Q_cond_sec']),
                ('W_hp_sec kW', self.hps_ss['W_comp_sec'], got['W_hp_sec']),
                ('m_lps_hp kg/s', self.hpp_ss['m_lps'], got['m_lps_hp']),
                ('m_hw_hp kg/s', self.hpp_ss['m_hw'], got['m_hw_hp'])]
        worst = max(abs(b - a)/abs(a) for _, a, b in rows if a)
        if verbose:
            print(f'PlantRig  solver {self.solver}  dt {DT:.0f} s  batch {self.batch}')
            print(f'  boiler  tau_q {TAU_BOILER:.2f} s (tau_bed {blr.TAU_BED:.1f} s), '
                  f'theta {THETA_FUEL:.0f} s   '
                  f'turbine  tau {TAU_STAGE:.2f} s, theta {THETA_STAGE:.2f} s')
            print(f'  interface  {T_HP:.2f} degC at {P_HP:.0f} bar, '
                  f'h = {blr.H_STEAM:.2f} kJ/kg')
            if LINK_DYNAMICS:
                #  Loop flow diagnostic only, not used anywhere in the model -- the
                #  intermediate water loop's implied flow at the sizing DT_LINK_K.
                _loop_flow = hps.Q_SEC_0 / (hx.CP_WATER * dp.DT_LINK_K)
                print(f'  HP2->HP1 link  theta {THETA_LINK:.1f} s   '
                      f'loop flow (diagnostic) {_loop_flow:.1f} kg/s at dT {dp.DT_LINK_K:.0f} K')
            print(f'  base point  fuel {self.fuel_base:.2f} kg/s   P_MPS {P_MPS_0:.1f} bar')
            for label, a, b in rows:
                #  Relative error is undefined on a zero reference, and zero is
                #  now a legal operating point: the secondary heat pump's floor
                #  was removed, so at Q_sec = 0 its duty, work and refrigerant
                #  flow are all genuinely zero. Report the absolute difference
                #  for those rows instead of dividing by zero.
                if a:
                    err = f'{abs(b-a)/abs(a)*100:8.5f} %'
                else:
                    err = f'{abs(b-a):8.5f} abs'
                print(f'  {label:14s} independent {a:12.4f}   plant {b:12.4f}   {err}')
            if worst > 1e-4:
                print(f'  WARNING self-check off by {worst*100:.4f} %')
            #  The point of solving fuel0/frac0/dt_sub0 against the base
            #  demands in __init__ rather than seeding off MFUEL_0/FRAC_T2_0:
            #  confirm the plant actually opens ON the site's setpoints, not
            #  just internally consistent with itself. Not folded into the
            #  worst/passed check above -- that one is a tight 1e-4 numerical
            #  identity, this one carries extra Newton-convergence and
            #  clamping tolerance and is reported for visibility, not gating.
            print(f'  demand match at t=0  MPS {got["m_mps_site"]:.3f} kg/s '
                  f'(target {D_MPS_0:.2f}, gap {got["m_mps_site"]-D_MPS_0:+.4f})   '
                  f'LPS {got["m_lps_total"]:.3f} kg/s '
                  f'(target {D_LPS_0:.2f}, gap {got["m_lps_total"]-D_LPS_0:+.4f})')
        return {'worst': worst, 'passed': worst < 1e-4}

    # ------------------------------------------------------------------ interface
    @staticmethod
    def _delivery_at_fuel(fuel, q_sec, t_hw, cyc, dh_hw_live):
        """Steady-state hot water and heat pump LPS at a given fuel rate.

        Pure arithmetic -- every property call is hoisted into cyc/dh_hw_live by
        the caller, so this is cheap enough to evaluate several times per sample.
        It is a steady-state predictor used as feedforward, the same standing as
        the fuel inverse: it says where the plant will end up, not where it is.
        """
        gas = fuel*blr.GAS_PER_FUEL
        T_stack = float(blr.stack_temp(fuel))
        Q_hx = gas*(blr.h_gas(T_stack) - blr.h_gas(hx.T_FG_OUT_SPEC))
        m_hx = Q_hx/(hx.CP_WATER*(t_hw - hx.T_W_IN))
        q_rec = max(m_hx*hx.CP_WATER*(hx.T_W_IN - hpp.T_RECYCLE_EVAP_OUT), 0.0)
        q_cond = (gas*hpp.DH_GAS_HP + q_rec + q_sec)/cyc['ratio']
        q_sub = q_cond*cyc['sub_frac']
        return dict(m_hw=m_hx + q_sub/dh_hw_live,
                    m_lps_hp=(q_cond - q_sub)/cyc['dh_lps'])

    def _state_at(self, fuel, dt_sub, d_mps, q_sec, t_hw, p_lps, dh_hw_live):
        """Steady-state MPS/LPS/HW the plant would settle at for (fuel, dt_sub).

        frac_T2 is not a free unknown here: it is whatever holds MPS exactly at
        its draw, so it is eliminated before the solve rather than carried
        through it. What is left is two unknowns against two requirements.
        """
        cyc = hpp.cycle_constants(p_lps, dt_sub)
        d = self._delivery_at_fuel(fuel, q_sec, t_hw, cyc, dh_hw_live)
        steam = blr.steam_from_fuel(fuel)
        return dict(steam=steam,
                    m_lps=max(steam - d_mps, 0.0) + d['m_lps_hp'],
                    m_hw=d['m_hw'], m_lps_hp=d['m_lps_hp'])

    def _solve_demands(self, d_mps, d_lps, hw_min, q_sec, t_hw, p_lps, dh_hw_live,
                       fuel_guess, dt_floor=None):
        """Fuel and subcooling that serve LPS exactly and clear the HW minimum.

        TWO handles, because one is not enough. Firing harder is the only way to
        make more hot water -- both sources are downstream of the flue gas -- but
        it also makes more steam, and that steam has to go somewhere. Subcooling
        is what absorbs it: raising dt_sub moves the primary's condenser duty off
        the LPS side and onto the hot water side, so the heat pump makes less LPS
        exactly when the boiler is being pushed to make more. The two move in
        opposite directions on LPS and the same direction on hot water, which is
        what lets both requirements be met at once instead of dumping the
        difference over a vent.

        The span is real: dt_sub over its 2-25 K range shifts the share of duty
        leaving as hot water from 0.037 to 0.296, an eight-fold swing, and it
        lowers the condenser duty as well, so it pulls LPS down twice over.

        Solved as a damped Newton on (fuel, dt_sub) with a numerical Jacobian.
        Every evaluation is arithmetic against cached cycle constants, so the
        whole thing costs well under a millisecond. Returns the pair plus flags
        for whichever handle ran out of travel.

        dt_floor (2026-09-27, for the RTO): the subcooling to use when hot water does not
        bind, and the lowest it may go when it does. None = DT_SUB_0 (10 K) as before.
        """
        #  MPS FLOOR ON THE FUEL. _state_at takes the turbine's LPS as
        #  max(steam - d_mps, 0), so once the boiler is making less steam than the
        #  MPS draw the turbine contributes nothing and the LPS residual depends
        #  only on the heat pump. The Newton would then happily walk fuel DOWN to
        #  trim an LPS oversupply -- starving MPS to fix LPS, which inverts the
        #  plant's own priority rule and was giving MPS misses of several kg/s
        #  whenever the secondary was run hard against a small LPS draw.
        #
        #  MPS is served first, so the fuel that serves it is a floor, not a
        #  trade. Below it there is nothing to solve. If LPS is still oversupplied
        #  at that floor the surplus is real and vents; it is not a reason to
        #  under-fire.
        f_mps, _ = blr.fuel_for_steam(d_mps, MFUEL_LO, MFUEL_HI)
        f_lo, f_hi = max(MFUEL_LO, f_mps), MFUEL_HI
        dt_hi = DT_SUB_HI
        dt_base = DT_SUB_0 if dt_floor is None else float(np.clip(dt_floor, DT_SUB_LO, DT_SUB_HI))
        dt_lo = DT_SUB_LO if dt_floor is None else dt_base

        def state(f, dt):
            return self._state_at(f, dt, d_mps, q_sec, t_hw, p_lps, dh_hw_live)

        #  First pass: subcooling at its default, fuel alone against LPS. Most
        #  operating points never leave here -- hot water is in surplus and the
        #  inequality costs nothing.
        f = float(np.clip(fuel_guess, f_lo, f_hi))
        dt = dt_base
        for _ in range(6):
            st = state(f, dt)
            r = st['m_lps'] - d_lps
            if abs(r) < 1e-7:
                break
            h = 1e-3
            slope = (state(min(f + h, f_hi), dt)['m_lps'] - st['m_lps'])/h
            if abs(slope) < 1e-9:
                break
            f = float(np.clip(f - r/slope, f_lo, f_hi))

        st = state(f, dt)
        if st['m_hw'] >= hw_min - 1e-6:
            return f, dt, False, (f >= f_hi - 1e-9 or f <= f_lo + 1e-9), False

        #  Hot water binds. Both handles now, both requirements at once.
        for _ in range(12):
            st = state(f, dt)
            r1 = st['m_lps'] - d_lps
            r2 = st['m_hw'] - hw_min
            if max(abs(r1), abs(r2)) < 1e-6:
                break
            hf, hd = 1e-3, 1e-3
            sf = state(min(f + hf, f_hi), dt)
            sd = state(f, min(dt + hd, dt_hi))
            j11, j21 = (sf['m_lps'] - st['m_lps'])/hf, (sf['m_hw'] - st['m_hw'])/hf
            j12, j22 = (sd['m_lps'] - st['m_lps'])/hd, (sd['m_hw'] - st['m_hw'])/hd
            det = j11*j22 - j12*j21
            if abs(det) < 1e-12:
                break
            df = (-r1*j22 + r2*j12)/det
            dd = (-r2*j11 + r1*j21)/det
            #  Damped: the LPS residual is very stiff in fuel, and an undamped
            #  full step overshoots into the clamp and oscillates there.
            f = float(np.clip(f + 0.7*df, f_lo, f_hi))
            dt = float(np.clip(dt + 0.7*dd, dt_lo, dt_hi))

        #  REPAIR PASS. Once subcooling hits a stop the 2x2 step is solving for a
        #  handle that cannot move, and it will happily walk fuel the wrong way
        #  chasing an LPS residual it can no longer trade against. So if the hot
        #  water minimum is still unmet with subcooling pinned, drop to one
        #  handle: fuel against hot water alone, and let LPS carry the surplus.
        #  That is the honest ordering -- the minimum is a requirement, the LPS
        #  match is what the second handle was buying, and when the handle is
        #  gone so is the match.
        st = state(f, dt)
        if st['m_hw'] < hw_min - 1e-6 and dt >= dt_hi - 1e-9:
            dt = dt_hi
            for _ in range(8):
                st = state(f, dt)
                r = st['m_hw'] - hw_min
                if abs(r) < 1e-6:
                    break
                h = 1e-3
                slope = (state(min(f + h, f_hi), dt)['m_hw'] - st['m_hw'])/h
                if abs(slope) < 1e-9:
                    break
                f = float(np.clip(f - r/slope, f_lo, f_hi))
            st = state(f, dt)

        fuel_clamped = f >= f_hi - 1e-9 or f <= f_lo + 1e-9
        dt_clamped = dt >= dt_hi - 1e-9 or dt <= dt_lo + 1e-9
        return f, dt, True, fuel_clamped, dt_clamped

    @staticmethod
    def _frac_t2_from_fuel(fuel, d_mps):
        """Turbine-2 admission fraction that leaves exactly d_mps on the MPS
        header, given the steam a solved fuel rate achieves.

        Pulled out of _invert so PlantRig.__init__ can reuse the exact same
        arithmetic when it solves the demand-consistent BASE point -- the seed
        and the running inverse must agree, or the plant starts on one number
        and immediately steps to another the instant Run is pressed.
        """
        m_steam_ach = blr.steam_from_fuel(fuel)
        frac = (m_steam_ach - d_mps)/m_steam_ach if m_steam_ach > 1e-9 else 0.0
        frac_clamped = not (FRAC_T2_LO <= frac <= FRAC_T2_HI)
        return turbine2_cutoff(float(np.clip(frac, FRAC_T2_LO, FRAC_T2_HI))), frac_clamped

    def _invert(self, d_mps, d_lps):
        """Fuel, turbine-2 admission and the subcooling split that serve the site.

        This is a static FEEDFORWARD calculation, not a solution. It produces
        targets; the boiler's own transport delay and heat transfer lag still
        stand between those targets and the steam arriving, so the demands are
        met at rest and missed during transients. That miss is deliberate -- it
        is the gap a controller is later judged on.

        TWO handles, solved together by _solve_demands: fuel, and the heat pump's
        subcooling. Subcooling is what lets the LPS header be matched without
        venting -- it moves the primary's condenser duty between raising LPS and
        making hot water, so when the hot water minimum bites the answer is to
        shift the heat pump toward hot water (which also drops its LPS, pulling
        more steam through the turbine) rather than to fire harder and dump the
        surplus. Venting only happens once subcooling has hit its stop.

        MPS is served first. When the boiler cannot raise enough steam for both,
        the shortfall lands entirely on LPS, which matches the rule the plant is
        built around.
        """
        p_lps = float(np.clip(self.plps_cmd_val, *hpp.P_LPS_RANGE))
        dh_hw_live = hpp.dh_hw(self.twout_cmd_val)

        fuel, dt, hw_binds, fuel_clamped, dt_clamped = self._solve_demands(
            d_mps, d_lps, self.hw_min_val, self.qsec_cmd_val,
            self.twout_cmd_val, p_lps, dh_hw_live, fuel_guess=self.mfuel_cmd_val,
            dt_floor=getattr(self, 'dt_floor', None))

        frac, frac_clamped = self._frac_t2_from_fuel(fuel, d_mps)

        self.hw_binds = hw_binds
        self.dt_clamped = dt_clamped
        return fuel, frac, float(dt), fuel_clamped, frac_clamped

    def step(self, d_mps, d_lps, hw_min, pmps_cmd, plps_cmd, twout_cmd,
             qsec_cmd=None, mfuel_cmd=None, frac_t2_cmd=None, dtsub_cmd=None,
             fuel_trim=0.0, frac_t2_trim=0.0):
        """Advance one batch and return a list of samples, one per DT.

        The site demands drive the plant. mfuel_cmd/frac_t2_cmd are only read when
        DEMAND_DRIVEN is False (or when they are passed explicitly), which is how
        the open-loop step test and the no-regulation baseline still get to move
        fuel by hand.

        Trims are absolute offsets, recalculated on every call, after the
        feedforward inverse (or explicit command) and before the existing
        dispatch/actuator dynamics. In explicit mode the untrimmed command is
        the baseline. Omitted explicit commands retain that baseline, so trims
        never accumulate implicitly. Saturation flags describe command clipping,
        separately from the inverse's existing feasibility flags.
        """
        if self.halted:
            raise RuntimeError('Plant stopped: '+self.stop_reason+'; reset before continuing')
        if not np.isfinite(fuel_trim) or not np.isfinite(frac_t2_trim):
            raise ValueError('feedback trims must be finite')
        self.d_mps_val = float(np.clip(d_mps, D_MPS_LO, D_MPS_HI))
        self.d_lps_val = float(np.clip(d_lps, D_LPS_LO, D_LPS_HI))
        self.hw_min_val = float(np.clip(hw_min, HW_MIN_LO, HW_MIN_HI))
        #  Live GEKKO side of the site demands -- the header mass balances
        #  need these as an actual model input, not just a NumPy argument to
        #  the feedforward inverse below.
        self.d_mps_param.value = self.d_mps_val
        self.d_lps_param.value = self.d_lps_val

        #  These are INPUTS to the inverse -- it needs the header pressure, the
        #  hot water setpoint and the secondary's duty to predict what the plant
        #  will deliver -- so they are resolved before it runs. Left below, the
        #  inverse was working off last step's slider positions, which put a
        #  one-sample stale answer into every move of P_LPS, T_HW or Q_sec.
        self.pmps_cmd_val = float(np.clip(pmps_cmd, P_MPS_LO, P_MPS_HI))
        self.plps_cmd_val = float(np.clip(plps_cmd, P_LPS_LO, P_LPS_HI))
        self.twout_cmd_val = float(np.clip(twout_cmd, TWOUT_LO, TWOUT_HI))
        qsec_cmd = self.qsec_cmd_val if qsec_cmd is None else qsec_cmd
        self.qsec_cmd_val = float(np.clip(qsec_cmd, Q_SEC_LO, Q_SEC_HI))

        if DEMAND_DRIVEN and mfuel_cmd is None and frac_t2_cmd is None:
            #  Subcooling comes out of the inverse now, not from a fixed
            #  constant: it is the second handle, the one that lets the heat
            #  pump back off its LPS output when hot water is forcing the
            #  boiler harder. dtsub_cmd is still accepted from the caller for
            #  the hand-driven mode below.
            (mfuel_cmd, frac_t2_cmd, dt_solved,
             self.fuel_clamped, self.frac_clamped) = self._invert(self.d_mps_val,
                                                                  self.d_lps_val)
            if dtsub_cmd is None:
                dtsub_cmd = dt_solved
        else:
            if mfuel_cmd is None:
                mfuel_cmd = getattr(self, 'fuel_feedforward', self.mfuel_cmd_val)
            if frac_t2_cmd is None:
                frac_t2_cmd = getattr(self, 'frac_t2_feedforward', self.frac_t2_cmd_val)
            self.fuel_clamped = self.frac_clamped = self.hw_binds = False
            self.dt_clamped = False

        self.fuel_feedforward = float(mfuel_cmd)
        self.frac_t2_feedforward = float(frac_t2_cmd)
        self.fuel_trim_requested = float(fuel_trim)
        self.frac_t2_trim_requested = float(frac_t2_trim)
        fuel_requested = float(mfuel_cmd) + fuel_trim
        frac_requested = float(frac_t2_cmd) + frac_t2_trim
        self.mfuel_cmd_val = float(np.clip(fuel_requested, MFUEL_LO, MFUEL_HI))
        self.frac_t2_cmd_val = turbine2_cutoff(float(np.clip(frac_requested, FRAC_T2_LO, FRAC_T2_HI)))
        self.fuel_saturated = bool(self.mfuel_cmd_val != fuel_requested)
        self.frac_t2_saturated = bool(self.frac_t2_cmd_val != frac_requested)
        dtsub_cmd = self.dtsub_cmd_val if dtsub_cmd is None else dtsub_cmd
        self.dtsub_cmd_val = float(np.clip(dtsub_cmd, DT_SUB_LO, DT_SUB_HI))
        self.mfuel_cmd.value = self.mfuel_cmd_val
        self.pmps_cmd.value = self.pmps_cmd_val
        self.frac_t2_cmd.value = self.frac_t2_cmd_val
        self.plps_cmd.value = self.plps_cmd_val
        self.twout_cmd.value = self.twout_cmd_val
        self.qsec_cmd.value = self.qsec_cmd_val
        self.dtsub_cmd.value = self.dtsub_cmd_val

        #  Retry before halting (2026-09-27). Solves fail at random, about once per ~1000: the
        #  same inputs, re-run, solve in one iteration (found re-running a halted FF+PID+RTO
        #  run: it failed at 7500 s in one run and passed in the next, inputs equal to 1e-9).
        #  A failed solve leaves the previous values in place, so a retry starts from the same
        #  point. Same solver first, then the other of APOPT/IPOPT; halt only if all fail.
        ok = False
        err = None
        solver = self.m.options.SOLVER
        for attempt_solver in (solver, solver, 3 if solver == 1 else 1):
            self.m.options.SOLVER = attempt_solver
            try:
                self.m.solve(disp=False)
                ok = True
                break
            except Exception as e:
                err = e
                self.solve_retries = getattr(self, 'solve_retries', 0) + 1
        self.m.options.SOLVER = solver
        if not ok:
            self.halted = True
            self.stop_reason = 'solver failure; state is invalid: ' + repr(err)[:200]

        samples = []
        for k in range(1, self.batch + 1):
            self.t += DT
            if ok:
                #  Removed 2026-09-24, then REINSTATED the same day. Removing it
                #  ("idc if its above some limit i don't want the simulation to stop")
                #  was tried; the very next thing found was MPS pressure, unopposed once
                #  its letdown valve saturates at 15 kg/s (a real, fixed mechanical cap --
                #  nothing else pushes back on MPS pressure once that's maxed), running
                #  past the boiler's own fixed 100 bar outlet before the solver finally
                #  gave up -- Turbine 1 computing a NEGATIVE pressure drop and negative
                #  power (dh -4839.7 kJ/kg, power -166292 kW) in the meantime, which is
                #  not "showing where it goes", it's the model producing numbers with no
                #  physical meaning. Confirmed independently: turbine_model.MOVING_P1_HI
                #  (58 bar) is turbine 2's own inlet-fit validity ceiling for the MPS
                #  header, tighter than the boiler-pressure argument alone. So the halt is
                #  back, but at the SAME widened thresholds from earlier the same day
                #  (mps_ceiling=57.5 / mps_domain_floor=22.5, now 8.5 / lps_domain_floor=1.8) --
                #  those already fixed the original complaint (stopping on perfectly
                #  ordinary slider positions, with the ceiling sitting exactly on the
                #  slider's own max/min); this was never a reason to remove the guard
                #  outright, just to widen it, and widening was already done.
                #  Ceilings later raised to 100 bar MPS / 10 bar LPS at Hamish's request
                #  (see header_closures.HeaderConfig), knowing the numbers above
                #  ~58 bar MPS / ~4.6 bar LPS are extrapolated. The stop message now
                #  names the header and pressure.
                sample = self._sample(k, ok=True)
                if not sample['domain_ok']:
                    self.halted = True
                    self.stop_reason = ('header pressure outside property/service domain '
                                        f"(MPS {sample['pmps_act']:.2f} bar, "
                                        f"LPS {sample['plps_act']:.2f} bar)")
                    sample.update(ok=False, terminal=True, stop_reason=self.stop_reason)
                    samples.append(sample)
                    break
                samples.append(sample)
            else:
                held = dict(self.last)
                held.update(t=self.t, mfuel_cmd=self.mfuel_cmd_val,
                            pmps_cmd=self.pmps_cmd_val,
                            frac_t2_cmd=self.frac_t2_cmd_val,
                            plps_cmd=self.plps_cmd_val,
                            twout_cmd=self.twout_cmd_val,
                            qsec_cmd=self.qsec_cmd_val,
                            d_mps=self.d_mps_val, d_lps=self.d_lps_val,
                            hw_min=self.hw_min_val,
                            dtsub_cmd=self.dtsub_cmd_val, ok=False)
                held.update(terminal=True, stop_reason=self.stop_reason, domain_ok=False,
                            mps_shortfall=float('nan'), lps_shortfall=float('nan'))
                samples.append(held)
                break
        for sample in samples:
            sample.update(self._trim_diagnostics())
        self.last = samples[-1]
        return samples

    def _trim_diagnostics(self):
        """Command-side anti-windup data, including on held/failed samples."""
        fuel_ff = getattr(self, 'fuel_feedforward', self.mfuel_cmd_val)
        frac_ff = getattr(self, 'frac_t2_feedforward', self.frac_t2_cmd_val)
        return dict(fuel_feedforward=fuel_ff,
                    fuel_trim_requested=getattr(self, 'fuel_trim_requested', 0.0),
                    fuel_trim_applied=self.mfuel_cmd_val - fuel_ff,
                    fuel_final=self.mfuel_cmd_val,
                    fuel_saturated=getattr(self, 'fuel_saturated', False),
                    frac_t2_feedforward=frac_ff,
                    frac_t2_trim_requested=getattr(self, 'frac_t2_trim_requested', 0.0),
                    frac_t2_trim_applied=self.frac_t2_cmd_val - frac_ff,
                    frac_t2_final=self.frac_t2_cmd_val,
                    frac_t2_saturated=getattr(self, 'frac_t2_saturated', False))

    def reset(self):
        """Restart from the settled base point."""
        self.halted = False
        self.stop_reason = ""
        self.m.options.TIME_SHIFT = 0
        self._reseed()
        self.mfuel_cmd.value = self.fuel_base
        self.pmps_cmd.value = P_MPS_0
        self.frac_t2_cmd.value = self.frac_t2_base
        self.plps_cmd.value = P_LPS_0
        self.twout_cmd.value = TWOUT_0
        self.qsec_cmd.value = Q_SEC_0
        self.dtsub_cmd.value = self.dt_sub_base
        #  Header states themselves are reseeded by _reseed() above (they are
        #  in self._seeds), but their ODE's own live inputs are not -- without
        #  this the header Equations would still see whatever d_mps/d_lps the
        #  just-ended run left the Params at.
        self.d_mps_param.value = D_MPS_0
        self.d_lps_param.value = D_LPS_0
        try:
            self.m.solve(disp=False)
        except Exception:
            #  Only reseed here, on failure -- reseeding unconditionally after
            #  a successful solve overwrote every state's freshly solved value
            #  with the raw scalar seed again, which _sample() below then
            #  choked on (a plain float isn't subscriptable the way a solved
            #  GEKKO value is).
            self.halted = True
            self.stop_reason = 'reset solve failed'
            raise RuntimeError(self.stop_reason)
        self.m.options.TIME_SHIFT = self.batch
        self.t = 0.0
        self.mfuel_cmd_val, self.pmps_cmd_val = self.fuel_base, P_MPS_0
        self.fuel_feedforward = self.fuel_base
        self.frac_t2_feedforward = self.frac_t2_base
        self.fuel_trim_requested = self.frac_t2_trim_requested = 0.0
        self.fuel_saturated = self.frac_t2_saturated = False
        self.frac_t2_cmd_val, self.plps_cmd_val = self.frac_t2_base, P_LPS_0
        self.twout_cmd_val = TWOUT_0
        self.qsec_cmd_val = Q_SEC_0
        self.dtsub_cmd_val = self.dt_sub_base
        self.d_mps_val, self.d_lps_val = D_MPS_0, D_LPS_0
        self.hw_min_val = HW_MIN_0
        self.fuel_clamped = self.frac_clamped = self.hw_binds = False
        self.dt_clamped = False
        self.last = self._sample(0, ok=True)
        return self.last


if __name__ == '__main__':
    import time

    rig = PlantRig()
    print()
    print('DEMAND STEP TEST  MPS 12 -> 16 kg/s at 300 s, LPS 34 -> 28 kg/s at 700 s, '
          'Q_sec Q_SEC_0 -> Q_SEC_HI at 1100 s (watch fuel fall while both demands stay '
          'met -- that substitution is the plant\'s economic lever), '
          'P_LPS 3 -> 2.2 bar at 1500 s')
    print(f"  {'t s':>6s}{'d_MPS':>7s}{'MPS':>7s}{'d_LPS':>7s}{'LPS':>7s}"
          f"{'HW':>7s}{'fuel':>7s}{'frac':>7s}{'Q_sec':>8s}{'HP1 kW':>9s}"
          f"{'W_hp':>8s}{'power':>9s}{'dMPS':>7s}{'dLPS':>7s}{'HWshort':>8s}")
    t0 = time.time()
    n = 500
    for k in range(n):
        t = k*DT*BATCH
        d_mps = D_MPS_0 if t < 300 else 16.0
        d_lps = D_LPS_0 if t < 700 else 28.0
        q_sec = Q_SEC_0 if t < 1100 else Q_SEC_HI
        p_lps = P_LPS_0 if t < 1500 else 2.2
        for s in rig.step(d_mps, d_lps, HW_MIN_0, P_MPS_0, p_lps, TWOUT_0,
                          qsec_cmd=q_sec):
            if abs(s['t'] % 100.0) < 1e-9:
                flag = ''
                if not s['ok']:
                    flag += '  FAIL'
                if s['fuel_clamped']:
                    flag += '  FUEL CLAMPED'
                if s['hw_short'] > 1e-6:
                    flag += '  HW BELOW MINIMUM'
                print(f"  {s['t']:6.0f}{s['d_mps']:7.2f}{s['m_mps_site']:7.2f}"
                      f"{s['d_lps']:7.2f}{s['m_lps_total']:7.2f}"
                      f"{s['m_hw_total']:7.1f}{s['mfuel_act']:7.2f}"
                      f"{s['frac_t2_act']:7.3f}{s['qsec_cmd']:8.0f}"
                      f"{s['Q_cond_pri']:9.0f}{s['W_hp_total']:8.0f}"
                      f"{s['power']:9.0f}{s['dm_mps']:7.2f}{s['dm_lps']:7.2f}"
                      f"{s['hw_short']:8.2f}{flag}")
        if rig.halted:
            print(f'  STOP: {rig.stop_reason}')
            break
    print(f'  elapsed {time.time()-t0:.1f} s')
