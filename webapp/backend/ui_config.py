"""UI configuration served to the frontend: sliders, modes, trend panels, colours, constants.

Everything here is data lifted from plant_operator.py (OperatorApp.__init__ `specs`/`SLOT`/mode list,
redraw()'s `panels`, the colour constants and the schematic's caption constants), so the browser
renders what the desktop dashboard renders without duplicating the rules in JavaScript.
"""
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parents[2]
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

import plant_rig as cfg
import header_closures as headers
import boiler as blr
import flue_gas_water_hx as hx
import heatpump_dyn as hpp
import heatpump_secondary as hps
import economics as econ

# --- modes (OperatorApp._mode_labels) ---------------------------------------------------------
MODE_LABELS = [('No control', 'no_control'), ('Feedforward', 'feedforward'),
               ('Feedforward+PID', 'feedforward_pid'), ('FF+PID+RTO', 'feedforward_pid_rto'),
               ('MPC (tracking)', 'mpc'), ('Economic MPC', 'economic_mpc')]
MODE_IDS = tuple(m for _, m in MODE_LABELS)
#  Modes in which a controller, not the HP2 slider, sets HP2 duty (OperatorApp._HP2_CONTROLLED).
HP2_CONTROLLED = ('feedforward_pid_rto', 'mpc', 'economic_mpc')
MPC_MODES = ('mpc', 'economic_mpc')

# --- sliders (OperatorApp.__init__ specs / SLOT) ----------------------------------------------
#  group: 'always' = live in every mode; 'explicit' = no_control only (explicit MV commands);
#  'pressure' = every mode except no_control (pressure targets). Pairs share a screen slot.
SLIDERS = [
    # key, label, lo, hi, whole, slot, group
    ('qsec_cmd', 'HP2 duty [kW]', cfg.Q_SEC_LO, cfg.Q_SEC_HI, True, 0, 'always'),
    ('pmps_cmd', 'MPS pressure target [bar]', cfg.P_MPS_LO, cfg.P_MPS_HI, False, 1, 'pressure'),
    ('mfuel_cmd', 'Biomass [kg/s]', cfg.MFUEL_LO, cfg.MFUEL_HI, False, 1, 'explicit'),
    ('plps_cmd', 'LPS pressure target [bar]', cfg.P_LPS_LO, cfg.P_LPS_HI, False, 2, 'pressure'),
    ('frac_t2_cmd', 'Turbine-2 split [-]', cfg.FRAC_T2_LO, cfg.FRAC_T2_HI, False, 2, 'explicit'),
    ('d_mps', 'MPS demand [kg/s]', cfg.D_MPS_LO, cfg.D_MPS_HI, False, 3, 'always'),
    ('d_lps', 'LPS demand [kg/s]', cfg.D_LPS_LO, cfg.D_LPS_HI, False, 4, 'always'),
    ('hw_min', 'HW minimum [kg/s]', cfg.HW_MIN_LO, cfg.HW_MIN_HI, False, 5, 'always'),
    ('twout_cmd', 'HW temperature [C]', cfg.TWOUT_LO, cfg.TWOUT_HI, False, 6, 'always'),
    ('dtsub_cmd', 'HP1 subcooling [K]', cfg.DT_SUB_LO, cfg.DT_SUB_HI, False, 7, 'explicit'),
]
SLIDER_KEYS = tuple(s[0] for s in SLIDERS)
SLIDER_LIMITS = {k: (lo, hi) for k, _, lo, hi, _, _, _ in SLIDERS}
SLIDER_WHOLE = {k: whole for k, _, _, _, whole, _, _ in SLIDERS}
EXPLICIT_KEYS = tuple(s[0] for s in SLIDERS if s[6] == 'explicit')
PRESSURE_KEYS = tuple(s[0] for s in SLIDERS if s[6] == 'pressure')
DEMAND_KEYS = tuple(econ.DEMAND_DRIVERS)

# --- colours (OperatorApp class constants) ----------------------------------------------------
COLORS = dict(
    MPS='#1f5573', LPS='#8a5a00', HW='#1f5c3c',
    MPS_FLOW='#6fb6de', LPS_FLOW='#f0b94a', HW_FLOW='#78c796',
    FUEL='#a13d2f', STEAM='#d98a3d', POWER='#4a3a6b', MARGIN='#2b6f6a', PRICE='#b5651d')

# --- trend panels (redraw()'s `panels`) -------------------------------------------------------
WINDOW_S = 2000.0
#  (Display smoothing of the trend lines lives in the frontend: src/lib/trend.ts, SMOOTH_SAMPLES.)


#  Series that are INPUTS, not plant responses: operator setpoints, site demands, controller
#  commands and the electricity price schedule. They are drawn exactly as they are (a setpoint
#  change is an instant step); only measured/computed plant responses get display smoothing.
UNSMOOTHED = ('pmps_cmd', 'plps_cmd', 'twout_cmd', 'd_mps', 'd_lps', 'hw_min',
              'mfuel_cmd', 'qsec_cmd', 'price_export')


def _series(rows):
    return [dict(key=k, label=lbl, color=c, role=role, smooth=k not in UNSMOOTHED)
            for k, lbl, c, role in rows]


def _axis(label, ylim0, rows):
    return dict(label=label, ylim0=list(ylim0), series=_series(rows))


def panels():
    C = COLORS
    mps_pad = .2 * (cfg.P_MPS_HI - cfg.P_MPS_LO)
    tw_pad = .1 * (cfg.TWOUT_HI - cfg.TWOUT_LO)
    return [
        dict(id='mps',
             left=_axis('MPS pressure [bar]', (cfg.P_MPS_0 - mps_pad, cfg.P_MPS_0 + mps_pad), [
                 ('pmps_act', 'MPS pressure', C['MPS'], 'primary'),
                 ('pmps_cmd', 'MPS setpoint', C['MPS'], 'secondary')]),
             right=_axis('MPS flow [kg/s]', (0, cfg.D_MPS_HI * 1.05), [
                 ('m_mps_delivered', 'MPS delivered', C['MPS_FLOW'], 'primary'),
                 ('d_mps', 'MPS setpoint', C['MPS_FLOW'], 'secondary')])),
        dict(id='lps',
             left=_axis('LPS pressure [bar]', (cfg.P_LPS_LO - .2, headers.DEFAULT.relief_full + .3), [
                 ('plps_act', 'LPS pressure', C['LPS'], 'primary'),
                 ('plps_cmd', 'LPS setpoint', C['LPS'], 'secondary')]),
             right=_axis('LPS flow [kg/s]', (0, cfg.D_LPS_HI * 1.05), [
                 ('m_lps_produced', 'LPS produced', C['LPS_FLOW'], 'primary'),
                 ('d_lps', 'LPS setpoint', C['LPS_FLOW'], 'secondary')])),
        dict(id='hw',
             left=_axis('HW temperature [C]', (cfg.TWOUT_LO - tw_pad, cfg.TWOUT_HI + tw_pad), [
                 ('twout', 'HW temperature', C['HW'], 'primary'),
                 ('twout_cmd', 'HW temp setpoint', C['HW'], 'secondary')]),
             right=_axis('HW flow [kg/s]', (0, cfg.HW_MIN_HI * .75), [
                 ('m_hw_total', 'HW delivered', C['HW_FLOW'], 'primary'),
                 ('hw_min', 'HW minimum', C['HW_FLOW'], 'secondary')])),
        dict(id='power',
             left=_axis('Power [kW]', (0, 25000), [
                 ('power1', 'Turbine 1', C['MPS'], 'primary'),
                 ('power2', 'Turbine 2', C['LPS'], 'primary'),
                 ('power', 'Total turbine power', C['POWER'], 'primary'),
                 ('W_hp_total', 'Heat pump work', C['HW'], 'secondary'),
                 ('net_export', 'Net export', C['POWER'], 'secondary')]),
             right=None),
        dict(id='fuel',
             left=_axis('Biomass / steam [kg/s]', (0, 90.), [
                 ('mfuel_act', 'Biomass actual', C['FUEL'], 'primary'),
                 ('mfuel_cmd', 'Biomass command', C['FUEL'], 'secondary'),
                 ('steam', 'Steam out', C['STEAM'], 'primary')]),
             right=None),
        dict(id='hp2',
             left=_axis('HP2 duty [kW]', (0, cfg.Q_SEC_HI * 1.05), [
                 ('Q_cond_sec', 'Actual', C['HW'], 'primary'),
                 ('qsec_cmd', 'Command', C['HW'], 'secondary')]),
             right=None),
        dict(id='margin',
             left=_axis('Margin [NZ$/h]', (0, 5000), [
                 ('margin_h', 'Margin', C['MARGIN'], 'primary')]),
             right=_axis('Electricity price [NZ$/MWh]', (0, 300), [
                 ('price_export', 'Electricity price', C['PRICE'], 'secondary')])),
    ]


def series_keys():
    """Every row key a trend line plots, in panel order, without duplicates."""
    seen = []
    for p in panels():
        for side in ('left', 'right'):
            ax = p[side]
            if ax:
                for s in ax['series']:
                    if s['key'] not in seen:
                        seen.append(s['key'])
    return seen


#  Row fields the schematic and the status bar read from the newest row (_update_schematic).
SCHEMATIC_KEYS = (
    't', 'duty', 'eta_b', 'Q_hx', 'power1', 'dh_act', 'power2', 'dh_act2', 'dQ_shortfall',
    'Q_cond_pri', 'COP_pri', 'W_hp_pri', 'COP_sec', 'Q_cond_sec', 'W_hp_sec', 'mfuel_act', 'steam',
    'gas', 'stack', 'pmps_act', 'T_out1', 'm_mps_site', 'mps_shortfall', 'frac_t2_act', 'm_lps',
    'plps_act', 'T_out2', 'm_lps_total', 'lps_shortfall', 'm_hw_total', 'm_lps_hp', 'm_letdown',
    'm_vent', 'm_hw_hp', 'mwater_act', 'twout', 'hw_short', 'service_ok')

#  Captions on the schematic that quote model constants (so they cannot drift from the model).
def schematic_constants():
    return dict(
        P_HP=cfg.P_HP, T_HP=cfg.T_HP, T_FW=blr.T_FW, T_W_IN=hx.T_W_IN,
        T_FG_IN_HP=hpp.T_FG_IN_HP, T_FG_OUT_HP_SPEC=hpp.T_FG_OUT_HP_SPEC, T_MU=hpp.T_MU,
        T_SRC_SEC=hps.T_SRC_SEC)


def build_config():
    return dict(
        modes=[dict(id=i, label=l, hp2_controlled=i in HP2_CONTROLLED) for l, i in MODE_LABELS],
        sliders=[dict(key=k, label=l, min=lo, max=hi, whole=w, slot=s, group=g,
                      decimals=0 if w else 2)
                 for k, l, lo, hi, w, s, g in SLIDERS],
        demand_keys=list(DEMAND_KEYS),
        panels=panels(),
        colors=COLORS,
        window_s=WINDOW_S,
        series_keys=series_keys(),
        schematic_keys=list(SCHEMATIC_KEYS),
        schematic_constants=schematic_constants(),
        biomass_options=list(econ.BIOMASS_PRICE_OPTIONS),
        price_modes=[dict(id='fixed', label='Fixed'), dict(id='daily', label='Daily loop')],
        demand_modes=[dict(id='fixed', label='Fixed'), dict(id='schedule', label='Schedule')],
    )
