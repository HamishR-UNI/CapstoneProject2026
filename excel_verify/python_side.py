"""Python side of the Excel verification: steady states built from the SAME component
modules the dynamic rig uses, so there is no separate copy that could drift.

plant_steady(c)  whole plant, Manual mode at rest: what OperatorRig('manual') settles to.
                 Same inputs as the sliders: d_mps, d_lps, hw_min, p_mps, p_lps, t_hw, q_sec.
component_*      one component at a time on identical inputs (used with the Excel sheets in
                 Standalone mode).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boiler as blr                    # noqa: E402
import turbine_model as tmod            # noqa: E402
import flue_gas_water_hx as hx          # noqa: E402
import heatpump_dyn as hpp              # noqa: E402
import heatpump_secondary as hps        # noqa: E402
import header_closures as headers       # noqa: E402
import plant_rig as rig_cfg             # noqa: E402

DRIVERS = ('d_mps', 'd_lps', 'hw_min', 'p_mps', 'p_lps', 't_hw', 'q_sec')
BASE = dict(d_mps=rig_cfg.D_MPS_0, d_lps=rig_cfg.D_LPS_0, hw_min=rig_cfg.HW_MIN_0,
            p_mps=rig_cfg.P_MPS_0, p_lps=rig_cfg.P_LPS_0, t_hw=rig_cfg.TWOUT_0,
            q_sec=rig_cfg.Q_SEC_0)
LIMITS = dict(d_mps=(rig_cfg.D_MPS_LO, rig_cfg.D_MPS_HI), d_lps=(rig_cfg.D_LPS_LO, rig_cfg.D_LPS_HI),
              hw_min=(rig_cfg.HW_MIN_LO, rig_cfg.HW_MIN_HI), p_mps=(rig_cfg.P_MPS_LO, rig_cfg.P_MPS_HI),
              p_lps=(rig_cfg.P_LPS_LO, rig_cfg.P_LPS_HI), t_hw=(rig_cfg.TWOUT_LO, rig_cfg.TWOUT_HI),
              q_sec=(rig_cfg.Q_SEC_LO, rig_cfg.Q_SEC_HI))


class _Inverter:
    """Just enough of PlantRig to run its own demand inverse without GEKKO."""
    _delivery_at_fuel = staticmethod(rig_cfg.PlantRig._delivery_at_fuel)
    _state_at = rig_cfg.PlantRig._state_at
    _solve_demands = rig_cfg.PlantRig._solve_demands

    def __init__(self):
        self.mfuel_cmd_val = rig_cfg.MFUEL_0


_INV = _Inverter()


def plant_steady(c):
    """Manual-mode steady state (calculated fuel, split, subcooling; no feedback).

    Mirrors OperatorRig.step: effective demands include the letdown / dump the header
    closures take at the TARGET pressures, then PlantRig._solve_demands (MPS floor, fuel on
    LPS, HW minimum acting only when it binds). The boiler is evaluated with the fitted map
    the dynamic rig itself settles to (boiler.duty / stack_temp / steam_from_fuel); the boiler component
    test separately measures that fit against the physics.
    """
    dm, dl, hw = c['d_mps'], c['d_lps'], c['hw_min']
    pm, pl, th, qs = c['p_mps'], c['p_lps'], c['t_hw'], c['q_sec']
    cl = headers.flows(pm, pl, dm, dl)
    dm_eff = dm + cl['letdown']
    dl_eff = max(dl + cl['dump'] - cl['letdown'], 0.0)
    p_hp = float(np.clip(pl, *hpp.P_LPS_RANGE))
    dh_hw = hpp.dh_hw(th)

    fuel, dt_sub, hw_binds, fuel_clamped, dt_clamped = _INV._solve_demands(
        dm_eff, dl_eff, hw, qs, th, p_hp, dh_hw, fuel_guess=_INV.mfuel_cmd_val,
        dt_floor=c.get('dt_floor'))   # RTO's subcooling floor; absent = 10 K default

    gas = fuel*blr.GAS_PER_FUEL
    steam = blr.steam_from_fuel(fuel)
    t_stack = float(blr.stack_temp(fuel))
    q_boiler = float(blr.duty(fuel))
    frac = float(np.clip((steam - dm_eff)/steam, 0.0, 1.0)) if steam > 1e-9 else 0.0
    frac = rig_cfg.turbine2_cutoff(frac)   # turbine 2 off below 1% (plant_rig.FRAC_T2_OFF)
    m_t2 = frac*steam

    t1 = tmod.steady_numeric(mdot=steam, p2_bar=pm, T1_C=rig_cfg.T_HP, P1_bar=rig_cfg.P_HP, eta=tmod.ETA)
    t2 = tmod.steady_numeric(mdot=m_t2, p2_bar=pl, T1_C=t1['T_out'], P1_bar=pm, eta=rig_cfg.ETA2)
    h = hx.steady_state(gas, t_stack, th)
    q_rec = hpp.q_evap_recycle(h['m_water'])
    q_evap = gas*hpp.DH_GAS_HP + q_rec + qs
    q_pri = hpp.q_cond_from_sources(q_evap, p_hp, dt_sub)
    p = hpp.steady_state(q_pri, gas, q_rec + qs, dt_sub, p_hp, th)
    s = hps.steady_state(qs)

    mps_supply = steam - m_t2                     # what reaches the MPS header
    lps_supply = m_t2 + p['m_lps']                # what reaches the LPS header
    power = t1['power'] + t2['power']
    w_hp = p['W'] + s['W_comp_sec']
    m_hw = h['m_water'] + p['m_hw']
    return dict(
        fuel=fuel, dt_sub=dt_sub, frac_t2=frac, steam=steam, steam_t2=m_t2, T_stack=t_stack,
        gas=gas, eta_boiler=float(blr.efficiency(fuel)), Q_boiler=q_boiler,
        power1=t1['power'], power2=t2['power'], power=power, T_out1=t1['T_out'], T_out2=t2['T_out'],
        Q_hx=h['Q'], m_water_hx=h['m_water'], Q_evap_rec=q_rec, Q_evap_demand=p['Q_evap_demand'],
        Q_cond_pri=q_pri, COP_pri=p['COP'], W_pri=p['W'], m_lps_hp=p['m_lps'], m_hw_hp=p['m_hw'],
        Q_cond_sec=qs, W_sec=s['W_comp_sec'], COP_sec=s['COP_sec'],
        m_mps_supply=mps_supply, m_lps_supply=lps_supply, m_hw_total=m_hw,
        W_hp_total=w_hp, power_net=power - w_hp,
        letdown=cl['letdown'], vent=cl['dump'], d_mps_eff=dm_eff, d_lps_eff=dl_eff,
        hw_short=max(hw - m_hw, 0.0),
        hw_binds=int(hw_binds), fuel_clamped=int(fuel_clamped), dt_clamped=int(dt_clamped),
        mps_gap=mps_supply - dm_eff, lps_gap=lps_supply - dl_eff)


# --------------------------------------------------------------------------- components

def component_boiler(fuel):
    b = blr.steady_state(fuel)
    return dict(gas=b['mdot_gas'], T_stack=b['T_stack'], steam=b['mdot_steam'], Q_duty=b['Q'],
                eta=b['eta'], fired=fuel*blr.Q_PER_FUEL, T_flame=blr.T_HOT,
                steam_map=blr.steam_from_fuel(fuel),
                T_stack_map=float(blr.stack_temp(fuel)))


def component_turbine(mdot, p_in, T_in, p_out, eta):
    t = tmod.steady_numeric(mdot=mdot, p2_bar=p_out, T1_C=T_in, P1_bar=p_in, eta=eta)
    #  Wet-outlet diagnostics (isentropic quality, superheat margin of the ACTUAL outlet), from
    #  CoolProp saturation data: the rig's own turbine only exposes them on the surrogate path.
    from CoolProp.CoolProp import PropsSI
    s_in = PropsSI('S', 'T', T_in + 273.15, 'P', p_in*1e5, 'Water')/1000
    h_in = PropsSI('H', 'T', T_in + 273.15, 'P', p_in*1e5, 'Water')/1000
    sf, sg = (PropsSI('S', 'P', p_out*1e5, 'Q', q, 'Water')/1000 for q in (0, 1))
    hg = PropsSI('H', 'P', p_out*1e5, 'Q', 1, 'Water')/1000
    return dict(power=t['power'], T_out=t['T_out'], dh_act=t['dh_act'], dh_is=t['dh_is'],
                quality_is=(s_in - sf)/(sg - sf), superheat_margin=(h_in - t['dh_act']) - hg)


def component_hx(gas, T_in, T_set, T_w_in, T_out_spec=hx.T_FG_OUT_SPEC):
    r = hx.steady_state(gas, T_in, T_set, T_fg_out=T_out_spec, T_w_in=T_w_in)
    return dict(Q=r['Q'], m_water=r['m_water'])


def component_hp1(Q, gas, q_sec, dt_sub, p_lps, t_hw):
    r = hpp.steady_state(Q, gas, q_sec, dt_sub, p_lps, t_hw)
    # Same duty, but hot water flow on the EXCEL basis (rise from the 20 C makeup, not from the
    # recycle inlet), so the sheet's arithmetic can be checked apart from that assumption.
    from CoolProp.CoolProp import PropsSI
    dh_mu = (PropsSI('H', 'T', t_hw + 273.15, 'P', hpp.P_HW*1e5, 'water')
             - PropsSI('H', 'T', hpp.T_MU + 273.15, 'P', hpp.P_HW*1e5, 'water'))/1000
    return dict(COP=r['COP'], W=r['W'], m_ref=r['m_ref'], Q_evap_demand=r['Q_evap_demand'],
                dh_sub=r['dh_sub'], Q_sub=r['Q_sub'], m_lps=r['m_lps'], m_hw=r['m_hw'],
                Q_gas=r['Q_evap_gas_avail'],
                m_hw_makeup=r['Q_sub']/dh_mu)


def component_hp2(Q):
    r = hps.steady_state(Q)
    return dict(COP=r['COP_sec'], W=r['W_comp_sec'], m_ref=r['m_ref'], Q_evap=Q - r['W_comp_sec'])
