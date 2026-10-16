"""Parity test: SimSession must reproduce the desktop OperatorApp row for row.

Both are driven through the same command script (slider moves, every mode, demand schedule,
prices, reset). The desktop app runs headless (Agg) and synchronously (threaded=False); the web
session runs through SimSession.step_once(). Every key of every row must match.

Run directly:   python webapp/backend/tests/test_parity.py [--quick]
or with pytest: pytest webapp/backend/tests/test_parity.py
"""
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # backend/
sys.path.insert(0, str(HERE.parents[2]))              # Models/

from plant_operator import OperatorApp                # noqa: E402
from sim_session import SimSession                    # noqa: E402
import ui_config as ui                                # noqa: E402

MODE_INDEX = {m: i for i, m in enumerate(ui.MODE_IDS)}
#  Wall-clock dependent, so legitimately different between two runs of the same inputs.
IGNORE = {'mpc_solve_s'}


def script(quick=False):
    s = [('step',), ('set', 'qsec_cmd', +500), ('step',), ('step',),
         ('mode', 'feedforward_pid'), ('step',), ('set_abs', 'plps_cmd', 3.2), ('step',),
         ('mode', 'no_control'), ('step',), ('set', 'mfuel_cmd', +0.5), ('step',),
         ('mode', 'feedforward_pid'), ('step',),
         ('mode', 'feedforward'), ('demand_interval', 10.0), ('demand', True), ('step',), ('step',),
         ('demand', False), ('price', 'daily'), ('step',), ('mean', 90.0), ('biomass', 18.0),
         ('step',), ('price', 'fixed')]
    if not quick:
        s += [('mode', 'feedforward_pid_rto'), ('step',), ('mode', 'mpc'), ('step',),
              ('mode', 'economic_mpc'), ('step',), ('mode', 'feedforward_pid'), ('step',)]
    s += [('reset',), ('step',), ('step',)]
    return s


def apply_desktop(a, op):
    kind = op[0]
    if kind == 'step':
        a.running = True
        a.tick()
    elif kind == 'set':
        a.sliders[op[1]].set_val(a.sliders[op[1]].val + op[2])
    elif kind == 'set_abs':
        a.sliders[op[1]].set_val(op[2])
    elif kind == 'mode':
        a.mode.set_active(MODE_INDEX[op[1]])
    elif kind == 'demand_interval':
        a.demand.interval_s = op[1]
    elif kind == 'demand':
        a.demand_radio.set_active(1 if op[1] else 0)
    elif kind == 'price':
        a.price_radio.set_active(1 if op[1] == 'daily' else 0)
    elif kind == 'mean':
        a.price_box.set_val(f'{op[1]:g}')
    elif kind == 'biomass':
        a.biomass_radio.set_active(list(map(float, ['12', '15', '18'])).index(op[1]))
    elif kind == 'reset':
        a.reset()


def apply_web(s, op):
    kind = op[0]
    if kind == 'step':
        s.start()
        s.step_once()
    elif kind == 'set':
        s.set_value(op[1], s.values[op[1]] + op[2])
    elif kind == 'set_abs':
        s.set_value(op[1], op[2])
    elif kind == 'mode':
        s.set_mode(op[1])
    elif kind == 'demand_interval':
        s.demand.interval_s = op[1]
    elif kind == 'demand':
        s.set_demand_mode(op[1])
    elif kind == 'price':
        s.set_price_mode(op[1])
    elif kind == 'mean':
        s.set_price_mean(op[1])
    elif kind == 'biomass':
        s.set_biomass_price(op[1])
    elif kind == 'reset':
        s.reset()


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def compare(rows_a, rows_b):
    assert len(rows_a) == len(rows_b), (len(rows_a), len(rows_b))
    worst = 0.0
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        assert set(ra) == set(rb), (i, set(ra) ^ set(rb))
        for k in ra:
            if k in IGNORE:
                continue
            assert same(ra[k], rb[k]), (i, ra['t'], k, ra[k], rb[k])
            if isinstance(ra[k], float) and not math.isnan(ra[k]):
                worst = max(worst, abs(ra[k] - rb[k]))
    return worst


def check_ui_matches_desktop(app, web):
    """The static UI definitions the web app serves must equal what the desktop app builds."""
    import re
    import economics as econ
    cfg = ui.build_config()
    n = 0
    # sliders: label, range, start-up value
    for c in cfg['sliders']:
        sd = app.sliders[c['key']]
        assert sd.label.get_text() == c['label'], (c['key'], sd.label.get_text(), c['label'])
        assert (sd.valmin, sd.valmax) == (c['min'], c['max']), c['key']
        assert abs(sd.valinit - web.defaults[c['key']]) < 1e-12, (c['key'], sd.valinit, web.defaults[c['key']])
        assert bool(sd.valstep) == c['whole'], c['key']
        n += 1
    assert len(app.sliders) == len(cfg['sliders'])
    # modes, which sliders swap, which modes hand HP2 to a controller, demand drivers, biomass options
    assert [(m['label'], m['id']) for m in cfg['modes']] == app._mode_labels
    assert tuple(m['id'] for m in cfg['modes'] if m['hp2_controlled']) == app._HP2_CONTROLLED
    assert ui.EXPLICIT_KEYS == app._explicit_keys and ui.PRESSURE_KEYS == app._pressure_keys
    assert tuple(cfg['demand_keys']) == tuple(app._DEMAND_KEYS)
    assert cfg['biomass_options'] == list(econ.BIOMASS_PRICE_OPTIONS)
    assert cfg['window_s'] == app.WINDOW
    # trend panels: axis titles, initial y ranges, every series' key / label / colour / line role
    desktop = app._trend_lines
    assert len(desktop) == len(cfg['panels']) == 7
    for entries, p in zip(desktop, cfg['panels']):
        axes = [p['left']] + ([p['right']] if p['right'] else [])
        assert len(entries) == len(axes), p['id']
        for (_ax, (label, ylim0, series), _lines), a in zip(entries, axes):
            assert label == a['label'], (p['id'], label, a['label'])
            assert tuple(ylim0) == tuple(a['ylim0']), (p['id'], ylim0, a['ylim0'])
            assert [tuple(s) for s in series] == [(s['key'], s['label'], s['color'], s['role']) for s in a['series']], p['id']
            n += 1
    # start-up state: mode, prices, demand schedule, status text
    assert app.rig.operator_mode == web.mode
    assert app.rig.prices.electricity.mode == web.rig.prices.electricity.mode
    assert app.rig.prices.electricity.mean == web.rig.prices.electricity.mean
    assert app.rig.prices.biomass_per_gj == web.rig.prices.biomass_per_gj
    assert app.demand_on == web.demand_on
    assert app.demand.interval_s == web.demand.interval_s and app.demand.seed == web.demand.seed
    return n


def norm_status(text):
    import re
    return re.sub(r'MPC solve [\d.]+ s', 'MPC solve X s', text)   # wall-clock time, differs run to run


def run_parity(quick=False):
    app = OperatorApp()
    app.timer.stop()
    web = SimSession()
    try:
        n_ui = check_ui_matches_desktop(app, web)
        print(f'UI definitions match the desktop app ({n_ui} sliders/axes compared)', flush=True)
        status_checks = 0
        for n, op in enumerate(script(quick)):
            apply_desktop(app, op)
            apply_web(web, op)
            #  Compare the slider panel after every command, and the rows after every step/reset.
            for key in ui.SLIDER_KEYS:
                sd = app.sliders[key]
                assert abs(sd.val - web.values[key]) < 1e-9, (n, op, key, sd.val, web.values[key])
                assert sd.active == web.active(key), (n, op, key, 'active', sd.active, web.active(key))
                assert sd.ax.get_visible() == web.visible(key), (n, op, key, 'visible')
            assert app.rig.operator_mode == web.mode, (n, op)
            assert app.running == web.running, (n, op, app.running, web.running)
            if op[0] in ('step', 'reset'):
                worst = compare(app.rows, web.rows)
            if op[0] in ('step', 'reset', 'mode', 'price', 'mean', 'biomass', 'demand'):
                # the desktop refreshes its status block on exactly these; the text must be identical
                st = web.status()
                want = norm_status(app.status.get_text())
                got = norm_status(st['line1'] + '\n' + st['line2'])
                assert want == got, (n, op, '\n' + want, '\n' + got)
                assert (app.status.get_color() == '#245c35') == st['ok'], (n, op, 'status colour')
                status_checks += 1
        print(f'PARITY PASS: {len(web.rows)} rows, {len(script(quick))} commands, '
              f'max abs difference {worst:.3g}, {status_checks} status texts identical', flush=True)
    finally:
        app.exit_app()
        web.close()


def test_parity_quick():
    run_parity(quick=True)


def test_parity_all_modes():
    run_parity(quick=False)


if __name__ == '__main__':
    run_parity(quick='--quick' in sys.argv)
