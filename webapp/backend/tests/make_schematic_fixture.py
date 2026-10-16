"""Capture the desktop schematic's text at three operating points, for the React comparison test.

Runs the real desktop OperatorApp (headless) and, at each point, records (a) the newest row's
schematic fields, (b) every non-empty text on the desktop schematic axes with the colour of the
value tags. webapp/frontend/src/components/Schematic.test.tsx renders the React schematic from the
same row and requires the same texts and colours.

    python webapp/backend/tests/make_schematic_fixture.py     (from Models/; about 3 minutes)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

from plant_operator import OperatorApp   # noqa: E402
from sim_session import clean            # noqa: E402
import ui_config as ui                   # noqa: E402

OUT = HERE.parents[1] / 'frontend' / 'src' / 'components' / 'schematic.fixture.json'
TAGS = ('d_boiler', 'd_hx', 'd_t1', 'd_t2', 'd_hpp', 'd_hps', 'd_fuel', 'd_hp', 'd_flue', 'd_power',
        'd_mps', 'd_frac', 'd_t1_out', 'd_t2_out', 'd_lps', 'd_hw_site', 'd_hpp_lps', 'd_letdown',
        'd_vent', 'd_hpp_hw', 'd_hx_hw', 'd_hp2')


def axes_of(app):
    """Each trend axis: its initial range, the range the desktop actually chose, and the min/max of
    the data it was fed (the desktop's autoscale depends on nothing else)."""
    out = []
    for i, entries in enumerate(app._trend_lines):
        for side, (ax, (label, ylim0, series), lines) in zip(('left', 'right'), entries):
            ys = [y for ln in lines for y in ln.get_ydata()]
            out.append(dict(panel=i, side=side, ylim0=list(ylim0), ylim=list(ax.get_ylim()),
                            xlim=list(ax.get_xlim()), dmin=float(min(ys)), dmax=float(max(ys)),
                            t_last=float(lines[0].get_xdata()[-1])))
    return out


def capture(app, name):
    r = app.rows[-1]
    texts = [t.get_text() for t in app.ax_d.texts if t.get_text()]
    tags = {k: dict(text=getattr(app, k).get_text(), color=getattr(app, k).get_color()) for k in TAGS}
    return dict(name=name, t=r['t'], mode=app.rig.operator_mode,
                last={k: clean(r.get(k)) for k in ui.SCHEMATIC_KEYS},
                texts=texts, tags=tags, axes=axes_of(app))


def ticks(app, n):
    app.running = True
    for _ in range(n):
        app.tick()


def main():
    app = OperatorApp()
    app.timer.stop()
    points = []
    ticks(app, 30)
    points.append(capture(app, 'feedforward, default inputs, t=300'))
    app.sliders['qsec_cmd'].set_val(15000)
    ticks(app, 2)
    points.append(capture(app, 'feedforward, HP2 stepped to 15000 kW (heat pump settling)'))
    app.mode.set_active(0)               # no_control
    app.sliders['mfuel_cmd'].set_val(5.0)
    ticks(app, 25)
    points.append(capture(app, 'no_control, biomass cut to 5 kg/s (unserved demand)'))
    app.exit_app()
    OUT.write_text(json.dumps(dict(constants=ui.schematic_constants(), points=points), indent=1))
    print('wrote', OUT, [p['t'] for p in points], flush=True)


if __name__ == '__main__':
    main()
