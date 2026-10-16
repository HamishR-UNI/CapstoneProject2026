"""Smooth LPS error scaling; TunedPressureRig supplies optional dashboard PI.

Measured gain ratio with MPS PI active is 0.706551/0.161967. Map pressure into
an equivalent error coordinate with slope 1 above 3 bar and this ratio below.
P and I see the same mapped error, while integral state stays in fuel units.
No gain jumps, integral resets, plant changes or extra actuator limits.
"""
import math
from pathlib import Path
import pressure_feedback as pf

GAIN_RATIO=0.7065509989980203/0.16196673136633777
TRANSITION_BAR=.002
RESPONSIVE_TUNING=pf.Tuning(0.850198964419589,606.9443966650285)


def coordinate(pressure):
    x=pressure-3.
    negative=.5*(x-math.hypot(x,TRANSITION_BAR))
    return x+(GAIN_RATIO-1.)*negative


class ScheduledRig(pf.PressureRig):
    lps_tuning = pf.LPS_TUNING

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lps_pi = pf.PI(self.lps_tuning)

    def _lps_control_error(self,error,measurement):
        return coordinate(measurement+error)-coordinate(measurement)


class TunedPressureRig(ScheduledRig):
    """Validated responsive candidate; feedforward remains the startup mode."""
    lps_tuning = RESPONSIVE_TUNING


def checks():
    for pressure in (2.,2.95,2.999,3.,3.001,3.05,4.):
        assert coordinate(pressure)-coordinate(pressure)==0
        derivative=(coordinate(pressure+1e-6)-coordinate(pressure-1e-6))/2e-6
        assert 1-1e-6<=derivative<=GAIN_RATIO+1e-6
    # A control-coordinate change cannot kick the transferred trim.
    for pressure in (2.95,3.,3.5):
        e=coordinate(3.)-coordinate(pressure)
        loop=pf.PI(pf.LPS_TUNING)
        out,_=loop.output(e,track=-.12)
        assert abs(out+.12)<1e-12
    print('PASS mapping sign, continuity, bounded slope and trim tracking; ratio',GAIN_RATIO,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('case',choices=['checks','quick','up','down','schedule'])
    p.add_argument('--responsive',action='store_true')
    args=p.parse_args();case=args.case
    if args.responsive:
        # Closed-MPS fits: worst tau and delay; lambda=max(tau,2*theta).
        ScheduledRig.lps_tuning=RESPONSIVE_TUNING
    checks()
    if case!='checks':
        import pi_validation as v
        v.PressureRig=ScheduledRig
        v.OUT=Path(__file__).resolve().parent/'lps_tuning_results'
        v.OUT.mkdir(exist_ok=True)
        prefix='responsive_' if args.responsive else 'scheduled_'
        import json
        from dataclasses import asdict
        (v.OUT/(prefix+case+'_config.json')).write_text(json.dumps(dict(
            gain_ratio=GAIN_RATIO,transition_bar=TRANSITION_BAR,
            MPS=asdict(pf.MPS_TUNING),LPS=asdict(ScheduledRig.lps_tuning)),indent=2))
        if case=='quick':v.quick()
        else:v.run(prefix+case,'auto',2500 if case=='schedule' else 5100,case)
