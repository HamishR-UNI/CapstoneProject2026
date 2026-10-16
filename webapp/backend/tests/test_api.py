"""API tests: REST + WebSocket protocol, against the real simulation (one session, shared)."""
import os
import sys
from pathlib import Path

import pytest

os.environ['WEB_PREWARM'] = '0'   # tests build their own session; no spare competing for CPU

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from starlette.testclient import TestClient  # noqa: E402
import app as appmod                          # noqa: E402
import ui_config as ui                        # noqa: E402


@pytest.fixture(scope='module')
def client():
    with TestClient(appmod.app) as c:
        yield c


@pytest.fixture(scope='module')
def sid(client):
    r = client.post('/api/sessions')
    assert r.status_code == 200
    s = r.json()['session_id']
    yield s
    client.delete(f'/api/sessions/{s}')


def recv_until(ws, kind, limit=20):
    for _ in range(limit):
        m = ws.receive_json()
        if m['type'] == kind:
            return m
    raise AssertionError(f'no {kind!r} message within {limit}')


def test_config(client):
    c = client.get('/api/config').json()
    assert [m['id'] for m in c['modes']] == list(ui.MODE_IDS)
    assert len(c['sliders']) == 10 and len(c['panels']) == 7
    assert c['panels'][0]['left']['series'][0]['key'] == 'pmps_act'
    assert 'T_FW' in c['schematic_constants']
    assert client.get('/api/health').json()['ok']


def test_unknown_session(client):
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect('/api/sessions/nope/ws') as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 4404   # readable by the browser, so the page starts a fresh session
    assert client.get('/api/sessions/nope/csv').status_code == 404


def test_snapshot_and_initial_state(client, sid):
    with client.websocket_connect(f'/api/sessions/{sid}/ws') as ws:
        snap = ws.receive_json()
        assert snap['type'] == 'snapshot' and snap['start'] == 0
        st = snap['state']
        assert st['mode'] == 'feedforward' and not st['running'] and st['n_rows'] == 1
        assert snap['cols']['t'] == [0.0] and len(snap['cols']['pmps_act']) == 1
        assert set(st['sliders']) == set(ui.SLIDER_KEYS)
        assert st['sliders']['pmps_cmd']['visible'] and not st['sliders']['mfuel_cmd']['visible']


def test_run_pause_inputs_mode_reset_csv(client, sid):
    with client.websocket_connect(f'/api/sessions/{sid}/ws') as ws:
        ws.receive_json()                                   # snapshot
        # inactive slider (explicit MV outside no_control) is ignored; HP2 duty is accepted + clipped
        ws.send_json(dict(type='inputs', values=dict(mfuel_cmd=11.0, qsec_cmd=1e9)))
        # a run: two batches arrive as 'rows' messages with contiguous start indices
        ws.send_json(dict(type='run'))
        a = recv_until(ws, 'rows')
        assert a['start'] == 1 and len(a['cols']['t']) == 10 and a['state']['running']
        assert a['state']['sliders']['qsec_cmd']['value'] == ui.SLIDER_LIMITS['qsec_cmd'][1]
        assert a['state']['sliders']['mfuel_cmd']['value'] != 11.0
        b = recv_until(ws, 'rows')
        assert b['start'] == a['start'] + 10
        ws.send_json(dict(type='pause'))
        # drain until the loop reports it stopped
        last = b
        while last['state']['running']:
            last = ws.receive_json()
            if last['type'] not in ('rows', 'state'):
                continue
        n = last['state']['n_rows']
        assert n >= 21

        # mode switch changes which sliders show (state message)
        ws.send_json(dict(type='mode', mode='no_control'))
        st = recv_until(ws, 'state')['state']
        assert st['mode'] == 'no_control'
        assert st['sliders']['mfuel_cmd']['visible'] and not st['sliders']['pmps_cmd']['visible']

        # economics / demand switches
        ws.send_json(dict(type='price_mode', mode='daily'))
        assert recv_until(ws, 'state')['state']['price_mode'] == 'daily'
        ws.send_json(dict(type='biomass', value=18.0))
        assert recv_until(ws, 'state')['state']['biomass_price'] == 18.0
        ws.send_json(dict(type='demand_mode', on=True))
        st = recv_until(ws, 'state')['state']
        assert st['demand_on'] and not st['sliders']['d_lps']['active']
        ws.send_json(dict(type='biomass', value=99.0))     # invalid -> error, connection survives
        assert recv_until(ws, 'error')['message']

        # CSV of the run so far, desktop format (one batch may still have been in flight at
        # pause; the mode switch above waited for it, so a resync now gives the settled count)
        ws.send_json(dict(type='resync'))
        n = recv_until(ws, 'snapshot')['state']['n_rows']
        assert n >= 21 and n % 10 == 1
        text = client.get(f'/api/sessions/{sid}/csv').text
        lines = [x for x in text.splitlines() if not x.startswith('#')]
        header = lines[0].split(',')
        assert {'t', 'margin_h', 'm_lps_produced', 'net_export'} <= set(header) and len(lines) == n + 1

        # reset: a fresh snapshot, mode kept, demand-schedule level applied
        ws.send_json(dict(type='reset'))
        snap = recv_until(ws, 'snapshot')
        assert snap['state']['n_rows'] == 1 and snap['state']['mode'] == 'no_control'
        assert snap['state']['epoch'] >= 1


def test_session_cap(client, monkeypatch):
    monkeypatch.setattr(appmod, 'MAX_SESSIONS', len(appmod.registry.runners))
    assert client.post('/api/sessions').status_code == 503
