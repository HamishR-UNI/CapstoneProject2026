# Plant operator web app (React + FastAPI)

A browser version of `plant_operator.py`. The desktop app is untouched; the simulation core
(`plant_rig.py`, `operator_control.py`, ...) is imported unchanged. Design and phase plan:
[../docs/WEBAPP_PLAN.md](../docs/WEBAPP_PLAN.md).

## Run it

One-off build of the frontend (needs Node.js 20+):

```
cd frontend
npm install
npm run build
```

Then, from `Models/`, double-click `webapp\run_web.bat` (or `cd webapp\backend` then
`python -m uvicorn app:app --port 8000`) and open <http://127.0.0.1:8000>.

Building a simulation session (the GEKKO model plus its initial solves) takes one to two minutes.
The server builds one spare at start-up, so the first browser tab usually attaches to a ready
session. Reload keeps your session (it is remembered per tab); closing the tab pauses it.

### Development

```
python -m uvicorn app:app --port 8000 --reload     # in webapp/backend
npm run dev                                         # in webapp/frontend -> http://localhost:5173
```

Vite proxies `/api` (REST and WebSocket) to port 8000.

## Layout

```
backend/
  app.py          FastAPI routes, WebSocket protocol, session registry (cap, idle timeout, spare)
  sim_session.py  SimSession: OperatorApp's logic without matplotlib (one rig per session)
  ui_config.py    sliders, modes, trend panels, colours, schematic constants served as /api/config
  tests/          test_parity.py (desktop vs web, row for row), test_api.py (REST + WebSocket)
frontend/src/
  store.ts        zustand store + WebSocket client (resync on gaps, reconnect)
  components/     Schematic (SVG), TrendPanel (uPlot), Controls (sliders, modes, economics), StatusBar
  lib/trend.ts    smoothing and autoscale ported from plant_operator.py (unit-tested)
```

## Line smoothing

The plant's responses are fast next to the 2000 s axis (10 s is about 3 px), so raw traces read as
straight runs with sharp corners. The header has a **Line smoothing** control (Off / Low / Medium /
High = Gaussian sigma of 0 / 6 / 14 / 28 s, default Medium, remembered in the browser) that rounds them
off. Only plant responses are smoothed: setpoints, site demands, controller commands and the price
schedule are inputs and are drawn exactly as they are (an instant step), see `UNSMOOTHED` in
`backend/ui_config.py`. It is display only: the schematic, status line and CSV always use the raw values. Because the
smoothing is centred, a trace starts to move a little before the moment of a step (about 2 sigma);
choose Off or Low when judging exact response times.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `WEB_MAX_SESSIONS` | 4 | simultaneous simulation sessions (each is a GEKKO process) |
| `WEB_IDLE_TIMEOUT_S` | 900 | close a session this long after its last viewer left |
| `WEB_BATCH` | 10 | seconds simulated per solve (desktop `--batch`) |
| `WEB_PREWARM` | 1 | keep one ready-built spare session (0 to disable) |

## Tests

```
python webapp/backend/tests/test_parity.py [--quick]   # from Models/, headless desktop vs SimSession
python -m pytest webapp/backend/tests/test_api.py      # protocol tests against the real simulation
cd webapp/frontend && npm test                          # trend maths
```

`test_parity.py` drives the desktop `OperatorApp` (Agg backend) and a `SimSession` through the same
command script (slider moves, all six modes, demand schedule, prices, reset) and requires every key
of every row to match; MPC solve wall-time (`mpc_solve_s`) is the only field excluded.

### Schematic overlap check

`frontend/src/components/Schematic.tsx` places every label by hand, so after changing a font size or
position, measure the rendered SVG rather than judging by eye. In the browser console (with a live
session loaded), collect `getBoundingClientRect()` of every tag card, caption and block title and test:
text against text, text crossing a block or turbine edge, a turbine title leaving its polygon, a pipe
passing through a label, and anything outside the viewBox. The last check on 2026-10-15 found none at
1366x768 and 1920x1080, with the plant running and long values (5-digit kW) on screen.

## Known limits

- The full run history is held in server memory (as the desktop app does) and CSV export builds
  the whole file in memory; very long runs (tens of MB of CSV) will be slow to download.
- No authentication. Put it behind a reverse proxy with HTTPS and a login before exposing it.
- Python 3.14 with GEKKO prints `Error: 'results.json' not found` once while a model is built;
  this comes from the existing plant code and also appears in the desktop app.
