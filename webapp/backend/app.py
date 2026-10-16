"""FastAPI server for the plant operator dashboard.

    GET    /api/config                  UI configuration (sliders, modes, panels, colours, ...)
    POST   /api/sessions                create a simulation session -> {session_id}
    DELETE /api/sessions/{id}           close it
    GET    /api/sessions/{id}/csv       full-resolution CSV of the run so far
    WS     /api/sessions/{id}/ws        live control + results (messages described below)

One browser tab = one SimSession = one GEKKO model. Run from this folder:

    python -m uvicorn app:app --port 8000

WebSocket protocol (JSON)
  server -> client
    {type:'snapshot', start, cols, last, state}   full resync: rolling window of series + state
    {type:'rows',     start, cols, last, state}   the rows of one new solve batch (row index `start`)
    {type:'state',    state}                      state change with no new rows
    {type:'error',    message}
  client -> server
    {type:'inputs', values:{key:number}}   slider / typed values
    {type:'run'} {type:'pause'} {type:'reset'} {type:'resync'}
    {type:'mode', mode}                    {type:'demand_mode', on:bool}
    {type:'price_mode', mode} {type:'price_mean', value} {type:'biomass', value}
`cols` is column-oriented ({key: [values]}); `last` is the newest row's schematic/status fields;
`state` is everything else (run/halt flags, per-slider value/visible/active, status text, prices).
A client that sees `start` != the number of rows it holds sends {type:'resync'}.
"""
import asyncio
import contextlib
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import ui_config as ui
from sim_session import SimSession

MAX_SESSIONS = int(os.environ.get('WEB_MAX_SESSIONS', '4'))
IDLE_TIMEOUT_S = float(os.environ.get('WEB_IDLE_TIMEOUT_S', '900'))
BATCH = int(os.environ['WEB_BATCH']) if os.environ.get('WEB_BATCH') else None   # seconds per solve
PREWARM = os.environ.get('WEB_PREWARM', '1') != '0'   # keep one ready-built session in reserve
FRONTEND_DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'


class SessionRunner:
    """A SimSession plus its run loop and the websocket subscribers watching it."""

    def __init__(self, sid, session):
        self.sid = sid
        self.session = session
        self.subs = set()
        self.wake = asyncio.Event()
        self.last_active = time.monotonic()
        self.task = asyncio.create_task(self._loop())

    # --- run loop: solve in a worker thread, publish after every batch -----------------------
    async def _loop(self):
        s = self.session
        while not s.closed:
            if s.running and not s.rig.halted:
                result = await asyncio.to_thread(s.step_once)
                if result is not None:
                    self.publish(await asyncio.to_thread(s.delta, *result))
                else:   # paused/halted/failed meanwhile: tell the clients why
                    self.publish(dict(type='state', state=s.state()))
                    self.wake.clear()
            else:
                await self.wake.wait()
                self.wake.clear()

    def kick(self):
        self.last_active = time.monotonic()
        self.wake.set()

    def publish(self, msg, only=None):
        for q in ([only] if only is not None else list(self.subs)):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:   # a stalled client: drop its backlog, resync it
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(self.session.snapshot())

    # --- commands -------------------------------------------------------------------------------
    async def handle(self, msg, q):
        s = self.session
        self.last_active = time.monotonic()
        kind = msg.get('type')
        if kind == 'inputs':
            s.set_values(msg.get('values', {}))
            return   # no echo: the client already shows its own slider position
        if kind == 'run':
            s.start()
            self.kick()
        elif kind == 'pause':
            s.pause()
        elif kind == 'reset':
            await asyncio.to_thread(s.reset)
            self.publish(await asyncio.to_thread(s.snapshot))
            return
        elif kind == 'resync':
            self.publish(await asyncio.to_thread(s.snapshot), only=q)
            return
        elif kind == 'mode':
            await asyncio.to_thread(s.set_mode, msg['mode'])
        elif kind == 'demand_mode':
            await asyncio.to_thread(s.set_demand_mode, bool(msg['on']))
        elif kind == 'price_mode':
            await asyncio.to_thread(s.set_price_mode, msg['mode'])
        elif kind == 'price_mean':
            await asyncio.to_thread(s.set_price_mean, msg['value'])
        elif kind == 'biomass':
            await asyncio.to_thread(s.set_biomass_price, msg['value'])
        else:
            raise ValueError(f'unknown message type {kind!r}')
        self.publish(dict(type='state', state=s.state()))

    async def close(self):
        self.session.running = False
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.task
        await asyncio.to_thread(self.session.close)


class Registry:
    """Session registry. Building a SimSession (the GEKKO model plus its initial solves) takes
    one to two minutes, so one spare is built ahead of time and handed to the next client."""

    def __init__(self):
        self.runners = {}
        self.pending = 0
        self.spare = None   # asyncio.Task building (or holding) a ready SimSession

    @staticmethod
    def _build():
        return asyncio.create_task(asyncio.to_thread(SimSession, BATCH))

    def prewarm(self):
        if PREWARM and self.spare is None and len(self.runners) + self.pending < MAX_SESSIONS:
            self.spare = self._build()

    async def create(self):
        if len(self.runners) + self.pending >= MAX_SESSIONS:
            raise HTTPException(503, f'Server is at its limit of {MAX_SESSIONS} simulation sessions; '
                                     'try again shortly.')
        self.pending += 1
        try:
            task, self.spare = self.spare or self._build(), None
            session = await task
        finally:
            self.pending -= 1
        sid = uuid.uuid4().hex
        self.runners[sid] = SessionRunner(sid, session)
        self.prewarm()
        return sid

    async def shutdown(self):
        for sid in list(self.runners):
            await self.remove(sid)
        if self.spare is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread((await self.spare).close)
            self.spare = None

    def get(self, sid):
        return self.runners.get(sid)

    async def remove(self, sid):
        runner = self.runners.pop(sid, None)
        if runner is not None:
            await runner.close()

    async def reap(self):
        """Pause sessions nobody is watching; close those idle past the timeout."""
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            for sid, r in list(self.runners.items()):
                if not r.subs:
                    r.session.pause()
                    if now - r.last_active > IDLE_TIMEOUT_S:
                        await self.remove(sid)


registry = Registry()


@contextlib.asynccontextmanager
async def lifespan(app):
    reaper = asyncio.create_task(registry.reap())
    registry.prewarm()
    yield
    reaper.cancel()
    await registry.shutdown()


app = FastAPI(title='Plant operator', lifespan=lifespan)


@app.get('/api/health')
def health():
    return dict(ok=True, sessions=len(registry.runners), max_sessions=MAX_SESSIONS)


@app.get('/api/config')
def config():
    return ui.build_config()


@app.post('/api/sessions')
async def create_session():
    return dict(session_id=await registry.create())


@app.delete('/api/sessions/{sid}')
async def delete_session(sid: str):
    await registry.remove(sid)
    return dict(ok=True)


@app.get('/api/sessions/{sid}/csv')
async def session_csv(sid: str):
    runner = registry.get(sid)
    if runner is None:
        raise HTTPException(404, 'No such session')
    text = await asyncio.to_thread(runner.session.csv_text)
    name = 'plant_operator_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    return Response(text, media_type='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})


@app.websocket('/api/sessions/{sid}/ws')
async def session_ws(ws: WebSocket, sid: str):
    runner = registry.get(sid)
    await ws.accept()   # accept first: closing before accept is an HTTP 403 the browser cannot read
    if runner is None:
        await ws.close(code=4404)
        return
    q = asyncio.Queue(maxsize=200)
    runner.subs.add(q)
    runner.last_active = time.monotonic()

    async def pump():
        while True:
            await ws.send_json(await q.get())

    pump_task = asyncio.create_task(pump())
    try:
        q.put_nowait(await asyncio.to_thread(runner.session.snapshot))
        while True:
            msg = await ws.receive_json()
            try:
                await runner.handle(msg, q)
            except Exception as e:   # bad command: report it, keep the connection
                runner.publish(dict(type='error', message=f'{type(e).__name__}: {e}'), only=q)
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        runner.subs.discard(q)
        runner.last_active = time.monotonic()
        if not runner.subs:
            runner.session.pause()   # nobody watching: stop burning CPU


class FrontendFiles(StaticFiles):
    """The built React app. index.html must never be served from a browser's cache, or a rebuilt
    app keeps loading the old (hash-named) JavaScript; the hashed files under /assets are
    immutable and can be cached."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if not path.replace('\\', '/').startswith('assets/'):   # (Windows paths use backslashes)
            response.headers['Cache-Control'] = 'no-cache'
        return response


if FRONTEND_DIST.is_dir():   # the built React app (npm run build), served by the same process
    app.mount('/', FrontendFiles(directory=FRONTEND_DIST, html=True), name='frontend')
