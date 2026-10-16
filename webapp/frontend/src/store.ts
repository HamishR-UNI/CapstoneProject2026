// Client state: UI config, the rolling window of series, the latest row, and the server's state.
// The WebSocket lives here too so every message goes through one place.
import { create } from 'zustand'
import type { ClientMsg, Cols, Config, LastRow, ServerMsg, ServerState } from './types'
import { DEFAULT_SMOOTHING, SMOOTHING_LEVELS, type SmoothingId } from './lib/trend'

const SESSION_KEY = 'plant-operator-session'
const SMOOTHING_KEY = 'plant-operator-smoothing'

function loadSmoothing(): SmoothingId {
  try {
    const v = localStorage.getItem(SMOOTHING_KEY)
    if (SMOOTHING_LEVELS.some((l) => l.id === v)) return v as SmoothingId
  } catch { /* storage unavailable: use the default */ }
  return DEFAULT_SMOOTHING
}
//  A value the user just set is kept for this long even if a (stale) server state disagrees.
const LOCAL_HOLD_MS = 1500

type Conn = 'connecting' | 'open' | 'closed' | 'error'

interface Store {
  config: Config | null
  sid: string | null
  conn: Conn
  state: ServerState | null
  cols: Cols
  nRows: number
  last: LastRow | null
  version: number           // bumped on every data change; charts redraw on it
  notice: string | null     // server-reported error, shown in the status bar
  localEdits: Record<string, { value: number; at: number }>
  smoothing: SmoothingId    // trend-line smoothing strength (display only), remembered per browser

  setSmoothing: (id: SmoothingId) => void
  init: () => Promise<void>
  send: (msg: ClientMsg) => void
  setSlider: (key: string, value: number) => void
  dismissNotice: () => void
}

let ws: WebSocket | null = null
let retry = 0
let retryTimer: ReturnType<typeof setTimeout> | null = null
let stopped = false
let initInFlight: Promise<void> | null = null

/** Drop rows older than the rolling window (the desktop only ever plots the visible window). */
export function trimWindow(cols: Cols, windowS: number): Cols {
  const t = cols.t
  if (!t || t.length === 0) return cols
  const tLast = t[t.length - 1] as number
  const lo = Math.max(0, tLast - windowS)
  let i0 = 0
  while (i0 < t.length && (t[i0] as number) < lo) i0++
  if (i0 === 0) return cols
  const out: Cols = {}
  for (const k of Object.keys(cols)) out[k] = cols[k].slice(i0)
  return out
}

export const useStore = create<Store>((set, get) => {
  const applyState = (s: ServerState) => {
    const now = Date.now()
    const edits = get().localEdits
    const kept: Store['localEdits'] = {}
    const sliders = { ...s.sliders }
    for (const [k, e] of Object.entries(edits)) {
      if (now - e.at < LOCAL_HOLD_MS && sliders[k]) {
        sliders[k] = { ...sliders[k], value: e.value }
        kept[k] = e
      }
    }
    return { state: { ...s, sliders }, localEdits: kept }
  }

  const handle = (msg: ServerMsg) => {
    const cfg = get().config
    if (!cfg) return
    switch (msg.type) {
      case 'snapshot':
        set((st) => ({
          ...applyState(msg.state),
          // a reset (new epoch) must not be overridden by slider values edited just before it
          ...(st.state && st.state.epoch !== msg.state.epoch
            ? { state: msg.state, localEdits: {} }
            : {}),
          cols: msg.cols,
          nRows: msg.state.n_rows,
          last: msg.last,
          version: st.version + 1,
        }))
        break
      case 'rows': {
        const { nRows, cols } = get()
        if (msg.start !== nRows) {          // missed or stale batch: ask for a full resync
          get().send({ type: 'resync' })
          break
        }
        const merged: Cols = {}
        for (const k of Object.keys(msg.cols)) merged[k] = (cols[k] ?? []).concat(msg.cols[k])
        set((st) => ({
          ...applyState(msg.state),
          cols: trimWindow(merged, cfg.window_s),
          nRows: nRows + (msg.cols.t?.length ?? 0),
          last: msg.last,
          version: st.version + 1,
        }))
        break
      }
      case 'state':
        set(applyState(msg.state))
        break
      case 'error':
        set({ notice: msg.message })
        break
    }
  }

  const connect = (sid: string) => {
    if (stopped) return
    set({ conn: 'connecting' })
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws = new WebSocket(`${proto}://${location.host}/api/sessions/${sid}/ws`)
    let opened = false
    ws.onopen = () => {
      opened = true
      retry = 0
      set({ conn: 'open' })
    }
    ws.onmessage = (ev) => handle(JSON.parse(ev.data) as ServerMsg)
    ws.onclose = (ev) => {
      set({ conn: 'closed' })
      if (stopped) return
      //  The server no longer has this session (4404), or it never accepted the connection at
      //  all (e.g. the server restarted and rejected an unknown id): start a fresh session.
      if (ev.code === 4404 || (!opened && retry >= 2)) {
        sessionStorage.removeItem(SESSION_KEY)
        retry = 0
        void get().init()
        return
      }
      retry = Math.min(retry + 1, 6)
      retryTimer = setTimeout(() => connect(sid), 500 * 2 ** retry)
    }
    ws.onerror = () => set({ conn: 'error' })
  }

  async function runInit() {
    try {
      let config = get().config
      if (!config) {
        const r = await fetch('/api/config')
        if (!r.ok) throw new Error(`config: HTTP ${r.status}`)
        config = (await r.json()) as Config
        set({ config })
      }
      let sid = sessionStorage.getItem(SESSION_KEY)
      if (!sid) {
        set({ conn: 'connecting', notice: null })
        const r = await fetch('/api/sessions', { method: 'POST' })
        if (!r.ok) {
          const detail = (await r.json().catch(() => ({}))).detail
          throw new Error(detail ?? `could not start a session (HTTP ${r.status})`)
        }
        sid = (await r.json()).session_id as string
        sessionStorage.setItem(SESSION_KEY, sid)
      }
      set({ sid, notice: null })
      stopped = false
      if (retryTimer) clearTimeout(retryTimer)
      ws?.close()
      connect(sid)
    } catch (e) {
      set({ conn: 'error', notice: String(e instanceof Error ? e.message : e) })
    }
  }

  return {
    config: null,
    sid: null,
    conn: 'connecting',
    state: null,
    cols: {},
    nRows: 0,
    last: null,
    version: 0,
    notice: null,
    localEdits: {},
    smoothing: loadSmoothing(),

    setSmoothing: (id) => {
      try { localStorage.setItem(SMOOTHING_KEY, id) } catch { /* not persisted: still applies */ }
      set({ smoothing: id })
    },

    // Idempotent while a start is in flight (React StrictMode runs effects twice in dev, and
    // starting a session takes a minute or two: it must not be started twice).
    init: () => {
      if (!initInFlight) initInFlight = runInit().finally(() => { initInFlight = null })
      return initInFlight
    },

    send: (msg) => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg))
    },

    setSlider: (key, value) => {
      set((st) => {
        const s = st.state
        if (!s || !s.sliders[key]) return {}
        return {
          state: { ...s, sliders: { ...s.sliders, [key]: { ...s.sliders[key], value } } },
          localEdits: { ...st.localEdits, [key]: { value, at: Date.now() } },
        }
      })
      get().send({ type: 'inputs', values: { [key]: value } })
    },

    dismissNotice: () => set({ notice: null }),
  }
})
