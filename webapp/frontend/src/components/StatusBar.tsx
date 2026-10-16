import { useStore } from '../store'
import { SMOOTHING_LEVELS } from '../lib/trend'

const CONN_TEXT: Record<string, string> = {
  connecting: 'Starting the plant simulation. This can take a minute or two the first time…',
  closed: 'Connection lost. Reconnecting…',
  error: 'Cannot reach the simulation server.',
}

export default function StatusBar() {
  const st = useStore((s) => s.state)
  const conn = useStore((s) => s.conn)
  const notice = useStore((s) => s.notice)
  const dismiss = useStore((s) => s.dismissNotice)
  const init = useStore((s) => s.init)
  const smoothing = useStore((s) => s.smoothing)
  const setSmoothing = useStore((s) => s.setSmoothing)
  return (
    <header className="status">
      <div className="status-top">
        <h1>Site supply: choose a regulation mode, HP2 duty and site demands</h1>
        <div className="segmented" role="radiogroup" aria-label="Trend line smoothing">
          <span className="segmented-label">Line smoothing</span>
          {SMOOTHING_LEVELS.map((l) => (
            <button key={l.id} role="radio" aria-checked={smoothing === l.id}
              className={smoothing === l.id ? 'on' : ''} onClick={() => setSmoothing(l.id)}>
              {l.label}
            </button>
          ))}
        </div>
      </div>
      {st ? (
        <div className={`status-lines ${st.status.ok ? 'ok' : 'bad'}`} role="status">
          <div>{st.status.line1}</div>
          <div>{st.status.line2}</div>
        </div>
      ) : (
        <div className="status-lines">{CONN_TEXT[conn] ?? ''}</div>
      )}
      {conn !== 'open' && st && <div className="banner warn">{CONN_TEXT[conn]}</div>}
      {st?.error && <div className="banner warn">{st.error}</div>}
      {notice && (
        <div className="banner warn">
          {notice}
          <button onClick={dismiss}>Dismiss</button>
          {conn === 'error' && <button onClick={() => void init()}>Retry</button>}
        </div>
      )}
    </header>
  )
}
