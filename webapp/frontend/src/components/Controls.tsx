// Mode selector, sliders, economics controls and the run/reset/CSV buttons
// (OperatorApp's mode RadioButtons, ten Sliders + value boxes, _build_economics_controls, Buttons).
import { useEffect, useState } from 'react'
import { useStore } from '../store'
import type { SliderCfg } from '../types'

function SliderRow({ cfg }: { cfg: SliderCfg }) {
  const st = useStore((s) => s.state?.sliders[cfg.key])
  const setSlider = useStore((s) => s.setSlider)
  const [text, setText] = useState<string | null>(null)   // non-null while the number box is being edited
  if (!st) return null

  const shown = st.value.toFixed(cfg.decimals)
  const commit = (raw: string) => {
    setText(null)
    const v = Number(raw)
    if (!Number.isFinite(v) || raw.trim() === '') return
    let c = Math.min(Math.max(v, cfg.min), cfg.max)
    if (cfg.whole) c = Math.round(c)
    setSlider(cfg.key, c)
  }
  const step = cfg.whole ? 1 : (cfg.max - cfg.min) / 1000

  return (
    <div className={`slider-row${st.active ? '' : ' inactive'}`}>
      <label htmlFor={`s-${cfg.key}`}>{cfg.label}</label>
      <div className="slider-track">
        <input id={`s-${cfg.key}`} type="range" min={cfg.min} max={cfg.max} step={step} value={st.value}
          disabled={!st.active} onChange={(e) => setSlider(cfg.key, Number(e.target.value))} />
        <div className="range-labels"><span>{cfg.min.toFixed(cfg.decimals)}</span><span>{cfg.max.toFixed(cfg.decimals)}</span></div>
      </div>
      <input className="value-box" type="text" inputMode="decimal" disabled={!st.active}
        aria-label={`${cfg.label} value`}
        value={text ?? shown}
        onFocus={(e) => { setText(shown); e.currentTarget.select() }}
        onChange={(e) => setText(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') { setText(null); e.currentTarget.blur() } }} />
    </div>
  )
}

export function SliderPanel() {
  const cfg = useStore((s) => s.config)
  const sliders = useStore((s) => s.state?.sliders)
  if (!cfg || !sliders) return null
  const rows = cfg.sliders.filter((c) => sliders[c.key]?.visible).sort((a, b) => a.slot - b.slot)
  return <div className="sliders">{rows.map((c) => <SliderRow key={c.key} cfg={c} />)}</div>
}

export function ModeSelector() {
  const modes = useStore((s) => s.config?.modes)
  const mode = useStore((s) => s.state?.mode)
  const send = useStore((s) => s.send)
  if (!modes) return null
  return (
    <fieldset className="radio-group">
      <legend>Regulation mode</legend>
      {modes.map((m) => (
        <label key={m.id}>
          <input type="radio" name="mode" checked={mode === m.id} onChange={() => send({ type: 'mode', mode: m.id })} />
          {m.label}
        </label>
      ))}
    </fieldset>
  )
}

export function EconomicsControls() {
  const cfg = useStore((s) => s.config)
  const st = useStore((s) => s.state)
  const send = useStore((s) => s.send)
  const [mean, setMean] = useState<string | null>(null)
  useEffect(() => { setMean(null) }, [st?.price_mean])
  if (!cfg || !st) return null
  const commitMean = (raw: string) => {
    const v = Number(raw)
    if (Number.isFinite(v) && v >= 0 && raw.trim() !== '') send({ type: 'price_mean', value: v })
    setMean(null)
  }
  return (
    <div className="econ">
      <fieldset className="radio-group">
        <legend>Electricity price</legend>
        {cfg.price_modes.map((m) => (
          <label key={m.id}>
            <input type="radio" name="price" checked={st.price_mode === m.id} onChange={() => send({ type: 'price_mode', mode: m.id })} />
            {m.label}
          </label>
        ))}
        <label className="inline">mean NZ$/MWh
          <input className="value-box" type="text" inputMode="decimal" value={mean ?? String(st.price_mean)}
            onChange={(e) => setMean(e.target.value)} onBlur={(e) => commitMean(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }} />
        </label>
      </fieldset>
      <fieldset className="radio-group">
        <legend>Site demand</legend>
        {cfg.demand_modes.map((m) => (
          <label key={m.id}>
            <input type="radio" name="demand" checked={st.demand_on === (m.id === 'schedule')}
              onChange={() => send({ type: 'demand_mode', on: m.id === 'schedule' })} />
            {m.label}
          </label>
        ))}
      </fieldset>
      <fieldset className="radio-group">
        <legend>Biomass NZ$/GJ</legend>
        <div className="row">
          {cfg.biomass_options.map((v) => (
            <label key={v}>
              <input type="radio" name="biomass" checked={st.biomass_price === v} onChange={() => send({ type: 'biomass', value: v })} />
              {v}
            </label>
          ))}
        </div>
      </fieldset>
    </div>
  )
}

export function Toolbar() {
  const st = useStore((s) => s.state)
  const sid = useStore((s) => s.sid)
  const conn = useStore((s) => s.conn)
  const send = useStore((s) => s.send)
  const disabled = !st || conn !== 'open'
  return (
    <div className="toolbar">
      <button className="primary" disabled={disabled || st?.halted}
        onClick={() => send({ type: st?.running ? 'pause' : 'run' })}>
        {st?.running ? 'Pause' : 'Run'}
      </button>
      <button disabled={disabled} onClick={() => send({ type: 'reset' })}>Reset</button>
      <a className={`button${sid ? '' : ' disabled'}`} href={sid ? `/api/sessions/${sid}/csv` : undefined} download>Save CSV</a>
    </div>
  )
}
