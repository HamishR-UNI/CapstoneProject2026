import { useEffect } from 'react'
import { useStore } from './store'
import StatusBar from './components/StatusBar'
import TrendPanel from './components/TrendPanel'
import Schematic from './components/Schematic'
import { EconomicsControls, ModeSelector, SliderPanel, Toolbar } from './components/Controls'

export default function App() {
  const init = useStore((s) => s.init)
  const config = useStore((s) => s.config)
  useEffect(() => { void init() }, [init])

  return (
    <div className="app">
      <StatusBar />
      <main className="layout">
        <section className="trends" aria-label="Trends">
          {config?.panels.map((p, i, all) => (
            <TrendPanel key={p.id} panel={p} showX={i === all.length - 1} />
          ))}
        </section>
        <section className="side">
          <div className="schematic-wrap"><Schematic /></div>
          <div className="controls">
            <div className="controls-left">
              <SliderPanel />
              <Toolbar />
            </div>
            <div className="controls-right">
              <ModeSelector />
              <EconomicsControls />
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
