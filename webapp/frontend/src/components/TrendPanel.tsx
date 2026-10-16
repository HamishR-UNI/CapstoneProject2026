// One trend panel: a uPlot chart with up to two y-axes, line roles (solid / dashed / dotted),
// the desktop's autoscale rules and a legend whose entries carry their axis unit.
import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { useStore } from '../store'
import { SMOOTHING_LEVELS, pickIndices, smooth, stepIndices, withAlpha, xRange, yRange } from '../lib/trend'
import type { AxisCfg, PanelCfg, Role } from '../types'

const ROLE: Record<Role, { width: number; alpha: number; dash?: number[] }> = {
  primary: { width: 1.6, alpha: 1 },
  secondary: { width: 1.3, alpha: 0.6, dash: [6, 4] },
  tertiary: { width: 1.2, alpha: 0.45, dash: [2, 3] },
}
const AXIS_W = 52
const unitOf = (label: string) => (label.includes('[') ? label.slice(label.indexOf('[')) : '')

function Legend({ axis }: { axis: AxisCfg }) {
  const unit = unitOf(axis.label)
  return (
    <>
      {axis.series.map((s) => (
        <span className="legend-item" key={s.key}>
          <svg width="22" height="8" aria-hidden>
            <line x1="0" y1="4" x2="22" y2="4" stroke={withAlpha(s.color, ROLE[s.role].alpha)}
              strokeWidth={ROLE[s.role].width + 0.4} strokeDasharray={ROLE[s.role].dash?.join(' ')} />
          </svg>
          {`${s.label} ${unit}`.trim()}
        </span>
      ))}
    </>
  )
}

export default function TrendPanel({ panel, showX }: { panel: PanelCfg; showX: boolean }) {
  const host = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const ranges = useRef<{ x: [number, number]; y: [number, number]; y2: [number, number] }>({
    x: [0, 2000], y: panel.left.ylim0, y2: panel.right?.ylim0 ?? [0, 1],
  })
  const cfg = useStore((s) => s.config)
  const version = useStore((s) => s.version)
  const smoothing = useStore((s) => s.smoothing)

  const all = [...panel.left.series, ...(panel.right?.series ?? [])]

  // Build the chart once per panel.
  useEffect(() => {
    const el = host.current
    if (!el) return
    const css = getComputedStyle(document.documentElement)
    const text = css.getPropertyValue('--muted').trim() || '#666'
    const grid = css.getPropertyValue('--grid').trim() || '#00000018'
    const frame = css.getPropertyValue('--plot-frame').trim() || '#aab6bf'
    const axisBase = { stroke: text, font: '10px system-ui, sans-serif', size: AXIS_W, ticks: { show: false } }
    const opts: uPlot.Options = {
      hooks: {
        // a thin frame around the plotting area, drawn over the grid and under nothing else
        draw: [(u) => {
          const { ctx } = u
          const { left, top, width, height } = u.bbox
          const px = Math.max(1, Math.round(devicePixelRatio))
          ctx.save()
          ctx.setLineDash([])   // the last dashed series would otherwise leave its dash on the frame
          ctx.strokeStyle = frame
          ctx.lineWidth = px
          ctx.strokeRect(left + px / 2, top + px / 2, width - px, height - px)
          ctx.restore()
        }],
      },
      width: el.clientWidth || 400,
      height: el.clientHeight || 120,
      legend: { show: false },
      cursor: { drag: { x: false, y: false }, points: { show: false } },
      padding: [8, panel.right ? 0 : AXIS_W, showX ? 0 : 8, 0],   // room for the first/last tick labels
      scales: {
        x: { time: false, range: () => ranges.current.x },
        y: { range: () => ranges.current.y },
        y2: { range: () => ranges.current.y2 },
      },
      axes: [
        { ...axisBase, show: showX, size: 22, grid: { stroke: grid, width: 1 } },
        { ...axisBase, scale: 'y', grid: { stroke: grid, width: 1 } },
        ...(panel.right ? [{ ...axisBase, scale: 'y2', side: 1, grid: { show: false } }] : []),
      ],
      series: [
        {},
        ...all.map((s, i) => ({
          label: s.label,
          scale: i < panel.left.series.length ? 'y' : 'y2',
          stroke: withAlpha(s.color, ROLE[s.role].alpha),
          width: ROLE[s.role].width,
          dash: ROLE[s.role].dash,
          points: { show: false },
          pxAlign: 0,   // do not snap path points to whole pixels: that serrates sloped lines
        })),
      ],
    }
    const u = new uPlot(opts, [[0], ...all.map(() => [null])] as uPlot.AlignedData, el)
    plot.current = u
    const ro = new ResizeObserver(() => {
      if (el.clientWidth > 0 && el.clientHeight > 0) u.setSize({ width: el.clientWidth, height: el.clientHeight })
    })
    ro.observe(el)
    return () => {
      ro.disconnect()
      u.destroy()
      plot.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panel.id, showX])

  // Feed it data on every new batch.
  useEffect(() => {
    const u = plot.current
    const cols = useStore.getState().cols
    if (!u || !cfg || !cols.t || cols.t.length === 0) return
    const t = cols.t as number[]
    const sigma = SMOOTHING_LEVELS.find((l) => l.id === smoothing)?.sigma ?? 0
    const nLeft = panel.left.series.length
    const raw = all.map((s) => cols[s.key] ?? t.map(() => null))
    // Inputs (setpoints, commands, schedules) are drawn as they are; measured responses smoothed.
    const full = raw.map((col, k) => (all[k].smooth === false ? col : smooth(col, sigma)))
    // Smoothed data has no detail finer than ~sigma, so thin it to at most about one point per
    // two screen pixels. With more points than pixels uPlot collapses each pixel column to a
    // min/max stroke, which looks serrated on a steep section. Both ends of every setpoint step
    // are kept, so a step is still an instant vertical edge.
    const width = host.current?.clientWidth ?? 600
    const stride = sigma > 0 ? Math.max(Math.ceil(sigma / 2), Math.ceil(t.length / Math.max(width / 2, 50))) : 1
    const steps: number[] = []
    if (stride > 1) {
      all.forEach((s, k) => {
        if (s.smooth !== false) return
        const [lo, hi] = (k < nLeft ? panel.left : panel.right!).ylim0
        steps.push(...stepIndices(raw[k], 0.02 * (hi - lo)))
      })
    }
    const idx = pickIndices(t.length, stride, steps)
    const xs = stride === 1 ? t : idx.map((i) => t[i])
    const ys = stride === 1 ? full : full.map((col) => idx.map((i) => col[i]))
    ranges.current = {
      x: xRange(t[t.length - 1], cfg.window_s),
      y: yRange(panel.left.ylim0, ys.slice(0, nLeft)),
      y2: panel.right ? yRange(panel.right.ylim0, ys.slice(nLeft)) : [0, 1],
    }
    u.setData([xs, ...ys] as uPlot.AlignedData)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, cfg, panel.id, smoothing])

  return (
    <div className="panel">
      <div className="panel-titles">
        <span>{panel.left.label}</span>
        {panel.right && <span>{panel.right.label}</span>}
      </div>
      <div className="panel-plot" ref={host} />
      <div className="panel-legend">
        <Legend axis={panel.left} />
        {panel.right && <Legend axis={panel.right} />}
      </div>
    </div>
  )
}
