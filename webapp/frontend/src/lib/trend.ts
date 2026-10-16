// Trend-panel maths ported from plant_operator.py: _smooth() and the per-axis autoscale in redraw().

/**
 * Display smoothing applied to EVERY plotted series. The plant's own dynamics are fast next to a
 * 2000 s axis (10 s is about 3 px), and the controllers act once per 10 s solve, so even a
 * continuous response reads as straight runs with sharp corners. A Gaussian kernel (sigma in
 * one-second samples) rounds those corners into smooth curves, which a moving average cannot
 * (it leaves trapezoids). The user picks the strength; 0 = off.
 * Display only: the schematic, status line and CSV keep the raw values.
 */
export const SMOOTHING_LEVELS = [
  { id: 'off', label: 'Off', sigma: 0 },
  { id: 'low', label: 'Low', sigma: 6 },
  { id: 'medium', label: 'Medium', sigma: 14 },
  { id: 'high', label: 'High', sigma: 28 },
] as const
export type SmoothingId = (typeof SMOOTHING_LEVELS)[number]['id']
export const DEFAULT_SMOOTHING: SmoothingId = 'medium'

/**
 * Gaussian-weighted average over a +-3 sigma window, truncated (and renormalised) at the ends of
 * the data so the newest points are not pulled towards zero. Gaps (null) are skipped, and a null
 * stays null.
 */
export function smooth(ys: (number | null)[], sigma: number): (number | null)[] {
  const n = ys.length
  if (sigma <= 0 || n < 3) return ys
  const h = Math.ceil(3 * sigma)
  const w = new Float64Array(h + 1)
  for (let k = 0; k <= h; k++) w[k] = Math.exp(-(k * k) / (2 * sigma * sigma))
  const out: (number | null)[] = new Array(n)
  for (let i = 0; i < n; i++) {
    if (ys[i] == null) { out[i] = null; continue }
    let sum = 0
    let wsum = 0
    const lo = Math.max(0, i - h)
    const hi = Math.min(n - 1, i + h)
    for (let j = lo; j <= hi; j++) {
      const y = ys[j]
      if (y == null) continue
      const wj = w[Math.abs(j - i)]
      sum += wj * y
      wsum += wj
    }
    out[i] = sum / wsum
  }
  return out
}

/**
 * Which sample indices to draw. Smoothed data has no detail finer than ~sigma, so it is thinned to
 * every `stride`-th sample (counting back from the newest, which is always kept). Indices in
 * `mustKeep` (both sides of every setpoint step) are always drawn too, so a step stays a true
 * vertical edge instead of a ramp across one stride. Returned ascending, without duplicates.
 */
export function pickIndices(n: number, stride: number, mustKeep: Iterable<number> = []): number[] {
  if (stride <= 1) return Array.from({ length: n }, (_, i) => i)
  const keep = new Set<number>()
  for (let i = n - 1; i >= 0; i -= stride) keep.add(i)
  for (const i of mustKeep) if (i >= 0 && i < n) keep.add(i)
  return [...keep].sort((a, b) => a - b)
}

/** Indices i and i-1 wherever |col[i] - col[i-1]| exceeds `threshold` (the two ends of a step). */
export function stepIndices(col: (number | null)[], threshold: number): number[] {
  const out: number[] = []
  for (let i = 1; i < col.length; i++) {
    const a = col[i - 1]
    const b = col[i]
    if (a != null && b != null && Math.abs(b - a) > threshold) out.push(i - 1, i)
  }
  return out
}

/**
 * Y range of one axis: starts from the panel's initial range and widens to the visible data, with
 * headroom above that scales with the data's own span (so a trace levelling off at its peak does
 * not sit flush against the top edge). Bottom only widens, as on the desktop.
 */
export function yRange(base: [number, number], data: (number | null)[][]): [number, number] {
  const [baseLo, baseHi] = base
  const span = baseHi - baseLo
  let min = Infinity
  let max = -Infinity
  for (const ys of data) {
    for (const v of ys) {
      if (v == null || !Number.isFinite(v)) continue
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  if (!Number.isFinite(min)) return [baseLo, baseHi + 0.02 * span]
  const topPad = Math.max(0.02 * span, 0.08 * (max - min))
  return [Math.min(baseLo, min - 0.02 * span), Math.max(baseHi + 0.02 * span, max + topPad)]
}

/** X range: opens at 0..window, then slides once the run passes `window` seconds. */
export function xRange(tLast: number, windowS: number): [number, number] {
  return [Math.max(0, tLast - windowS), Math.max(windowS, tLast)]
}

/** '#rrggbb' + alpha -> 'rgba(...)' (uPlot has no per-series opacity). */
export function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`
}
