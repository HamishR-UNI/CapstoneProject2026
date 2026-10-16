// The web trend panels must choose the same axis ranges as the desktop for the same data.
// Fixture: backend/tests/make_schematic_fixture.py (the real desktop app's axes after a run).
import { describe, expect, it } from 'vitest'
import { xRange, yRange } from './trend'
import fixture from '../components/schematic.fixture.json'

describe.each(fixture.points)('axis ranges match the desktop: $name', (pt) => {
  it('y range (initial range widened to the data, 8% headroom) equals matplotlib\'s', () => {
    expect(pt.axes.length).toBe(11)   // 7 panels, 4 of them with a right-hand axis
    for (const a of pt.axes) {
      const [lo, hi] = yRange(a.ylim0 as [number, number], [[a.dmin, a.dmax]])
      expect(lo, `panel ${a.panel} ${a.side} low`).toBeCloseTo(a.ylim[0], 9)
      expect(hi, `panel ${a.panel} ${a.side} high`).toBeCloseTo(a.ylim[1], 9)
    }
  })
  it('x range (0..2000 s, then a sliding 2000 s window) equals matplotlib\'s', () => {
    for (const a of pt.axes) {
      const [lo, hi] = xRange(a.t_last, 2000)
      expect(lo).toBeCloseTo(a.xlim[0], 9)
      expect(hi).toBeCloseTo(a.xlim[1], 9)
    }
  })
})
