import { describe, expect, it } from 'vitest'
import { pickIndices, smooth, stepIndices, xRange, yRange, withAlpha } from './trend'

describe('smooth (Gaussian)', () => {
  it('is the identity when off, and for short series', () => {
    expect(smooth([1, 2, 3, 4], 0)).toEqual([1, 2, 3, 4])
    expect(smooth([1, 2], 10)).toEqual([1, 2])
  })
  it('keeps a constant constant, including at the ends (renormalised)', () => {
    const s = smooth(Array(100).fill(7), 14) as number[]
    expect(Math.max(...s.map((v) => Math.abs(v - 7)))).toBeLessThan(1e-9)
  })
  it('keeps a straight line straight away from the ends and does not shift it', () => {
    const ys = Array.from({ length: 200 }, (_, i) => 3 * i)
    const s = smooth(ys, 10) as number[]
    expect(s[100]).toBeCloseTo(300, 6)
  })
  it('turns a step into a monotone S-curve centred on the step, within the data range', () => {
    const ys = Array.from({ length: 200 }, (_, i) => (i < 100 ? 0 : 10))
    const s = smooth(ys, 10) as number[]
    for (let i = 1; i < s.length; i++) expect(s[i]).toBeGreaterThanOrEqual(s[i - 1] - 1e-12)
    expect(s[99]).toBeGreaterThan(0)
    expect(s[100]).toBeLessThan(10)
    expect(Math.abs(s[100] - 5)).toBeLessThan(0.5)   // crosses the middle at the step
    expect(s[10]).toBeCloseTo(0, 6)
    expect(s[190]).toBeCloseTo(10, 6)
  })
  it('skips gaps and keeps a null null', () => {
    const s = smooth([1, 1, null, 1, 1, 1], 2)
    expect(s[2]).toBeNull()
    expect(s[3]).toBeCloseTo(1, 9)
  })
})

describe('pickIndices / stepIndices', () => {
  it('keeps everything when the stride is 1', () => {
    expect(pickIndices(5, 1)).toEqual([0, 1, 2, 3, 4])
  })
  it('thins from the newest sample backwards, ascending, newest always kept', () => {
    expect(pickIndices(10, 4)).toEqual([1, 5, 9])
  })
  it('always keeps the two samples either side of a step', () => {
    const col = Array.from({ length: 20 }, (_, i) => (i < 11 ? 40 : 48))
    const steps = stepIndices(col, 1)
    expect(steps).toEqual([10, 11])
    const idx = pickIndices(20, 7, steps)
    expect(idx).toContain(10)
    expect(idx).toContain(11)
    expect(idx).toContain(19)
    expect([...idx].sort((a, b) => a - b)).toEqual(idx)
    expect(new Set(idx).size).toBe(idx.length)
  })
  it('ignores changes below the threshold and gaps', () => {
    expect(stepIndices([1, 1.001, 1.002, null, 5], 0.5)).toEqual([])
  })
})

describe('yRange', () => {
  it('keeps the base range for data inside it, plus top headroom', () => {
    const [lo, hi] = yRange([0, 100], [[10, 20, 30]])
    expect(lo).toBe(0)
    expect(hi).toBeCloseTo(102)
  })
  it('widens past the base range with 8% headroom', () => {
    const [lo, hi] = yRange([0, 10], [[5, 50]])
    expect(lo).toBeCloseTo(0)
    expect(hi).toBeCloseTo(50 + 0.08 * 45)
  })
  it('ignores nulls', () => {
    expect(yRange([0, 10], [[null, null]])).toEqual([0, 10.2])
  })
})

describe('xRange / withAlpha', () => {
  it('slides after the window', () => {
    expect(xRange(500, 2000)).toEqual([0, 2000])
    expect(xRange(3000, 2000)).toEqual([1000, 3000])
  })
  it('converts hex', () => {
    expect(withAlpha('#1f5573', 0.5)).toBe('rgba(31, 85, 115, 0.5)')
  })
})
