// The React schematic must show exactly the texts (and warning colours) the desktop schematic shows
// for the same plant state. Fixture: backend/tests/make_schematic_fixture.py (real desktop app).
import { describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import fixture from './schematic.fixture.json'

// Schematic only reads `last` and the schematic constants from the store; give it fixed state.
const state = vi.hoisted(() => ({ last: null as unknown, config: null as unknown }))
vi.mock('../store', () => ({
  useStore: (selector: (s: typeof state) => unknown) => selector(state),
}))
import Schematic from './Schematic'

const unescape = (s: string) =>
  s.replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')

/** Every <text> in the markup: its lines joined by \n, its fill, and whether it is a value tag. */
function textsOf(markup: string) {
  const out: { text: string; fill: string; tag: boolean }[] = []
  for (const m of markup.matchAll(/<text\b([^>]*)>(.*?)<\/text>/gs)) {
    const lines = [...m[2].matchAll(/<tspan\b[^>]*>(.*?)<\/tspan>/gs)].map((t) => unescape(t[1]))
    out.push({
      text: lines.join('\n'),
      fill: /fill="([^"]*)"/.exec(m[1])?.[1] ?? '',
      tag: /font-family/.test(m[1]),
    })
  }
  return out
}

const sorted = (xs: string[]) => [...xs].sort()

describe.each(fixture.points)('schematic matches the desktop: $name', (pt) => {
  state.last = pt.last
  state.config = { schematic_constants: fixture.constants }
  const found = textsOf(renderToStaticMarkup(<Schematic />))

  it('has the same number of labels as the desktop diagram', () => {
    expect(found.length).toBe(pt.texts.length)
  })

  it("shows exactly the desktop's texts (captions, block titles and live values)", () => {
    // desktop: every non-empty text on the schematic axes, compared as a multiset
    expect(sorted(found.map((f) => f.text))).toEqual(sorted(pt.texts))
  })

  it('colours the value tags like the desktop (normal, warning, settling)', () => {
    const web = new Map(found.filter((f) => f.tag).map((f) => [f.text, f.fill]))
    for (const [name, tag] of Object.entries(pt.tags)) {
      expect(web.has(tag.text), `${name}: ${JSON.stringify(tag.text)} missing`).toBe(true)
      expect(web.get(tag.text), `${name} colour for ${JSON.stringify(tag.text)}`).toBe(tag.color)
    }
  })
})
