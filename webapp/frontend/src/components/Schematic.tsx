// The plant block diagram, ported from OperatorApp._build_schematic / _update_schematic.
// Coordinates are the desktop's own drawing units (x -20..235, y 0..182, y up); Y() flips them
// for SVG. Font sizes are in drawing units and were chosen so no label overlaps another label,
// a block title, a block edge or a pipe (checked by measuring the rendered SVG, see
// webapp/README.md "Schematic overlap check"). Tag cards are sized from the real rendered text.
import { useLayoutEffect, useRef, useState } from 'react'
import { useStore } from '../store'

const H = 182
const Y = (y: number) => H - y

const C_PIPE = '#5a5a58'
const COMBUST = { fc: '#c1584a', ec: '#8f3f34' }
const TURBINE = { fc: '#3d7ea6', ec: '#2a5a78' }
const HP = { fc: '#4a8f6b', ec: '#356b4e' }
const HEADER = { fc: '#d9a441', ec: '#a97b2b' }
const SITE = { fc: '#9aa0a6', ec: '#6c7278' }
const DARK_TXT = '#3a2f12'
const TAG_TXT = '#1d3d52'
const WARN_TXT = '#a45a13'

const FS = {
  block: 5.0, blockSm: 4.6, header: 4.0, stream: 3.6, streamSm: 3.3,
  tag: 3.8, tagSm: 3.1, tagDense: 3.0, site: 6.0, turbine: 4.6,
}
const MONO = 'Consolas, "Courier New", monospace'
const LINE = 1.2   // line height, in font sizes

type Ha = 'left' | 'center' | 'right'
type Va = 'top' | 'center' | 'bottom'

const anchor = (ha: Ha) => (ha === 'left' ? 'start' : ha === 'right' ? 'end' : 'middle')

/** Multi-line text placed like matplotlib's text(x, y, ha, va): (x, y) in drawing units. */
function Txt({ x, y, lines, fs, ha = 'center', va = 'bottom', color = C_PIPE, mono = false, textRef }: {
  x: number; y: number; lines: string[]; fs: number; ha?: Ha; va?: Va; color?: string; mono?: boolean
  textRef?: React.Ref<SVGTextElement>
}) {
  const lh = fs * LINE
  const h = lines.length * lh
  const top = va === 'top' ? Y(y) : va === 'bottom' ? Y(y) - h : Y(y) - h / 2
  return (
    <text ref={textRef} x={x} textAnchor={anchor(ha)} fill={color} fontSize={fs}
      fontFamily={mono ? MONO : undefined} style={{ pointerEvents: 'none' }}>
      {lines.map((ln, i) => (
        <tspan key={i} x={x} y={top + (i + 0.5) * lh} dominantBaseline="central">{ln}</tspan>
      ))}
    </text>
  )
}

/** A live value on a light card; the card is sized from the measured text, not an estimate. */
function Tag({ x, y, text, fs, ha = 'center', color = TAG_TXT }: {
  x: number; y: number; text: string; fs: number; ha?: Ha; color?: string
}) {
  const lines = text.split('\n')
  const ref = useRef<SVGTextElement>(null)
  const [w, setW] = useState(Math.max(...lines.map((l) => l.length)) * fs * 0.6)
  useLayoutEffect(() => {
    const b = ref.current?.getBBox()
    if (b && b.width > 0 && Math.abs(b.width - w) > 0.05) setW(b.width)
  })
  const pad = 0.28 * fs
  const bw = w + 2 * pad
  const bh = lines.length * fs * LINE + 2 * pad
  const bx = ha === 'center' ? x - bw / 2 : ha === 'left' ? x - pad : x - bw + pad
  return (
    <g>
      <rect x={bx} y={Y(y) - bh / 2} width={bw} height={bh} rx={pad} fill="#f4f6f8" stroke="#c3ccd3" strokeWidth={0.25} />
      <Txt x={x} y={y} lines={lines} fs={fs} ha={ha} va="center" color={color} mono textRef={ref} />
    </g>
  )
}

function Pipe({ p0, p1, lw = 1.8, head = true }: { p0: [number, number]; p1: [number, number]; lw?: number; head?: boolean }) {
  return (
    <line x1={p0[0]} y1={Y(p0[1])} x2={p1[0]} y2={Y(p1[1])} stroke={C_PIPE} strokeWidth={lw * 0.6}
      markerEnd={head ? 'url(#arrow)' : undefined} />
  )
}

function Block({ x, y, w, h, label, colors = { fc: '#2f6f92', ec: '#1d4d67' }, fs = FS.block, center = false }: {
  x: number; y: number; w: number; h: number; label: string; colors?: { fc: string; ec: string }; fs?: number; center?: boolean
}) {
  return (
    <g>
      <rect x={x} y={Y(y + h)} width={w} height={h} rx={2.2} fill={colors.fc} stroke={colors.ec} strokeWidth={0.5} />
      <Txt x={x + w / 2} y={center ? y + h / 2 : y + h - 1.4} lines={[label]} fs={fs} va={center ? 'center' : 'top'} color="white" />
    </g>
  )
}

function Header({ x, y, w, h, label }: { x: number; y: number; w: number; h: number; label: string }) {
  return (
    <g>
      <rect x={x} y={Y(y + h)} width={w} height={h} rx={1.6} fill={HEADER.fc} stroke={HEADER.ec} strokeWidth={0.5} />
      <Txt x={x + 3.5} y={y + h / 2} lines={[label]} fs={FS.header} ha="left" va="center" color={DARK_TXT} />
    </g>
  )
}

const edgeY = (x: number, [x0, y0]: [number, number], [x1, y1]: [number, number]) => y0 + ((y1 - y0) * (x - x0)) / (x1 - x0)

const T1: [number, number][] = [[100, 147], [100, 160], [150, 167], [150, 141]]
const T2_DX = -4
const T2: [number, number][] = [[100 + T2_DX, 86], [100 + T2_DX, 100], [150 + T2_DX, 108], [150 + T2_DX, 80]]
const T_LABEL_DX = 3
const T_TAP_FRAC = 1 / 3

const f = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '–')
const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : 0)

function Poly({ pts }: { pts: [number, number][] }) {
  return <polygon points={pts.map(([x, y]) => `${x},${Y(y)}`).join(' ')} fill={TURBINE.fc} stroke={TURBINE.ec} strokeWidth={0.5} />
}

export default function Schematic() {
  const last = useStore((s) => s.last)
  const k = useStore((s) => s.config?.schematic_constants)
  if (!last || !k) return <div className="schematic-empty">Waiting for the plant…</div>
  const r = last as Record<string, unknown>

  const xTap = 150 - T_TAP_FRAC * 50
  const t1Lx = 125 + T_LABEL_DX
  const t1TopMid = edgeY(t1Lx, T1[1], T1[2])
  const t1BotMid = edgeY(t1Lx, T1[0], T1[3])
  const t1TopTap = edgeY(xTap, T1[1], T1[2])
  const t1BotTap = edgeY(xTap, T1[0], T1[3])
  const t2Mid = 125 + T2_DX
  const xTap2 = xTap + T2_DX
  const t2TopMid = edgeY(t2Mid, T2[1], T2[2])
  const t2Lx = t2Mid + T_LABEL_DX
  const t2LblTop = edgeY(t2Lx, T2[1], T2[2])
  const t2BotMid = edgeY(t2Lx, T2[0], T2[3])
  const t2BotTap = edgeY(xTap2, T2[0], T2[3])

  const mx = 45, my = 116, mw = 115, mh = 14
  const lx = 45, ly = 56, lw = 115, lh = 14
  const wx = 45, wy = 2, ww = 115, wh = 14
  const HP_PROD_X = 67
  const LETDOWN_X = 157
  const VENT_X = 75
  const HX_CY = 93, HX_WO = 3
  const HX_RET_X = -5

  const settling = Math.abs(num(r.dQ_shortfall)) > 0.02 * Math.max(num(r.Q_cond_pri), 1)
  const mpsShort = num(r.mps_shortfall) > 0.05
  const lpsShort = num(r.lps_shortfall) > 0.05

  return (
    <svg className="schematic" viewBox={`-20 -6 255 ${H + 6}`} role="img" aria-label="Plant schematic" preserveAspectRatio="xMidYMid meet">
      <defs>
        <marker id="arrow" markerUnits="userSpaceOnUse" markerWidth="4.6" markerHeight="4" refX="4.4" refY="2" orient="auto">
          <path d="M0,0 L4.6,2 L0,4 z" fill={C_PIPE} />
        </marker>
      </defs>

      {/* Row A: boiler -> turbine 1 */}
      <Block x={10} y={138} w={50} h={22} label="Biomass boiler" colors={COMBUST} />
      <Tag x={35} y={145.2} fs={FS.tagSm} text={`${f(num(r.duty) / 1e3, 1)} MW\neta ${f(r.eta_b, 3)}`} />
      <Pipe p0={[20, 170]} p1={[20, 160]} />
      <Tag x={20} y={173.6} fs={FS.tagSm} text={`${f(r.mfuel_act, 2)} kg/s`} />
      <Txt x={20} y={177.4} lines={['biomass in']} fs={FS.stream} />
      <Pipe p0={[50, 170]} p1={[50, 160]} />
      <Txt x={50} y={171.4} lines={['water source', `${f(k.T_FW, 0)}°C`]} fs={FS.stream} />

      <Poly pts={T1} />
      <Txt x={t1Lx} y={t1TopMid - 2.6} lines={['Turbine 1']} fs={FS.turbine} va="top" color="white" />
      <Tag x={t1Lx} y={t1BotMid + 6.6} fs={FS.tagSm} text={`${f(r.power1, 0)} kW\ndh ${f(r.dh_act, 1)} kJ/kg`} />
      <Pipe p0={[60, 151]} p1={[99, 151]} />
      <Tag x={80} y={157} fs={FS.tagSm} text={`${f(r.steam, 2)} kg/s\n${f(k.P_HP, 0)} bar  ${f(k.T_HP, 1)} C`} />
      <Pipe p0={[xTap, t1TopTap]} p1={[xTap, 176]} />
      <Txt x={xTap} y={177} lines={['power']} fs={FS.stream} />
      <Tag x={xTap + 3} y={171.5} ha="left" fs={FS.tagSm} text={`${f(r.power1, 0)} kW`} />

      {/* Row B: MPS header */}
      <Header x={mx} y={my} w={mw} h={mh} label="MPS header" />
      <Pipe p0={[xTap, t1BotTap]} p1={[xTap, my + mh]} />
      <Txt x={xTap + 3} y={136.4} lines={['MPS out']} fs={FS.streamSm} ha="left" va="center" />
      <Tag x={xTap - 3} y={136.4} ha="right" fs={FS.tagSm} text={`${f(r.steam, 2)} kg/s`} />
      <Pipe p0={[mx + mw, my + mh / 2]} p1={[190, my + mh / 2]} />
      <Tag x={mx + mw / 2} y={my + mh / 2} fs={FS.tagDense} color={mpsShort ? WARN_TXT : TAG_TXT}
        text={`${f(r.pmps_act, 1)} bar  ${f(r.T_out1, 1)} C\n${f(r.m_mps_site, 2)} kg/s to site\nunserved ${f(r.mps_shortfall, 2)} kg/s`} />
      <Pipe p0={[t2Mid, my]} p1={[t2Mid, t2TopMid]} />
      <Tag x={t2Mid - 3} y={109.8} ha="right" fs={FS.tagSm} text={`frac ${f(r.frac_t2_act, 2)}\n${f(r.m_lps, 2)} kg/s`} />

      {/* Row C: heat exchanger, turbine 2 */}
      <Block x={10} y={82} w={50} h={22} label="Heat exchanger" colors={COMBUST} />
      <Tag x={35} y={89.2} fs={FS.tagSm} text={`${f(num(r.Q_hx) / 1e3, 2)} MW`} />
      <Pipe p0={[25, 138]} p1={[25, 104]} />
      <Txt x={27} y={122} lines={['flue gas']} fs={FS.streamSm} ha="left" va="center" />
      <Tag x={28} y={112} ha="left" fs={FS.tagSm} text={`${f(r.gas, 1)} kg/s\n${f(r.stack, 1)} C`} />
      <Pipe p0={[-13, HX_CY + HX_WO]} p1={[9, HX_CY + HX_WO]} />
      <Txt x={-14.5} y={HX_CY + HX_WO + 1.5} lines={['site recycle', `${f(k.T_W_IN, 0)}°C`]} fs={FS.streamSm} ha="left" va="bottom" />

      <Poly pts={T2} />
      <Txt x={t2Lx} y={t2LblTop - 2.6} lines={['Turbine 2']} fs={FS.turbine} va="top" color="white" />
      <Tag x={t2Lx} y={t2BotMid + 6.8} fs={FS.tagSm} text={`${f(r.power2, 0)} kW\ndh ${f(r.dh_act2, 1)} kJ/kg`} />

      {/* Row D: LPS header */}
      <Header x={lx} y={ly} w={lw} h={lh} label="LPS header" />
      <Pipe p0={[xTap2, t2BotTap]} p1={[xTap2, ly + lh]} />
      {/* caption right of the pipe, value left of it: stacked they do not fit between the header
          and the turbine's sloping bottom edge */}
      <Txt x={xTap2 + 3} y={75.2} lines={['LPS out']} fs={FS.streamSm} ha="left" va="center" />
      <Tag x={xTap2 - 3} y={75.2} ha="right" fs={FS.tagSm} text={`${f(r.m_lps, 2)} kg/s`} />
      <Pipe p0={[lx + lw, ly + lh / 2]} p1={[190, ly + lh / 2]} />
      <Tag x={lx + lw / 2} y={ly + lh / 2} fs={FS.tagDense} color={lpsShort ? WARN_TXT : TAG_TXT}
        text={`${f(r.plps_act, 1)} bar  ${f(r.T_out2, 1)} C\n${f(r.m_lps_total, 2)} kg/s to site\nunserved ${f(r.lps_shortfall, 2)} kg/s`} />

      <Pipe p0={[HP_PROD_X, 44]} p1={[HP_PROD_X, ly]} />
      <Txt x={HP_PROD_X + 3} y={53.5} lines={['HP1 LPS']} fs={FS.streamSm} ha="left" va="center" />
      <Tag x={HP_PROD_X + 3} y={48} ha="left" fs={FS.tagSm} text={`${f(r.m_lps_hp, 2)} kg/s`} />

      <Pipe p0={[LETDOWN_X, my]} p1={[LETDOWN_X, ly + lh]} />
      <Txt x={LETDOWN_X + 3} y={95} lines={['letdown']} fs={FS.streamSm} ha="left" va="center" />
      <Tag x={LETDOWN_X + 3} y={88.5} ha="left" fs={FS.tagSm} text={`${f(r.m_letdown, 2)} kg/s`} />

      <Pipe p0={[VENT_X, ly + lh]} p1={[VENT_X, ly + lh + 11]} />
      <Txt x={VENT_X + 3} y={ly + lh + 3.2} lines={['vent']} fs={FS.streamSm} ha="left" va="center" />
      <Tag x={VENT_X + 3} y={ly + lh + 8.6} ha="left" fs={FS.tagSm} text={`${f(r.m_vent, 2)} kg/s`} />

      {/* Row E: the two heat pumps */}
      <Block x={10} y={22} w={60} h={22} label="Heat pump (primary)" colors={HP} />
      <Tag x={40} y={29.2} fs={FS.tagSm} color={settling ? '#8a6d1f' : TAG_TXT}
        text={`Q ${f(r.Q_cond_pri, 0)} kW  COP ${f(r.COP_pri, 2)}\nW ${f(r.W_hp_pri, 0)} kW`} />
      <Pipe p0={[25, 82]} p1={[25, 44]} />
      <Txt x={27} y={62} lines={['flue gas', `${f(k.T_FG_IN_HP, 0)}°C`]} fs={FS.streamSm} ha="left" va="center" />
      <Pipe p0={[16, 22]} p1={[16, 12]} />
      <Txt x={19} y={15.5} lines={['flue gas out', `${f(k.T_FG_OUT_HP_SPEC, 0)}°C`]} fs={FS.streamSm} ha="left" va="center" />
      <Pipe p0={[36, 54]} p1={[36, 44]} />
      <Txt x={38} y={50} lines={['water source', `${f(k.T_MU, 0)}°C`]} fs={FS.streamSm} ha="left" va="center" />

      <Block x={100} y={22} w={60} h={22} label="Heat pump (secondary)" colors={HP} fs={FS.blockSm} />
      <Tag x={130} y={29.2} fs={FS.tagSm} text={`Q ${f(r.Q_cond_sec, 0)} kW  COP ${f(r.COP_sec, 2)}\nW ${f(r.W_hp_sec, 0)} kW`} />
      <Pipe p0={[100, 33]} p1={[71, 33]} />
      <Txt x={85} y={36.2} lines={['HP2 duty']} fs={FS.streamSm} />
      <Tag x={85} y={29} fs={FS.tagSm} text={`${f(r.Q_cond_sec, 0)} kW`} />
      <Pipe p0={[182, 33]} p1={[161, 33]} lw={1.4} />
      <Txt x={175} y={35.5} lines={['water source', `${f(k.T_SRC_SEC, 0)}°C`]} fs={FS.streamSm} va="bottom" />

      {/* Row F: hot water header */}
      <Header x={wx} y={wy} w={ww} h={wh} label="HW header" />
      <Pipe p0={[HP_PROD_X, 22]} p1={[HP_PROD_X, wy + wh]} />
      <Txt x={HP_PROD_X - 3} y={19} lines={['HW out']} fs={FS.streamSm} ha="right" va="center" />
      <Tag x={HP_PROD_X + 3} y={19} ha="left" fs={FS.tagSm} text={`${f(r.m_hw_hp, 2)} kg/s`} />
      <Pipe p0={[10, HX_CY - HX_WO]} p1={[HX_RET_X, HX_CY - HX_WO]} head={false} />
      <Pipe p0={[HX_RET_X, HX_CY - HX_WO]} p1={[HX_RET_X, wy + wh / 2]} head={false} />
      <Pipe p0={[HX_RET_X, wy + wh / 2]} p1={[wx, wy + wh / 2]} />
      <Txt x={-2.5} y={79.4} lines={['HX hot water']} fs={FS.streamSm} ha="left" va="center" />
      <Tag x={-2} y={71.6} ha="left" fs={FS.tagSm} text={`${f(r.mwater_act, 2)} kg/s\n${f(r.twout, 1)} C`} />
      <Pipe p0={[wx + ww, wy + wh / 2]} p1={[190, wy + wh / 2]} />
      <Tag x={wx + ww / 2} y={wy + wh / 2} fs={FS.tag} text={`${f(r.m_hw_total, 2)} kg/s to site`} />

      {/* Site: one tall block collecting all three headers */}
      <Block x={190} y={2} w={40} h={128} label="Site" colors={SITE} fs={FS.site} center />
    </svg>
  )
}
