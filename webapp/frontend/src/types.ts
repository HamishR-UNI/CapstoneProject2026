// Shapes of the data the FastAPI server sends (see webapp/backend/app.py and ui_config.py).

export type Role = 'primary' | 'secondary' | 'tertiary'

export interface SliderCfg {
  key: string
  label: string
  min: number
  max: number
  whole: boolean
  slot: number
  group: 'always' | 'explicit' | 'pressure'
  decimals: number
}

/** smooth: false for inputs (setpoints, commands, schedules), which are drawn as the true steps. */
export interface SeriesCfg { key: string; label: string; color: string; role: Role; smooth?: boolean }
export interface AxisCfg { label: string; ylim0: [number, number]; series: SeriesCfg[] }
export interface PanelCfg { id: string; left: AxisCfg; right: AxisCfg | null }
export interface ModeCfg { id: string; label: string; hp2_controlled: boolean }

export interface Config {
  modes: ModeCfg[]
  sliders: SliderCfg[]
  demand_keys: string[]
  panels: PanelCfg[]
  colors: Record<string, string>
  window_s: number
  series_keys: string[]
  schematic_keys: string[]
  schematic_constants: Record<string, number>
  biomass_options: number[]
  price_modes: { id: string; label: string }[]
  demand_modes: { id: string; label: string }[]
}

export interface SliderState { value: number; visible: boolean; active: boolean }

export interface ServerState {
  running: boolean
  halted: boolean
  stop_reason: string
  mode: string
  t: number
  n_rows: number
  epoch: number
  sliders: Record<string, SliderState>
  demand_on: boolean
  price_mode: string
  price_mean: number
  biomass_price: number
  status: { line1: string; line2: string; ok: boolean }
  step_ms: number
  error: string | null
}

export type Cols = Record<string, (number | null)[]>
export type LastRow = Record<string, number | boolean | null>

export type ServerMsg =
  | { type: 'snapshot'; start: number; cols: Cols; last: LastRow; state: ServerState }
  | { type: 'rows'; start: number; cols: Cols; last: LastRow; state: ServerState }
  | { type: 'state'; state: ServerState }
  | { type: 'error'; message: string }

export type ClientMsg =
  | { type: 'inputs'; values: Record<string, number> }
  | { type: 'run' | 'pause' | 'reset' | 'resync' }
  | { type: 'mode'; mode: string }
  | { type: 'demand_mode'; on: boolean }
  | { type: 'price_mode'; mode: string }
  | { type: 'price_mean'; value: number }
  | { type: 'biomass'; value: number }
