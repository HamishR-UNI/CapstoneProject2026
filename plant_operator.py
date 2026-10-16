"""HP2-centred operator interface. Run: python plant_operator.py"""
import bisect,csv,json,math,sys,threading,time,weakref
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.widgets import Slider,Button,RadioButtons,TextBox
from matplotlib.patches import FancyBboxPatch, Polygon
from operator_control import OperatorRig, LPS_FLOW_TUNING
import plant_rig as cfg
import header_closures as headers
import boiler as blr
import flue_gas_water_hx as hx
import heatpump_dyn as hpp
import heatpump_secondary as hps
import economics as econ

#  ECONOMICS SETTINGS (2026-09-27, see economics.py). The two on-screen switches pick Fixed or
#  changing for the electricity price and the site demands; everything else is set here.
PRICE_MODE_DEFAULT = 'fixed'         # 'fixed' or 'daily' at start-up
PRICE_DAY_LENGTH_S = 6*3600.0        # simulated seconds per price day (24 h, 6 h or 2 h)
PRICE_START_HOUR = 6.0               # time of day at t = 0
DEMAND_SCHEDULE_DEFAULT = False      # site demands follow the schedule at start-up
DEMAND_INTERVAL_S = 3600.0           # a new demand level every simulated hour
DEMAND_STEP_FRAC = 0.10              # each move up to +/-10% of the slider range
DEMAND_SEED = 582                    # same seed = same sequence for every controller

#  The live Excel comparison table (2026-09-25) was removed 2026-10-01 (Hamish: too much
#  text in the middle of the dashboard). Its row list is in git history if needed again.


class FastTextBox(TextBox):
    """TextBox whose per-keystroke cursor render blits instead of doing a
    full, ~500ms canvas.draw() -- see OperatorApp._blit for why that number
    is real, not a guess (measured: 243 Text + 205 Patch artists across 35
    axes on this figure).

    Every OTHER widget here (Slider, and TextBox's own set_val) gates its
    redraw behind `self.drawon`, so OperatorApp can just flip that off and
    drive the blit itself. TextBox._rendercursor() is the one exception --
    it calls `fig.canvas.draw()` directly, unconditionally, with no drawon
    check, because it needs a renderer to measure where the cursor glyph
    should sit. Swapping canvas.draw to a no-op for the DURATION of that one
    call keeps matplotlib's own (version-matched, exact) measuring logic
    untouched -- only the wasteful full-figure paint it does afterward is
    skipped, replaced with our own blit of just the text and cursor.
    """
    def __init__(self,*a,app=None,**kw):
        self._app=app
        super().__init__(*a,**kw)

    def _rendercursor(self):
        canvas=self.ax.get_figure(root=True).canvas
        real_draw=canvas.draw
        canvas.draw=lambda *a,**k:None
        try:
            super()._rendercursor()
        finally:
            canvas.draw=real_draw
        if self._app is not None:
            self._app._blit(self.ax.patch,self.text_disp,self.cursor)

    def stop_typing(self):
        #  TextBox._click() calls THIS on every box, on every click ANYWHERE
        #  in the figure (not just clicks on that box, and not only while it
        #  was actually focused) -- and stop_typing() itself draws
        #  unconditionally, same as _rendercursor. With ten boxes on this
        #  figure that's up to ten full-figure draws on a single click on,
        #  say, a completely unrelated slider. Same swap-and-restore trick.
        canvas=self.ax.get_figure(root=True).canvas
        real_draw=canvas.draw
        canvas.draw=lambda *a,**k:None
        try:
            super().stop_typing()
        finally:
            canvas.draw=real_draw
        if self._app is not None:
            self._app._blit(self.ax.patch,self.text_disp,self.cursor)


class OperatorApp:
    #  One colour family per physical quantity, reused everywhere that
    #  quantity appears (pressure panel, flow panel, schematic tag) so the
    #  same colour always means the same thing across the whole dashboard.
    #  Chosen to echo the schematic's own block-colour families (turbine
    #  blue, header amber, heat-pump green, combustion red) but darkened/
    #  saturated for legibility as line plots on a white background, where
    #  the schematic's pastel fills would wash out.
    #  Panels that pair a fluid property with its flow put two quantities on
    #  one plot (twin y-axes), so each family has a DARK shade for the
    #  property (pressure / temperature) and a LIGHT shade for the flow.
    MPS_COLOR  = '#1f5573'   # dark blue: MPS pressure
    LPS_COLOR  = '#8a5a00'   # dark amber: LPS pressure
    HW_COLOR   = '#1f5c3c'   # dark green: HW temperature, HP2 duty
    MPS_FLOW_COLOR = '#6fb6de'   # light blue: MPS flow
    LPS_FLOW_COLOR = '#f0b94a'   # light amber: LPS flow
    HW_FLOW_COLOR  = '#78c796'   # light green: HW flow
    FUEL_COLOR = '#a13d2f'   # matches schematic combustion red
    STEAM_COLOR = '#d98a3d'  # warm orange: boiler steam output, alongside biomass flow
    POWER_COLOR = '#4a3a6b'  # purple: gross / net electrical power
    MARGIN_COLOR = '#2b6f6a' # dark teal: margin NZ$/h
    PRICE_COLOR = '#b5651d'  # brown-orange: electricity price
    #  Heat pump work follows the subcooling command, which is updated once per
    #  10 s solve, so the raw trace is a staircase (steps of up to ~1 MW while the
    #  LPS protection is active). Plotted as an 11-sample centred moving average
    #  (one solve wide); the CSV and the schematic keep the raw values.
    SMOOTH_KEYS = ('W_hp_total','net_export')
    SMOOTH_SAMPLES = 11

    @classmethod
    def _smooth(cls,ys):
        n=len(ys)
        if n<3:return ys
        h=cls.SMOOTH_SAMPLES//2
        csum=[0.0]
        for y in ys:csum.append(csum[-1]+y)
        out=[]
        for i in range(n):
            lo,hi=max(0,i-h),min(n,i+h+1)
            out.append((csum[hi]-csum[lo])/(hi-lo))
        return out
    WINDOW = 2000.0          # s, width of the rolling time axis

    #  Within a panel: the ACTUAL/DELIVERED series is solid and fully opaque
    #  (it is the thing that happened); a TARGET/COMMAND/REQUIRED series is
    #  dashed and faded, so it reads as a reference line the actual trace is
    #  being judged against, not a second, equally-important measurement.
    #  A third, tertiary reference (e.g. the feedforward's own fuel basis)
    #  fades further still and is dotted, so three co-plotted lines never
    #  compete for attention -- the eye lands on solid-and-dark first.
    ROLE_STYLE = dict(
        primary=dict(linestyle='-', alpha=1.0, linewidth=1.6, zorder=3),
        secondary=dict(linestyle='--', alpha=0.6, linewidth=1.3, zorder=2),
        tertiary=dict(linestyle=':', alpha=0.45, linewidth=1.2, zorder=1))

    def __init__(self,batch=None,threaded=False):
        #  batch = samples (seconds) per GEKKO solve; a solve has ~0.8 s fixed cost, so a
        #  larger batch simulates faster per wall second but samples the controllers
        #  less often (control_dt = batch s). Default is the plant's own 10.
        self.threaded=threaded
        prices=econ.Prices(electricity=econ.ElectricityPrice(
            mode=PRICE_MODE_DEFAULT,day_length_s=PRICE_DAY_LENGTH_S,start_hour=PRICE_START_HOUR))
        self.demand=econ.DemandSchedule(interval_s=DEMAND_INTERVAL_S,step_frac=DEMAND_STEP_FRAC,
                                        seed=DEMAND_SEED)
        self.demand_on=DEMAND_SCHEDULE_DEFAULT
        self.rig=OperatorRig(verbose=False,prices=prices,**({} if batch is None else {'batch':batch}))
        self.running=False;self.rows=[dict(self.rig.last)]
        self._last_step_ms=0.0
        self.fig=plt.figure(figsize=(15,10))
        self.fig.suptitle('Site supply: choose a regulation mode, HP2 duty and site demands', y=.995, fontsize=12)
        self.status=self.fig.text(.055,.925,'',fontsize=9)

        #  Left column: the six trend panels stacked in one column, not a
        #  2x3 grid -- each one gets the full column width, which reads
        #  better for a time series than a half-width tile. Right column:
        #  the schematic on top, sliders/mode/buttons stacked directly
        #  under it -- the controls now sit under the diagram they drive,
        #  rather than spanning the full figure width under everything.
        col_l, col_r = .05, .40
        top, bot = .87, .045
        rgap = .014
        n_panels = 7   # 7th (2026-09-27): margin NZ$/h with the electricity price
        row_h = ((top-bot)-(n_panels-1)*rgap)/n_panels
        #  Each row reserves TITLE_H of its own height for the horizontal axis
        #  title(s) above the plot, rather than the axes filling the whole row --
        #  set_title()'s default position is genuinely above the axes (unlike the
        #  old rotated ylabel, which used horizontal margin instead), and with no
        #  room reserved for it, it was bleeding into the panel stacked above
        #  (rgap is only 0.014, nowhere near enough on its own).
        TITLE_H = .026
        #  Axes narrowed to leave a legend margin between the plot and col_r --
        #  NOT the full col_r..right_l gap (0.40 to 0.56): slider labels (the
        #  longest, "MPS pressure target [bar]", worst) grow LEFT from right_l
        #  and reach back to about 0.44, so a legend anchored too far right runs
        #  into them, not into genuinely empty space. Found by rendering and
        #  measuring against the actual longest label, not assumed.
        #  .75 -> .72 (2026-09-25) to make room for the live comparison table, back to
        #  .75 (2026-10-01) now the table is gone.
        PLOT_FRAC = .75
        plot_w = (col_r-col_l)*PLOT_FRAC
        self.axes=[]
        for row in range(n_panels):
            y = top-row*(row_h+rgap)-row_h
            self.axes.append(self.fig.add_axes([col_l, y, plot_w, row_h-TITLE_H]))
        #  The first three panels (MPS, LPS, hot water) each pair a fluid
        #  property with its flow, so they get a twin right-hand axis; the rest
        #  get None. Biomass (index 4) shares one axis with steam flow -- both
        #  kg/s, no twin needed (see the panel definition in redraw()).
        self.axes_r=[ax.twinx() if i in (0,1,2,6) else None for i,ax in enumerate(self.axes)]

        #  right_l sits well clear of col_r: matplotlib's Slider draws its
        #  label OUTSIDE the slider's own axes, growing LEFT from a fixed
        #  axes-relative anchor -- with too little gap here, long labels
        #  ("LPS pressure target [bar]") were bleeding back over the trend
        #  column's right edge.
        right_l, right_r = .56, .965
        #  The schematic gets most of the column's height (its internal
        #  drawing is fixed at 255x182 data units regardless of the box it
        #  is stretched into, so a short box just crowds the same amount of
        #  text into less room -- this is what made the header/heat-pump
        #  tags look oversized before).
        self._build_schematic(x0=right_l, y0=.30, w=right_r-right_l, h=.87-.30)

        #  Ten sliders total. Five are live inputs in every mode (HP2 duty and the
        #  four site-demand/target quantities). The other five are two PAIRS that
        #  share a screen slot: pressure targets (pmps_cmd/plps_cmd) occupy it in
        #  feedforward/feedforward_pid, the explicit MV commands (mfuel_cmd/
        #  frac_t2_cmd) occupy it in no_control -- only one of each pair is ever
        #  visible. dtsub_cmd has no counterpart and is visible in no_control only.
        #  This is the "slider panel swaps, diagram/graphs don't" rule from the
        #  build brief: which controls SHOW is the only thing that changes here.
        self.sliders={}
        specs=[('qsec_cmd','HP2 duty [kW]',cfg.Q_SEC_LO,cfg.Q_SEC_HI,round(cfg.Q_SEC_0),True),
            ('pmps_cmd','MPS pressure target [bar]',cfg.P_MPS_LO,cfg.P_MPS_HI,cfg.P_MPS_0,False),
            ('mfuel_cmd','Biomass [kg/s]',cfg.MFUEL_LO,cfg.MFUEL_HI,self.rig.fuel_base,False),
            ('plps_cmd','LPS pressure target [bar]',cfg.P_LPS_LO,cfg.P_LPS_HI,cfg.P_LPS_0,False),
            ('frac_t2_cmd','Turbine-2 split [-]',cfg.FRAC_T2_LO,cfg.FRAC_T2_HI,self.rig.frac_t2_base,False),
            ('d_mps','MPS demand [kg/s]',cfg.D_MPS_LO,cfg.D_MPS_HI,cfg.D_MPS_0,False),
            ('d_lps','LPS demand [kg/s]',cfg.D_LPS_LO,cfg.D_LPS_HI,cfg.D_LPS_0,False),
            ('hw_min','HW minimum [kg/s]',cfg.HW_MIN_LO,cfg.HW_MIN_HI,cfg.HW_MIN_0,False),
            ('twout_cmd','HW temperature [C]',cfg.TWOUT_LO,cfg.TWOUT_HI,cfg.TWOUT_0,False),
            ('dtsub_cmd','HP1 subcooling [K]',cfg.DT_SUB_LO,cfg.DT_SUB_HI,self.rig.dt_sub_base,False)]
        #  Shared screen slot per pair: pmps_cmd/mfuel_cmd share slot 1, plps_cmd/
        #  frac_t2_cmd share slot 2 -- everything else gets its own slot.
        SLOT={'qsec_cmd':0,'pmps_cmd':1,'mfuel_cmd':1,'plps_cmd':2,'frac_t2_cmd':2,
              'd_mps':3,'d_lps':4,'hw_min':5,'twout_cmd':6,'dtsub_cmd':7}
        slider_top, slider_pitch = .255, .0275
        #  CTRL_W is the combined slider+value footprint -- kept equal to the
        #  old slider-only width (.26) so mode_l/button positions below are
        #  untouched. slider_w (the actual drag track) is narrowed to make
        #  room for an editable value box in the space matplotlib's own
        #  (read-only) valtext used to occupy just past the track's right
        #  edge -- that built-in text is hidden per-slider below so the two
        #  don't double up.
        CTRL_W = .26
        TEXT_GAP, TEXT_W = .006, .048
        slider_w = CTRL_W-TEXT_GAP-TEXT_W
        self.textboxes={}
        self._slider_fmt={}
        self._range_labels={}
        for key,label,lo,hi,value,whole in specs:
            y=slider_top-SLOT[key]*slider_pitch
            valfmt='%1.0f' if whole else '%1.2f'
            ax=self.fig.add_axes([right_l, y, slider_w, .017])
            s=Slider(ax,label,lo,hi,valinit=value,valfmt=valfmt,valstep=1 if whole else None)
            s.valtext.set_visible(False)  # replaced by the editable box below
            #  drawon=False: set_val() (drag OR programmatic) then only updates
            #  the poly/handle geometry and fires on_changed, skipping its own
            #  ~500ms full-figure canvas.draw_idle(). _sync_textbox (wired to
            #  on_changed below) does the actual screen update, via _blit.
            s.drawon=False
            #  Min/max range, small and faint, tucked into the gap just under the
            #  track's own two ends -- horizontal Slider leaves this strip empty
            #  (its label sits left of the axes and valtext right of it, both
            #  vertically centred, not above/below), so there's no other widget
            #  to collide with. Lets a typed value's valid range be read at a
            #  glance without reaching for the report/config to look it up.
            #  Figure coords, not axes-fraction: the gap to the NEXT row is only
            #  ~0.0105 fig-fraction, and a fraction of THIS axes' own tiny .017
            #  height overshot into that next row's track (found by rendering,
            #  not assumed) -- absolute figure coords make the offset exact.
            #  fig.text()'s children are NOT owned by `ax`, so a paired slot's
            #  hidden slider (see _apply_mode_visibility) does not hide these on
            #  its own -- found the same way, as an overlap with the OTHER
            #  member of the pair, which shares this same y. Stored per key and
            #  hidden alongside its slider/box below.
            lo_txt=self.fig.text(right_l,y-.002,valfmt%lo,fontsize=6,color='.5',ha='left',va='top')
            hi_txt=self.fig.text(right_l+slider_w,y-.002,valfmt%hi,fontsize=6,color='.5',ha='right',va='top')
            self._range_labels[key]=(lo_txt,hi_txt)
            self.sliders[key]=s
            self._slider_fmt[key]=valfmt
            tax=self.fig.add_axes([right_l+slider_w+TEXT_GAP, y, TEXT_W, .017])
            tb=FastTextBox(tax,'',initial=valfmt%value,textalignment='center',app=self)
            tb.text_disp.set_fontsize(8)
            #  _motion (hover facecolor swap) is the one TextBox path that
            #  DOES respect drawon -- unlike _rendercursor/stop_typing above,
            #  which needed overriding instead. Off, so passing the mouse
            #  over any of the ten boxes doesn't cost a full draw either;
            #  the color/hovercolor difference is faint enough to not miss.
            tb.drawon=False
            #  ax.patch (the box's own background, blitted alongside its text
            #  in _sync_textbox/_rendercursor/stop_typing so a shorter new
            #  string properly erases a longer old one) is created directly
            #  by Axes.__init__, not through add_patch/add_artist, so it never
            #  gets its .axes backref set -- draw_artist(patch) needs one.
            tb.ax.patch.axes=tb.ax
            s.on_changed(lambda _v,k=key: self._sync_textbox(k))
            tb.on_submit(lambda text,k=key: self._submit_textbox(k,text))
            self.textboxes[key]=tb
        #  Explicit-command sliders (no_control only) shade differently from the
        #  always-live ones once mode is known, in _apply_mode_visibility below.
        self._explicit_keys=('mfuel_cmd','frac_t2_cmd','dtsub_cmd')
        self._pressure_keys=('pmps_cmd','plps_cmd')
        mode_l = right_l+CTRL_W+.03
        #  Plain PID mode removed 2026-09-25 (see operator_control's docstring).
        #  RTO and the two MPC modes added 2026-09-27 (rto.py, mpc.py); the box grew from 3 to 6
        #  entries, so it is taller and starts lower, and the help text moved below it.
        self._mode_labels=[('No control','no_control'),('Feedforward','feedforward'),
                           ('Feedforward+PID','feedforward_pid'),('FF+PID+RTO','feedforward_pid_rto'),
                           ('MPC (tracking)','mpc'),('Economic MPC','economic_mpc')]
        start_idx=[m for _,m in self._mode_labels].index(self.rig.operator_mode)
        self.mode=RadioButtons(self.fig.add_axes([mode_l,.10,right_r-mode_l,.185]),
                               [label for label,_ in self._mode_labels],active=start_idx)
        for lab in self.mode.labels:lab.set_fontsize(9)
        self.mode.on_clicked(self.change_mode)
        #  y=.005, not .035: with no_control's three extra sliders shown, slot 7
        #  (dtsub_cmd, bottom at .255-7*.0275=.0625) started overlapping a button
        #  row anchored at .035+.036=.071. Moved down to clear it.
        BTN_Y = .005
        self.buttons={}
        btn_w=(CTRL_W-3*.012)/4
        for i,(label,callback) in enumerate([('Run / Pause',self.toggle_run),('Reset',self.reset),('Save CSV',self.export),('Exit',self.exit_app)]):
            b=Button(self.fig.add_axes([right_l+i*(btn_w+.012),BTN_Y,btn_w,.036]),label)
            b.on_clicked(callback);self.buttons[label]=b
        self._build_economics_controls()
        self._apply_mode_visibility()
        self.timer=self.fig.canvas.new_timer(interval=100);self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect('close_event',self.close)
        #  A resize changes every pixel position, invalidating the cached
        #  blit background -- drop it so the next interaction falls back to
        #  a real draw (_blit's own None check) and recaptures a current one --
        #  and rescales every font to the new window size.
        self._font_base=weakref.WeakKeyDictionary();self._font_scale=1.
        self.fig.canvas.mpl_connect('resize_event',self._rescale_text)
        self.redraw();self.timer.start()

    #  Every font size in this file is chosen for the figsize=(15,10) figure. Text is sized
    #  in points, so without this it stays the same size while the window shrinks or grows
    #  around it (2026-10-01, Hamish: text should resize with the window).
    FIG_BASE_IN=(15.,10.)

    def _rescale_text(self,_evt=None):
        """Scale all figure text by the window size relative to FIG_BASE_IN (the smaller
        of the width and height ratios, so text never outgrows the layout)."""
        self._bg=None
        w,h=self.fig.get_size_inches()
        s=min(w/self.FIG_BASE_IN[0],h/self.FIG_BASE_IN[1])
        if abs(s-self._font_scale)<1e-3:return
        for t in self.fig.findobj(Text):
            #  First sight of an artist records its size at the current scale, so the base
            #  is right even for text created after an earlier resize.
            if t not in self._font_base:self._font_base[t]=t.get_fontsize()/self._font_scale
            t.set_fontsize(self._font_base[t]*s)
        #  Tick labels are recreated as the axes limits move, so they are scaled through
        #  tick_params (which applies to future ticks too) rather than per artist.
        tick_pt=FontProperties(size=plt.rcParams['xtick.labelsize']).get_size_in_points()
        for ax in self.fig.axes:ax.tick_params(labelsize=tick_pt*s)
        self._font_scale=s

    #  ------------------------------------------------------------------
    #  Economics controls (2026-09-27): two switches -- electricity price Fixed / Daily loop,
    #  site demand Fixed / Schedule -- and editable electricity (mean) and biomass prices. They
    #  sit in the free band under the live comparison table, left of the slider labels (found by
    #  rendering: the table ends near y=.42 and the longest slider label starts near x=.44 at
    #  y=.27). The live price, margin and cumulative NZ$ go on the status line.
    #  ------------------------------------------------------------------
    _DEMAND_KEYS=econ.DEMAND_DRIVERS   # sliders the demand schedule drives

    def _build_economics_controls(self):
        x0,w_r=.42,.075
        self.fig.text(x0,.392,'Electricity price',fontsize=8,weight='bold',color='#1d3d52')
        pe=self.rig.prices.electricity
        self.price_radio=RadioButtons(self.fig.add_axes([x0,.343,w_r,.045],frameon=False),
                                      ('Fixed','Daily loop'),active=0 if pe.mode=='fixed' else 1)
        self.fig.text(x0,.328,'Site demand',fontsize=8,weight='bold',color='#1d3d52')
        self.demand_radio=RadioButtons(self.fig.add_axes([x0,.279,w_r,.045],frameon=False),
                                       ('Fixed','Schedule'),active=1 if self.demand_on else 0)
        for rb in (self.price_radio,self.demand_radio):
            for lab in rb.labels:lab.set_fontsize(8)
        self.price_radio.on_clicked(self._change_price_mode)
        self.demand_radio.on_clicked(self._change_demand_mode)
        bx=x0+w_r+.005
        self.fig.text(bx,.377,'mean NZ$/MWh',fontsize=6.5,color='.35')
        self.price_box=FastTextBox(self.fig.add_axes([bx,.354,.045,.02]),'',
                                   initial=f'{pe.mean:g}',textalignment='center',app=self)
        #  Biomass price: one of econ.BIOMASS_PRICE_OPTIONS (12/15/18, 2026-09-28), a row of
        #  radio buttons in the slot the free-text box had. Kept left of x=.558: the schematic
        #  starts at x=.56 from y=.30 up.
        self.fig.text(bx,.313,'biomass NZ$/GJ',fontsize=6.5,color='.35')
        opts=[f'{v:g}' for v in econ.BIOMASS_PRICE_OPTIONS]
        cur=min(range(len(opts)),key=lambda i:abs(econ.BIOMASS_PRICE_OPTIONS[i]-self.rig.prices.biomass_per_gj))
        #  Starts left of bx, in the gap after the Site demand labels (they end near x=.475),
        #  so all three options fit before x=.558.
        self.biomass_radio=RadioButtons(self.fig.add_axes([.478,.284,.08,.024],frameon=False),
                                        opts,active=cur,layout='horizontal')
        for lab in self.biomass_radio.labels:lab.set_fontsize(7.5)
        self.biomass_radio.on_clicked(self._change_biomass)
        self.price_box.text_disp.set_fontsize(8);self.price_box.drawon=False
        self.price_box.ax.patch.axes=self.price_box.ax
        self.price_box.on_submit(lambda s:self._submit_price('electricity',s))
        self._apply_demand_mode()

    @staticmethod
    def _nzd(v):
        return f"{'-' if v<0 else ''}NZ${abs(v):,.0f}"

    def _change_price_mode(self,label):
        self.rig.prices.electricity.mode='fixed' if label=='Fixed' else 'daily'
        self.redraw()

    def _submit_price(self,which,text):
        try:v=float(text)
        except ValueError:v=None
        if v is not None and v>=0:self.rig.prices.electricity.mean=v
        self.price_box.text_disp.set_text(f'{self.rig.prices.electricity.mean:g}')
        self.redraw()

    def _change_biomass(self,label):
        self.rig.prices.biomass_per_gj=float(label)
        self.redraw()

    def _change_demand_mode(self,label):
        self.demand_on=label=='Schedule'
        self._apply_demand_mode()
        self.redraw()

    def _apply_demand_mode(self):
        """Schedule on: the three demand sliders show the scheduled values and are greyed out
        (not draggable). Schedule off: they are ordinary sliders again, starting from wherever
        the schedule left them."""
        for key in self._DEMAND_KEYS:
            s=self.sliders[key];tb=self.textboxes[key]
            s.set_active(not self.demand_on);tb.set_active(not self.demand_on)
            s.poly.set_alpha(.35 if self.demand_on else 1.)
        if self.demand_on:self._apply_demand_schedule()

    def _apply_demand_schedule(self):
        """Move the demand sliders to the schedule's level at the plant's current time."""
        for key,v in self.demand.levels(self.rig.t).items():
            if self.sliders[key].val!=v:self.sliders[key].set_val(v)

    #  The per-mode help text under the mode selector was removed 2026-10-01 (Hamish);
    #  docs/PLAN_A_regulation_models.md / CONTROL_STRATEGIES.md describe each mode.
    #  Modes in which a controller, not the HP2 slider, sets HP2 duty (slider greyed, shows it).
    _HP2_CONTROLLED=('feedforward_pid_rto','mpc','economic_mpc')

    #  ------------------------------------------------------------------
    #  Blitting: dragging a slider or typing a value used to trigger
    #  matplotlib's default full-figure canvas.draw_idle() on every single
    #  pixel of motion / every keystroke -- measured at ~500-550ms per call
    #  on this figure (243 Text + 205 Patch artists across 35 axes), which
    #  is what made the dashboard feel laggy even while paused, since that
    #  cost has nothing to do with whether the simulation is running.
    #  Sliders/textboxes are wired (drawon=False, FastTextBox) to skip that
    #  and call _blit instead: restore a cached snapshot of everything
    #  static, redraw only the few artists that actually changed on top of
    #  it, and push just that to screen. redraw() (real simulation steps,
    #  mode switches, reset) still does a full draw -- real data actually
    #  changes there -- and re-captures the snapshot afterward via
    #  _capture_bg so the next idle interaction blits against current
    #  content, not a stale frame.
    _bg=None

    def _capture_bg(self):
        self.fig.canvas.draw()
        self._bg=self.fig.canvas.copy_from_bbox(self.fig.bbox)

    #  Capped: a Slider's round handle (matplotlib's own clip_on=False marker)
    #  measurably degrades its own rendering after repeated draw_artist calls
    #  AT DIFFERENT positions -- isolated by testing artist subsets: handle
    #  alone left ~4900 stray pixels after a 200-step smooth drag (diffed
    #  against an authoritative draw() of the identical final state), track/
    #  poly/vline alone left ~300, padding and pixel-snapping the restore
    #  bbox changed neither number. Not chasing further into Agg's marker
    #  rasterizer for a technique that's inherently approximate by design --
    #  bounding the drift with a periodic full recapture is the standard,
    #  reliable answer: a real mouse drag fires far more than _BLIT_CAP
    #  motion events, but each burst of that many is invisible on its own
    #  (~25-step subset of the same test stayed under the single-call
    #  baseline), and the occasional full-draw pause it costs is still
    #  planted between many cheap ones, not on every single one.
    _BLIT_CAP=25
    _blit_n=0

    def _blit(self,*artists):
        if self._bg is None or self._blit_n>=self._BLIT_CAP:
            self._capture_bg();self._blit_n=0;return
        self._blit_n+=1
        #  Scoped to the union of the involved axes' own bboxes (a slider's
        #  tiny row, not the 1500x1000px figure) -- restore_region/blit both
        #  take a sub-bbox of the cached full-figure snapshot. Measured:
        #  dropping this scoping (always passing the whole fig.bbox) still
        #  worked, just slower -- ~18-38ms/call instead of low single digits,
        #  from copying/pushing ~6MB of pixels for a one-slider change.
        #  Padded 15px beyond the axes' own tight bbox and snapped to whole
        #  pixels (ax.bbox's extents are fractional, e.g. y0=172.5) -- kept
        #  even though neither moved the handle-degradation number above,
        #  since both are correct/cheap on their own merits (the padding
        #  does cover the handle's genuine clip_on=False overhang; it just
        #  isn't where THIS artifact comes from).
        axes={a.axes for a in artists if a.axes is not None}
        bbox=mtransforms.Bbox.union([ax.bbox for ax in axes]).padded(15) if axes else self.fig.bbox
        x0,y0,x1,y1=bbox.extents
        bbox=mtransforms.Bbox([[math.floor(x0),math.floor(y0)],[math.ceil(x1),math.ceil(y1)]])
        self.fig.canvas.restore_region(self._bg,bbox=bbox)
        for a in artists:
            a.axes.draw_artist(a)
        self.fig.canvas.blit(bbox)

    def _sync_textbox(self,key):
        """Keep a slider's value box matching its current position -- fires
        on every set_val (drag, typed submit, or a programmatic reset/mode
        switch elsewhere), so the box never goes stale regardless of what
        moved the slider. Writes the Text artist directly rather than going
        through TextBox.set_val, which itself re-fires 'submit' on any
        change and would loop back into _submit_textbox for no reason.

        Blits track+poly+vline+handle (matplotlib's own creation order --
        matters, since draw_artist has no z-sorting of its own and just
        paints in the order given), not just poly+handle: poly only covers
        the FILLED portion, so on a decreasing drag the region it no longer
        covers would otherwise keep showing the stale (wider) fill baked
        into the cached background -- track is unchanging but opaque, so
        redrawing it first correctly erases that before poly/vline/handle go
        back on top. vline (the thin initial-value reference mark) is easy
        to assume static-and-skippable since its OWN geometry never moves --
        wrong: redrawing the also-static track over it, without redrawing
        vline too, was silently erasing it on the first slider move, caught
        by diffing a blitted frame against an authoritative full draw()
        pixel-for-pixel, not by eye. Same idea for the textbox's own axes
        background (ax.patch) ahead of its text, clearing any stale longer
        string underneath a shorter one."""
        s=self.sliders[key];tb=self.textboxes[key]
        text=self._slider_fmt[key]%s.val
        if tb.text!=text:
            tb.text_disp.set_text(text)
        self._blit(s.track,s.poly,s.vline,s._handle,tb.ax.patch,tb.text_disp)

    def _submit_textbox(self,key,text):
        """Typed value: parse, clip to the slider's own range (and round to
        its step, for whole-number sliders -- set_val doesn't do this on
        its own), then apply via the slider's set_val so it takes the exact
        same path a drag does. That fires on_changed, which re-syncs the box
        to the clipped/formatted value -- so an out-of-range or fractional
        entry on a whole-number slider visibly snaps back rather than
        silently taking a value the slider itself could never reach."""
        s=self.sliders[key]
        try:v=float(text)
        except ValueError:self._sync_textbox(key);return
        v=min(max(v,s.valmin),s.valmax)
        if s.valstep:v=round(v)
        s.set_val(v)

    def _apply_mode_visibility(self):
        """Which sliders are live is the only thing that changes with mode --
        the diagram and every trend graph stay exactly as they are (see the
        build brief's interface section)."""
        mode=self.rig.operator_mode
        no_control=mode=='no_control'
        #  set_active(False) as well as hiding the axes: paired sliders share a
        #  screen rect (see the specs comment above), so the hidden one of each
        #  pair must also stop taking mouse events or a drag meant for the
        #  visible slider would move both.
        for key in self._explicit_keys:
            self.sliders[key].ax.set_visible(no_control);self.sliders[key].set_active(no_control)
            self.textboxes[key].ax.set_visible(no_control);self.textboxes[key].set_active(no_control)
            for t in self._range_labels[key]:t.set_visible(no_control)
        for key in self._pressure_keys:
            self.sliders[key].ax.set_visible(not no_control);self.sliders[key].set_active(not no_control)
            self.textboxes[key].ax.set_visible(not no_control);self.textboxes[key].set_active(not no_control)
            for t in self._range_labels[key]:t.set_visible(not no_control)
        hp2_ctl=mode in self._HP2_CONTROLLED
        self.sliders['qsec_cmd'].set_active(not hp2_ctl);self.textboxes['qsec_cmd'].set_active(not hp2_ctl)
        self.sliders['qsec_cmd'].poly.set_alpha(.35 if hp2_ctl else 1.)

    def change_mode(self,_=None):
        self._finish_solve()
        if self.rig.halted:return
        mode=dict(self._mode_labels)[self.mode.value_selected]
        was_no_control=self.rig.operator_mode=='no_control'
        self.rig.set_mode(mode)
        if mode=='no_control':
            #  Entering no_control: pressure targets go inert (nothing tracks
            #  them there), so pin them at nominal rather than leave a stale
            #  slider position that silently does nothing.
            self.sliders['pmps_cmd'].reset();self.sliders['plps_cmd'].reset()
        elif was_no_control:
            #  Leaving no_control: sync the explicit sliders to what the plant
            #  is actually doing now, so a later return to no_control doesn't
            #  start from a stale, disconnected command.
            r=self.rig.last
            self.sliders['mfuel_cmd'].set_val(r['mfuel_cmd'])
            self.sliders['frac_t2_cmd'].set_val(r['frac_t2_cmd'])
            self.sliders['dtsub_cmd'].set_val(r['dtsub_cmd'])
        self._apply_mode_visibility()
        self.redraw()

    def exit_app(self,_=None):
        plt.close(self.fig)   # fires close_event -> close(): stops timer, frees the model

    def toggle_run(self,_=None):
        if not self.rig.halted:self.running=not self.running
        self.redraw()

    def reset(self,_=None):
        self.running=False;self._finish_solve();self.rig.reset()
        for key,s in self.sliders.items():s.reset()
        if self.demand_on:self._apply_demand_schedule()   # t = 0: the schedule's base level
        self.rows=[dict(self.rig.last)];self.redraw()

    #  Threaded stepping (the live window): the GEKKO solve is a subprocess wait, so it
    #  runs in a worker while the main thread draws the previous batch. A tick is then
    #  about max(solve, draw) instead of solve + draw. Tests and scripts keep the
    #  synchronous path (threaded=False), where tick() steps and draws before returning.
    threaded=False
    _worker=None

    def _solve(self,u):
        t0=time.perf_counter()
        try:
            self._result=(self.rig.step(**u),None,time.perf_counter()-t0)
        except Exception as e:
            self._result=(None,e,time.perf_counter()-t0)

    def _collect(self):
        rows,err,secs=self._result
        self._worker=None
        if err is not None:
            self.running=False;print('Step failed:',err,flush=True);return
        self.rows.extend(rows)
        self._last_step_ms=secs*1000/self.rig.batch
        if self.rig.operator_mode in self._HP2_CONTROLLED:
            #  show the controller's HP2 duty on the (greyed) slider
            s=self.sliders['qsec_cmd']
            v=min(max(round(rows[-1]['qsec_cmd']),s.valmin),s.valmax)
            if s.val!=v:s.set_val(v)
        if self.rig.halted:self.running=False

    def _finish_solve(self):
        """Wait for an in-flight solve and take its rows (before reset/mode change/close)."""
        w=self._worker
        if w is not None:
            w.join();self._collect()

    def tick(self):
        if self.threaded:
            collected=False
            if self._worker is not None:
                if self._worker.is_alive():return
                self._collect();collected=True
            if self.running and not self.rig.halted:
                if self.demand_on:self._apply_demand_schedule()
                u={k:s.val for k,s in self.sliders.items()}
                self._worker=threading.Thread(target=self._solve,args=(u,),daemon=True)
                self._worker.start()
            elif self.rig.halted:
                self.running=False;collected=True
            if collected:self.redraw()   # drawn while the next solve runs
            return
        if not self.running:return
        if self.rig.halted:self.running=False;self.redraw();return
        if self.demand_on:self._apply_demand_schedule()
        u={k:s.val for k,s in self.sliders.items()}
        self._result=None
        self._solve(u);self._collect()
        self.redraw()

    _trend_lines=None

    def _build_trends(self,panels):
        """Create every trend line and legend once; redraw() only updates their data."""
        self._trend_lines=[]
        for i,(ax,(left,right)) in enumerate(zip(self.axes,panels)):
            entries=[]
            for a,spec,grid,title_loc in ([(ax,left,True,'left')]
                    +([(self.axes_r[i],right,False,'right')] if right is not None else [])):
                label_y,_,series=spec
                #  Legend entries carry their axis's unit (2026-09-25), so on a
                #  dual-axis panel it is clear which side each line is read against.
                unit=label_y[label_y.find('['):] if '[' in label_y else ''
                lines=[a.plot([],[],label=f'{label} {unit}'.strip(),color=color,**self.ROLE_STYLE[role])[0]
                       for _,label,color,role in series]
                #  Axis title aligned along the x-axis (horizontal), not matplotlib's
                #  default 90-degree ylabel -- the built-in subplot-title mechanism
                #  (loc='left'/'right') sits just above the plot instead of rotated
                #  in the left tick-label margin. Two titles share one row on a twin
                #  panel: left-quantity at loc='left', right-quantity at loc='right'.
                a.set_title(label_y, loc=title_loc, fontsize=7.5)
                a.ticklabel_format(axis='y',style='plain',useOffset=False)
                if grid:
                    a.grid(alpha=.2)
                else:
                    a.yaxis.tick_right()
                    a.xaxis.set_visible(False)
                entries.append((a,spec,lines))
            handles=[ln for e in entries for ln in e[2]]
            labels=[ln.get_label() for ln in handles]
            #  Legend outside the plotted trace area, one column (not two),
            #  anchored in FIGURE coordinates (not axes-fraction) so its position
            #  doesn't depend on how wide this panel's axes happen to be: a fixed
            #  gap clear of the axes' own right edge -- and clear of a twin axis's
            #  own tick numbers, which draw in that same margin -- rather than
            #  right against it. The trend column ends at col_r=.40 with nothing
            #  else until the schematic column at .56, so this room already
            #  exists unused; the legend uses some of it instead of crowding the
            #  plot itself.
            pos=ax.get_position()
            #  .02, clear of a twin axis's tick labels (the margin panel's price axis has
            #  3-digit ones). Was .015 on panels 0-5 while the comparison table was there.
            LEGEND_GAP=.02
            ax.legend(handles,labels,fontsize=6.0,ncol=1,frameon=False,
                     loc='center left',bbox_to_anchor=(pos.x1+LEGEND_GAP,(pos.y0+pos.y1)/2),
                     bbox_transform=self.fig.transFigure)
            if i==len(panels)-1:
                ax.set_xlabel('Time [s]')
            else:
                ax.tick_params(labelbottom=False)
            self._trend_lines.append(entries)

    def redraw(self):
        rows=self.rows
        #  LPS produced = turbine 2 + heat pump + letdown (the header's total
        #  inflow). It exceeds what the site takes whenever the surplus is
        #  vented or stored as pressure.
        for v in rows:
            if 'm_lps_produced' not in v:
                v['m_lps_produced']=v['m_lps']+v['m_lps_hp']+v['m_letdown']
                v['net_export']=v['power']-v['W_hp_total']
        t=[r['t'] for r in rows];r=rows[-1]
        #  ylim0 is the panel's INITIAL y-range -- wide enough to show a
        #  meaningful fraction of the physical/slider envelope from the
        #  first sample, rather than autoscaling tight to whatever narrow
        #  band the run happens to occupy so far (which made a flat start
        #  look like noise once the trace finally moved). Pressures use
        #  +/-20% of their valid domain width around the fixed target;
        #  flows/duties use 0 up to a bit past their slider maximum. Every
        #  panel still WIDENS past ylim0 if the actual data goes further,
        #  via the same min(base_lo,data-pad)/max(base_hi,data+pad) pattern
        #  the pressure panels already used.
        mps_pad=.2*(cfg.P_MPS_HI-cfg.P_MPS_LO)
        #  Each panel is (left axis, right axis or None); an axis is
        #  (label, initial y-range, series). Dark = property, light = flow.
        tw_pad=.1*(cfg.TWOUT_HI-cfg.TWOUT_LO)
        panels=[
            (('MPS pressure [bar]', (cfg.P_MPS_0-mps_pad,cfg.P_MPS_0+mps_pad), [
                ('pmps_act','MPS pressure',self.MPS_COLOR,'primary'),
                ('pmps_cmd','MPS setpoint',self.MPS_COLOR,'secondary')]),
             ('MPS flow [kg/s]', (0,cfg.D_MPS_HI*1.05), [
                ('m_mps_delivered','MPS delivered',self.MPS_FLOW_COLOR,'primary'),
                ('d_mps','MPS setpoint',self.MPS_FLOW_COLOR,'secondary')])),
            #  LPS range fixed wide (2026-09-24): from just under the target slider's
            #  minimum to above the relief valve's full-lift point, so the whole relief
            #  band is always in view instead of the trace running into a tight top edge.
            #  Still widens further if LPS goes beyond it.
            (('LPS pressure [bar]', (cfg.P_LPS_LO-.2,headers.DEFAULT.relief_full+.3), [
                ('plps_act','LPS pressure',self.LPS_COLOR,'primary'),
                ('plps_cmd','LPS setpoint',self.LPS_COLOR,'secondary')]),
             ('LPS flow [kg/s]', (0,cfg.D_LPS_HI*1.05), [
                ('m_lps_produced','LPS produced',self.LPS_FLOW_COLOR,'primary'),
                ('d_lps','LPS setpoint',self.LPS_FLOW_COLOR,'secondary')])),
            (('HW temperature [C]', (cfg.TWOUT_LO-tw_pad,cfg.TWOUT_HI+tw_pad), [
                ('twout','HW temperature',self.HW_COLOR,'primary'),
                ('twout_cmd','HW temp setpoint',self.HW_COLOR,'secondary')]),
             ('HW flow [kg/s]', (0,cfg.HW_MIN_HI*.75), [
                ('m_hw_total','HW delivered',self.HW_FLOW_COLOR,'primary'),
                ('hw_min','HW minimum',self.HW_FLOW_COLOR,'secondary')])),
            (('Power [kW]', (0,25000), [
                ('power1','Turbine 1',self.MPS_COLOR,'primary'),
                ('power2','Turbine 2',self.LPS_COLOR,'primary'),
                ('power','Total turbine power',self.POWER_COLOR,'primary'),
                ('W_hp_total','Heat pump work',self.HW_COLOR,'secondary'),
                ('net_export','Net export',self.POWER_COLOR,'secondary')]),None),
            #  Biomass and steam flow share one axis (both kg/s) rather than a dual
            #  axis with a temperature -- Hamish's own call, 2026-09-23: shows the
            #  fuel-to-steam conversion and its lag directly, no second scale to
            #  read. Steam runs several times higher than fuel at any given
            #  operating point, so the initial range covers steam's own span; fuel
            #  reads as the lower band of the same axis rather than filling it.
            (('Biomass / steam [kg/s]', (0,90.), [
                ('mfuel_act','Biomass actual',self.FUEL_COLOR,'primary'),
                ('mfuel_cmd','Biomass command',self.FUEL_COLOR,'secondary'),
                ('steam','Steam out',self.STEAM_COLOR,'primary')]),None),
            (('HP2 duty [kW]', (0,cfg.Q_SEC_HI*1.05), [
                ('Q_cond_sec','Actual',self.HW_COLOR,'primary'),
                ('qsec_cmd','Command',self.HW_COLOR,'secondary')]),None),
            #  Economics (2026-09-27): margin = product revenue + electricity - fuel, with the
            #  electricity export price on the right so the price cycle and the plant's
            #  response are read together.
            (('Margin [NZ$/h]', (0,5000), [
                ('margin_h','Margin',self.MARGIN_COLOR,'primary')]),
             ('Electricity price [NZ$/MWh]', (0,300), [
                ('price_export','Electricity price',self.PRICE_COLOR,'secondary')]))]
        #  Shared time axis: every panel plots the same t, so give them all
        #  the same xlim explicitly (not just matching by coincidence) and
        #  only label/tick the bottom one -- the other five would just be
        #  repeating it.
        #  Rolling window as in plant_manual: the axis opens at 0-WINDOW and
        #  then slides once the run passes WINDOW seconds.
        t_lo,t_hi=max(0.,t[-1]-self.WINDOW),max(self.WINDOW,t[-1])
        #  Performance: the axes, lines and legends are built once and each redraw only
        #  swaps the line data and the limits (clear()+re-plot+legend rebuild of nine
        #  axes cost about 0.6 s per tick), and only the visible window is plotted.
        if self._trend_lines is None:self._build_trends(panels)
        i0=bisect.bisect_left(t,t_lo)
        tv=t[i0:];rows_v=rows[i0:]
        for entries in self._trend_lines:
            for ax,(_,(base_lo,base_hi),series),lines in entries:
                vis=[]
                for (key,*_),ln in zip(series,lines):
                    ys=[v[key] for v in rows_v]
                    if key in self.SMOOTH_KEYS:ys=self._smooth(ys)
                    ln.set_data(tv,ys);vis+=ys
                #  Autoscale only over the visible window so old excursions
                #  don't pin the range after they slide off the left edge.
                span=base_hi-base_lo
                #  Top always carries the same .02*span headroom, not just once a
                #  trace exceeds base_hi -- a line sitting exactly AT the panel's
                #  fixed ceiling (never having "excurted" past it, e.g. Power
                #  approaching its fixed 25000 top) used to touch the box's top
                #  edge with zero gap, since max(base_hi, data+pad) picked base_hi
                #  itself whenever data+pad fell short of it. Bottom is left as-is
                #  (Hamish only flagged the top; several panels floor at a genuine
                #  physical zero, where sitting flush at the bottom edge is fine).
                #  Headroom above the data scales with the data's own visible range too
                #  (8%), not only the panel's small base span: with just .02*span (0.016
                #  bar on LPS) a trace levelling off at its peak sat flush against the top
                #  edge and read as a hard cap -- reported as "a hard stop at 4.5 bar"
                #  when LPS was simply settling where the relief valve matched the surplus.
                top_pad=max(.02*span,.08*(max(vis)-min(vis)))
                lim=(min(base_lo,min(vis)-.02*span),max(base_hi+.02*span,max(vis)+top_pad))
                if lim!=ax.get_ylim():ax.set_ylim(*lim)
                ax.set_xlim(t_lo,t_hi)
        stop=('STOPPED: '+self.rig.stop_reason) if self.rig.halted else ('Running' if self.running else 'Paused')
        #  One two-line status block (2026-10-01, Hamish: replaces the three-line status here
        #  and the monospace block over the schematic). Line 1: is the plant running and is
        #  the site served. Line 2: the money. Values already on the diagram (fuel, vent,
        #  letdown, split) are not repeated; controller internals (fuel driver, PI trims,
        #  ms/step) were dropped.
        if r['service_ok']:
            service='Demand met'
        else:
            service=f"SHORT  MPS {r['mps_shortfall']:.2f} / LPS {r['lps_shortfall']:.2f} / HW {r['hw_short']:.2f} kg/s"
        if r.get('hw_override_latched'):service+='  [HW override armed]'
        mode=self.rig.operator_mode.replace('_',' ').upper()
        pe=self.rig.prices.electricity
        price_txt=(f"NZ${r['price_export']:.0f}/MWh ({pe.clock_text(r['t'])})"
                   if pe.mode=='daily' else f"NZ${r['price_export']:.0f}/MWh")
        extra=''
        if self.demand_on:
            extra+=f" | demand change in {(self.demand.next_change(r['t'])-r['t'])/60:.0f} min"
        if self.rig.operator_mode in ('mpc','economic_mpc') and self.rig.mpc.solve_secs:
            extra+=f" | MPC solve {self.rig.mpc.solve_secs[-1]:.1f} s, {self.rig.mpc_fallbacks} fallbacks"
        elif self.rig.operator_mode=='feedforward_pid_rto' and self.rig.rto_decision:
            extra+=f" | RTO subcooling floor {self.rig.rto_decision['dt_floor']:.1f} K"
        self.status.set_text(f"{stop} | {mode} | t={r['t']:.0f} s | {service}\n"
                             f"Electricity {price_txt} | net export {r['power']-r['W_hp_total']:.0f} kW"
                             f" | margin {self._nzd(r['margin_h'])}/h | since reset {self._nzd(r['margin_cum'])}{extra}")
        self.status.set_color('#a83232' if self.rig.halted or not r['service_ok'] else '#245c35')
        self._update_schematic(r)
        #  Real content changed (line data, ylim, schematic tags, maybe which
        #  sliders/boxes are even visible) -- a full draw is unavoidable here
        #  regardless of the blitting path above, so do it and recapture the
        #  background in one go rather than draw_idle() (deferred, so a
        #  copy_from_bbox() right after would risk grabbing a stale frame).
        self._capture_bg()

    # ------------------------------------------------------------------
    #  Schematic. Ported from the archived plant_manual.py's block diagram
    #  (same plant, same topology), with every live field remapped onto
    #  plant_rig.py's current row fields, and the status line built on fields
    #  that are live and meaningful in every one of the four operator modes.
    #  PlantRig's own fuel_clamped/frac_clamped come from its internal
    #  demand inverse, which OperatorRig never lets run (it always passes an
    #  explicit mfuel_cmd/frac_t2_cmd -- see operator_control.py), so those
    #  two stay stale here; OperatorRig's own hw_binds/dt_clamped (set only
    #  when it calls _invert() itself, i.e. feedforward/feedforward_pid) and
    #  service_ok/mps_shortfall/lps_shortfall/hw_short/hw_override_latched
    #  are correct in all four modes and used instead.
    # ------------------------------------------------------------------

    #  One colour family per category: red for combustion/heat-recovery,
    #  blue for turbines, green for heat pumps, amber for headers, grey for
    #  the site. Equipment blocks get white text (fills are saturated
    #  enough); headers/site get dark text (fills are pale).
    C_BLOCK, C_EDGE, C_PIPE = '#2f6f92', '#1d4d67', '#5a5a58'
    C_COMBUST, C_COMBUST_EDGE = '#c1584a', '#8f3f34'
    C_TURBINE, C_TURBINE_EDGE = '#3d7ea6', '#2a5a78'
    C_HP, C_HP_EDGE = '#4a8f6b', '#356b4e'
    C_HEADER, C_HEADER_EDGE = '#d9a441', '#a97b2b'
    C_SITE, C_SITE_EDGE = '#9aa0a6', '#6c7278'
    C_DARK_TXT = '#3a2f12'

    FS_BLOCK, FS_BLOCK_SM = 9.5, 9.0
    FS_HEADER, FS_STREAM, FS_STREAM_SM = 7.5, 7.0, 6.6
    #  Header/heat-pump tags carry 3-4 lines in a box only 14-22 data units
    #  tall, so they get a smaller size than the single/two-line tags
    #  elsewhere (FS_TAG); FS_TAG_SM already covers most of those.
    FS_TAG, FS_TAG_SM, FS_TAG_DENSE = 7.5, 6.3, 5.6
    FS_NOTE = 8.6

    #  Both turbines made taller 2026-10-01 (Hamish): the name and the kW/dh tag overlapped.
    #  T1's bottom edge also moved up 2-3 units so the 'MPS out' label under it clears it.
    T1_POLY = [(100, 147), (100, 160), (150, 167), (150, 141)]
    #  Turbine 2 sits T2_DX to the left of turbine 1 (2026-09-25), carrying its inlet
    #  pipe, outlet tap and their labels with it -- more room from the letdown line.
    T2_DX = -4.0
    T2_POLY = [(100 + T2_DX, 86), (100 + T2_DX, 100), (150 + T2_DX, 108), (150 + T2_DX, 80)]
    #  Turbine name and tag sit right of centre, where the trapezoid is taller.
    T_LABEL_DX = 3.0
    T_TAP_FRAC = 1.0/3.0

    @staticmethod
    def _fs(artist, base):
        artist.set_fontsize(base)
        return artist

    def _tag(self, x, y, ha='center', va='center', colour='#1d3d52', fontsize=8):
        """A live value label with a light card behind it."""
        t = self.ax_d.text(
            x, y, '', ha=ha, va=va, family='monospace',
            color=colour, zorder=5,
            bbox=dict(boxstyle='round,pad=0.35', fc='#f4f6f8',
                      ec='#c3ccd3', lw=0.7))
        return self._fs(t, fontsize)

    def _stream_text(self, x, y, s, fontsize=None, ha='center', va='bottom',
                     colour='#5a5a58'):
        """A static caption naming a stream."""
        fontsize = self.FS_STREAM if fontsize is None else fontsize
        return self._fs(self.ax_d.text(x, y, s, ha=ha, va=va, color=colour), fontsize)

    def _pipe(self, p0, p1, lw=1.8, head=True):
        """A straight connector; head=False for an elbow's non-terminal leg,
        so a multi-segment run only shows one arrowhead, at its real end."""
        self.ax_d.annotate('', xy=p1, xytext=p0, zorder=2,
                           arrowprops=dict(arrowstyle='-|>' if head else '-',
                                           lw=lw, color=self.C_PIPE,
                                           shrinkA=0, shrinkB=0))

    @staticmethod
    def _edge_y(x, p_left, p_right):
        """Height of a turbine's sloping top or bottom edge at station x."""
        (x0, y0), (x1, y1) = p_left, p_right
        return y0 + (y1 - y0)*(x - x0)/(x1 - x0)

    def _block(self, xy, w, h, label, fontsize=None, fc=None, ec=None,
               label_at='top'):
        """An equipment block: rounded rect with its name across the top."""
        x, y = xy
        fontsize = self.FS_BLOCK if fontsize is None else fontsize
        fc = self.C_BLOCK if fc is None else fc
        ec = self.C_EDGE if ec is None else ec
        self.ax_d.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=2.2',
            fc=fc, ec=ec, lw=1.0, zorder=3))
        cx = x + w/2
        if label_at == 'center':
            ty, va = y + h/2, 'center'
        else:
            ty, va = y + h - 1.6, 'top'
        self._fs(self.ax_d.text(cx, ty, label, ha='center', va=va,
                                color='white', zorder=4), fontsize)
        return x, y, w, h, cx, y + h*0.40

    def _header(self, xy, w, h, label):
        """A steam/water header: a flat amber bar."""
        x, y = xy
        self.ax_d.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6',
            fc=self.C_HEADER, ec=self.C_HEADER_EDGE, lw=1.0, zorder=3))
        self._fs(self.ax_d.text(x + 3.5, y + h/2, label, ha='left', va='center',
                                color=self.C_DARK_TXT, zorder=4), self.FS_HEADER)
        return x, y, w, h

    def _build_schematic(self, x0, y0, w, h):
        """Block diagram of the connected plant, with live values on the
        streams. Internal drawing coordinates (0-235 x, 0-182 y) are
        unchanged from plant_manual.py; only the axes' position within this
        figure (x0,y0,w,h, figure-fraction) is new."""
        ax = self.fig.add_axes([x0, y0, w, h])
        self.ax_d = ax
        ax.set_xlim(-20, 235)
        ax.set_ylim(0, 182)
        ax.axis('off')

        x_tap = 150 - self.T_TAP_FRAC*(150 - 100)      # 133.33

        # Row A (y=138-160): boiler -> turbine 1
        *_, bcx, bcy = self._block((10, 138), 50, 22, 'Biomass boiler',
                                   fc=self.C_COMBUST, ec=self.C_COMBUST_EDGE)
        self.d_boiler = self._tag(bcx, bcy, ha='center', fontsize=self.FS_TAG_SM)

        #  Boiler inlets (2026-10-01, Hamish): each caption stacked ABOVE its pipe --
        #  biomass as its value tag with 'biomass in' over it, feedwater as its two-line
        #  caption -- tops level, pipes symmetric about the boiler's centre (x=35).
        self._pipe((20, 170), (20, 160))
        self.d_fuel = self._tag(20, 172.8, ha='center', fontsize=self.FS_TAG_SM)
        self._stream_text(20, 176, 'biomass in', ha='center')

        self._pipe((50, 170), (50, 160))
        self._stream_text(50, 171.8, f'water source\n{blr.T_FW:.0f}°C', ha='center')

        ax.add_patch(Polygon(self.T1_POLY, closed=True, fc=self.C_TURBINE,
                             ec=self.C_TURBINE_EDGE, lw=1.0, zorder=3))
        t1_lx = 125 + self.T_LABEL_DX
        t1_top_mid = self._edge_y(t1_lx, self.T1_POLY[1], self.T1_POLY[2])
        self._fs(ax.text(t1_lx, t1_top_mid - 2.2, 'Turbine 1', ha='center', va='top',
                         color='white', zorder=4), self.FS_BLOCK)
        self.d_t1 = self._tag(t1_lx, t1_top_mid - 13.0, ha='center', fontsize=self.FS_TAG_SM)

        self._pipe((60, 151), (99, 151))
        self.d_hp = self._tag(80, 156.5, ha='center', fontsize=self.FS_TAG_SM)

        t1_top_tap = self._edge_y(x_tap, self.T1_POLY[1], self.T1_POLY[2])
        self._pipe((x_tap, t1_top_tap), (x_tap, 176))
        self._stream_text(x_tap, 177, 'power', ha='center')
        self.d_power = self._tag(x_tap + 3, 171, ha='left', fontsize=self.FS_TAG_SM)

        # Row B (y=116-130): MPS header
        mx, my, mw, mh = self._header((45, 116), 115, 14, 'MPS header')
        t1_bot_tap = self._edge_y(x_tap, self.T1_POLY[0], self.T1_POLY[3])
        self._pipe((x_tap, t1_bot_tap), (x_tap, my + mh))
        self._stream_text(x_tap + 3, 138.8, 'MPS out', ha='left', va='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_t1_out = self._tag(x_tap + 3, 133.8, ha='left',
                                  fontsize=self.FS_TAG_SM)
        self._pipe((mx+mw, my+mh/2), (190, my+mh/2))
        self.d_mps = self._tag(mx+mw/2, my+mh/2, ha='center', fontsize=self.FS_TAG_DENSE)

        t2_mid = 125 + self.T2_DX                  # turbine 2's centre line
        x_tap2 = x_tap + self.T2_DX                # turbine 2's outlet tap
        t2_top_mid = self._edge_y(t2_mid, self.T2_POLY[1], self.T2_POLY[2])
        self._pipe((t2_mid, my), (t2_mid, t2_top_mid))
        #  Left of the inlet pipe (2026-10-01): the taller turbine's top edge rises to the
        #  right, into where this tag used to sit.
        self.d_frac = self._tag(t2_mid - 3, 109.6, ha='right', fontsize=self.FS_TAG_SM)

        # Row C (y=82-104): heat exchanger, turbine 2
        *_, hcx, hcy = self._block((10, 82), 50, 22, 'Heat exchanger',
                                   fc=self.C_COMBUST, ec=self.C_COMBUST_EDGE)
        self.d_hx = self._tag(hcx, hcy, ha='center', fontsize=self.FS_TAG_SM)

        self._pipe((25, 138), (25, 104))
        self._stream_text(27, 120, 'flue gas', ha='left', va='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_flue = self._tag(28, 112, ha='left', fontsize=self.FS_TAG_SM)

        HX_CY, HX_WATER_OFFSET = 93.0, 3.0
        #  Starts further left than the HX hot-water return column (x=-1), and
        #  the caption sits clear of the exchanger block's left edge.
        self._pipe((-13.0, HX_CY + HX_WATER_OFFSET), (9, HX_CY + HX_WATER_OFFSET))
        self._stream_text(
            -14.5, HX_CY + HX_WATER_OFFSET + 1.5,
            f'site recycle\n{hx.T_W_IN:.0f}°C',
            ha='left', va='bottom', fontsize=self.FS_STREAM_SM)

        ax.add_patch(Polygon(self.T2_POLY, closed=True, fc=self.C_TURBINE,
                             ec=self.C_TURBINE_EDGE, lw=1.0, zorder=3))
        t2_lx = t2_mid + self.T_LABEL_DX
        t2_lbl_top = self._edge_y(t2_lx, self.T2_POLY[1], self.T2_POLY[2])
        self._fs(ax.text(t2_lx, t2_lbl_top - 2.2, 'Turbine 2', ha='center', va='top',
                         color='white', zorder=4), self.FS_BLOCK)
        self.d_t2 = self._tag(t2_lx, t2_lbl_top - 13.0, ha='center', fontsize=self.FS_TAG_SM)

        # Row D (y=56-70): LPS header
        lx, ly, lw_, lh = self._header((45, 56), 115, 14, 'LPS header')
        t2_bot_tap = self._edge_y(x_tap2, self.T2_POLY[0], self.T2_POLY[3])
        self._pipe((x_tap2, t2_bot_tap), (x_tap2, ly + lh))
        self._stream_text(x_tap2 + 3, 77.8, 'LPS out', ha='left', va='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_t2_out = self._tag(x_tap2 + 3, 72.9, ha='left',
                                  fontsize=self.FS_TAG_SM)
        self._pipe((lx+lw_, ly+lh/2), (190, ly+lh/2))
        self.d_lps = self._tag(lx+lw_/2, ly+lh/2, ha='center', fontsize=self.FS_TAG_DENSE)

        #  60 -> 67 (2026-10-01): the primary heat pump's 'water source' caption (left
        #  of this pipe) ran into the HP1 LPS riser.
        HP_PROD_X = 67.0
        self._pipe((HP_PROD_X, 44), (HP_PROD_X, ly))
        self._stream_text(HP_PROD_X + 3, 52, 'HP1 LPS', ha='left', va='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_hpp_lps = self._tag(HP_PROD_X + 3, 47, ha='left',
                                   fontsize=self.FS_TAG_SM)

        #  Letdown (MPS -> LPS, above mps_nominal) and vent/dump (LPS -> sink,
        #  above lps_nominal/relief_start) -- both real, always-on plant
        #  equipment (header_closures.flows()), not a control action, and both
        #  previously only showed up as text in the top-left status line
        #  ("spill .. dump .. kg/s"). Added 2026-09-24: with the target
        #  pressures pushed to their extremes it's easy to have most of
        #  Turbine 1's output leave as letdown rather than to site, and most
        #  of the LPS header's inflow leave as vent rather than to site --
        #  invisible on the diagram before this, which read as the header not
        #  balancing when it always did.
        #  Letdown: clear of both turbines (T1_POLY/T2_POLY span x 100-150)
        #  and short of the to-site branch point (x=mx+mw=160), so it neither
        #  crosses a turbine nor the horizontal to-site runs.
        #  152 -> 157 (2026-10-01), clear of the LPS-out tag's right edge.
        LETDOWN_X = 157
        self._pipe((LETDOWN_X, my), (LETDOWN_X, ly + lh))
        self._stream_text(LETDOWN_X + 3, (my + ly + lh)/2, 'letdown',
                          ha='left', va='center', fontsize=self.FS_STREAM_SM)
        self.d_letdown = self._tag(LETDOWN_X + 3, (my + ly + lh)/2 - 4.5,
                                   ha='left', fontsize=self.FS_TAG_SM)
        #  Vent: a short stub with nowhere it connects to (same convention as
        #  'flue gas out' on the heat pump above) -- it leaves the modelled
        #  system here, at the sink pressure, same as letdown does not.
        #  Moved 2026-09-25 to the TOP of the LPS header, towards its left end,
        #  pointing up (steam leaving to atmosphere): x=75 sits in the clear gap
        #  between the heat exchanger (x 10-60) and turbine 2 (x 96-146), below
        #  the MPS header.
        VENT_X = 75
        self._pipe((VENT_X, ly + lh), (VENT_X, ly + lh + 11))
        self._stream_text(VENT_X + 3, ly + lh + 3.5, 'vent', ha='left', va='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_vent = self._tag(VENT_X + 3, ly + lh + 8.5, ha='left',
                                fontsize=self.FS_TAG_SM)

        # Row E (y=22-44): the two heat pumps
        *_, pcx, pcy = self._block((10, 22), 60, 22, 'Heat pump (primary)',
                                   fc=self.C_HP, ec=self.C_HP_EDGE)
        self.d_hpp = self._tag(pcx, pcy, ha='center', fontsize=self.FS_TAG_SM)

        self._pipe((25, 82), (25, 44))
        self._stream_text(27, 62, f'flue gas\n{hpp.T_FG_IN_HP:.0f}°C',
                          ha='left', va='center', fontsize=self.FS_STREAM_SM)

        self._pipe((16, 22), (16, 12))
        self._stream_text(19, 15.5,
                          f'flue gas out\n{hpp.T_FG_OUT_HP_SPEC:.0f}°C',
                          ha='left', va='center', fontsize=self.FS_STREAM_SM)

        HP_MU_X = 36.0
        self._pipe((HP_MU_X, 54), (HP_MU_X, 44))
        self._stream_text(HP_MU_X + 2, 50,
                          f'water source\n{hpp.T_MU:.0f}°C',
                          ha='left', va='center', fontsize=self.FS_STREAM_SM)

        *_, scx, scy = self._block((100, 22), 60, 22, 'Heat pump (secondary)',
                                   fc=self.C_HP, ec=self.C_HP_EDGE,
                                   fontsize=self.FS_BLOCK_SM)
        self.d_hps = self._tag(scx, scy, ha='center', fontsize=self.FS_TAG_SM)

        self._pipe((100, 33), (71, 33))
        self._stream_text(85, 36.5, 'HP2 duty', ha='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_hp2 = self._tag(85, 29.5, ha='center', fontsize=self.FS_TAG_SM)

        SEC_CY = 22 + 22/2
        self._pipe((182, SEC_CY), (161, SEC_CY), lw=1.4)
        self._stream_text(175, SEC_CY + 2.5,
                          f'water source\n{hps.T_SRC_SEC:.0f}°C',
                          ha='center', va='bottom', fontsize=self.FS_STREAM_SM)

        # Row F (y=2-16): hot water header
        wx, wy, ww, wh = self._header((45, 2), 115, 14, 'HW header')
        self._pipe((HP_PROD_X, 22), (HP_PROD_X, wy+wh))
        #  Caption left of the riser, value right of it (2026-10-01): side by side on
        #  one side they overlapped.
        self._stream_text(HP_PROD_X - 3, 19, 'HW out', ha='right', va='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_hpp_hw = self._tag(HP_PROD_X + 3, 19, ha='left',
                                  fontsize=self.FS_TAG_SM)

        #  Return column -1 -> -5 (2026-10-01): its caption and tag sat across the
        #  flue gas pipe at x=25.
        HX_RET_X = -5.0
        self._pipe((10, HX_CY - HX_WATER_OFFSET), (HX_RET_X, HX_CY - HX_WATER_OFFSET),
                   head=False)
        self._pipe((HX_RET_X, HX_CY - HX_WATER_OFFSET), (HX_RET_X, wy+wh/2), head=False)
        self._pipe((HX_RET_X, wy+wh/2), (wx, wy+wh/2))
        self._stream_text(-2.5, 79.8, 'HX hot water', ha='left', va='center',
                          fontsize=self.FS_STREAM_SM)
        self.d_hx_hw = self._tag(-2, 72.8, ha='left', fontsize=self.FS_TAG_SM)

        self._pipe((wx+ww, wy+wh/2), (190, wy+wh/2))
        self.d_hw_site = self._tag(wx+ww/2, wy+wh/2, ha='center', fontsize=self.FS_TAG)

        # Site -- one tall block on the right, collecting all three headers
        self._block((190, 2), 40, 128, 'Site', fc=self.C_SITE, ec=self.C_SITE_EDGE,
                    fontsize=11, label_at='center')

    def _update_schematic(self, r):
        self.d_boiler.set_text(f'{r["duty"]/1e3:.1f} MW\neta {r["eta_b"]:.3f}')
        self.d_hx.set_text(f'{r["Q_hx"]/1e3:.2f} MW')
        self.d_t1.set_text(f'{r["power1"]:.0f} kW\ndh {r["dh_act"]:.1f} kJ/kg')
        self.d_t2.set_text(f'{r["power2"]:.0f} kW\ndh {r["dh_act2"]:.1f} kJ/kg')
        #  Thermal inertia, not infeasibility: Q_cond follows its sources
        #  through TAU_DUTY, so during a transient the instantaneous
        #  evaporator demand and supply differ, and at rest they do not.
        #  Trimmed 2026-10-01 to Q/COP/W, matching the secondary box: the old
        #  evap/gas/rec/HP2/lag lines overflowed the block and covered its title
        #  (HP2 duty is already on its own arrow, evap in the comparison table).
        #  Settling now shows only as the text colour.
        resid = r['dQ_shortfall']
        settling = abs(resid) > 0.02*max(r['Q_cond_pri'], 1.0)
        #  The LPS-pressure/critical-temperature warning that used to sit here was
        #  removed 2026-09-24 (Hamish: no high-pressure warning; the run only stops
        #  at header_closures' 10 bar LPS / 100 bar MPS ceilings).
        hpp_color = '#8a6d1f' if settling else '#1d3d52'
        self.d_hpp.set_text(f'Q {r["Q_cond_pri"]:.0f} kW  COP {r["COP_pri"]:.2f}\n'
                            f'W {r["W_hp_pri"]:.0f} kW')
        self.d_hpp.set_color(hpp_color)
        self.d_hps.set_text(f'Q {r["Q_cond_sec"]:.0f} kW  COP {r["COP_sec"]:.2f}\n'
                            f'W {r["W_hp_sec"]:.0f} kW')

        self.d_fuel.set_text(f'{r["mfuel_act"]:.2f} kg/s')
        self.d_hp.set_text(f'{r["steam"]:.2f} kg/s\n'
                           f'{cfg.P_HP:.0f} bar  {cfg.T_HP:.1f} C')
        self.d_flue.set_text(f'{r["gas"]:.1f} kg/s\n{r["stack"]:.1f} C')
        self.d_power.set_text(f'{r["power1"]:.0f} kW')
        self.d_mps.set_text(f'{r["pmps_act"]:.1f} bar  {r["T_out1"]:.1f} C\n'
                            f'{r["m_mps_site"]:.2f} kg/s to site\n'
                            f'unserved {r["mps_shortfall"]:.2f} kg/s')
        self.d_mps.set_color('#a45a13' if r['mps_shortfall'] > 0.05 else '#1d3d52')
        self.d_frac.set_text(f'frac {r["frac_t2_act"]:.2f}\n{r["m_lps"]:.2f} kg/s')
        self.d_t1_out.set_text(f'{r["steam"]:.2f} kg/s')
        self.d_t2_out.set_text(f'{r["m_lps"]:.2f} kg/s')
        self.d_lps.set_text(f'{r["plps_act"]:.1f} bar  {r["T_out2"]:.1f} C\n'
                            f'{r["m_lps_total"]:.2f} kg/s to site\n'
                            f'unserved {r["lps_shortfall"]:.2f} kg/s')
        self.d_lps.set_color('#a45a13' if r['lps_shortfall'] > 0.05 else '#1d3d52')
        self.d_hw_site.set_text(f'{r["m_hw_total"]:.2f} kg/s to site')
        self.d_hpp_lps.set_text(f'{r["m_lps_hp"]:.2f} kg/s')
        self.d_letdown.set_text(f'{r["m_letdown"]:.2f} kg/s')
        self.d_vent.set_text(f'{r["m_vent"]:.2f} kg/s')
        self.d_hpp_hw.set_text(f'{r["m_hw_hp"]:.2f} kg/s')
        self.d_hx_hw.set_text(f'{r["mwater_act"]:.2f} kg/s\n{r["twout"]:.1f} C')
        self.d_hp2.set_text(f'{r["Q_cond_sec"]:.0f} kW')
        #  The monospace note block over the schematic (t, fuel act/cmd, driver, trims,
        #  shortfall flags) was merged into the status line 2026-10-01.

    def save_csv(self,path):
        with Path(path).open('w',newline='') as f:
            f.write('# Operator control; no_control/feedforward/feedforward_pid; actual mode recorded per row\n')
            f.write('# See docs/07_regulation_models_report.md for what each mode drives biomass/split/subcooling from\n')
            f.write('# LPS tuning '+str(self.rig.lps_pi.tuning)+'; guards '+str(self.rig.guards)+'\n')
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        return Path(path)

    def export(self,_=None):
        path=Path(__file__).parent/('plant_operator_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')
        print('Saved',self.save_csv(path),flush=True)

    def close(self,_=None):
        self.running=False;self.timer.stop()
        self._finish_solve()
        self.rig.m.cleanup()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='HP2-centred operator interface')
    p.add_argument('--batch',type=int,default=None,help='seconds per solve (default 10; 20 is about 1.5x faster per simulated second)')
    a=p.parse_args()
    app=OperatorApp(batch=a.batch,threaded=True);plt.show()
