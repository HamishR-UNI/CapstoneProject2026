"""Raw dynamic constants for the CHP plant model -- one place, no imports.

Each constant carries its basis. [S] = sourced (see SOURCES), [D] = derived from sourced
values, [L] = layout/engineering assumption (no plant exists), with impact noted.

SOURCES
-------
Mas'a, V. et al. (2011). "Blocks and streamlined models for optimising and controlling
    biomass-fired boilers." Chemical Engineering Transactions, 25. Measured FOPTD fits for
    a 1 MW moving-grate hot-water boiler (fuel-increase and fuel-decrease cases). Cited via
    Hosovsky, A. (2015), "Mathematical modelling of biomass combustion..." review paper.
Xia, K. et al. (2025). Grate-fired 30 MW biomass boiler dynamic model and control study --
    feed -> heat -> main steam pressure -> power staging, validated on plant data. Used only
    for the multi-stage STRUCTURE, not for any numeric time constant here (its valve/steam
    terms describe a boiler-internal regulating stage, not this project's turbine).
Yang, Y.B. et al. (2005). "Effect of fuel properties on biomass combustion." Devolatilisation
    timing for wood-chip beds.
KLEAA (fixed-bed wood chip combustion reference data). Ignition delay for a wood-chip bed.
Saastamoinen, J. et al. (2000). "Ignition-front propagation in packed beds of biomass."
    Front speed vs fuel moisture content.
Fu, Z. et al. (2026). Grate residence time and conversion-length study for a moving-grate
    biomass boiler.
IEA Bioenergy Task 32. Volatile matter share of as-fired biomass heating value.
Spirax Sarco. Steam Engineering Tutorials -- recommended steam velocities (saturated and
    superheated) for pipe sizing.
CIBSE Guide B1 (2016), Table 1.A1.4 -- recommended water velocities for heating pipework.
Qv, D. et al. (2024). System identification of a field water-source heat pump (Shanxi):
    compressor and condenser step-response transfer functions.
IEEE PES-TR1 (2013) and IEEE Working Group (1991), "Dynamic models for steam and hydro
    turbines in power system studies" -- steam-chest charging time (TCH), sub-second to
    low seconds for a grid-synchronised non-reheat unit.
Astrom, K.J. and Bell, R.D. (2000). "Drum-boiler dynamics." Automatica 36. Header/drum
    pressure storage as a capacitance state -- out of scope here (section 11).
Aastroem/Kortela, J. and Jamsae-Jounela, S.-L. Grate boiler drum-pressure control papers --
    fuel-drying-rate-limited pressure settling.
Bauer, R. et al. (2010). Two-stage (drying / devolatilisation) grate bed model.
ControlGuru (controlguru.com) heat exchanger FOPDT training example -- a simulated
    shell-side exchanger with its own sensor/pipe arrangement, used only as a sanity
    comparison, not a physical-plant benchmark.
"""

# --- Plug-flow reduction [D] ---------------------------------------------------------
# Plug-flow liquid side + uniform heat-flux step -> outlet ramps linearly over one residence
# time t_res. Smith two-point fit (28.3/63.2 %) of a unit ramp, exactly:
#   tau = 1.5*(0.632-0.283)*t_res = 0.5235 t_res,  theta = (0.632-0.5235)*t_res = 0.1085 t_res
PLUG_TAU_FRAC   = 1.5 * (0.632 - 0.283)
PLUG_THETA_FRAC = 0.632 - PLUG_TAU_FRAC

# --- Actuators [L, low impact] ------------------------------------------------------------
# Valve response is characterised by T86 (~2 time constants) per ISA-TR75.25.02; no generic
# numeric value is published, so these are assumptions. Impact: seconds, vs a 10-min boiler.
THETA_VALVE_S = 0.3
TAU_VALVE_S   = 2.0     # existing model value, unchanged
# LPS relief valve lift response (FOPTD on the valve's flow, plant_rig), added 2026-09-24.
# ASSUMED, no relief-valve data. Faster than the actuated control valves above: a
# spring-loaded relief valve has no actuator/positioner to stroke. First tried at the
# control-valve values (0.3 s / 2.0 s): against the small LPS header (~20 kg/bar) and
# the relief's steep curve that rang for ~25 s (damping ~0.2, ~5 s period, peak 5.3
# bar on a 29 kg/s surplus step).
THETA_RELIEF_S = 0.1
TAU_RELIEF_S   = 0.5
TAU_FAN_S     = 10.0    # FD/ID fan speed response [L, low impact]
THETA_FAN_S   = 1.0     # [L, low impact]

# --- Demand-driven dispatch delay [L, low impact] ------------------------------------------
# plant_rig.PlantRig._invert recomputes fuel/frac_T2 from the site demand sliders on every
# solve and, without this, applies the result immediately -- nothing stands between a demand
# change and a new target the way a real supervisory calculation-and-dispatch cycle would.
# No literature fit exists for that cycle time; this is a plain assumption, kept deliberately
# short and separate from everything physical downstream of it (feeder ~60 s, fuel bed
# ~600 s, admission valve ~2 s) -- a dead time only, no smoothing, since it stands for a
# calculation-and-issue delay, not a mechanical response.
THETA_DISPATCH_S = 5.0

# --- Biomass grate: fuel-bed heat-release response ------------------------------------------
THETA_FEEDER_S       = 60.0        # [S] Masa et al. 2011 (via Hosovsky 2015): ~60 s to feed fuel
T_IGNITION_S         = 120.0       # [S] KLEAA fixed-bed wood chips: ignition delay ~2 min
                                   #     (Yang et al. 2005: devolatilisation starts 2.5-3 min)
BED_HEIGHT_M         = 0.20        # [S] grate fuel layer 10-30 cm (grate-firing references, mid value)
V_IGNITION_FRONT_MPS = 0.25e-3     # [S] reported front speeds ~0.25-0.33 mm/s; speed falls with
                                   #     moisture (Saastamoinen et al. 2000) -> low end for 40 % moisture
T_CONVERSION_END_S   = 0.62 * 75 * 60   # [S] Fu et al. 2026: ~75 min on grate, conversion
                                        #     complete at ~62 % of grate length (pure wood)
F_VOLATILE_HEAT      = 0.70        # [S] IEA Bioenergy Task 32: volatiles ~70 % of heating value
V_HP_STEAM_MPS       = 50.0        # [S] Spirax Sarco: superheated steam 50-70 m/s
L_HP_STEAM_M         = 30.0        # [L] boiler -> turbine 1 main steam line (0.6 s, negligible)

# --- Turbine ------------------------------------------------------------------------------
L_STAGE_PATH_M, V_STAGE_STEAM_MPS = 4.0, 40.0   # [L] steam chest + blading transit (0.1 s)

# --- Flue gas and water/steam distribution ----------------------------------------------------
L_FG_DUCT_M, V_FG_DUCT_MPS = 20.0, 15.0    # [L, low impact] boiler exit -> flue gas HX (1.3 s)
L_HW_RUN_M                 = 40.0          # [L] equipment -> site hot water header
V_HW_PIPE_MPS              = 1.5           # [S] CIBSE Guide B1 (2016) Table 1.A1.4 limits;
                                           #     heating circuits typically 1.0-1.5 m/s
L_LPS_PIPE_M               = 50.0          # [L] HP1 steam raiser -> LPS header
V_LPS_STEAM_MPS            = 25.0          # [S] Spirax Sarco: saturated steam 25-40 m/s

# --- Heat pumps -------------------------------------------------------------------------------
THETA_COMP_S = 10.0   # [S] Qv et al. 2024: identified compressor model e^-10s/(1+20s)
TAU_COMP_S   = 20.0   # [S] same
REF_CHARGE_KG_PER_KW = 0.10  # [L] cross-check only (charge / refrigerant flow), NOT used in the model

# --- HP2 condenser -> HP1 evaporator intermediate water loop ------------------------------------
DT_LINK_K   = 10.0    # [S] Qv et al. 2024: condenser supply/return difference usually limited to 10 K
L_LINK_M    = 40.0    # [L]
V_LINK_MPS  = 2.0     # [S] within CIBSE B1 limit for steel pipe > 50 mm (3.0 m/s)
