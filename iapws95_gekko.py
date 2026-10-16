"""Native IAPWS-95 water / steam properties for GEKKO.

This module implements the complete 2018 revision of the IAPWS-95
dimensionless Helmholtz-energy equation (all 56 residual terms).  It is
written directly with GEKKO expressions, so APMonitor can differentiate the
property equations during simulation or optimisation.

The fundamental equation is single-phase.  ``state_ph_smooth`` adds a
differentiable pressure-enthalpy flash for subcooled liquid, the saturation
dome, and superheated vapour.  Saturated liquid and vapour are calculated by
the IAPWS-95 equality-of-pressure and equality-of-Gibbs-energy conditions.
Within the dome, mixture properties use vapour quality.  A configurable,
very small dT/dh regularises the otherwise exactly horizontal isobar, and
smooth min/max functions blend the three regions.

Units
-----
    temperature       K
    pressure          kPa
    density           kg/m^3
    specific energy   kJ/kg
    specific entropy  kJ/(kg K)

GEKKO supplies the symbolic model.  SciPy is used for accurate first-time-point
initialisation; CoolProp is only used by the separate regression tests.

Reference
---------
Wagner, W. and Pruß, A. (2002), J. Phys. Chem. Ref. Data 31, 387-535,
including the IAPWS R6-95(2018) revision.
"""

from dataclasses import dataclass
from functools import lru_cache
import math


# IAPWS-95 reducing constants and specific gas constant.
TC = 647.096                 # K
RHOC = 322.0                 # kg/m^3
PC_KPA = 22064.0             # kPa
MOLAR_MASS = 0.018015268     # kg/mol
R = 8.314371357587 / MOLAR_MASS / 1000.0  # kJ/(kg K)
T_TRIPLE = 273.16            # K
P_TRIPLE_KPA = 0.6116548008968684


# Ideal-gas Helmholtz contribution, Table 4 of IAPWS-95.
IDEAL_LEAD = (-8.3204464837497, 6.6832105275932)
IDEAL_LOG_TAU = 3.00632
IDEAL_PLANCK_N = (0.012436, 0.97315, 1.27950, 0.96956, 0.24873)
IDEAL_PLANCK_GAMMA = (
    1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105,
)


# Residual power / exponential terms 1-51, Table 5.
POWER_N = (
    0.012533547935523, 7.8957634722828, -8.7803203303561,
    0.31802509345418, -0.26145533859358, -0.0078199751687981,
    0.0088089493102134, -0.66856572307965, 0.20433810950965,
    -6.6212605039687e-05, -0.19232721156002, -0.25709043003438,
    0.16074868486251, -0.040092828925807, 3.9343422603254e-07,
    -7.5941377088144e-06, 0.00056250979351888,
    -1.5608652257135e-05, 1.1537996422951e-09,
    3.6582165144204e-07, -1.3251180074667998e-12,
    -6.2639586912454e-10, -0.10793600908932, 0.017611491008752,
    0.22132295167546, -0.40247669763528, 0.58083399985759,
    0.0049969146990806, -0.031358700712549, -0.74315929710341,
    0.4780732991548, 0.020527940895948, -0.13636435110343,
    0.014180634400617, 0.0083326504880713, -0.029052336009585,
    0.038615085574206, -0.020393486513704, -0.0016554050063734,
    0.0019955571979541, 0.00015870308324157, -1.638856834253e-05,
    0.043613615723811, 0.034994005463765, -0.076788197844621,
    0.022446277332006, -6.2689710414685e-05,
    -5.571111856564501e-10, -0.19905718354408, 0.31777497330738,
    -0.11841182425981,
)
POWER_D = (
    1, 1, 1, 2, 2, 3, 4, 1, 1, 1, 2, 2, 3, 4, 4, 5, 7, 9, 10, 11,
    13, 15, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 9, 9, 9, 9, 9,
    10, 10, 12, 3, 4, 4, 5, 14, 3, 6, 6, 6,
)
POWER_T = (
    -0.5, 0.875, 1, 0.5, 0.75, 0.375, 1, 4, 6, 12, 1, 5, 4, 2,
    13, 9, 3, 4, 11, 4, 13, 1, 7, 1, 9, 10, 10, 3, 7, 10, 10, 6,
    10, 10, 1, 2, 3, 4, 8, 6, 9, 8, 16, 22, 23, 23, 10, 50, 44,
    46, 50,
)
POWER_L = (
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 3, 3, 3, 3, 4, 6, 6, 6, 6,
)


# Gaussian terms 52-54.
GAUSS_N = (-31.306260323435, 31.546140237781, -2521.3154341695)
GAUSS_D = (3, 3, 3)
GAUSS_T = (0, 1, 4)
GAUSS_ETA = (20, 20, 20)
GAUSS_BETA = (150, 150, 250)
GAUSS_GAMMA = (1.21, 1.21, 1.25)
GAUSS_EPSILON = (1, 1, 1)


# Non-analytic critical terms 55-56.
CRIT_N = (-0.14874640856724, 0.31806110878444)
CRIT_A = (0.32, 0.32)
CRIT_B = (0.2, 0.2)
CRIT_C = (28, 32)
CRIT_D = (700, 800)
CRIT_a = (3.5, 3.5)
CRIT_b = (0.85, 0.95)
CRIT_beta = (0.3, 0.3)


# IAPWS saturation ancillary equations.  These are used only for initial
# guesses; symbolic saturation states are refined with the full EOS.
SAT_P_N = (-7.85951783, 1.84408259, -11.7866497,
           22.6807411, -15.9618719, 1.80122502)
SAT_P_T = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
SAT_RHOL_N = (1.99274064, 1.09965342, -0.510839303,
              -1.75493479, -45.5170352, -6.74694450e5)
SAT_RHOL_T = (1/3, 2/3, 5/3, 16/3, 43/3, 110/3)
SAT_RHOV_N = (-2.03150240, -2.68302940, -5.38626492,
              -17.2991605, -44.7586581, -63.9201063)
SAT_RHOV_T = (1/3, 2/3, 4/3, 3, 37/6, 71/6)


@dataclass
class HelmholtzDerivatives:
    """Dimensionless Helmholtz energy and derivatives."""

    phi0: object
    phi0_t: object
    phi0_tt: object
    phir: object
    phir_d: object
    phir_dd: object
    phir_t: object
    phir_tt: object
    phir_dt: object


@dataclass
class State:
    """Thermodynamic state; fields may be floats or GEKKO expressions."""

    T: object
    P: object
    rho: object
    h: object
    s: object
    u: object
    g: object
    cv: object
    cp: object
    w: object
    quality: object = None
    phase: str = 'single'


@dataclass
class SaturationState:
    """Coexisting liquid and vapour states at a pressure."""

    T: object
    P: object
    liquid: State
    vapor: State


def _numeric(value):
    """Return a scalar from a number or GEKKO object, when available."""
    if isinstance(value, (int, float)):
        return float(value)
    val = getattr(value, 'value', None)
    if val is None:
        return None
    try:
        return float(val[0])
    except (TypeError, IndexError):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


def _exp(m, value):
    return math.exp(value) if m is None else m.exp(value)


def _log(m, value):
    return math.log(value) if m is None else m.log(value)


def _sqrt(m, value):
    return math.sqrt(value) if m is None else m.sqrt(value)


def _named_intermediate(m, value, name):
    return value if m is None else m.Intermediate(value, name=name)


def _chunked_sum(m, values, name, chunk_size=6):
    """Sum long symbolic expressions without exceeding APMonitor line limits."""
    if m is None:
        return sum(values)
    chunks = []
    for index in range(0, len(values), chunk_size):
        chunks.append(m.Intermediate(
            sum(values[index:index+chunk_size]),
            name=f'{name}_{index//chunk_size}'))
    return m.Intermediate(sum(chunks), name=name)


def smooth_abs(m, value, epsilon):
    """Differentiable absolute value, analogous to Pyomo smooth_abs."""
    return _sqrt(m, value*value + epsilon*epsilon)


def smooth_min(m, left, right, epsilon=1e-3):
    """Twice differentiable approximation to min(left, right)."""
    return 0.5*(left + right - smooth_abs(m, left-right, epsilon))


def smooth_max(m, left, right, epsilon=1e-3):
    """Twice differentiable approximation to max(left, right)."""
    return 0.5*(left + right + smooth_abs(m, left-right, epsilon))


def smooth_clip(m, value, lower, upper, epsilon=1e-5):
    """Differentiably clip value to [lower, upper]."""
    return smooth_min(m, smooth_max(m, value, lower, epsilon),
                      upper, epsilon)


def smooth_step(m, value, width):
    """Smoothly move from zero to one as value crosses zero."""
    return 0.5*(1.0 + value/smooth_abs(m, value, width))


def helmholtz(T, rho, m=None, name='iapws', critical_epsilon=1e-20):
    """Evaluate IAPWS-95 Helmholtz energy and first/second derivatives.

    ``m=None`` evaluates ordinary floats.  Supplying a GEKKO model returns
    symbolic Intermediates.  ``critical_epsilon`` only regularises the exact
    critical-point singularity of terms 55-56; away from (Tc, rhoc) its
    numerical effect is negligible.
    """
    delta = rho/RHOC
    tau = TC/T

    # Ideal contribution and tau derivatives.
    phi0 = _log(m, delta) + IDEAL_LEAD[0] + IDEAL_LEAD[1]*tau
    phi0_t = IDEAL_LEAD[1] + IDEAL_LOG_TAU/tau
    phi0_tt = -IDEAL_LOG_TAU/(tau*tau)
    phi0 += IDEAL_LOG_TAU*_log(m, tau)
    for ni, gamma in zip(IDEAL_PLANCK_N, IDEAL_PLANCK_GAMMA):
        eg = _exp(m, gamma*tau)
        phi0 += ni*_log(m, 1.0 - 1.0/eg)
        phi0_t += ni*gamma/(eg - 1.0)
        phi0_tt -= ni*gamma*gamma*eg/((eg-1.0)*(eg-1.0))

    phir_terms = []
    phir_d_terms = []
    phir_dd_terms = []
    phir_t_terms = []
    phir_tt_terms = []
    phir_dt_terms = []

    # Terms 1-51.
    for ni, di, ti, li in zip(POWER_N, POWER_D, POWER_T, POWER_L):
        damp = 1.0 if li == 0 else _exp(m, -(delta**li))
        term = ni*(delta**di)*(tau**ti)*damp
        ad = di/delta
        add = -di/(delta*delta)
        if li:
            ad -= li*(delta**(li-1))
            add -= li*(li-1)*(delta**(li-2))
        at = ti/tau
        att = -ti/(tau*tau)

        phir_terms.append(term)
        phir_d_terms.append(term*ad)
        phir_dd_terms.append(term*(ad*ad + add))
        phir_t_terms.append(term*at)
        phir_tt_terms.append(term*(at*at + att))
        phir_dt_terms.append(term*ad*at)

    # Terms 52-54.
    for ni, di, ti, eta, beta, gamma, eps in zip(
            GAUSS_N, GAUSS_D, GAUSS_T, GAUSS_ETA, GAUSS_BETA,
            GAUSS_GAMMA, GAUSS_EPSILON):
        exponent = -eta*(delta-eps)**2 - beta*(tau-gamma)**2
        term = ni*(delta**di)*(tau**ti)*_exp(m, exponent)
        ad = di/delta - 2.0*eta*(delta-eps)
        add = -di/(delta*delta) - 2.0*eta
        at = ti/tau - 2.0*beta*(tau-gamma)
        att = -ti/(tau*tau) - 2.0*beta

        phir_terms.append(term)
        phir_d_terms.append(term*ad)
        phir_dd_terms.append(term*(ad*ad + add))
        phir_t_terms.append(term*at)
        phir_tt_terms.append(term*(at*at + att))
        phir_dt_terms.append(term*ad*at)

    # Terms 55-56.  Log-derivative expressions keep the implementation
    # compact while retaining all first and second derivatives.
    x = delta - 1.0
    q = x*x + critical_epsilon
    for ni, Ai, Bi, Ci, Di, ai, bi, betai in zip(
            CRIT_N, CRIT_A, CRIT_B, CRIT_C, CRIT_D,
            CRIT_a, CRIT_b, CRIT_beta):
        mi = 1.0/(2.0*betai)
        theta = (1.0-tau) + Ai*(q**mi)
        theta_d = Ai*2.0*mi*x*(q**(mi-1.0))
        theta_dd = Ai*2.0*mi*(q**(mi-2.0)) * (
            q + 2.0*(mi-1.0)*x*x)

        Delta = theta*theta + Bi*(q**ai) + critical_epsilon
        Delta_d = (2.0*theta*theta_d
                   + Bi*2.0*ai*x*(q**(ai-1.0)))
        Delta_dd = (2.0*(theta_d*theta_d + theta*theta_dd)
                    + Bi*2.0*ai*(q**(ai-2.0))
                    * (q + 2.0*(ai-1.0)*x*x))
        Delta_t = -2.0*theta
        Delta_tt = 2.0
        Delta_dt = -2.0*theta_d

        psi = _exp(m, -Ci*x*x - Di*(tau-1.0)**2)
        term = ni*delta*(Delta**bi)*psi

        ld = 1.0/delta + bi*Delta_d/Delta - 2.0*Ci*x
        lt = bi*Delta_t/Delta - 2.0*Di*(tau-1.0)
        ldd = (-1.0/(delta*delta)
               + bi*(Delta_dd/Delta-(Delta_d/Delta)**2) - 2.0*Ci)
        ltt = (bi*(Delta_tt/Delta-(Delta_t/Delta)**2) - 2.0*Di)
        ldt = bi*(Delta_dt/Delta-Delta_d*Delta_t/(Delta*Delta))

        phir_terms.append(term)
        phir_d_terms.append(term*ld)
        phir_dd_terms.append(term*(ld*ld + ldd))
        phir_t_terms.append(term*lt)
        phir_tt_terms.append(term*(lt*lt + ltt))
        phir_dt_terms.append(term*(ld*lt + ldt))

    phir = _chunked_sum(m, phir_terms, f'{name}_phir')
    phir_d = _chunked_sum(m, phir_d_terms, f'{name}_phird')
    phir_dd = _chunked_sum(m, phir_dd_terms, f'{name}_phirdd')
    phir_t = _chunked_sum(m, phir_t_terms, f'{name}_phirt')
    phir_tt = _chunked_sum(m, phir_tt_terms, f'{name}_phirtt')
    phir_dt = _chunked_sum(m, phir_dt_terms, f'{name}_phirdt')

    return HelmholtzDerivatives(
        _named_intermediate(m, phi0, f'{name}_phi0'),
        _named_intermediate(m, phi0_t, f'{name}_phi0t'),
        _named_intermediate(m, phi0_tt, f'{name}_phi0tt'),
        phir,
        phir_d,
        phir_dd,
        phir_t,
        phir_tt,
        phir_dt,
    )


def properties_trho(T, rho, m=None, name='iapws'):
    """Return a complete single-phase state from temperature and density."""
    d = rho/RHOC
    tau = TC/T
    a = helmholtz(T, rho, m=m, name=name)

    pressure = rho*R*T*(1.0 + d*a.phir_d)
    internal_energy = R*T*tau*(a.phi0_t + a.phir_t)
    entropy = R*(tau*(a.phi0_t+a.phir_t) - a.phi0 - a.phir)
    enthalpy = R*T*(1.0 + tau*(a.phi0_t+a.phir_t) + d*a.phir_d)
    gibbs = enthalpy - T*entropy

    cv_r = -tau*tau*(a.phi0_tt+a.phir_tt)
    denominator = 1.0 + 2.0*d*a.phir_d + d*d*a.phir_dd
    numerator = 1.0 + d*a.phir_d - d*tau*a.phir_dt
    cp_r = cv_r + numerator*numerator/denominator
    sound_bracket = denominator - (
        numerator*numerator/(tau*tau*(a.phi0_tt+a.phir_tt)))
    sound_squared = 1000.0*R*T*sound_bracket
    # Mathematical roots inside the spinodal can have w^2 < 0.  Numeric
    # initialisation solvers still need p/h/s there to bracket a stable root.
    sound_speed = (math.sqrt(sound_squared) if m is None and sound_squared >= 0
                   else (float('nan') if m is None else m.sqrt(
                       smooth_max(m, sound_squared, 0.0, 1e-8)+1e-12)))

    values = [pressure, enthalpy, entropy, internal_energy, gibbs,
              R*cv_r, R*cp_r, sound_speed]
    if m is not None:
        labels = ('P', 'h', 's', 'u', 'g', 'cv', 'cp', 'w')
        values = [_named_intermediate(m, value, f'{name}_{label}')
                  for value, label in zip(values, labels)]
    return State(T, values[0], rho, values[1], values[2], values[3],
                 values[4], values[5], values[6], values[7])


def saturation_pressure_guess(T):
    """IAPWS ancillary saturation pressure in kPa (initial guesses only)."""
    if not T_TRIPLE <= T <= TC:
        raise ValueError(f'saturation temperature must be in [{T_TRIPLE}, {TC}] K')
    theta = 1.0-T/TC
    exponent = (TC/T)*sum(n*theta**t for n, t in zip(SAT_P_N, SAT_P_T))
    return PC_KPA*math.exp(exponent)


def saturation_temperature_guess(P_kPa):
    """Invert the saturation-pressure ancillary by bisection."""
    if not P_TRIPLE_KPA <= P_kPa <= PC_KPA:
        raise ValueError(
            f'saturation pressure must be in [{P_TRIPLE_KPA}, {PC_KPA}] kPa')
    lo, hi = T_TRIPLE, TC
    for _ in range(100):
        mid = 0.5*(lo+hi)
        if saturation_pressure_guess(mid) < P_kPa:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


def saturation_density_guesses(T):
    """Return ancillary estimates (rho_liquid, rho_vapour), kg/m^3."""
    theta = 1.0-T/TC
    rhol = RHOC*(1.0 + sum(n*theta**t
                           for n, t in zip(SAT_RHOL_N, SAT_RHOL_T)))
    rhov = RHOC*math.exp(sum(n*theta**t
                            for n, t in zip(SAT_RHOV_N, SAT_RHOV_T)))
    return rhol, rhov


def solve_rho_tp_numeric(T, P_kPa, phase='auto', rho_guess=None):
    """Solve the full numeric EOS for density at T and P.

    This supplies consistent initial values for GEKKO dynamic models, whose
    algebraic variables are fixed at their initial values at the first time
    point.  SciPy is used only for these scalar initialisation solves; all
    model equations remain native GEKKO expressions.
    """
    from scipy.optimize import brentq, root_scalar

    selected = phase.lower()
    if selected == 'auto':
        if P_kPa < PC_KPA and T < TC:
            selected = ('liquid' if P_kPa > saturation_pressure_guess(T)
                        else 'vapor')
        else:
            selected = 'supercritical'

    def residual(rho):
        return properties_trho(T, rho).P-P_kPa

    tiny = 1e-10
    if selected in ('vapor', 'vapour'):
        lower, upper = tiny, RHOC*(1.0-1e-9)
        guess = P_kPa/(R*T)
    elif selected == 'liquid':
        lower, upper = RHOC*(1.0+1e-9), 1300.0
        guess = (saturation_density_guesses(T)[0]
                 if T_TRIPLE <= T < TC else 1000.0)
    else:
        lower, upper = tiny, 1500.0
        if T_TRIPLE <= T < TC:
            guess = saturation_density_guesses(T)[0]
        else:
            guess = P_kPa/(R*T)
    if rho_guess is not None:
        guess = rho_guess

    if selected == 'vapor' and rho_guess is None and T < TC:
        #  Below Tc the isotherm on (0, RHOC) can cross P more than once: the stable
        #  vapour root, then the unstable van der Waals loop root(s). A brentq over the
        #  whole interval may return a loop root, and one wrong density makes
        #  _solve_px_numeric's own residual discontinuous in T -- its brentq then
        #  "converges" onto the jump (found 2026-09-25: solve_ps_numeric at 21.19 bar
        #  returned T 620 K with s 6.918 for a target s of 6.695). The vapour root is the
        #  LOWEST-density crossing, so walk up from below the ideal-gas density to the
        #  first sign change and bracket only that.
        lo_r = 0.5*guess
        if residual(lo_r) < 0:
            hi_r = lo_r
            while hi_r < upper:
                hi_r = min(hi_r*1.02, upper)
                if residual(hi_r) >= 0:
                    return brentq(residual, lo_r, hi_r, xtol=1e-11, rtol=1e-13)
                lo_r = hi_r

    f_lower, f_upper = residual(lower), residual(upper)
    if f_lower*f_upper <= 0:
        return brentq(residual, lower, upper, xtol=1e-11, rtol=1e-13)

    # An unrestricted supercritical interval can contain several mathematical
    # roots below Tc.  A local secant solve follows the stable-root seed.
    second = min(upper, max(lower, guess*(1.0+1e-3)+1e-5))
    solution = root_scalar(residual, x0=guess, x1=second, method='secant',
                           xtol=1e-11, rtol=1e-13, maxiter=100)
    if not solution.converged or not lower <= solution.root <= upper:
        raise RuntimeError('numeric IAPWS-95 density solve did not converge')
    return solution.root


def solve_tp_numeric(T, P_kPa, phase='auto'):
    """Return a numeric IAPWS-95 State from T and P."""
    rho = solve_rho_tp_numeric(T, P_kPa, phase=phase)
    state = properties_trho(T, rho)
    state.P = P_kPa
    state.phase = phase
    return state


#  Unbounded, not 128. The saturation solve is the single most expensive call in
#  the numeric path -- two coupled brentq root-finds on the Helmholtz surface --
#  and an envelope sweep visits hundreds of distinct pressures, so a small cache
#  thrashes and every point pays full price. The keys are floats and the values
#  are small immutable states; even a long session holds a few thousand.
@lru_cache(maxsize=None)
def solve_saturation_numeric(P_kPa):
    """Solve exact numeric IAPWS-95 liquid-vapour coexistence."""
    from scipy.optimize import root

    P_kPa = float(P_kPa)
    if not P_TRIPLE_KPA <= P_kPa < PC_KPA:
        raise ValueError('numeric saturation solve requires subcritical pressure')
    T0 = saturation_temperature_guess(P_kPa)
    rhol0, rhov0 = saturation_density_guesses(T0)

    def residual(values):
        T, rhol, rhov = values
        liquid = properties_trho(T, rhol)
        vapor = properties_trho(T, rhov)
        return ((liquid.P-P_kPa)/PC_KPA,
                (vapor.P-P_kPa)/PC_KPA,
                (liquid.g-vapor.g)/(R*TC))

    solution = root(residual, (T0, rhol0, rhov0), method='hybr',
                    options={'xtol': 1e-10, 'maxfev': 1000})
    if not solution.success:
        raise RuntimeError(f'numeric IAPWS-95 saturation solve failed: {solution.message}')
    T, rhol, rhov = map(float, solution.x)
    liquid = properties_trho(T, rhol)
    vapor = properties_trho(T, rhov)
    liquid.P = vapor.P = P_kPa
    liquid.phase = 'liquid'
    vapor.phase = 'vapor'
    return SaturationState(T, P_kPa, liquid, vapor)


def _solve_px_numeric(P_kPa, target, prop, phase):
    """Shared numeric P-h / P-s temperature solve."""
    from scipy.optimize import brentq

    selected = phase.lower()
    if selected == 'liquid':
        upper = solve_saturation_numeric(float(P_kPa)).T
        lower = T_TRIPLE
    elif selected in ('vapor', 'vapour'):
        lower = solve_saturation_numeric(float(P_kPa)).T
        upper = 1273.0
        selected = 'vapor'
    else:
        lower, upper = T_TRIPLE, 1273.0
        selected = 'supercritical'

    def residual(T):
        return getattr(solve_tp_numeric(T, P_kPa, selected), prop)-target

    T = brentq(residual, lower, upper, xtol=2e-10, rtol=1e-13)
    return solve_tp_numeric(T, P_kPa, selected)


def solve_ph_numeric(P_kPa, h, phase):
    """Return a numeric single-phase state from P and h."""
    return _solve_px_numeric(P_kPa, h, 'h', phase)


def solve_ps_numeric(P_kPa, s, phase):
    """Return a numeric single-phase state from P and s."""
    return _solve_px_numeric(P_kPa, s, 's', phase)


def saturation_state(m, P_kPa, name='sat', initial_P=None):
    """Create an exact IAPWS-95 liquid-vapour coexistence calculation.

    Pressure must remain between the triple and critical pressures.  The
    ancillary equations only seed the solve; all three defining equations use
    the full IAPWS-95 fundamental equation.

    initial_P overrides the numeric seed pressure when P_kPa is a live GEKKO
    expression with no usable .value yet (an Intermediate, before the first
    solve) -- same override pattern state_ph/state_ps already use.
    """
    pressure0 = (_numeric(initial_P) if initial_P is not None
                else _numeric(P_kPa))
    if pressure0 is None:
        pressure0 = 0.1*PC_KPA
    if not P_TRIPLE_KPA <= pressure0 < PC_KPA:
        raise ValueError('saturation_state requires subcritical pressure')
    seed = solve_saturation_numeric(float(pressure0))
    temperature0 = seed.T
    rhol0, rhov0 = seed.liquid.rho, seed.vapor.rho

    temperature = m.Var(value=temperature0, lb=T_TRIPLE, ub=TC-1e-5,
                        name=f'{name}_T')
    rho_l = m.Var(value=rhol0, lb=RHOC, ub=1300.0, name=f'{name}_rhol')
    rho_v = m.Var(value=rhov0, lb=1e-8, ub=RHOC, name=f'{name}_rhov')
    liquid = properties_trho(temperature, rho_l, m=m, name=f'{name}_l')
    vapor = properties_trho(temperature, rho_v, m=m, name=f'{name}_v')

    m.Equation((liquid.P-P_kPa)/PC_KPA == 0)
    m.Equation((vapor.P-P_kPa)/PC_KPA == 0)
    m.Equation((liquid.g-vapor.g)/(R*TC) == 0)
    return SaturationState(temperature, P_kPa, liquid, vapor)


def state_tp(m, T, P_kPa, phase='auto', name='tp', initial_T=None,
             initial_P=None):
    """Create a single-phase IAPWS-95 state from T and P.

    ``phase`` may be ``'liquid'``, ``'vapor'``, ``'supercritical'`` or
    ``'auto'``.  Explicit phase selection is recommended when T or P is a
    decision variable that could admit multiple density roots.
    """
    temperature0 = (_numeric(initial_T) if initial_T is not None
                    else _numeric(T))
    pressure0 = (_numeric(initial_P) if initial_P is not None
                 else _numeric(P_kPa))
    if temperature0 is None:
        temperature0 = 500.0
    if pressure0 is None:
        pressure0 = 101.325

    selected = phase.lower()
    if selected == 'auto':
        if pressure0 < PC_KPA and temperature0 < TC:
            selected = ('liquid' if pressure0 > saturation_pressure_guess(temperature0)
                        else 'vapor')
        else:
            # The ideal-gas seed works well for hot, gas-like supercritical
            # states; dense states should be selected explicitly as liquid.
            selected = 'supercritical'

    ideal_rho = pressure0/(R*temperature0)
    if selected in ('vapor', 'vapour'):
        rho0 = min(max(ideal_rho, 1e-6), 0.95*RHOC)
        bounds = (1e-8, RHOC)
    elif selected == 'liquid':
        if T_TRIPLE <= temperature0 < TC:
            rho0 = saturation_density_guesses(temperature0)[0]
        else:
            rho0 = 1000.0
        bounds = (RHOC, 1300.0)
    elif selected == 'supercritical':
        if T_TRIPLE <= temperature0 < TC:
            # Above pc but below Tc is a dense, liquid-like state.
            rho0 = saturation_density_guesses(temperature0)[0]
        else:
            rho0 = max(ideal_rho, 1e-6)
        bounds = (1e-8, 1500.0)
    else:
        raise ValueError("phase must be auto, liquid, vapor, or supercritical")

    rho0 = solve_rho_tp_numeric(
        temperature0, pressure0, phase=selected, rho_guess=rho0)

    rho = m.Var(value=rho0, lb=bounds[0], ub=bounds[1], name=f'{name}_rho')
    state = properties_trho(T, rho, m=m, name=name)
    m.Equation((state.P-P_kPa)/PC_KPA == 0)
    state.P = P_kPa
    state.phase = selected
    return state


def _state_ph_branch(m, P_kPa, h_target, T0, rho0, phase, name):
    """Solve one clamped single-phase P-h branch."""
    if phase == 'liquid':
        rho_bounds = (RHOC, 1300.0)
    elif phase == 'vapor':
        rho_bounds = (1e-8, RHOC)
    else:
        rho_bounds = (1e-8, 1500.0)
    temperature = m.Var(value=T0, lb=T_TRIPLE, ub=1273.0,
                        name=f'{name}_T')
    rho = m.Var(value=rho0, lb=rho_bounds[0], ub=rho_bounds[1],
                name=f'{name}_rho')
    state = properties_trho(temperature, rho, m=m, name=name)
    m.Equation((state.P-P_kPa)/PC_KPA == 0)
    m.Equation((state.h-h_target)/(R*TC) == 0)
    state.P = P_kPa
    state.h = h_target
    state.phase = phase
    return state


def state_ph(m, P_kPa, h, phase='auto', name='ph_single', initial_h=None,
             initial_P=None):
    """Create an IAPWS-95 state from pressure and enthalpy.

    With numeric initial values, ``phase='auto'`` selects liquid, vapour,
    supercritical, or the smooth two-phase flash.  If P or h can move between
    regions during optimisation, call ``state_ph_smooth`` explicitly for a
    subcritical trajectory rather than relying on the initial auto choice.
    """
    pressure0 = (_numeric(initial_P) if initial_P is not None
                 else _numeric(P_kPa))
    enthalpy0 = _numeric(initial_h) if initial_h is not None else _numeric(h)
    if pressure0 is None:
        pressure0 = 0.1*PC_KPA
    if enthalpy0 is None:
        enthalpy0 = 2500.0

    selected = phase.lower()
    if pressure0 < PC_KPA:
        sat0 = solve_saturation_numeric(float(max(pressure0, P_TRIPLE_KPA)))
        hf0, hg0 = sat0.liquid.h, sat0.vapor.h
    if selected == 'auto':
        if pressure0 >= PC_KPA:
            selected = 'supercritical'
        elif enthalpy0 < hf0:
            selected = 'liquid'
        elif enthalpy0 > hg0:
            selected = 'vapor'
        else:
            return state_ph_smooth(m, P_kPa, h, name=name)
    if selected in ('smooth', 'two-phase', 'two_phase'):
        return state_ph_smooth(m, P_kPa, h, name=name)
    if pressure0 >= PC_KPA and selected != 'supercritical':
        raise ValueError('above critical pressure, select phase=supercritical')
    if selected in ('vapor', 'vapour'):
        selected = 'vapor'
    if selected not in ('liquid', 'vapor', 'supercritical'):
        raise ValueError('phase must be auto, liquid, vapor, supercritical, or smooth')
    seed = solve_ph_numeric(pressure0, enthalpy0, selected)
    T0, rho0 = seed.T, seed.rho
    return _state_ph_branch(m, P_kPa, h, T0, rho0, selected, name)


def state_ps(m, P_kPa, s, phase='auto', name='ps', initial_s=None,
             initial_P=None):
    """Create a single-phase IAPWS-95 state from pressure and entropy.

    For a phase-changing expansion use ``state_ps_smooth``.  The direct
    formulation is substantially smaller and faster when the phase is known.
    """
    pressure0 = (_numeric(initial_P) if initial_P is not None
                 else _numeric(P_kPa))
    entropy0 = _numeric(initial_s) if initial_s is not None else _numeric(s)
    if pressure0 is None:
        pressure0 = 0.1*PC_KPA
    if entropy0 is None:
        entropy0 = 6.0

    selected = phase.lower()
    if pressure0 < PC_KPA:
        sat0 = solve_saturation_numeric(float(max(pressure0, P_TRIPLE_KPA)))
        sf0, sg0 = sat0.liquid.s, sat0.vapor.s
    if selected == 'auto':
        if pressure0 >= PC_KPA:
            selected = 'supercritical'
        elif entropy0 < sf0:
            selected = 'liquid'
        elif entropy0 > sg0:
            selected = 'vapor'
        else:
            return state_ps_smooth(m, P_kPa, s, name=name)
    if selected in ('smooth', 'two-phase', 'two_phase'):
        return state_ps_smooth(m, P_kPa, s, name=name)
    if pressure0 >= PC_KPA and selected != 'supercritical':
        raise ValueError('above critical pressure, select phase=supercritical')
    if selected == 'liquid':
        rho_bounds = (RHOC, 1300.0)
    elif selected in ('vapor', 'vapour'):
        selected = 'vapor'
        rho_bounds = (1e-8, RHOC)
    elif selected == 'supercritical':
        rho_bounds = (1e-8, 1500.0)
    else:
        raise ValueError('phase must be auto, liquid, vapor, supercritical, or smooth')

    seed = solve_ps_numeric(pressure0, entropy0, selected)
    T0, rho0 = seed.T, seed.rho
    temperature = m.Var(value=T0, lb=T_TRIPLE, ub=1273.0,
                        name=f'{name}_T')
    rho = m.Var(value=rho0, lb=rho_bounds[0], ub=rho_bounds[1],
                name=f'{name}_rho')
    state = properties_trho(temperature, rho, m=m, name=name)
    m.Equation((state.P-P_kPa)/PC_KPA == 0)
    m.Equation((state.s-s)/(R) == 0)
    state.P = P_kPa
    state.s = s
    state.phase = selected
    return state


def state_ph_smooth(m, P_kPa, h, name='ph', transition_h=0.1,
                    two_phase_dTdh=1e-6, initial_P=None, initial_h=None,
                    sat=None):
    """Differentiable IAPWS-95 P-h state including the saturation dome.

    Parameters
    ----------
    transition_h : float
        Enthalpy width [kJ/kg] of each smooth liquid/two-phase/vapour
        transition.  Smaller values approach a piecewise-exact flash but are
        more demanding numerically.
    two_phase_dTdh : float
        Artificial slope [K/(kJ/kg)] within the saturation dome.  The exact
        equilibrium value is zero.  A small positive value prevents a flat
        T(h) relation in equation-oriented optimisation.  It is anchored at
        saturated liquid and continued as a negligible constant offset into
        the vapour branch so dT/dh remains positive at both transitions.
    initial_P, initial_h : float, optional
        Override the numeric seed when P_kPa/h are live GEKKO expressions
        with no usable .value yet (e.g. an Intermediate before the first
        solve) -- same override pattern state_ph/state_ps already use.
    sat : SaturationState, optional
        A coexistence state already built at this same P_kPa (e.g. by an
        earlier call to ``saturation_state`` or ``state_ps_smooth`` for the
        same pressure). When supplied, the (expensive -- two full IAPWS-95
        Helmholtz evaluations plus the pressure/Gibbs equations) saturation
        solve is skipped and this one is reused instead. Only valid when
        ``sat`` really was built at this P_kPa; passing one from a different
        pressure silently gives the wrong dome location.

    Notes
    -----
    This function is for subcritical pressure.  The fundamental EOS functions
    remain valid above the critical point; use ``state_tp`` or a direct
    single-phase P-h solve there because no phase-change region exists.
    """
    if transition_h <= 0:
        raise ValueError('transition_h must be positive')
    if two_phase_dTdh < 0:
        raise ValueError('two_phase_dTdh must be non-negative')

    if sat is None:
        sat = saturation_state(m, P_kPa, name=f'{name}_sat', initial_P=initial_P)
    hf, hg = sat.liquid.h, sat.vapor.h
    latent = hg-hf

    # Clamp the branch enthalpies with Pyomo-style smooth min/max.  Thus both
    # branch inversions remain in a physically valid single-phase region even
    # while h passes through the saturation dome.
    h_liquid = _named_intermediate(
        m, smooth_min(m, h, hf, transition_h), f'{name}_h_liq')
    h_vapor = _named_intermediate(
        m, smooth_max(m, h, hg, transition_h), f'{name}_h_vap')

    pressure0 = _numeric(initial_P) if initial_P is not None else _numeric(P_kPa)
    enthalpy0 = _numeric(initial_h) if initial_h is not None else _numeric(h)
    if pressure0 is None:
        pressure0 = 0.1*PC_KPA
    sat_seed = solve_saturation_numeric(float(pressure0))
    hf0, hg0 = sat_seed.liquid.h, sat_seed.vapor.h
    if enthalpy0 is None:
        enthalpy0 = 0.5*(hf0+hg0)
    h_liquid0 = smooth_min(None, enthalpy0, hf0, transition_h)
    h_vapor0 = smooth_max(None, enthalpy0, hg0, transition_h)

    #  The single-phase P-h solve is accurate and should be preferred whenever
    #  it succeeds (h_liquid0/h_vapor0 sit far from saturation whenever the
    #  true h does, e.g. genuinely subcooled liquid or superheated vapor, and
    #  the coexistence-based fallback below would be a much worse seed
    #  there). It only becomes numerically degenerate when clamped to right
    #  at the saturation boundary (h near/inside the dome on that branch's
    #  side): forcing a single-phase branch solve exactly at the saturation
    #  temperature can converge to a spurious root near the critical density
    #  instead of the true liquid/vapor density. In that case the branch's
    #  seed is essentially the saturation state anyway (by construction, it's
    #  clamped to within transition_h of it), so fall back to the already-
    #  robust coexistence solve above instead.
    try:
        liquid_seed = solve_ph_numeric(pressure0, h_liquid0, 'liquid')
    except ValueError:
        liquid_seed = sat_seed.liquid
    try:
        vapor_seed = solve_ph_numeric(pressure0, h_vapor0, 'vapor')
    except ValueError:
        vapor_seed = sat_seed.vapor

    liquid = _state_ph_branch(
        m, P_kPa, h_liquid, liquid_seed.T, liquid_seed.rho,
                              'liquid', f'{name}_liq')
    vapor = _state_ph_branch(m, P_kPa, h_vapor, vapor_seed.T, vapor_seed.rho,
                             'vapor', f'{name}_vap')

    # Smooth phase weights.  w_two is nearly one between hf and hg, while
    # w_liquid and w_vapor take over outside the dome.
    w_liquid = 1.0-smooth_step(m, h-hf, transition_h)
    w_vapor = smooth_step(m, h-hg, transition_h)
    w_two = 1.0-w_liquid-w_vapor
    qeps = transition_h/latent
    quality = smooth_clip(m, (h-hf)/latent, 0.0, 1.0, qeps)

    rho_two = 1.0/((1.0-quality)/sat.liquid.rho
                  + quality/sat.vapor.rho)
    s_two = (1.0-quality)*sat.liquid.s + quality*sat.vapor.s
    # Anchor the regularised line at saturated liquid and carry its very small
    # offset into the vapour branch.  This preserves positive dT/dh through
    # both smooth boundaries.  With the default slope, the full latent interval
    # introduces only about 0.002 K of vapour-side offset at typical pressures.
    T_two = sat.T + two_phase_dTdh*(h-hf)
    T_vapor_regularized = vapor.T + two_phase_dTdh*latent
    cp_two = 1.0/max(two_phase_dTdh, 1e-12)

    temperature = _named_intermediate(
        m, w_liquid*liquid.T+w_two*T_two
        + w_vapor*T_vapor_regularized, f'{name}_T')
    density = _named_intermediate(
        m, w_liquid*liquid.rho+w_two*rho_two+w_vapor*vapor.rho,
        f'{name}_rho')
    entropy = _named_intermediate(
        m, w_liquid*liquid.s+w_two*s_two+w_vapor*vapor.s, f'{name}_s')
    cv = _named_intermediate(
        m, w_liquid*liquid.cv+w_two*cp_two+w_vapor*vapor.cv,
        f'{name}_cv')
    cp = _named_intermediate(
        m, w_liquid*liquid.cp+w_two*cp_two+w_vapor*vapor.cp,
        f'{name}_cp')
    sound = _named_intermediate(
        m, w_liquid*liquid.w+w_two*0.0+w_vapor*vapor.w, f'{name}_w')
    internal_energy = _named_intermediate(
        m, h-P_kPa/density, f'{name}_u')
    gibbs = _named_intermediate(m, h-temperature*entropy, f'{name}_g')
    quality = _named_intermediate(m, quality, f'{name}_quality')

    return State(temperature, P_kPa, density, h, entropy, internal_energy,
                 gibbs, cv, cp, sound, quality=quality, phase='smooth')


def state_ps_smooth(m, P_kPa, s, name='ps_smooth', transition_h=0.1,
                    two_phase_dTdh=1e-6, initial_P=None, initial_s=None,
                    sat=None):
    """Differentiable pressure-entropy flash built on ``state_ph_smooth``.

    This is convenient for isentropic turbine outlets.  Enthalpy is introduced
    as an algebraic variable and the smooth P-h state is constrained to the
    requested entropy.

    initial_P, initial_s : float, optional
        Override the numeric seed when P_kPa/s are live GEKKO expressions
        with no usable .value yet -- same override pattern state_ph/state_ps
        already use.
    sat : SaturationState, optional
        Forwarded to ``state_ph_smooth`` -- pass a coexistence state already
        built at this P_kPa to skip re-solving it. See that function's
        docstring.
    """
    pressure0 = _numeric(initial_P) if initial_P is not None else _numeric(P_kPa)
    entropy0 = _numeric(initial_s) if initial_s is not None else _numeric(s)
    if pressure0 is None:
        pressure0 = 0.1*PC_KPA
    if not P_TRIPLE_KPA <= pressure0 < PC_KPA:
        raise ValueError('state_ps_smooth requires subcritical pressure')
    sat_seed = solve_saturation_numeric(float(pressure0))
    sl, sv = sat_seed.liquid, sat_seed.vapor
    if entropy0 is None:
        entropy0 = 0.5*(sl.s+sv.s)
    if entropy0 < sl.s:
        h0 = solve_ps_numeric(pressure0, entropy0, 'liquid').h
    elif entropy0 > sv.s:
        h0 = solve_ps_numeric(pressure0, entropy0, 'vapor').h
    else:
        quality0 = (entropy0-sl.s)/(sv.s-sl.s)
        h0 = sl.h + quality0*(sv.h-sl.h)

    enthalpy = m.Var(value=h0, lb=-100.0, ub=5000.0, name=f'{name}_h')
    state = state_ph_smooth(
        m, P_kPa, enthalpy, name=name, transition_h=transition_h,
        two_phase_dTdh=two_phase_dTdh, initial_P=pressure0, initial_h=h0,
        sat=sat)
    m.Equation((state.s-s)/R == 0)
    state.s = s
    return state


__all__ = [
    'TC', 'RHOC', 'PC_KPA', 'R', 'T_TRIPLE', 'P_TRIPLE_KPA',
    'State', 'SaturationState', 'helmholtz', 'properties_trho',
    'saturation_pressure_guess', 'saturation_temperature_guess',
    'saturation_density_guesses', 'solve_rho_tp_numeric', 'solve_tp_numeric',
    'solve_saturation_numeric', 'solve_ph_numeric', 'solve_ps_numeric',
    'saturation_state', 'state_tp',
    'state_ph', 'state_ps', 'state_ph_smooth', 'state_ps_smooth',
    'smooth_abs', 'smooth_min', 'smooth_max',
    'smooth_clip', 'smooth_step',
]
