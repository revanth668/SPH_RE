"""
REAL-SCALE 2D SPH + N-body simulation:
    an impactor on a collision course with the Earth flies past Mars on the
    way in. Its gravity tugs Mars onto a NEW ORBIT, Mars tugs it back, and
    the deflected impactor then goes on to hit the Earth and merge with it.

Bodies (all real values, astronomical units: AU, years, solar masses):
    Sun     1 Msun,        R = 695,700 km
    Earth   5.97e24 kg,    R = 6,371 km,   on its 1 AU orbit
    Mars    6.42e23 kg,    R = 3,390 km,   real orbit a = 1.524 AU, e = 0.093
    Theia   0.3 Earths,    R ~ 4,565 km    (density 4,500 kg/m^3)

Same SPH + gravity core as sph_orbit_collision_real_scale.py:
  - cubic spline kernel, summation density, symmetrised pressure gradient,
    Monaghan-Gingold artificial viscosity, kick-drift-kick leapfrog,
    contact merge (mass, momentum, volume conserved), unsoftened gravity,
    adaptive time step that shrinks during close encounters.

New in this version:
  1. Mars, on its real orbit, as a fourth body.
  2. A designed close flyby: the impactor passes ~10 Mars radii behind Mars.
     Mars is pulled backwards, loses orbital energy and drops onto a
     smaller, more eccentric orbit.
  3. Targeting through the flyby. A close flyby amplifies tiny errors by
     a factor of thousands, so the starting conditions come from
       (a) integrating the WHOLE system backwards from the chosen impact,
           then
       (b) a "shooting" correction that re-runs the real integrator and
           nudges the impactor's starting speed until it is aimed at the
           Earth's centre.
  4. A "ghost Mars": the same Mars integrated WITHOUT the impactor, so the
     animation and plots show exactly how much the flyby changed its orbit.
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp

try:
    import cupy as xp
    GPU = xp.cuda.is_available()
except Exception:
    GPU = False
if not GPU:
    import numpy as xp
print(f"[backend] using {'GPU (CuPy)' if GPU else 'CPU (NumPy)'} for array ops")


def to_host(a):
    return xp.asnumpy(a) if GPU else np.array(a, copy=True)


# ======================================================================
# 0. UNITS AND REAL-WORLD CONSTANTS
# ======================================================================
AU_M = 1.495978707e11
YR_S = 365.25 * 86400.0
DAY = 1.0 / 365.25
MSUN_KG = 1.98847e30
G = 4.0 * np.pi ** 2                                # AU^3 / (Msun yr^2)
KMS = AU_M / YR_S / 1e3                             # 1 AU/yr = 4.74 km/s
JOULE = MSUN_KG * (AU_M / YR_S) ** 2
AU_KM = AU_M / 1e3


def sphere_density(m, r):
    return m / (4.0 / 3.0 * np.pi * r ** 3)


def sphere_radius(m, rho):
    return (3.0 * m / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


M_SUN, R_SUN = 1.0, 695_700.0 / AU_KM
M_EARTH, R_EARTH = 5.9722e24 / MSUN_KG, 6_371.0 / AU_KM
M_MARS, R_MARS = 6.4171e23 / MSUN_KG, 3_389.5 / AU_KM
A_MARS, E_MARS = 1.52368, 0.0934                   # Mars's real orbit
M_THEIA = 0.3 * M_EARTH                            # impactor: 0.3 Earth masses
RHO_THEIA_SI = 4500.0                              # kg/m^3 (rocky, iron-poor)
RHO_UNIT = MSUN_KG / AU_M ** 3
RHO_THEIA = RHO_THEIA_SI / RHO_UNIT
R_THEIA = sphere_radius(M_THEIA, RHO_THEIA)

RHO = {"Sun": sphere_density(M_SUN, R_SUN), "Earth": sphere_density(M_EARTH, R_EARTH),
       "Mars": sphere_density(M_MARS, R_MARS), "Theia": RHO_THEIA}
MASS = {"Sun": M_SUN, "Earth": M_EARTH, "Mars": M_MARS, "Theia": M_THEIA}
NAMES = ["Sun", "Earth", "Mars", "Theia"]


# ----------------------------------------------------------------------
# Cubic spline kernel (2D normalisation)
# ----------------------------------------------------------------------
def kernel_W(r, h):
    q = r / h
    sigma = 10.0 / (7.0 * xp.pi * h ** 2)
    W = xp.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    W[m1] = sigma[m1] * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
    W[m2] = sigma[m2] * 0.25 * (2.0 - q[m2]) ** 3
    return W


def kernel_gradW(rij_vec, r, h):
    q = r / h
    sigma = 10.0 / (7.0 * xp.pi * h ** 2)
    dWdq = xp.zeros_like(q)
    m1 = (q > 1e-12) & (q <= 1.0)
    m2 = (q > 1.0) & (q <= 2.0)
    dWdq[m1] = sigma[m1] * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)
    dWdq[m2] = -sigma[m2] * 0.75 * (2.0 - q[m2]) ** 2
    if not GPU:
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = xp.where(r > 0.0, dWdq / (h * r), 0.0)
    else:
        factor = xp.where(r > 0.0, dWdq / (h * r), 0.0)
    return rij_vec * factor[..., None]


# ----------------------------------------------------------------------
# Orbital elements relative to the Sun
# ----------------------------------------------------------------------
def orbital_elements(r_rel, v_rel, mu):
    r_rel = np.asarray(r_rel, float)
    v_rel = np.asarray(v_rel, float)
    r = np.linalg.norm(r_rel)
    v2 = v_rel @ v_rel
    eps = 0.5 * v2 - mu / r
    h_z = r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]
    e = np.sqrt(max(0.0, 1.0 + 2.0 * eps * h_z ** 2 / mu ** 2))
    bound = eps < 0.0
    if bound:
        a = -mu / (2.0 * eps)
        peri, apo = a * (1 - e), a * (1 + e)
        period = 2 * np.pi * np.sqrt(a ** 3 / mu)
    else:
        a = apo = period = np.inf
        peri = h_z ** 2 / mu / (1 + e)
    e_vec = ((v2 - mu / r) * r_rel - (r_rel @ v_rel) * v_rel) / mu
    return dict(a=a, e=e, peri=peri, apo=apo, period=period, bound=bound,
                omega=np.arctan2(e_vec[1], e_vec[0]), energy=eps)


def ellipse_xy(el, n=400):
    th = np.linspace(0, 2 * np.pi, n)
    rr = el["a"] * (1 - el["e"] ** 2) / (1 + el["e"] * np.cos(th))
    return np.c_[rr * np.cos(th + el["omega"]), rr * np.sin(th + el["omega"])]


# ----------------------------------------------------------------------
# SPH + gravity system
# ----------------------------------------------------------------------
class SPHSystem:
    def __init__(self, pos, vel, mass, rho_bulk, names, h, k_gas=1.0, gamma=1.0,
                 alpha=1.0, beta=2.0, G=0.0, exclude_pairs=(), allow_merge=True):
        # xp.array (not asarray) -> private copies, so the caller's arrays
        # are never modified in place by the integration
        self.pos = xp.array(pos, dtype=xp.float64)
        self.vel = xp.array(vel, dtype=xp.float64)
        self.mass = xp.array(mass, dtype=xp.float64)
        self.rho_bulk = xp.array(rho_bulk, dtype=xp.float64)
        self.names = list(names)
        self.h, self.k_gas, self.gamma = h, k_gas, gamma
        self.c_s = float(np.sqrt(gamma * k_gas))
        self.alpha, self.beta, self.G = alpha, beta, G
        self.eta2 = 0.01 * h ** 2
        # Pairs whose mutual gravity is switched off (used only while aiming)
        self.exclude_pairs = {frozenset(p) for p in exclude_pairs}
        self.allow_merge = allow_merge
        self.merge_events = []
        self._build_mask()

    def _build_mask(self):
        n = len(self.names)
        mask = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if frozenset((self.names[i], self.names[j])) in self.exclude_pairs:
                    mask[i, j] = 0.0
        self.grav_mask = xp.asarray(mask)

    def _pairwise(self):
        rij = self.pos[:, None, :] - self.pos[None, :, :]
        return rij, xp.linalg.norm(rij, axis=-1)

    def density(self, r):
        W = kernel_W(r, xp.full_like(r, self.h))
        return xp.maximum((self.mass[None, :] * W).sum(axis=1), 1e-30)

    def particle_radii(self):
        return (3.0 * self.mass / (4.0 * xp.pi * self.rho_bulk)) ** (1.0 / 3.0)

    def pressure(self, rho):
        return self.k_gas * rho ** self.gamma

    def viscosity(self, rij, r, vij, rho):
        vr = (vij * rij).sum(axis=-1)
        mu_ab = (self.h * vr) / (r ** 2 + self.eta2)
        rho_ab = 0.5 * (rho[:, None] + rho[None, :])
        Pi = (-self.alpha * self.c_s * mu_ab + self.beta * mu_ab ** 2) / rho_ab
        Pi = xp.where(vr < 0.0, Pi, 0.0)
        xp.fill_diagonal(Pi, 0.0)
        return Pi

    def accelerations(self):
        rij, r = self._pairwise()
        rho = self.density(r)
        P = self.pressure(rho)
        gradW = kernel_gradW(rij, r, xp.full_like(r, self.h))
        vij = self.vel[:, None, :] - self.vel[None, :, :]
        Pi = self.viscosity(rij, r, vij, rho)
        coeff = P[None, :] / rho[None, :] ** 2 + P[:, None] / rho[:, None] ** 2 + Pi
        xp.fill_diagonal(coeff, 0.0)
        acc = -(self.mass[None, :, None] * coeff[..., None] * gradW).sum(axis=1)
        if self.G != 0.0:
            r_safe = r.copy()
            xp.fill_diagonal(r_safe, xp.inf)
            w = self.grav_mask * r_safe ** -3
            acc += -self.G * (self.mass[None, :, None] * rij * w[..., None]).sum(axis=1)
        return acc, rho

    def choose_dt(self, dt_max, eta_ff=0.02, eta_gap=0.3, dt_min=1e-10):
        """Smaller of 2% of every pair's free-fall time and 30% of its time to contact."""
        if len(self.mass) < 2:
            return dt_max
        rij, r = self._pairwise()
        vrel = xp.linalg.norm(self.vel[:, None, :] - self.vel[None, :, :], axis=-1)
        radii = self.particle_radii()
        gap = xp.maximum(r - (radii[:, None] + radii[None, :]), 0.0)
        msum = self.mass[:, None] + self.mass[None, :]
        iu = xp.triu_indices(len(self.mass), 1)
        live = self.grav_mask[iu] > 0          # ignore pairs switched off for aiming
        t_ff = xp.where(live, xp.sqrt(r[iu] ** 3 / (self.G * msum[iu])), xp.inf)
        t_gap = xp.where(live, gap[iu] / xp.maximum(vrel[iu], 1e-30), xp.inf)
        dt = min(dt_max, eta_ff * float(t_ff.min()), eta_gap * float(t_gap.min()))
        return max(dt, dt_min)

    def step(self, dt):
        acc, _ = self.accelerations()
        self.vel += 0.5 * dt * acc
        self.pos += dt * self.vel
        acc, rho = self.accelerations()
        self.vel += 0.5 * dt * acc
        if self.allow_merge:
            self.merge()
        return rho

    def merge(self):
        """Contact merge; conserves mass, momentum and total volume."""
        merged_any = True
        while merged_any and len(self.mass) > 1:
            merged_any = False
            _, r = self._pairwise()
            radii = self.particle_radii()
            gap = r - (radii[:, None] + radii[None, :])
            xp.fill_diagonal(gap, xp.inf)
            i, j = (int(k) for k in xp.unravel_index(xp.argmin(gap), gap.shape))
            if gap[i, j] < 0.0:
                if self.mass[j] > self.mass[i]:
                    i, j = j, i
                m_a, m_b = self.mass[i], self.mass[j]
                v_a, v_b = self.vel[i].copy(), self.vel[j].copy()
                m_new = float(m_a + m_b)
                v_new = (m_a * v_a + m_b * v_b) / m_new
                r_new = (m_a * self.pos[i] + m_b * self.pos[j]) / m_new
                rho_new = m_new / float(m_a / self.rho_bulk[i] + m_b / self.rho_bulk[j])
                name_new = f"{self.names[i]}+{self.names[j]}"
                self.merge_events.append(dict(
                    name=name_new, parents=(self.names[i], self.names[j]), mass=m_new,
                    vel=to_host(v_new), pos=to_host(r_new),
                    parent_mass=(float(m_a), float(m_b)),
                    parent_vel=(to_host(v_a), to_host(v_b)),
                    parent_radii=(float(radii[i]), float(radii[j])),
                    ke_before=float(0.5 * m_a * (v_a @ v_a) + 0.5 * m_b * (v_b @ v_b)),
                    ke_after=float(0.5 * m_new * (v_new @ v_new))))
                keep = [k for k in range(len(self.mass)) if k not in (i, j)]
                self.pos = xp.vstack([self.pos[keep], xp.asarray(r_new).reshape(1, 2)])
                self.vel = xp.vstack([self.vel[keep], xp.asarray(v_new).reshape(1, 2)])
                self.mass = xp.concatenate([self.mass[keep], xp.array([m_new])])
                self.rho_bulk = xp.concatenate([self.rho_bulk[keep], xp.array([rho_new])])
                self.names = [self.names[k] for k in keep] + [name_new]
                self._build_mask()
                merged_any = True

    def sep(self, a, b):
        """Centre-to-centre distance between two named bodies (inf if absent)."""
        if a in self.names and b in self.names:
            d = self.pos[self.names.index(a)] - self.pos[self.names.index(b)]
            return float(xp.sqrt((d * d).sum()))
        return np.inf


# ----------------------------------------------------------------------
# High-accuracy N-body integrator (SciPy DOP853), used only for planning
# the scenario and for the "ghost Mars" comparison run.
# ----------------------------------------------------------------------
def nbody_rhs_factory(masses, mask):
    masses = np.asarray(masses)
    n = len(masses)

    def rhs(_, y):
        p = y[:2 * n].reshape(n, 2)
        v = y[2 * n:].reshape(n, 2)
        d = p[:, None, :] - p[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)
        r = np.where(mask > 0, r, np.inf)          # masked pairs feel nothing
        a = -G * (masses[None, :, None] * d / r[..., None] ** 3).sum(axis=1)
        return np.r_[v.ravel(), a.ravel()]
    return rhs


def nbody_integrate(pos, vel, masses, t_span, mask=None, t_eval=None):
    n = len(masses)
    mask = np.ones((n, n)) if mask is None else mask
    sol = solve_ivp(nbody_rhs_factory(masses, mask), t_span, np.r_[pos.ravel(), vel.ravel()],
                    method="DOP853", rtol=1e-12, atol=1e-16, t_eval=t_eval)
    return sol


def kepler(r0, v0, mu, t, event_r=None):
    """Two-body orbit about a fixed mass; optional stop when |r| reaches event_r."""
    def rhs(_, y):
        return np.r_[y[2:], -mu * y[:2] / np.linalg.norm(y[:2]) ** 3]
    events = None
    if event_r is not None:
        ev = lambda _, y: np.linalg.norm(y[:2]) - event_r
        ev.terminal = True
        events = ev
    sol = solve_ivp(rhs, (0.0, t), np.r_[r0, v0], method="DOP853",
                    rtol=1e-12, atol=1e-15, events=events)
    if event_r is not None and sol.t_events[0].size:
        return sol.t_events[0][0], sol.y_events[0][0][:2], sol.y_events[0][0][2:]
    return sol.t[-1], sol.y[:2, -1], sol.y[2:, -1]


def state_on_ellipse(point, a, e, mu, outbound=True):
    """Velocity of a body on the ellipse (a, e) that passes through `point`."""
    r = np.linalg.norm(point)
    p = a * (1 - e ** 2)
    f = np.arccos(np.clip((p / r - 1) / e, -1, 1)) * (1 if outbound else -1)
    th = np.arctan2(point[1], point[0])
    rhat, that = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    k = np.sqrt(mu / p)
    return k * e * np.sin(f) * rhat + k * (1 + e * np.cos(f)) * that


# ======================================================================
# 1. DESIGN THE SCENARIO
# ======================================================================
print(f"Radii: Earth {R_EARTH*AU_KM:,.0f} km, Mars {R_MARS*AU_KM:,.0f} km, "
      f"Theia {R_THEIA*AU_KM:,.0f} km, Sun {R_SUN*AU_KM:,.0f} km")

mu_E, mu_M, mu_T = (G * (M_SUN + m) for m in (M_EARTH, M_MARS, M_THEIA))

# --- 1a. The Earth impact we want -------------------------------------
HIT_POINT = np.array([1.0, 0.0])
V_EARTH_HIT = np.array([0.0, np.sqrt(mu_E)])          # 29.8 km/s, circular
DV_THEIA = np.array([-1.20, 0.60])                    # AU/yr vs Earth (6.4 km/s)
V_THEIA_HIT = V_EARTH_HIT + DV_THEIA

# --- 1b. Where was Theia when it crossed Mars's orbit? -----------------
R_CROSS = 1.50                                        # AU (Mars ranges 1.38-1.67)
t_ev, P_CROSS, V_CROSS = kepler(HIT_POINT, V_THEIA_HIT, mu_T, -2.0, event_r=R_CROSS)
DT_CROSS = -t_ev                                      # flyby -> impact travel time
T_FLYBY = 0.30                                        # flyby happens 0.3 yr in
T_HIT = T_FLYBY + DT_CROSS
print(f"Planned: Mars flyby at t={T_FLYBY:.3f} yr, Earth impact at t={T_HIT:.3f} yr "
      f"({DT_CROSS*365.25:.0f} days later)")

# --- 1c. Put Mars on its real orbit, B_FLYBY to the side of Theia's path
B_FLYBY = 10.0 * R_MARS          # planned closest approach (straight-line)
MARS_SIDE = +1                   # +1: Theia passes BEHIND Mars -> Mars slows down
V_MARS = np.array([-np.sin(np.arctan2(*P_CROSS[::-1])),
                   np.cos(np.arctan2(*P_CROSS[::-1]))]) * np.sqrt(mu_M / R_CROSS)
for _ in range(5):               # n_hat depends on Mars's velocity -> iterate
    v_rel = V_CROSS - V_MARS
    n_hat = np.array([-v_rel[1], v_rel[0]]) / np.linalg.norm(v_rel)
    if MARS_SIDE * (n_hat @ V_MARS) < 0:
        n_hat = -n_hat
    P_MARS = P_CROSS + B_FLYBY * n_hat
    V_MARS = state_on_ellipse(P_MARS, A_MARS, E_MARS, mu_M, outbound=True)

# --- 1d. Full N-body BACKWARDS from the impact to t = 0 ----------------
#     Earth-Theia gravity is switched off while aiming: they overlap at
#     the impact instant. The real run has it on, which only bends Theia
#     further INTO the Earth (gravitational focusing).
m_arr = np.array([MASS[n] for n in NAMES])
AIM_MASK = np.ones((4, 4)); AIM_MASK[1, 3] = AIM_MASK[3, 1] = 0.0


def backward_plan(V_mars_flyby):
    """Mars state at the flyby -> state at impact -> whole system backwards to t=0."""
    _, p_mh, v_mh = kepler(P_MARS, V_mars_flyby, mu_M, DT_CROSS)
    pos_hit = np.array([[0, 0], HIT_POINT, p_mh, HIT_POINT], float)
    vel_hit = np.array([[0, 0], V_EARTH_HIT, v_mh, V_THEIA_HIT], float)
    vel_hit[0] = -(m_arr[1:, None] * vel_hit[1:]).sum(0) / M_SUN
    sol = solve_ivp(nbody_rhs_factory(m_arr, AIM_MASK), (T_HIT, 0.0),
                    np.r_[pos_hit.ravel(), vel_hit.ravel()], method="DOP853",
                    rtol=1e-12, atol=1e-16, dense_output=True)
    return sol


def mars_pre_flyby(sol, lead=0.04):
    """Mars's undisturbed (pre-flyby) orbit, carried forward to the flyby time."""
    y = sol.sol(T_FLYBY - lead)
    p, v = y[:8].reshape(4, 2), y[8:].reshape(4, 2)
    _, pp, vp = kepler(p[2] - p[0], v[2] - v[0], mu_M, lead)
    return pp, vp


# The flyby kicks Mars, so the state AT the flyby is not the state Mars had
# before it. Iterate: adjust Mars's flyby-time velocity until its orbit
# BEFORE the encounter is the real one (a = 1.524 AU, e = 0.093).
for it in range(6):
    sol = backward_plan(V_MARS)
    P_pre, V_pre = mars_pre_flyby(sol)
    el_pre = orbital_elements(P_pre, V_pre, mu_M)
    print(f"  Mars pre-flyby orbit, iter {it}: a={el_pre['a']:.5f} AU  e={el_pre['e']:.5f}")
    if abs(el_pre["a"] - A_MARS) < 1e-5 and abs(el_pre["e"] - E_MARS) < 1e-5:
        break
    V_MARS = V_MARS + (state_on_ellipse(P_pre, A_MARS, E_MARS, mu_M,
                                        outbound=(P_pre @ V_pre) > 0) - V_pre)
POS0 = sol.y[:8, -1].reshape(4, 2)
VEL0 = sol.y[8:, -1].reshape(4, 2)


def initial_state(s):
    """Initial state with Theia's starting speed scaled by (1+s); zero momentum."""
    pos, vel = POS0.copy(), VEL0.copy()
    vel[3] = vel[3] * (1.0 + s)
    vel[0] = -(m_arr[1:, None] * vel[1:]).sum(0) / M_SUN
    return pos, vel


H_SPH = R_EARTH
C_SOUND = 1.0 / KMS                    # 1 km/s
RHO0 = np.array([RHO[n] for n in NAMES])
DT_MAX = 5e-5                          # yr (~26 min) -- longest step


def make_system(pos, vel, aiming=False):
    return SPHSystem(pos, vel, m_arr, RHO0, NAMES, h=H_SPH, k_gas=C_SOUND ** 2,
                     gamma=1.0, alpha=0.5, beta=1.0, G=G,
                     exclude_pairs=[("Earth", "Theia")] if aiming else (),
                     allow_merge=not aiming)


# --- 1e. Shooting: aim with the REAL integrator -------------------------
def signed_miss(s):
    """Run the real integrator (merges off, Earth-Theia gravity off) and
    return Theia's signed miss distance from the Earth's centre."""
    sys_a = make_system(*initial_state(s), aiming=True)
    t, best = 0.0, (np.inf, None)
    while t < T_HIT + 0.03:
        dt = sys_a.choose_dt(DT_MAX)
        sys_a.step(dt)
        t += dt
        d = sys_a.sep("Earth", "Theia")
        if d < best[0]:
            p, v = to_host(sys_a.pos), to_host(sys_a.vel)
            best = (d, (p[3] - p[1], v[3] - v[1]))
        elif best[0] < 0.02 and d > best[0] + 1e-3:   # closest approach is behind us
            break
    r_rel, v_rel = best[1]
    return float(r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]) / np.linalg.norm(v_rel)


print("Aiming through the Mars flyby (shooting with the real integrator):")
t0 = time.time()
s_prev, m_prev = 0.0, signed_miss(0.0)
print(f"  iter 0: s={0.0:+.3e}  miss = {m_prev/R_EARTH:+10.2f} Earth radii")
s_cur = 1e-6
for it in range(1, 8):
    m_cur = signed_miss(s_cur)
    print(f"  iter {it}: s={s_cur:+.3e}  miss = {m_cur/R_EARTH:+10.2f} Earth radii")
    if abs(m_cur) < 0.05 * R_EARTH:
        break
    s_prev, m_prev, s_cur = s_cur, m_cur, s_cur - m_cur * (s_cur - s_prev) / (m_cur - m_prev)
S_AIM = s_cur
print(f"  aimed in {time.time()-t0:.1f}s; starting-speed correction = {S_AIM:+.2e}")

# ======================================================================
# 2. MAIN RUN
# ======================================================================
T_TOTAL = 3.0
OUTPUT_DT = 1.0 * DAY
SLOW_MARS = 60 * R_MARS          # slow motion when Theia is within 60 Mars radii
SLOW_EARTH = 40 * R_EARTH        # ...or within 40 Earth radii of the Earth
DT_SLOWMO = 1e-5
POST_SLOWMO = 0.0015

pos0, vel0 = initial_state(S_AIM)
sys_ = make_system(pos0, vel0)
p0 = (m_arr[:, None] * vel0).sum(0)

snaps, merge_records = [], []


def record(t, slow):
    snaps.append(dict(t=t, pos=to_host(sys_.pos), vel=to_host(sys_.vel),
                      mass=to_host(sys_.mass), radii=to_host(sys_.particle_radii()),
                      names=list(sys_.names), slow=slow))


t, next_out, n_steps = 0.0, OUTPUT_DT, 0
record(0.0, False)
t_start = time.time()
while t < T_TOTAL - 1e-12:
    t_m = merge_records[0]["t"] if merge_records else None
    slow = (sys_.sep("Theia", "Mars") < SLOW_MARS or sys_.sep("Theia", "Earth") < SLOW_EARTH
            or (t_m is not None and t - t_m < POST_SLOWMO))
    dt = sys_.choose_dt(DT_MAX)
    if slow:
        dt = min(dt, DT_SLOWMO)
    hit_output = t + dt >= next_out
    if hit_output:
        dt = next_out - t
    prev = len(sys_.merge_events)
    sys_.step(dt)
    t += dt
    n_steps += 1
    for ev in sys_.merge_events[prev:]:
        merge_records.append({"t": t, **ev})
        print(f"  MERGE at t={t:.5f} yr (day {t/DAY:.2f}): "
              f"{ev['parents'][0]} + {ev['parents'][1]} -> {ev['name']}")
    if hit_output:
        record(t, False)
        next_out += OUTPUT_DT
    elif slow:
        record(t, True)
print(f"Main run: {n_steps} steps, {len(snaps)} snapshots, {time.time()-t_start:.1f}s")

# ---- ghost run: identical start, but no impactor ----------------------
T_ARR = np.array([s["t"] for s in snaps])
g_pos, g_vel = pos0[:3].copy(), vel0[:3].copy()
g_vel[0] = -(m_arr[1:3, None] * g_vel[1:]).sum(0) / M_SUN
gsol = nbody_integrate(g_pos, g_vel, m_arr[:3], (0.0, T_ARR[-1]), t_eval=T_ARR)
GHOST = gsol.y[:6].T.reshape(-1, 3, 2)
GHOST_V = gsol.y[6:].T.reshape(-1, 3, 2)

# ======================================================================
# 3. DIAGNOSTICS
# ======================================================================
p_final = to_host((sys_.mass[:, None] * sys_.vel).sum(axis=0))
print(f"Total momentum: initial={p0}  final={p_final}")
print(f"Total mass:     initial={m_arr.sum():.12f}  final={float(sys_.mass.sum()):.12f}")


def body(k, name):
    s = snaps[k]
    if name not in s["names"]:
        return None, None
    i = s["names"].index(name)
    return s["pos"][i], s["vel"][i]


def rel_sun(k, name):
    p, v = body(k, name)
    ps, vs = body(k, "Sun")
    return (p - ps, v - vs) if p is not None else (None, None)


MARS_EL = [orbital_elements(*rel_sun(k, "Mars"), mu_M) for k in range(len(snaps))]
GHOST_EL = [orbital_elements(GHOST[k, 2] - GHOST[k, 0], GHOST_V[k, 2] - GHOST_V[k, 0], mu_M)
            for k in range(len(snaps))]
SEP_TM = np.array([np.linalg.norm(body(k, "Theia")[0] - body(k, "Mars")[0])
                   if "Theia" in snaps[k]["names"] else np.nan for k in range(len(snaps))])
SEP_TE = np.array([np.linalg.norm(body(k, "Theia")[0] - body(k, "Earth")[0])
                   if "Theia" in snaps[k]["names"] else np.nan for k in range(len(snaps))])

k_fly = int(np.nanargmin(SEP_TM))
t_fly = T_ARR[k_fly]
q_fly = SEP_TM[k_fly]
v_fly = np.linalg.norm(body(k_fly, "Theia")[1] - body(k_fly, "Mars")[1]) * KMS
k_after = int(np.searchsorted(T_ARR, t_fly + 0.05))
el_b, el_a = MARS_EL[0], MARS_EL[k_after]
dv_mars = np.linalg.norm(rel_sun(k_after, "Mars")[1] - (GHOST_V[k_after, 2] - GHOST_V[k_after, 0]))
print(f"Mars flyby at t={t_fly:.4f} yr (day {t_fly/DAY:.1f}): closest approach "
      f"{q_fly*AU_KM:,.0f} km = {q_fly/R_MARS:.1f} Mars radii, relative speed {v_fly:.2f} km/s")
print(f"Mars orbit before: a={el_b['a']:.4f} AU  e={el_b['e']:.4f}  "
      f"peri={el_b['peri']:.4f}  apo={el_b['apo']:.4f}  T={el_b['period']*365.25:.1f} d")
print(f"Mars orbit after:  a={el_a['a']:.4f} AU  e={el_a['e']:.4f}  "
      f"peri={el_a['peri']:.4f}  apo={el_a['apo']:.4f}  T={el_a['period']*365.25:.1f} d")
print(f"Mars velocity change vs no-flyby Mars, 0.05 yr later: {dv_mars*KMS*1000:.0f} m/s")
i_end = len(snaps) - 1
gap_end = np.linalg.norm(rel_sun(i_end, "Mars")[0] - (GHOST[i_end, 2] - GHOST[i_end, 0]))
print(f"At the end, Mars is {gap_end:.3f} AU ({gap_end*AU_KM/1e6:.0f} million km) "
      f"from where it would have been")

MERGED, merged_el = None, None
if merge_records:
    ev = merge_records[0]
    MERGED = ev["name"]
    va, vb = ev["parent_vel"]
    Ra, Rb = ev["parent_radii"]
    v_imp = np.linalg.norm(va - vb) * KMS
    e_heat = (ev["ke_before"] - ev["ke_after"]) * JOULE
    R_new = (Ra ** 3 + Rb ** 3) ** (1 / 3)
    k0 = next(i for i, s in enumerate(snaps) if MERGED in s["names"])
    merged_el = orbital_elements(*rel_sun(k0, MERGED), G * (M_SUN + ev["mass"]))
    print(f"Earth impact at t={ev['t']:.4f} yr (day {ev['t']/DAY:.1f}), speed {v_imp:.2f} km/s, "
          f"heat {e_heat:.2e} J")
    print(f"Merged body: {ev['mass']/M_EARTH:.3f} Earth masses, radius {R_new*AU_KM:,.0f} km; "
          f"orbit a={merged_el['a']:.4f} AU e={merged_el['e']:.4f} "
          f"peri={merged_el['peri']:.4f} apo={merged_el['apo']:.4f}")
else:
    print("No Earth impact -- adjust DV_THEIA / B_FLYBY.")

# ======================================================================
# 4. ANIMATION
# ======================================================================
COL = {"Sun": "#ffcc33", "Earth": "#3b82f6", "Mars": "#e4572e", "Theia": "#2ec4b6",
       "ghost": "#9a9aa8"}
if MERGED:
    COL[MERGED] = "#b46be0"
LABEL = {"Sun": "Sun", "Earth": "Earth", "Mars": "Mars", "Theia": "Impactor", MERGED: "Earth+Impactor"}
BODIES = [n for n in ("Earth", "Mars", "Theia", MERGED) if n]


def track(name):
    out = np.full((len(snaps), 2), np.nan)
    for k, s in enumerate(snaps):
        if name in s["names"]:
            out[k] = s["pos"][s["names"].index(name)]
    return out


TRACK = {n: track(n) for n in BODIES}
SUN_T = track("Sun")

# frames: every 3rd day + 140 evenly-timed frames per slow-motion segment
FRAME_EVERY, N_SLOW = 3, 140
reg = [k for k, s in enumerate(snaps) if not s["slow"]][::FRAME_EVERY]
slow_idx = [k for k, s in enumerate(snaps) if s["slow"]]
segments, cur = [], []
for k in slow_idx:
    if cur and T_ARR[k] - T_ARR[cur[-1]] > 0.01:
        segments.append(cur); cur = []
    cur.append(k)
if cur:
    segments.append(cur)
slow_sel, SLOW_FACTOR = set(), {}
for seg in segments:
    ts = T_ARR[seg]
    for x in np.linspace(ts[0], ts[-1], N_SLOW):
        slow_sel.add(seg[int(np.argmin(np.abs(ts - x)))])
    f = FRAME_EVERY * DAY / ((ts[-1] - ts[0]) / N_SLOW)
    for k in seg:
        SLOW_FACTOR[k] = f
if MERGED:
    k_hit = next(i for i, s in enumerate(snaps) if MERGED in s["names"])
    slow_sel |= {k_hit - 1, k_hit}
FRAMES = sorted(set(reg) | slow_sel)
print(f"Animation: {len(FRAMES)} frames, {len(segments)} slow-motion segments")
T_SWITCH = 0.5 * (t_fly + (merge_records[0]["t"] if merge_records else T_HIT))

plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(15.5, 6.8))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05], hspace=0.42, wspace=0.24,
                      left=0.04, right=0.97, top=0.89, bottom=0.09)
ax = fig.add_subplot(gs[:, 0])
axz = fig.add_subplot(gs[:, 1])
ax_m = fig.add_subplot(gs[0, 2])
ax_s = fig.add_subplot(gs[1, 2])
fig.suptitle("True scale: an impactor bound for Earth swings past Mars, knocks it onto a new orbit, "
             "then hits the Earth", fontsize=11.5, fontweight="bold")

# ---- Panel 1: inner solar system ------------------------------------
LIM = 1.85
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
ax.set_facecolor("#0b0b1a")
ax.set_title("Inner solar system, TRUE SCALE (AU)")
ax.set_xlabel("x (AU)   |   discs drawn true size (Sun < 2 px, planets << 1 px); rings mark positions",
              fontsize=7.5)
ax.plot(*ellipse_xy(GHOST_EL[0]).T, ":", color=COL["Mars"], lw=0.8, alpha=0.6,
        label="Mars orbit before flyby")
ax.plot(*ellipse_xy(dict(a=1.0, e=0.0, omega=0.0)).T, ":", color=COL["Earth"], lw=0.7, alpha=0.5)
mars_new_line, = ax.plot([], [], "--", color=COL["Mars"], lw=1.0, label="Mars orbit after flyby")
merged_line, = ax.plot([], [], "--", color=COL.get(MERGED, "w"), lw=0.9,
                       label="Earth+Impactor orbit")
sun_disc = ax.add_patch(Circle((0, 0), R_SUN, color=COL["Sun"], zorder=6))
discs = {n: ax.add_patch(Circle((np.nan, np.nan), 0.0, color=COL[n], zorder=6)) for n in BODIES}
rings = {n: ax.scatter([], [], s=80, facecolors="none", edgecolors=COL[n], linewidths=1.2,
                       zorder=5) for n in BODIES + ["Sun"]}
tags = {n: ax.text(0, 0, LABEL[n], color=COL[n], fontsize=8, zorder=7) for n in BODIES + ["Sun"]}
trails = {n: ax.plot([], [], "-", color=COL[n], lw=0.9, alpha=0.7)[0] for n in BODIES}
ghost_ring = ax.scatter([], [], s=80, facecolors="none", edgecolors=COL["ghost"],
                        linewidths=1.0, linestyle="--", zorder=5)
ghost_tag = ax.text(0, 0, "Mars if no flyby", color=COL["ghost"], fontsize=7.5)
ghost_trail, = ax.plot([], [], "-", color=COL["ghost"], lw=0.8, alpha=0.6)
info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", color="white",
               family="monospace", fontsize=8.5)
ax.legend(loc="upper right", fontsize=7, facecolor="#222", labelcolor="white")
MARS_NEW_XY = ellipse_xy(el_a)
MERGED_XY = ellipse_xy(merged_el) if merged_el and merged_el["bound"] else None

# ---- Panel 2: close-up (Mars during the flyby, Earth for the impact) --
axz.set_aspect("equal")
axz.set_facecolor("#05050d")
z_title = axz.set_title("")
planet_disc = axz.add_patch(Circle((0, 0), 1.0, zorder=5))
theia_disc = axz.add_patch(Circle((np.nan, np.nan), 1.0, color=COL["Theia"], zorder=5))
outline = axz.add_patch(Circle((0, 0), 1.0, fill=False, ls="--", lw=0.8, ec=COL["Earth"],
                               alpha=0.0, zorder=6))
flash = axz.add_patch(Circle((0, 0), 1.0, fill=False, ec="orange", lw=2, alpha=0.0, zorder=7))
rel_trail, = axz.plot([], [], "-", color=COL["Theia"], lw=1.0, alpha=0.8)
z_text = axz.text(0.02, 0.98, "", transform=axz.transAxes, va="top", color="white",
                  family="monospace", fontsize=8.5)
sun_arrow = axz.annotate("", xy=(0.5, 0.5), xytext=(0.5, 0.5), xycoords="axes fraction",
                         textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="->", color=COL["Sun"], lw=1.4))
sun_txt = axz.text(0.5, 0.5, "to Sun", transform=axz.transAxes, color=COL["Sun"],
                   fontsize=8, ha="center", va="center")

# ---- Panel 3: Mars's distance from the Sun, actual vs no-flyby -------
DAYS = T_ARR / DAY
r_act = np.array([np.linalg.norm(rel_sun(k, "Mars")[0]) for k in range(len(snaps))])
r_gh = np.linalg.norm(GHOST[:, 2] - GHOST[:, 0], axis=1)
ax_m.plot(DAYS, r_gh, ":", color=COL["ghost"], lw=1.4, label="Mars if no flyby")
ax_m.plot(DAYS, r_act, "-", color=COL["Mars"], lw=1.5, label="Mars, actual")
ax_m.set_ylabel("Mars distance from Sun (AU)")
ax_m.set_title("Mars's orbit: actual vs. if the impactor never came")
ax_m.set_xlim(0, DAYS[-1])
ax_m.legend(fontsize=7.5, loc="lower left")
ax_m.grid(alpha=0.25)
ax_m.text(0.99, 0.04,
          f"a: {el_b['a']:.3f} -> {el_a['a']:.3f} AU\n"
          f"e: {el_b['e']:.3f} -> {el_a['e']:.3f}\n"
          f"year: {el_b['period']*365.25:.0f} -> {el_a['period']*365.25:.0f} days",
          transform=ax_m.transAxes, ha="right", va="bottom", fontsize=7.5,
          family="monospace", bbox=dict(fc="white", ec="#ccc", alpha=0.9))
cur_m = ax_m.axvline(0, color="k", lw=0.8)

# ---- Panel 4: separations --------------------------------------------
ax_s.semilogy(DAYS, SEP_TM * AU_KM, color=COL["Mars"], lw=1.2, label="impactor - Mars")
ax_s.semilogy(DAYS, SEP_TE * AU_KM, color=COL["Earth"], lw=1.2, label="impactor - Earth")
ax_s.axvline(t_fly / DAY, color=COL["Mars"], ls="--", lw=0.9)
ax_s.text(t_fly / DAY, 1.4e3, " flyby", color=COL["Mars"], fontsize=7.5)
if merge_records:
    for a_ in (ax_m, ax_s):
        a_.axvline(merge_records[0]["t"] / DAY, color="orange", ls="--", lw=0.9)
    ax_s.text(merge_records[0]["t"] / DAY, 4e3, " impact", color="darkorange", fontsize=7.5)
ax_m.axvline(t_fly / DAY, color=COL["Mars"], ls="--", lw=0.9)
ax_s.set_xlabel("time (days)")
ax_s.set_ylabel("distance (km)")
ax_s.set_title("How close is the impactor?")
ax_s.set_xlim(0, DAYS[-1]); ax_s.set_ylim(1e3, None)
ax_s.legend(fontsize=7.5, loc="upper right")
ax_s.grid(alpha=0.25, which="both")
cur_s = ax_s.axvline(0, color="k", lw=0.8)

TRAIL = 150 * DAY


def update(k):
    s = snaps[k]
    t_now, nm = s["t"], s["names"]
    sun_p = s["pos"][nm.index("Sun")]
    sun_disc.center = tuple(sun_p)
    rings["Sun"].set_offsets([sun_p])
    tags["Sun"].set_position((sun_p[0] + 0.06, sun_p[1] + 0.06))
    lo = np.searchsorted(T_ARR, t_now - TRAIL)
    for n in BODIES:
        if n in nm:
            p = s["pos"][nm.index(n)]
            discs[n].center = tuple(p)
            discs[n].set_radius(s["radii"][nm.index(n)])
            rings[n].set_offsets([p])
            off = {"Theia": (0.07, -0.14), "Mars": (0.07, 0.07)}.get(n, (0.07, 0.07))
            tags[n].set_position((p[0] + off[0], p[1] + off[1]))
            tags[n].set_visible(True)
        else:
            rings[n].set_offsets(np.empty((0, 2)))
            tags[n].set_visible(False)
        seg = TRACK[n][lo:k + 1]
        trails[n].set_data(seg[:, 0], seg[:, 1])

    after_fly = t_now > t_fly + 0.01
    if after_fly:
        gp = GHOST[k, 2]
        ghost_ring.set_offsets([gp])
        ghost_tag.set_position((gp[0] + 0.07, gp[1] - 0.14))
        ghost_tag.set_visible(True)
        g_lo = np.searchsorted(T_ARR, max(t_fly, t_now - TRAIL))
        ghost_trail.set_data(GHOST[g_lo:k + 1, 2, 0], GHOST[g_lo:k + 1, 2, 1])
        mars_new_line.set_data(MARS_NEW_XY[:, 0] + sun_p[0], MARS_NEW_XY[:, 1] + sun_p[1])
    else:
        ghost_ring.set_offsets(np.empty((0, 2)))
        ghost_tag.set_visible(False)
        ghost_trail.set_data([], [])
        mars_new_line.set_data([], [])
    merged = MERGED is not None and MERGED in nm
    if merged and MERGED_XY is not None:
        merged_line.set_data(MERGED_XY[:, 0] + sun_p[0], MERGED_XY[:, 1] + sun_p[1])
    else:
        merged_line.set_data([], [])

    # ---- close-up
    mars_phase = t_now < T_SWITCH
    centre = "Mars" if mars_phase else ("Earth" if "Earth" in nm else MERGED)
    unit = R_MARS if mars_phase else R_EARTH
    uname = "Mars radii = 3,390 km" if mars_phase else "Earth radii = 6,371 km"
    c = s["pos"][nm.index(centre)]
    z_title.set_text(f"Close-up centred on {'Mars (flyby)' if mars_phase else 'the Earth (impact)'}")
    axz.set_xlabel(f"x ({uname})")
    planet_disc.set_radius(s["radii"][nm.index(centre)] / unit)
    planet_disc.set_color(COL[centre])
    theia_disc.set_radius(R_THEIA / unit)
    sep = (SEP_TM[k] if mars_phase else SEP_TE[k]) / unit
    if "Theia" in nm:
        lim = float(np.clip(1.25 * sep, 6.0 if mars_phase else 4.0, 90.0 if mars_phase else 45.0))
        theia_disc.center = tuple((s["pos"][nm.index("Theia")] - c) / unit)
        lo_z = np.searchsorted(T_ARR, t_now - 0.02)
        rel = (TRACK["Theia"][lo_z:k + 1] - TRACK[centre if centre != MERGED else "Earth"][lo_z:k + 1]) / unit
        rel_trail.set_data(rel[:, 0], rel[:, 1])
        who = "Mars" if mars_phase else "Earth"
        z_text.set_text(f"impactor: {sep:,.1f} {who} radii\n          ({sep*unit*AU_KM:,.0f} km)")
    else:
        lim = 4.0
        theia_disc.center = (np.nan, np.nan)
        rel_trail.set_data([], [])
        z_text.set_text(f"merged body radius {s['radii'][nm.index(centre)]*AU_KM:,.0f} km\n"
                        f"(dashed = Earth before impact)")
    axz.set_xlim(-lim, lim); axz.set_ylim(-lim, lim)
    outline.set_alpha(0.8 if merged else 0.0)
    u = (sun_p - c) / np.linalg.norm(sun_p - c)
    sun_arrow.xy = (0.5 + 0.30 * u[0], 0.5 + 0.30 * u[1])
    sun_arrow.set_position((0.5 + 0.19 * u[0], 0.5 + 0.19 * u[1]))
    sun_txt.set_position((0.5 + 0.36 * u[0], 0.5 + 0.36 * u[1]))
    if merge_records and 0 <= t_now - merge_records[0]["t"] < POST_SLOWMO:
        age = (t_now - merge_records[0]["t"]) / POST_SLOWMO
        flash.set_radius(1.2 + 3 * age); flash.set_alpha(1 - age)
    else:
        flash.set_alpha(0.0)

    # ---- status text
    if merged:
        status = "after impact"
    elif t_now < t_fly - 0.003:
        status = "impactor approaching Mars"
    elif t_now < t_fly + 0.003:
        status = "MARS FLYBY"
    elif "Theia" in nm and SEP_TE[k] < SLOW_EARTH:
        status = "IMPACT imminent"
    else:
        status = "deflected, heading for Earth"
    sm = f"SLOW MOTION x{SLOW_FACTOR[k]:,.0f}" if k in SLOW_FACTOR else ""
    info.set_text(f"day {t_now/DAY:8.2f} ({t_now:5.3f} yr)\nbodies = {len(nm)}\n{status}\n{sm}")
    cur_m.set_xdata([t_now / DAY] * 2)
    cur_s.set_xdata([t_now / DAY] * 2)
    return []


ani = animation.FuncAnimation(fig, lambda i: update(FRAMES[i]), frames=len(FRAMES),
                              interval=33, blit=False)
t0 = time.time()
OUT = "sph_mars_flyby_impact"
try:
    ani.save(f"{OUT}.mp4", writer=animation.FFMpegWriter(fps=30, bitrate=2400), dpi=90)
    print(f"Saved {OUT}.mp4 ({time.time()-t0:.0f}s)")
except Exception as exc:
    print(f"MP4 failed ({exc}); writing GIF with Pillow")
    ani.save(f"{OUT}.gif", writer=animation.PillowWriter(fps=30), dpi=70)
    print(f"Saved {OUT}.gif")

# stills: during the Mars flyby, just before impact, final frame
update(k_fly)                       # the moment of closest approach
fig.savefig(f"{OUT}_flyby.png", dpi=110)
if MERGED:
    near = [i for i in slow_idx if i < k_hit and 3.0 < SEP_TE[i] / R_EARTH < 7.0]
    update(near[-1] if near else k_hit - 1)
    fig.savefig(f"{OUT}_impact.png", dpi=110)
update(len(snaps) - 1)
fig.savefig(f"{OUT}_final.png", dpi=110)
print("Saved stills.")
