"""
REAL-SCALE 2D SPH + N-body simulation: the Sun, the Earth on its real
1 AU orbit, and a Mars-sized protoplanet ("Theia") on a crossing orbit.
Theia hits the Earth, the two merge (perfectly inelastic: mass, momentum
and volume conserved) and the merged body carries on around the Sun.

This is the "giant-impact" picture of how the Moon formed, run with real
numbers: real masses, real radii, the real Earth-Sun distance and real
orbital speeds.

Built on top of sph_orbit_collision_sim.py -- same SPH core:
  - Cubic spline kernel (Monaghan & Lattanzio 1985), 2D normalisation
  - Summation density, symmetrised pressure gradient (Monaghan 1992, Sec. 3)
  - Monaghan-Gingold artificial viscosity (Sec. 4)
  - Leapfrog (kick-drift-kick) integrator
  - Contact-based, momentum-conserving merge rule
  - Unsoftened Newtonian gravity between every pair of particles

What changed compared to the toy-scale version:
  1. ASTRONOMICAL UNITS. Length = 1 AU, time = 1 year, mass = 1 solar mass.
     In these units G = 4*pi^2 and the Earth's orbital period is exactly
     1 year. Every input is a real measured value converted into them.
  2. REAL SIZES. Each body gets its real 3D bulk density; its radius is
     R = (3 m / (4 pi rho))^(1/3), so Sun, Earth and Theia come out at
     their true radii (695,700 km, 6,371 km and 3,390 km). On a merge,
     VOLUME (not 2D area) is additive, as it is for real spheres.
  3. ADAPTIVE TIME STEP. At true scale the Earth is only 4.3e-5 AU wide,
     so a fixed step big enough to cover several years would let the
     planets pass straight through each other. The step now shrinks
     automatically as two bodies close in, and grows back afterwards.
  4. TWO-SCALE VISUALS. A true-scale solar-system view (where the Earth is
     far smaller than a pixel), an Earth-centred close-up that zooms in
     automatically during the impact, and the encounter in slow motion.
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# ----------------------------------------------------------------------
# Backend: CuPy on GPU if available, else NumPy. With only 3 particles
# NumPy will be faster -- the GPU path is kept so the code scales.
# ----------------------------------------------------------------------
try:
    import cupy as xp
    GPU = xp.cuda.is_available()
except Exception:
    GPU = False
if not GPU:
    import numpy as xp
print(f"[backend] using {'GPU (CuPy)' if GPU else 'CPU (NumPy)'} for array ops")


def to_host(a):
    """Independent NumPy copy of an array (device->host if needed)."""
    return xp.asnumpy(a) if GPU else np.array(a, copy=True)


# ======================================================================
# 0. UNITS AND REAL-WORLD CONSTANTS
# ======================================================================
AU_M = 1.495978707e11          # 1 astronomical unit in metres
YR_S = 365.25 * 86400.0        # 1 Julian year in seconds
DAY = 1.0 / 365.25             # 1 day in years
MSUN_KG = 1.98847e30           # solar mass in kg
G = 4.0 * np.pi ** 2           # gravitational constant in AU^3 / (Msun yr^2)

KMS_PER_CODE_V = AU_M / YR_S / 1e3                 # 1 AU/yr = 4.74 km/s
JOULE_PER_CODE_E = MSUN_KG * (AU_M / YR_S) ** 2    # energy unit in joules
KGM3_PER_CODE_RHO = MSUN_KG / AU_M ** 3            # density unit in kg/m^3

# Real masses (kg) and mean radii (km)
SUN_KG, SUN_KM = 1.98847e30, 695_700.0
EARTH_KG, EARTH_KM = 5.9722e24, 6_371.0
THEIA_KG, THEIA_KM = 6.4171e23, 3_389.5     # Theia modelled as Mars-sized

M_SUN = SUN_KG / MSUN_KG
M_EARTH = EARTH_KG / MSUN_KG                 # 3.0e-6 Msun
M_THEIA = THEIA_KG / MSUN_KG                 # 3.2e-7 Msun (0.107 Earths)
R_SUN = SUN_KM * 1e3 / AU_M                  # 0.00465 AU
R_EARTH = EARTH_KM * 1e3 / AU_M              # 4.26e-5 AU
R_THEIA = THEIA_KM * 1e3 / AU_M


def density_for(mass, radius):
    """3D bulk density that reproduces the real radius for a given mass."""
    return mass / (4.0 / 3.0 * np.pi * radius ** 3)


RHO_SUN = density_for(M_SUN, R_SUN)          # = 1408 kg/m^3
RHO_EARTH = density_for(M_EARTH, R_EARTH)    # = 5513 kg/m^3
RHO_THEIA = density_for(M_THEIA, R_THEIA)    # = 3934 kg/m^3


# ----------------------------------------------------------------------
# Cubic spline kernel (Monaghan & Lattanzio 1985), 2D normalisation
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
# Orbital elements of a body relative to the Sun (2-body approximation)
# ----------------------------------------------------------------------
def orbital_elements(r_rel, v_rel, mu):
    """
    r_rel, v_rel: position/velocity of the body relative to the Sun (2D).
    mu = G * (M_sun + m_body).
    Returns specific energy, semi-major axis a, eccentricity e, periapsis,
    apoapsis, period, whether the orbit is bound, and argument of periapsis.
    """
    r_rel = np.asarray(r_rel, float)
    v_rel = np.asarray(v_rel, float)
    r = np.linalg.norm(r_rel)
    v2 = v_rel @ v_rel
    eps = 0.5 * v2 - mu / r                          # specific orbital energy
    h_z = r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]  # specific angular momentum
    e = np.sqrt(max(0.0, 1.0 + 2.0 * eps * h_z ** 2 / mu ** 2))
    bound = eps < 0.0
    if bound:
        a = -mu / (2.0 * eps)
        peri, apo = a * (1.0 - e), a * (1.0 + e)
        period = 2.0 * np.pi * np.sqrt(a ** 3 / mu)
    else:
        a, apo, period = np.inf, np.inf, np.inf
        peri = h_z ** 2 / mu / (1.0 + e)
    e_vec = ((v2 - mu / r) * r_rel - (r_rel @ v_rel) * v_rel) / mu
    omega = np.arctan2(e_vec[1], e_vec[0])
    return dict(energy=eps, h=h_z, a=a, e=e, peri=peri, apo=apo,
                period=period, bound=bound, omega=omega)


# ----------------------------------------------------------------------
# SPH + gravity system
# ----------------------------------------------------------------------
class SPHSystem:
    def __init__(self, pos, vel, mass, rho_bulk, names, h, k_gas=1.0,
                 gamma=1.0, alpha=1.0, beta=2.0, G=0.0):
        self.pos = xp.asarray(pos, dtype=xp.float64)        # (N,2) AU
        self.vel = xp.asarray(vel, dtype=xp.float64)        # (N,2) AU/yr
        self.mass = xp.asarray(mass, dtype=xp.float64)      # (N,)  Msun
        # Per-particle 3D bulk density (Msun / AU^3) -> sets the true radius
        self.rho_bulk = xp.asarray(rho_bulk, dtype=xp.float64)
        self.names = list(names)                             # labels, follow merges
        self.h = h                                           # SPH smoothing length (AU)
        self.k_gas = k_gas
        self.gamma = gamma
        self.c_s = float(np.sqrt(gamma * k_gas))             # sound speed (AU/yr)
        self.alpha = alpha
        self.beta = beta
        self.G = G
        self.eta2 = 0.01 * h ** 2
        self.merge_events = []

    def _pairwise(self):
        rij = self.pos[:, None, :] - self.pos[None, :, :]
        r = xp.linalg.norm(rij, axis=-1)
        return rij, r

    def density(self, r):
        W = kernel_W(r, xp.full_like(r, self.h))
        return xp.maximum((self.mass[None, :] * W).sum(axis=1), 1e-30)

    def particle_radii(self):
        """True physical radius of a sphere: R = (3 m / (4 pi rho))^(1/3)."""
        return (3.0 * self.mass / (4.0 * xp.pi * self.rho_bulk)) ** (1.0 / 3.0)

    def pressure(self, rho):
        return self.k_gas * rho ** self.gamma

    def viscosity(self, rij, r, vij, rho):
        vij_dot_rij = (vij * rij).sum(axis=-1)
        approaching = vij_dot_rij < 0.0
        mu_ab = (self.h * vij_dot_rij) / (r ** 2 + self.eta2)
        rho_ab = 0.5 * (rho[:, None] + rho[None, :])
        Pi = (-self.alpha * self.c_s * mu_ab + self.beta * mu_ab ** 2) / rho_ab
        Pi = xp.where(approaching, Pi, 0.0)
        xp.fill_diagonal(Pi, 0.0)
        return Pi

    def accelerations(self):
        rij, r = self._pairwise()
        rho = self.density(r)
        P = self.pressure(rho)
        gradW = kernel_gradW(rij, r, xp.full_like(r, self.h))
        vij = self.vel[:, None, :] - self.vel[None, :, :]
        Pi = self.viscosity(rij, r, vij, rho)

        coeff = (P[None, :] / rho[None, :] ** 2
                 + P[:, None] / rho[:, None] ** 2
                 + Pi)
        xp.fill_diagonal(coeff, 0.0)
        acc = -(self.mass[None, :, None] * coeff[..., None] * gradW).sum(axis=1)

        if self.G != 0.0:
            r_safe = r.copy()
            xp.fill_diagonal(r_safe, xp.inf)
            inv_r3 = r_safe ** -3
            acc += -self.G * (self.mass[None, :, None] * rij * inv_r3[..., None]).sum(axis=1)
        return acc, rho

    def choose_dt(self, dt_max, eta_ff=0.02, eta_gap=0.3, dt_min=1e-10):
        """
        Adaptive time step. For every pair of bodies take the smaller of
          - eta_ff  * free-fall time  sqrt(r^3 / G(m_i+m_j))   (resolves the
            strong gravity of a close encounter), and
          - eta_gap * time-to-contact gap / |v_rel|            (never lets two
            bodies jump past each other inside one step).
        """
        if len(self.mass) < 2:
            return dt_max
        rij, r = self._pairwise()
        vij = self.vel[:, None, :] - self.vel[None, :, :]
        vrel = xp.linalg.norm(vij, axis=-1)
        radii = self.particle_radii()
        gap = xp.maximum(r - (radii[:, None] + radii[None, :]), 0.0)
        msum = self.mass[:, None] + self.mass[None, :]
        n = len(self.mass)
        iu = xp.triu_indices(n, 1)
        t_ff = xp.sqrt(r[iu] ** 3 / (self.G * msum[iu]))
        t_gap = gap[iu] / xp.maximum(vrel[iu], 1e-30)
        dt = min(dt_max, eta_ff * float(t_ff.min()), eta_gap * float(t_gap.min()))
        return max(dt, dt_min)

    def step(self, dt):
        acc, _ = self.accelerations()
        self.vel += 0.5 * dt * acc
        self.pos += dt * self.vel
        acc, rho = self.accelerations()
        self.vel += 0.5 * dt * acc
        self.merge()
        return rho

    def merge(self):
        """Contact-based merge; conserves mass, momentum and total VOLUME."""
        merged_any = True
        while merged_any and len(self.mass) > 1:
            merged_any = False
            _, r = self._pairwise()
            radii = self.particle_radii()
            gap = r - (radii[:, None] + radii[None, :])
            xp.fill_diagonal(gap, xp.inf)
            i, j = xp.unravel_index(xp.argmin(gap), gap.shape)
            i, j = int(i), int(j)
            if gap[i, j] < 0.0:
                if self.mass[j] > self.mass[i]:       # heavier body named first
                    i, j = j, i
                m_a, m_b = self.mass[i], self.mass[j]
                v_a, v_b = self.vel[i].copy(), self.vel[j].copy()
                m_new = float(m_a + m_b)
                v_new = (m_a * v_a + m_b * v_b) / m_new
                r_new = (m_a * self.pos[i] + m_b * self.pos[j]) / m_new
                # volume is additive -> derive the merged body's bulk density
                vol_new = m_a / self.rho_bulk[i] + m_b / self.rho_bulk[j]
                rho_new = m_new / float(vol_new)

                ke_before = float(0.5 * m_a * (v_a @ v_a) + 0.5 * m_b * (v_b @ v_b))
                ke_after = float(0.5 * m_new * (v_new @ v_new))
                name_new = f"{self.names[i]}+{self.names[j]}"

                self.merge_events.append({
                    "name": name_new,
                    "parents": (self.names[i], self.names[j]),
                    "mass": m_new,
                    "vel": to_host(v_new),
                    "pos": to_host(r_new),
                    "parent_mass": (float(m_a), float(m_b)),
                    "parent_vel": (to_host(v_a), to_host(v_b)),
                    "parent_radii": (float(radii[i]), float(radii[j])),
                    "ke_before": ke_before,
                    "ke_after": ke_after,
                })

                keep = [k for k in range(len(self.mass)) if k not in (i, j)]
                self.pos = xp.vstack([self.pos[keep], xp.asarray(r_new).reshape(1, 2)])
                self.vel = xp.vstack([self.vel[keep], xp.asarray(v_new).reshape(1, 2)])
                self.mass = xp.concatenate([self.mass[keep], xp.array([m_new])])
                self.rho_bulk = xp.concatenate([self.rho_bulk[keep], xp.array([rho_new])])
                self.names = [self.names[k] for k in keep] + [name_new]
                merged_any = True


# ======================================================================
# SCENARIO: Sun + Earth + Theia, all at real scale
# ======================================================================
print(f"Radii: Sun {R_SUN:.5f} AU ({SUN_KM:,.0f} km), "
      f"Earth {R_EARTH:.2e} AU ({EARTH_KM:,.0f} km), "
      f"Theia {R_THEIA:.2e} AU ({THEIA_KM:,.0f} km)")
print(f"Densities: Sun {RHO_SUN*KGM3_PER_CODE_RHO:.0f}, Earth "
      f"{RHO_EARTH*KGM3_PER_CODE_RHO:.0f}, Theia {RHO_THEIA*KGM3_PER_CODE_RHO:.0f} kg/m^3")
print(f"1 AU = {1/R_SUN:.0f} solar radii = {1/R_EARTH:,.0f} Earth radii")

# --- 1. Choose the collision itself ------------------------------------
R_HIT = 1.0                                   # impact at 1 AU, at (1, 0)
mu_E = G * (M_SUN + M_EARTH)
mu_T = G * (M_SUN + M_THEIA)
v_circ = np.sqrt(mu_E / R_HIT)                # 2*pi AU/yr = 29.8 km/s
V_EARTH_HIT = np.array([0.0, v_circ])         # Earth: circular, counter-clockwise
# Theia: a few km/s faster than the Earth and dropping inward, so it
# catches the Earth from behind and outside (relative speed ~3.2 km/s
# before the Earth's gravity speeds it up to ~10 km/s at contact).
DV_THEIA = np.array([-0.60, 0.30])            # AU/yr relative to the Earth
V_THEIA_HIT = V_EARTH_HIT + DV_THEIA
T_COLLIDE = 0.60                              # years after t=0 (about 219 days)


# --- 2. Integrate each body BACKWARD from the collision point ----------
def integrate_kepler(r0, v0, mu, t):
    """Two-body orbit around a fixed point mass at the origin. t<0 = backwards."""
    def rhs(_, y):
        r = y[:2]
        a = -mu * r / np.linalg.norm(r) ** 3
        return np.r_[y[2:], a]
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(rhs, (0.0, t), np.r_[r0, v0], method="DOP853",
                        rtol=1e-12, atol=1e-15)
        y = sol.y[:, -1]
        return y[:2], y[2:]
    except ImportError:                       # fall back to a fine leapfrog
        r, v = np.array(r0, float), np.array(v0, float)
        dt = 1e-6
        n = int(round(abs(t) / dt))
        d = np.sign(t) * dt
        for _ in range(n):
            v += 0.5 * d * rhs(0, np.r_[r, v])[2:]
            r += d * v
            v += 0.5 * d * rhs(0, np.r_[r, v])[2:]
        return r, v


hit_point = np.array([R_HIT, 0.0])
rE0, vE0 = integrate_kepler(hit_point, V_EARTH_HIT, mu_E, -T_COLLIDE)
rT0, vT0 = integrate_kepler(hit_point, V_THEIA_HIT, mu_T, -T_COLLIDE)

# --- 3. Sun at the origin with a tiny counter-velocity so total momentum
#        is zero (otherwise the whole system slowly drifts). ------------
v_sun0 = -(M_EARTH * vE0 + M_THEIA * vT0) / M_SUN

pos0 = np.array([[0.0, 0.0], rE0, rT0])
vel0 = np.array([v_sun0, vE0, vT0])
mass0 = np.array([M_SUN, M_EARTH, M_THEIA])
rho0 = np.array([RHO_SUN, RHO_EARTH, RHO_THEIA])
names0 = ["Sun", "Earth", "Theia"]

# SPH parameters, scaled to the planets: smoothing length = 1 Earth radius,
# "sound speed" sqrt(k_gas) = 1 km/s. Because there is one particle per
# body, pressure/viscosity only act in the last ~half Earth radius before
# contact -- orbital motion is pure gravity.
H_SPH = R_EARTH
C_SOUND = 1.0 / KMS_PER_CODE_V                # 1 km/s in AU/yr
sys_ = SPHSystem(pos0, vel0, mass0, rho0, names0,
                 h=H_SPH, k_gas=C_SOUND ** 2, gamma=1.0, alpha=0.5, beta=1.0, G=G)

el_E = orbital_elements(rE0, vE0 - v_sun0, mu_E)
el_T = orbital_elements(rT0, vT0 - v_sun0, mu_T)
for nm, el in (("Earth", el_E), ("Theia", el_T)):
    print(f"{nm:6s} before: a={el['a']:.4f} AU  e={el['e']:.4f}  "
          f"peri={el['peri']:.4f}  apo={el['apo']:.4f} AU  "
          f"T={el['period']*365.25:.1f} days")

# --- 4. Run ------------------------------------------------------------
T_TOTAL = 3.0                 # years (~2.4 orbits after the impact)
DT_MAX = 2e-4                 # largest step: ~1.75 hours
OUTPUT_DT = 1.0 * DAY         # regular snapshot every day
SLOWMO_SEP = 40 * R_EARTH     # record every step when Earth-Theia < 40 R_E
DT_SLOWMO = 1e-5              # ...and cap the step at ~5 minutes there
POST_SLOWMO = 0.0015          # keep slow-motion for ~13 h after impact

p0 = (mass0[:, None] * vel0).sum(axis=0)
m0_total = mass0.sum()

snaps = []                    # list of dicts: t, pos, vel, mass, radii, names, slow


def record(t, slow):
    snaps.append(dict(t=t, pos=to_host(sys_.pos), vel=to_host(sys_.vel),
                      mass=to_host(sys_.mass), radii=to_host(sys_.particle_radii()),
                      names=list(sys_.names), slow=slow))


def earth_theia_sep():
    if "Earth" in sys_.names and "Theia" in sys_.names:
        p = to_host(sys_.pos)
        return float(np.linalg.norm(p[sys_.names.index("Earth")]
                                    - p[sys_.names.index("Theia")]))
    return np.inf


merge_records = []
t, next_out, n_steps = 0.0, OUTPUT_DT, 0
record(0.0, False)
t_start = time.time()
while t < T_TOTAL - 1e-12:
    t_merge = merge_records[0]["t"] if merge_records else None
    slow = (earth_theia_sep() < SLOWMO_SEP) or (
        t_merge is not None and t - t_merge < POST_SLOWMO)
    dt = sys_.choose_dt(DT_MAX)
    if slow:
        dt = min(dt, DT_SLOWMO)
    hit_output = False
    if t + dt >= next_out:                    # land exactly on the next snapshot
        dt, hit_output = next_out - t, True

    prev = len(sys_.merge_events)
    sys_.step(dt)
    t += dt
    n_steps += 1
    for ev in sys_.merge_events[prev:]:
        merge_records.append({"t": t, **ev})
        print(f"  MERGE at t={t:.6f} yr (day {t/DAY:.2f}): {ev['parents'][0]} + "
              f"{ev['parents'][1]} -> {ev['name']}")
    if hit_output:
        record(t, False)
        next_out += OUTPUT_DT
    elif slow:
        record(t, True)
    if n_steps % 5000 == 0:
        print(f"  step {n_steps:6d}  t={t:.3f} yr  N={len(sys_.mass)}  "
              f"elapsed={time.time()-t_start:5.1f}s")
print(f"Integration done: {n_steps} steps, {len(snaps)} snapshots, "
      f"{time.time()-t_start:.1f}s")

# --- 5. Diagnostics ----------------------------------------------------
p_final = to_host((sys_.mass[:, None] * sys_.vel).sum(axis=0))
print(f"Total momentum: initial={p0}  final={p_final}")
print(f"Total mass:     initial={m0_total:.12f}  final={float(sys_.mass.sum()):.12f}")

merged_el, MERGED = None, None
if merge_records:
    ev = merge_records[0]
    MERGED = ev["name"]
    va, vb = ev["parent_vel"]
    m_a, m_b = ev["parent_mass"]
    Ra, Rb = ev["parent_radii"]
    v_imp = np.linalg.norm(va - vb) * KMS_PER_CODE_V
    v_esc = np.sqrt(2 * G * (m_a + m_b) / (Ra + Rb)) * KMS_PER_CODE_V
    e_heat = (ev["ke_before"] - ev["ke_after"]) * JOULE_PER_CODE_E
    R_new = (Ra ** 3 + Rb ** 3) ** (1 / 3)
    print(f"Impact speed {v_imp:.2f} km/s (mutual escape speed {v_esc:.2f} km/s)")
    print(f"Kinetic energy turned into heat: {e_heat:.2e} J")
    print(f"Merged body: mass {ev['mass']/M_EARTH:.3f} Earths, "
          f"radius {R_new*AU_M/1e3:,.0f} km")

    # Orbit of the merged body from the first snapshot after the impact
    k0 = next(i for i, s in enumerate(snaps) if MERGED in s["names"])
    s = snaps[k0]
    js, jm = s["names"].index("Sun"), s["names"].index(MERGED)
    merged_el = orbital_elements(s["pos"][jm] - s["pos"][js],
                                 s["vel"][jm] - s["vel"][js],
                                 G * (s["mass"][js] + s["mass"][jm]))
    print(f"Merged orbit: a={merged_el['a']:.4f} AU  e={merged_el['e']:.4f}  "
          f"peri={merged_el['peri']:.4f}  apo={merged_el['apo']:.4f} AU  "
          f"T={merged_el['period']*365.25:.1f} days  bound={merged_el['bound']}")
    d = [np.linalg.norm(q["pos"][q["names"].index(MERGED)] - q["pos"][q["names"].index("Sun")])
         for q in snaps[k0:]]
    stable = max(d) < 3 * R_HIT and min(d) > R_SUN
    print(f"Measured after merge: r_min={min(d):.4f} AU  r_max={max(d):.4f} AU "
          f"-> {'STABLE bound orbit' if stable else 'not stable'}")
else:
    print("No collision happened -- try adjusting DV_THEIA / T_COLLIDE.")


# ======================================================================
# Animation
# ======================================================================
COLORS = {"Sun": "#ffcc33", "Earth": "#3b82f6", "Theia": "#e4572e"}
if MERGED:
    COLORS[MERGED] = "#b46be0"
BODY_LABEL = {"Sun": "Sun", "Earth": "Earth", "Theia": "Theia", MERGED: "Earth+Theia"}

T_ARR = np.array([s["t"] for s in snaps])


def track(name, ref="Sun"):
    """(n_snaps, 2) position history of `name` relative to `ref` (NaN if absent)."""
    out = np.full((len(snaps), 2), np.nan)
    for k, s in enumerate(snaps):
        if name in s["names"] and (ref is None or ref in s["names"]):
            origin = s["pos"][s["names"].index(ref)] if ref else 0.0
            out[k] = s["pos"][s["names"].index(name)] - origin
    return out


SUN_TRACK = track("Sun", ref=None)
TRACKS = {nm: track(nm, ref=None) for nm in ("Earth", "Theia", MERGED) if nm}
DIST = {nm: np.linalg.norm(track(nm), axis=1) for nm in TRACKS}
THEIA_REL = (TRACKS["Theia"] - TRACKS["Earth"]) / R_EARTH     # in Earth radii
SEP_RE = np.linalg.norm(THEIA_REL, axis=1)

# ---- choose animation frames: every 3rd day + an evenly-timed slow-mo
FRAME_EVERY = 3                                  # days per regular frame
N_SLOW = 160                                     # frames for the impact sequence
reg = [k for k, s in enumerate(snaps) if not s["slow"]][::FRAME_EVERY]
slow_idx = [k for k, s in enumerate(snaps) if s["slow"]]
slow_sel = []
if slow_idx:
    ts = T_ARR[slow_idx]
    targets = np.linspace(ts[0], ts[-1], N_SLOW)
    slow_sel = sorted({slow_idx[int(np.argmin(np.abs(ts - x)))] for x in targets})
    if merge_records:                            # make sure the contact frame is in
        k_hit = next(i for i, s in enumerate(snaps) if MERGED in s["names"])
        slow_sel = sorted(set(slow_sel) | {k_hit - 1, k_hit})
    SLOWMO_FACTOR = (FRAME_EVERY * DAY) / ((ts[-1] - ts[0]) / N_SLOW)
FRAMES = sorted(set(reg) | set(slow_sel))
print(f"Animation: {len(FRAMES)} frames ({len(slow_sel)} in slow motion)")

plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05], hspace=0.42, wspace=0.22,
                      left=0.04, right=0.985, top=0.89, bottom=0.09)
ax = fig.add_subplot(gs[:, 0])
axz = fig.add_subplot(gs[:, 1])
ax_r = fig.add_subplot(gs[0, 2])
ax_s = fig.add_subplot(gs[1, 2])
fig.suptitle("Giant impact at true scale: a Mars-sized Theia hits the Earth on its real 1 AU orbit",
             fontsize=12, fontweight="bold")

# ---- Panel 1: solar system at true scale ------------------------------
LIM = 1.35
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal")
ax.set_facecolor("#0b0b1a")
ax.set_title("Solar-system view, TRUE SCALE (AU)")
ax.set_xlabel("x (AU)")
sun_disc = Circle((0, 0), R_SUN, color=COLORS["Sun"], zorder=6)
ax.add_patch(sun_disc)
planet_discs = {nm: ax.add_patch(Circle((np.nan, np.nan), 0.0, color=COLORS[nm], zorder=6))
                for nm in TRACKS}
locators = {nm: ax.scatter([], [], s=90, facecolors="none", edgecolors=COLORS[nm],
                           linewidths=1.2, zorder=5) for nm in list(TRACKS) + ["Sun"]}
loc_text = {nm: ax.text(0, 0, BODY_LABEL[nm], color=COLORS[nm], fontsize=8, zorder=7)
            for nm in list(TRACKS) + ["Sun"]}
trail_lines = {nm: ax.plot([], [], "-", color=COLORS[nm], lw=0.9, alpha=0.65)[0]
               for nm in TRACKS}
ellipse_line, = ax.plot([], [], "--", color="white", lw=0.7, alpha=0.45,
                        label="predicted post-impact orbit")
ax.text(0.02, 0.02,
        "Filled discs are drawn at true size: the Sun is the\n"
        "tiny dot at the centre and the planets are far below\n"
        "one pixel. Rings only mark where the bodies are.",
        transform=ax.transAxes, color="#bbbbcc", fontsize=7.5, va="bottom")
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                    color="white", family="monospace", fontsize=9)
ax.legend(loc="upper right", fontsize=7.5, facecolor="#222", labelcolor="white")

if merged_el is not None and merged_el["bound"]:
    th = np.linspace(0, 2 * np.pi, 400)
    a_, e_, w_ = merged_el["a"], merged_el["e"], merged_el["omega"]
    rr = a_ * (1 - e_ ** 2) / (1 + e_ * np.cos(th))
    ELLIPSE_XY = np.c_[rr * np.cos(th + w_), rr * np.sin(th + w_)]
else:
    ELLIPSE_XY = None

# ---- Panel 2: Earth-centred close-up, units of Earth radii ------------
axz.set_aspect("equal")
axz.set_facecolor("#05050d")
axz.set_title("Close-up centred on the Earth (Earth radii, true scale)")
axz.set_xlabel("x (Earth radii = 6,371 km)")
earth_disc = axz.add_patch(Circle((0, 0), 1.0, color=COLORS["Earth"], zorder=5))
theia_disc = axz.add_patch(Circle((np.nan, np.nan), R_THEIA / R_EARTH,
                                  color=COLORS["Theia"], zorder=5))
old_outline = axz.add_patch(Circle((0, 0), 1.0, fill=False, ls="--", lw=0.8,
                                   ec=COLORS["Earth"], alpha=0.0, zorder=6))
theia_rel_trail, = axz.plot([], [], "-", color=COLORS["Theia"], lw=1.0, alpha=0.7)
zoom_text = axz.text(0.02, 0.98, "", transform=axz.transAxes, va="top",
                     color="white", family="monospace", fontsize=9)
sun_arrow = axz.annotate("", xy=(0.5, 0.5), xytext=(0.5, 0.5), xycoords="axes fraction",
                         textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="->", color=COLORS["Sun"], lw=1.4))
sun_arrow_txt = axz.text(0.5, 0.5, "to Sun", transform=axz.transAxes,
                         color=COLORS["Sun"], fontsize=8, ha="center", va="center")
flash = axz.add_patch(Circle((0, 0), 1.0, fill=False, ec="orange", lw=2, alpha=0.0, zorder=7))

# ---- Panel 3: distance from the Sun ----------------------------------
DAYS = T_ARR / DAY
for nm, d in DIST.items():
    if np.isfinite(d).any():
        ax_r.plot(DAYS, d, color=COLORS[nm], lw=1.2, label=BODY_LABEL[nm])
if merge_records:
    for a_ in (ax_r, ax_s):
        a_.axvline(merge_records[0]["t"] / DAY, color="orange", ls="--", lw=1)
ax_r.set_ylabel("distance from Sun (AU)")
ax_r.set_title("Orbital radius vs time")
ax_r.set_xlim(0, DAYS[-1])
ax_r.legend(fontsize=7.5, loc="lower right")
ax_r.grid(alpha=0.25)
cursor_r = ax_r.axvline(0, color="k", lw=0.8)

# ---- Panel 4: Earth-Theia separation (log) ---------------------------
ax_s.semilogy(DAYS, SEP_RE, color=COLORS["Theia"], lw=1.2, label="Earth-Theia distance")
ax_s.axhline((R_EARTH + R_THEIA) / R_EARTH, color="k", ls=":", lw=1,
             label="contact (1.53 R_E)")
ax_s.axhline(60.3, color="grey", ls="--", lw=0.8, label="Moon's distance today (60 R_E)")
hill = (M_EARTH / (3 * M_SUN)) ** (1 / 3) / R_EARTH
ax_s.axhline(hill, color="tab:green", ls="--", lw=0.8,
             label=f"Earth's Hill sphere ({hill:.0f} R_E)")
ax_s.set_xlabel("time (days)")
ax_s.set_ylabel("separation (Earth radii)")
ax_s.set_title("How close is Theia?")
ax_s.set_xlim(0, DAYS[-1])
ax_s.legend(fontsize=7, loc="upper right")
ax_s.grid(alpha=0.25, which="both")
cursor_s = ax_s.axvline(0, color="k", lw=0.8)

TRAIL_DAYS = 120


def update(k):
    s = snaps[k]
    t_now = s["t"]
    nm = s["names"]
    sun_p = s["pos"][nm.index("Sun")]
    sun_disc.center = tuple(sun_p)
    locators["Sun"].set_offsets([sun_p])
    loc_text["Sun"].set_position((sun_p[0] + 0.05, sun_p[1] + 0.05))

    lo = np.searchsorted(T_ARR, t_now - TRAIL_DAYS * DAY)
    for name in TRACKS:
        if name in nm:
            p = s["pos"][nm.index(name)]
            planet_discs[name].center = tuple(p)
            planet_discs[name].set_radius(s["radii"][nm.index(name)])
            locators[name].set_offsets([p])
            dx = 0.06 if name != "Theia" else 0.06
            dy = 0.06 if name != "Theia" else -0.11
            loc_text[name].set_position((p[0] + dx, p[1] + dy))
            loc_text[name].set_visible(True)
        else:
            locators[name].set_offsets(np.empty((0, 2)))
            loc_text[name].set_visible(False)
        seg = TRACKS[name][lo:k + 1]
        trail_lines[name].set_data(seg[:, 0], seg[:, 1])

    merged = MERGED is not None and MERGED in nm
    if merged and ELLIPSE_XY is not None:
        ellipse_line.set_data(ELLIPSE_XY[:, 0] + sun_p[0], ELLIPSE_XY[:, 1] + sun_p[1])
    else:
        ellipse_line.set_data([], [])

    # ---- close-up panel
    centre_name = "Earth" if "Earth" in nm else MERGED
    c = s["pos"][nm.index(centre_name)]
    sep = SEP_RE[k] if np.isfinite(SEP_RE[k]) else 0.0
    lim = float(np.clip(1.25 * sep, 4.0, 45.0)) if not merged else 4.0
    axz.set_xlim(-lim, lim); axz.set_ylim(-lim, lim)
    earth_disc.set_radius(s["radii"][nm.index(centre_name)] / R_EARTH)
    earth_disc.set_color(COLORS[centre_name])
    old_outline.set_alpha(0.8 if merged else 0.0)
    if "Theia" in nm:
        rel = (s["pos"][nm.index("Theia")] - c) / R_EARTH
        theia_disc.center = tuple(rel)
        lo_z = np.searchsorted(T_ARR, t_now - 0.02)
        theia_rel_trail.set_data(THEIA_REL[lo_z:k + 1, 0], THEIA_REL[lo_z:k + 1, 1])
        km = sep * EARTH_KM
        zoom_text.set_text(f"Theia: {sep:,.1f} R_E away\n       ({km:,.0f} km)")
    else:
        theia_disc.center = (np.nan, np.nan)
        theia_rel_trail.set_data([], [])
        R_now = s["radii"][nm.index(centre_name)] * AU_M / 1e3
        zoom_text.set_text(f"Merged body: radius {R_now:,.0f} km\n"
                           f"(dashed = Earth before impact)")
    u = sun_p - c
    u = u / np.linalg.norm(u)
    # small arrow just outside the planet, pointing towards the Sun
    sun_arrow.xy = (0.5 + 0.30 * u[0], 0.5 + 0.30 * u[1])
    sun_arrow.set_position((0.5 + 0.19 * u[0], 0.5 + 0.19 * u[1]))
    sun_arrow_txt.set_position((0.5 + 0.36 * u[0], 0.5 + 0.36 * u[1]))
    if merge_records:
        age = t_now - merge_records[0]["t"]
        if 0 <= age < POST_SLOWMO:
            flash.set_radius(1.2 + 3.0 * age / POST_SLOWMO)
            flash.set_alpha(max(0.0, 1 - age / POST_SLOWMO))
        else:
            flash.set_alpha(0.0)

    # ---- text + cursors
    if merged:
        status = "after impact: merged body orbits on"
    elif "Theia" in nm and sep < 45:
        status = "Theia falling onto Earth"
    else:
        status = "approaching..."
    sm = f"SLOW MOTION x{SLOWMO_FACTOR:,.0f}" if s["slow"] else ""
    info_text.set_text(f"day {t_now/DAY:8.2f}  ({t_now:5.3f} yr)\n"
                       f"bodies = {len(nm)}\n{status}\n{sm}")
    cursor_r.set_xdata([t_now / DAY] * 2)
    cursor_s.set_xdata([t_now / DAY] * 2)
    return []


ani = animation.FuncAnimation(fig, lambda i: update(FRAMES[i]), frames=len(FRAMES),
                              interval=33, blit=False)
t0 = time.time()
out_mp4 = "sph_orbit_collision_real_scale.mp4"
try:
    ani.save(out_mp4, writer=animation.FFMpegWriter(fps=30, bitrate=2400), dpi=90)
    print(f"Saved animation to {out_mp4}  ({time.time()-t0:.0f}s)")
except Exception as exc:                       # no ffmpeg -> GIF instead
    print(f"MP4 failed ({exc}); writing GIF instead")
    ani.save("sph_orbit_collision_real_scale.gif",
             writer=animation.PillowWriter(fps=30), dpi=70)
    print("Saved animation to sph_orbit_collision_real_scale.gif")

# Static summary images: the moment just before contact, and the final frame
if merge_records:
    k_hit = next(i for i, s in enumerate(snaps) if MERGED in s["names"])
    k_pre = max(i for i in slow_sel if i < k_hit) if slow_sel else k_hit - 1
    # a frame where Theia is a few Earth radii out looks best
    near = [i for i in slow_idx if i < k_hit and 3.0 < SEP_RE[i] < 6.0]
    update(near[-1] if near else k_pre)
    fig.savefig("sph_orbit_collision_real_scale_impact.png", dpi=110)
update(len(snaps) - 1)
fig.savefig("sph_orbit_collision_real_scale_final.png", dpi=110)
print("Saved stills: sph_orbit_collision_real_scale_impact.png, "
      "sph_orbit_collision_real_scale_final.png")
