"""
2D SPH + N-body simulation: a central "Sun" with two orbiting bodies on a
collision course. When the two bodies physically touch they merge
(perfectly inelastic, mass + momentum conserved) and the merged body
settles into a new, bound Keplerian orbit around the Sun.

Built on top of sph_merge_sim_2.py -- same SPH core:
  - Cubic spline kernel (Monaghan & Lattanzio 1985), 2D normalisation
  - Summation density, symmetrised pressure gradient (Monaghan 1992, Sec. 3)
  - Monaghan-Gingold artificial viscosity (Sec. 4)
  - Leapfrog (kick-drift-kick) integrator
  - Contact-based, momentum-conserving merge rule
  - Unsoftened Newtonian gravity between every pair of particles

What's new compared to sph_merge_sim_2.py:
  1. Per-particle bulk density (rho_bulk is an array, not a scalar), so the
     Sun can be far more compact per unit mass than the planets. Otherwise a
     1000x heavier Sun would be ~32x wider than a planet and swallow the
     orbits. On merge, AREA is still exactly additive:
         area_new = m_a/rho_a + m_b/rho_b,   rho_new = m_new / area_new
  2. Collision-course initial conditions: we choose WHERE and HOW the two
     bodies should meet, then integrate each body BACKWARDS in time around
     the Sun to find where it must start. Running the simulation forward
     then brings them together at the chosen point.
  3. Orbital-element diagnostics (semi-major axis, eccentricity, periapsis,
     apoapsis) before and after the collision, to show the merged body is
     on a stable, bound orbit that clears the Sun.
  4. Animation with trails, colour-coded bodies, the predicted post-merge
     ellipse, and a live distance-from-Sun plot.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
            factor = xp.where(r > 1e-12, dWdq / (h * r), 0.0)
    else:
        factor = xp.where(r > 1e-12, dWdq / (h * r), 0.0)
    return rij_vec * factor[..., None]


# ----------------------------------------------------------------------
# Orbital elements of a body relative to the Sun (2-body approximation)
# ----------------------------------------------------------------------
def orbital_elements(r_rel, v_rel, mu):
    """
    r_rel, v_rel: position/velocity of the body relative to the Sun (2D).
    mu = G * (M_sun + m_body).
    Returns dict with specific energy, semi-major axis a, eccentricity e,
    periapsis, apoapsis, period, and whether the orbit is bound.
    """
    r_rel = np.asarray(r_rel, float)
    v_rel = np.asarray(v_rel, float)
    r = np.linalg.norm(r_rel)
    v2 = v_rel @ v_rel
    eps = 0.5 * v2 - mu / r                       # specific orbital energy
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
    # argument of periapsis (direction of the eccentricity vector)
    e_vec = ((v2 - mu / r) * r_rel - (r_rel @ v_rel) * v_rel) / mu
    omega = np.arctan2(e_vec[1], e_vec[0])
    return dict(energy=eps, h=h_z, a=a, e=e, peri=peri, apo=apo,
                period=period, bound=bound, omega=omega)


# ----------------------------------------------------------------------
# SPH + gravity system
# ----------------------------------------------------------------------
class SPHSystem:
    def __init__(self, pos, vel, mass, rho_bulk, names, h=1.0, k_gas=1.0,
                 gamma=1.0, alpha=1.0, beta=2.0, radius_scale=1.0, G=0.0):
        self.pos = xp.asarray(pos, dtype=xp.float64)        # (N,2)
        self.vel = xp.asarray(vel, dtype=xp.float64)        # (N,2)
        self.mass = xp.asarray(mass, dtype=xp.float64)      # (N,)
        # Per-particle bulk density -> lets the Sun be compact.
        self.rho_bulk = xp.asarray(rho_bulk, dtype=xp.float64)  # (N,)
        self.names = list(names)                             # labels, follow merges
        self.h = h
        self.k_gas = k_gas
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.G = G
        self.radius_scale = radius_scale
        self.eta2 = 0.01 * h ** 2
        self.merge_events = []

    def _pairwise(self):
        rij = self.pos[:, None, :] - self.pos[None, :, :]
        r = xp.linalg.norm(rij, axis=-1)
        return rij, r

    def density(self, r):
        W = kernel_W(r, xp.full_like(r, self.h))
        return xp.maximum((self.mass[None, :] * W).sum(axis=1), 1e-8)

    def particle_radii(self):
        """R_i = radius_scale * sqrt(m_i / (pi * rho_bulk_i))."""
        return self.radius_scale * xp.sqrt(self.mass / (xp.pi * self.rho_bulk))

    def pressure(self, rho):
        return self.k_gas * rho ** self.gamma

    def viscosity(self, rij, r, vij, rho):
        vij_dot_rij = (vij * rij).sum(axis=-1)
        approaching = vij_dot_rij < 0.0
        mu_ab = (self.h * vij_dot_rij) / (r ** 2 + self.eta2)
        rho_ab = 0.5 * (rho[:, None] + rho[None, :])
        c_ab = 1.0
        Pi = (-self.alpha * c_ab * mu_ab + self.beta * mu_ab ** 2) / rho_ab
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

    def step(self, dt):
        acc, _ = self.accelerations()
        self.vel += 0.5 * dt * acc
        self.pos += dt * self.vel
        acc, rho = self.accelerations()
        self.vel += 0.5 * dt * acc
        self.merge()
        return rho

    def merge(self):
        """Contact-based merge; conserves mass, momentum and total area."""
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
                m_a, m_b = self.mass[i], self.mass[j]
                v_a, v_b = self.vel[i].copy(), self.vel[j].copy()
                m_new = float(m_a + m_b)
                v_new = (m_a * v_a + m_b * v_b) / m_new
                r_new = (m_a * self.pos[i] + m_b * self.pos[j]) / m_new
                # area is additive -> derive the merged body's bulk density
                area_new = m_a / self.rho_bulk[i] + m_b / self.rho_bulk[j]
                rho_new = m_new / float(area_new)

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
# SCENARIO: Sun + two planets on a collision course
# ======================================================================
G = 1.0
M_SUN = 1000.0
M_A = 1.0            # planet A
M_B = 1.0            # planet B
RHO_SUN = 30.0       # compact Sun   -> radius ~ 3.3
RHO_PLANET = 1.5     # fluffier rock -> radius ~ 0.46

# --- 1. Choose the collision itself ------------------------------------
R_HIT = 20.0                          # collision happens at (R_HIT, 0)
v_circ = np.sqrt(G * M_SUN / R_HIT)   # circular speed there (~7.07)
# Planet A: on a circular, counter-clockwise orbit.
V_A_HIT = np.array([0.0, v_circ])
# Planet B: on an eccentric orbit, falling inward when it meets A. Its
# velocity at impact is rotated ~30 deg from A's, so they hit at an angle.
V_B_HIT = np.array([-4.0, 6.5])
T_COLLIDE = 12.0     # time (sim units) at which the collision should occur

# --- 2. Integrate each planet BACKWARD from the collision point ---------
def integrate_kepler(r0, v0, mu, t, dt=1e-3):
    """Leapfrog around a fixed point mass at the origin. Negative t = backwards."""
    r, v = np.array(r0, float), np.array(v0, float)
    n = int(round(abs(t) / dt))
    d = np.sign(t) * dt
    acc = lambda p: -mu * p / np.linalg.norm(p) ** 3
    for _ in range(n):
        v += 0.5 * d * acc(r)
        r += d * v
        v += 0.5 * d * acc(r)
    return r, v

hit_point = np.array([R_HIT, 0.0])
rA0, vA0 = integrate_kepler(hit_point, V_A_HIT, G * (M_SUN + M_A), -T_COLLIDE)
rB0, vB0 = integrate_kepler(hit_point, V_B_HIT, G * (M_SUN + M_B), -T_COLLIDE)

# --- 3. Sun starts at origin with a tiny counter-velocity so the system's
#        total momentum is zero (no drift of the whole picture). ---------
v_sun0 = -(M_A * vA0 + M_B * vB0) / M_SUN

pos0 = np.array([[0.0, 0.0], rA0, rB0])
vel0 = np.array([v_sun0, vA0, vB0])
mass0 = np.array([M_SUN, M_A, M_B])
rho0 = np.array([RHO_SUN, RHO_PLANET, RHO_PLANET])
names0 = ["Sun", "A", "B"]
N_INITIAL = len(mass0)

sys_ = SPHSystem(pos0, vel0, mass0, rho0, names0,
                 h=1.0, k_gas=0.005, gamma=1.0, alpha=0.5, beta=1.0,
                 radius_scale=1.0, G=G)

# Pre-collision orbits, for the log
mu_A, mu_B = G * (M_SUN + M_A), G * (M_SUN + M_B)
el_A = orbital_elements(rA0, vA0 - v_sun0, mu_A)
el_B = orbital_elements(rB0, vB0 - v_sun0, mu_B)
R_sun = float(to_host(sys_.particle_radii())[0])
print(f"Sun radius = {R_sun:.2f}, planet radius = "
      f"{float(to_host(sys_.particle_radii())[1]):.2f}")
for nm, el in (("A", el_A), ("B", el_B)):
    print(f"Planet {nm} before: a={el['a']:.2f}  e={el['e']:.3f}  "
          f"peri={el['peri']:.2f}  apo={el['apo']:.2f}  T={el['period']:.2f}")

# --- 4. Run ------------------------------------------------------------
dt = 0.005
T_TOTAL = 75.0                      # ~4 orbits of the merged body
n_steps = int(T_TOTAL / dt)

p0 = (mass0[:, None] * vel0).sum(axis=0)
m0_total = mass0.sum()

frames_pos, frames_mass, frames_radii, frames_names = [], [], [], []
merge_records = []
t_start = time.time()
for step in range(n_steps):
    prev = len(sys_.merge_events)
    sys_.step(dt)
    for ev in sys_.merge_events[prev:]:
        merge_records.append({"step": step, "t": (step + 1) * dt, **ev})
        print(f"  MERGE at t={(step+1)*dt:.3f}: {ev['parents'][0]} + "
              f"{ev['parents'][1]} -> {ev['name']}  (m={ev['mass']:.2f}, "
              f"KE {ev['ke_before']:.2f} -> {ev['ke_after']:.2f})")
    frames_pos.append(to_host(sys_.pos))
    frames_mass.append(to_host(sys_.mass))
    frames_radii.append(to_host(sys_.particle_radii()))
    frames_names.append(list(sys_.names))
    if step % 2000 == 0:
        print(f"  step {step:6d}/{n_steps}  N={len(sys_.mass)}  "
              f"elapsed={time.time()-t_start:5.1f}s")

# --- 5. Diagnostics ----------------------------------------------------
p_final = to_host((sys_.mass[:, None] * sys_.vel).sum(axis=0))
print(f"Total momentum: initial={p0}  final={p_final}")
print(f"Total mass:     initial={m0_total:.6f}  final={float(sys_.mass.sum()):.6f}")

merged_el = None
if merge_records:
    names_f = sys_.names
    i_sun = names_f.index("Sun") if "Sun" in names_f else None
    i_m = names_f.index("A+B") if "A+B" in names_f else (
          names_f.index("B+A") if "B+A" in names_f else None)
    if i_sun is not None and i_m is not None:
        # Elements from the state right after the collision
        ev = merge_records[0]
        k = ev["step"]
        # Sun velocity isn't stored per frame; reconstruct from momentum
        # conservation (total momentum is ~0 by construction).
        m_arr, p_arr = frames_mass[k], frames_pos[k]
        nm_k = frames_names[k]
        js, jm = nm_k.index("Sun"), nm_k.index(ev["name"])
        v_sun_k = -(ev["mass"] * ev["vel"]) / m_arr[js]
        merged_el = orbital_elements(p_arr[jm] - p_arr[js], ev["vel"] - v_sun_k,
                                     G * (m_arr[js] + ev["mass"]))
        print(f"Merged body orbit: a={merged_el['a']:.2f}  e={merged_el['e']:.3f}  "
              f"peri={merged_el['peri']:.2f}  apo={merged_el['apo']:.2f}  "
              f"T={merged_el['period']:.2f}  bound={merged_el['bound']}")
        # Measured min/max distance after the merge
        d = [np.linalg.norm(frames_pos[s][frames_names[s].index(ev['name'])]
                            - frames_pos[s][frames_names[s].index('Sun')])
             for s in range(k + 1, n_steps)]
        print(f"Measured after merge: r_min={min(d):.2f}  r_max={max(d):.2f}  "
              f"(Sun radius {R_sun:.2f}) -> "
              f"{'STABLE bound orbit' if max(d) < 3 * R_HIT and min(d) > R_sun else 'not stable'}")
else:
    print("No collision happened -- try adjusting V_B_HIT / T_COLLIDE.")

# ======================================================================
# Animation
# ======================================================================
COLORS = {"Sun": "gold", "A": "royalblue", "B": "crimson",
          "A+B": "mediumorchid", "B+A": "mediumorchid"}

fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(13, 6.5),
                               gridspec_kw=dict(width_ratios=[1.15, 1]))
LIM = 36.0
XLIM = YLIM = (-LIM, LIM)
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
ax.set_aspect("equal")
ax.set_facecolor("#0b0b1a")
ax.set_title("Sun + two planets on a collision course (SPH merge)")

# Predicted post-merge Kepler ellipse (shown after the collision)
ellipse_line, = ax.plot([], [], "--", color="white", lw=0.8, alpha=0.5,
                        label="predicted post-merge orbit")
if merged_el is not None and merged_el["bound"]:
    th = np.linspace(0, 2 * np.pi, 400)
    a_, e_, w_ = merged_el["a"], merged_el["e"], merged_el["omega"]
    rr = a_ * (1 - e_ ** 2) / (1 + e_ * np.cos(th))
    ELLIPSE_XY = np.c_[rr * np.cos(th + w_), rr * np.sin(th + w_)]
else:
    ELLIPSE_XY = None

TRAIL = 1200   # steps of trail
trail_lines = {nm: ax.plot([], [], "-", color=c, lw=1.0, alpha=0.6)[0]
               for nm, c in COLORS.items() if nm != "B+A"}
scat = ax.scatter([], [], s=[], edgecolors="white", linewidths=0.4, zorder=5)
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                    color="white", family="monospace", fontsize=9)
flash = ax.scatter([], [], s=[], facecolors="none", edgecolors="orange",
                   linewidths=2, zorder=6)
ax.legend(loc="lower right", fontsize=8, facecolor="#222", labelcolor="white")

POINTS_PER_DATA_UNIT = (fig.get_size_inches()[0] * 72.0 * 0.5) / (XLIM[1] - XLIM[0])


def radii_to_sizes(radii):
    s = np.pi * (radii * POINTS_PER_DATA_UNIT) ** 2
    return np.maximum(s, 25.0)   # keep tiny planets visible


# Distance-from-Sun panel
times = np.arange(n_steps) * dt
def dist_series(name):
    out = np.full(n_steps, np.nan)
    for s in range(n_steps):
        nm = frames_names[s]
        if name in nm and "Sun" in nm:
            out[s] = np.linalg.norm(frames_pos[s][nm.index(name)]
                                    - frames_pos[s][nm.index("Sun")])
    return out

for nm in ("A", "B", "A+B", "B+A"):
    d = dist_series(nm)
    if np.isfinite(d).any():
        ax_r.plot(times, d, color=COLORS[nm], lw=1.2,
                  label=f"{nm} (merged)" if "+" in nm else f"planet {nm}")
ax_r.axhline(R_sun, color="gold", ls=":", lw=1, label="Sun surface")
if merge_records:
    ax_r.axvline(merge_records[0]["t"], color="orange", ls="--", lw=1,
                 label="collision")
ax_r.set_xlabel("time"); ax_r.set_ylabel("distance from Sun")
ax_r.set_title("Orbital radius vs time")
ax_r.set_ylim(0, None)
ax_r.legend(fontsize=8)
time_cursor = ax_r.axvline(0, color="k", lw=0.8)


def build_track(*aliases):
    """(n_steps, 2) position history of a named body, NaN when absent."""
    out = np.full((n_steps, 2), np.nan)
    for s in range(n_steps):
        for a in aliases:
            if a in frames_names[s]:
                out[s] = frames_pos[s][frames_names[s].index(a)]
                break
    return out

TRACKS = {nm: build_track(nm, "B+A") if nm == "A+B" else build_track(nm)
          for nm in trail_lines}


def init():
    scat.set_offsets(np.empty((0, 2)))
    flash.set_offsets(np.empty((0, 2)))
    for ln in trail_lines.values():
        ln.set_data([], [])
    ellipse_line.set_data([], [])
    return (scat, flash, info_text, ellipse_line, time_cursor, *trail_lines.values())


def update(step):
    p, rad, nm = frames_pos[step], frames_radii[step], frames_names[step]
    scat.set_offsets(p)
    scat.set_sizes(radii_to_sizes(rad))
    scat.set_facecolors([COLORS.get(n, "white") for n in nm])

    # trails: last TRAIL steps of each body's precomputed track (NaN gaps
    # where the body doesn't exist simply aren't drawn)
    lo = max(0, step - TRAIL)
    for name, ln in trail_lines.items():
        seg = TRACKS[name][lo:step + 1]
        ln.set_data(seg[:, 0], seg[:, 1])

    merged = bool(merge_records) and step >= merge_records[0]["step"]
    if merged and ELLIPSE_XY is not None:
        sun_p = p[nm.index("Sun")]
        ellipse_line.set_data(ELLIPSE_XY[:, 0] + sun_p[0], ELLIPSE_XY[:, 1] + sun_p[1])
    # orange flash ring for a short while after impact
    if merge_records and 0 <= step - merge_records[0]["step"] < 300:
        age = step - merge_records[0]["step"]
        flash.set_offsets([merge_records[0]["pos"]])
        flash.set_sizes([60 + 8 * age])
        flash.set_alpha(max(0.0, 1 - age / 300))
    else:
        flash.set_offsets(np.empty((0, 2)))

    status = "post-merge orbit" if merged else "approaching..."
    info_text.set_text(f"t = {step*dt:6.2f}\nbodies = {len(nm)}\n{status}")
    time_cursor.set_xdata([step * dt, step * dt])
    return (scat, flash, info_text, ellipse_line, time_cursor, *trail_lines.values())


SKIP = 30
render_frames = range(0, n_steps, SKIP)
ani = animation.FuncAnimation(fig, lambda i: update(render_frames[i]),
                              frames=len(render_frames), init_func=init,
                              interval=33, blit=True)
fig.tight_layout()
out_path = "sph_orbit_collision.gif"
ani.save(out_path, writer=animation.PillowWriter(fps=30))
print(f"Saved animation to {out_path}")

# Static summary image (final frame)
update(n_steps - 1)
fig.savefig("sph_orbit_collision_final.png", dpi=110)
print("Saved final frame to sph_orbit_collision_final.png")
