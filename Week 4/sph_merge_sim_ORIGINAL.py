"""
Implements:
  - Merging rule: two particles merge once they physically overlap, i.e.
    once their centre separation drops below the SUM OF THEIR RADII
    (not an arbitrary constant). The merger conserves mass and linear
    momentum exactly, i.e. the new particle sits at, and moves with,
    the centre of mass of the two colliding particles:
        m_new = m_a + m_b
        v_new = (m_a v_a + m_b v_b) / m_new
        r_new = (m_a r_a + m_b r_b) / m_new
    Because rho_bulk is constant, area = mass / rho_bulk is exactly additive
    across a merger: area_new = area_a + area_b, so two particles of equal
    radius R combine into one of radius sqrt(2)*R, not 2*R -- the way a real
    constant-density accreting body would grow, rather than an arbitrary
    jump in drawn size.
    Kinetic energy is NOT conserved (a perfectly inelastic collision loses
    KE to "heat") -- this is physically correct and is logged so you can
    see it happen.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import cupy as xp
    GPU = xp.cuda.is_available()
except Exception:
    GPU = False
if not GPU:
    import numpy as xp
print(f"[backend] using {'GPU (CuPy)' if GPU else 'CPU (NumPy)'} for array ops")


def to_host(a):
    """Bring an array back to NumPy/host memory (no-op if already NumPy)."""
    return xp.asnumpy(a) if GPU else np.asarray(a)


rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Cubic spline kernel (Monaghan & Lattanzio 1985), 2D normalization
# ----------------------------------------------------------------------
def kernel_W(r, h):
    """W(r,h) for the cubic spline kernel, r and h can be arrays."""
    q = r / h
    sigma = 10.0 / (7.0 * xp.pi * h ** 2)  # 2D normalization constant
    W = xp.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    W[m1] = sigma[m1] * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
    W[m2] = sigma[m2] * 0.25 * (2.0 - q[m2]) ** 3
    return W


def kernel_gradW(rij_vec, r, h):
    """
    Gradient of W with respect to particle a's position.
    rij_vec: (...,2) array of r_a - r_b
    r      : (...)   array of |r_a - r_b|
    h      : (...)   smoothing length used for the pair
    Returns (...,2) gradient array.
    """
    q = r / h
    sigma = 10.0 / (7.0 * xp.pi * h ** 2)
    dWdq = xp.zeros_like(q)
    m1 = (q > 1e-12) & (q <= 1.0)
    m2 = (q > 1.0) & (q <= 2.0)
    dWdq[m1] = sigma[m1] * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)
    dWdq[m2] = -sigma[m2] * 0.75 * (2.0 - q[m2]) ** 2
    # NumPy warns (doesn't raise) on the 0/0 at r=0; CuPy has no errstate
    # context manager, and neither backend actually errors here since the
    # np.where below immediately replaces every non-finite entry with 0.
    if not GPU:
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = xp.where(r > 1e-12, dWdq / (h * r), 0.0)
    else:
        factor = xp.where(r > 1e-12, dWdq / (h * r), 0.0)
    return rij_vec * factor[..., None]


# ----------------------------------------------------------------------
# SPH system
# ----------------------------------------------------------------------
class SPHSystem:
    def __init__(self, pos, vel, mass, h=1.0, k_gas=1.0, gamma=1.0,
                 alpha=1.0, beta=2.0, rho_bulk=0.5, radius_scale=1.0):
        # xp.asarray (not .astype) so plain NumPy inputs get moved onto the
        # GPU device when the CuPy backend is active.
        self.pos = xp.asarray(pos, dtype=xp.float64)      # (N,2)
        self.vel = xp.asarray(vel, dtype=xp.float64)      # (N,2)
        self.mass = xp.asarray(mass, dtype=xp.float64)    # (N,)
        self.h = h                        # smoothing length (fixed, global)
        self.k_gas = k_gas                # equation-of-state constant
        self.gamma = gamma                # polytropic index (1 = isothermal-like)
        self.alpha = alpha                # artificial viscosity linear term
        self.beta = beta                  # artificial viscosity quadratic term
        # Fixed bulk material density: area = mass / rho_bulk. Distinct from
        # the compressible SPH density used for pressure -- see class docstring.
        self.rho_bulk = rho_bulk
        # Extra dimensionless knob if you want to visually inflate/deflate
        # the drawn radius without changing the physical rho_bulk.
        self.radius_scale = radius_scale
        self.eta2 = 0.01 * h ** 2         # viscosity softening, Sec. 4
        self.merge_events = []            # log of merges: dict(mass, vel, pos, parents, radii, ke_before, ke_after)

    # -- pairwise geometry -------------------------------------------------
    def _pairwise(self):
        rij = self.pos[:, None, :] - self.pos[None, :, :]   # r_a - r_b
        r = xp.linalg.norm(rij, axis=-1)
        return rij, r

    # -- density (summation interpolant, Sec. 3 continuity option 1) ------
    def density(self, r):
        h_arr = xp.full_like(r, self.h)
        W = kernel_W(r, h_arr)
        rho = (self.mass[None, :] * W).sum(axis=1)
        return xp.maximum(rho, 1e-8)

    # -- physical size of each particle, tied to mass AND bulk density ----
    def particle_radii(self):
        return xp.sqrt(self.mass / (xp.pi * self.rho_bulk))

    # -- pressure: simple polytropic equation of state ---------------------
    def pressure(self, rho):
        return self.k_gas * rho ** self.gamma

    # -- artificial viscosity (Monaghan & Gingold, Sec. 4) ------------------
    def viscosity(self, rij, r, vij, rho):
        h_arr = xp.full_like(r, self.h)
        vij_dot_rij = (vij * rij).sum(axis=-1)
        approaching = vij_dot_rij < 0.0  # only when particles move together

        mu_ab = (h_arr * vij_dot_rij) / (r ** 2 + self.eta2)
        rho_ab = 0.5 * (rho[:, None] + rho[None, :])
        c_ab = 1.0  # constant sound speed proxy for this toy demo

        Pi = (-self.alpha * c_ab * mu_ab + self.beta * mu_ab ** 2) / rho_ab
        Pi = xp.where(approaching, Pi, 0.0)
        xp.fill_diagonal(Pi, 0.0)
        return Pi

    # -- accelerations from the symmetrized momentum equation (Sec. 3) -----
    def accelerations(self):
        n = len(self.mass)
        rij, r = self._pairwise()
        rho = self.density(r)
        P = self.pressure(rho)

        h_arr = xp.full_like(r, self.h)
        gradW = kernel_gradW(rij, r, h_arr)  # (n,n,2)

        vij = self.vel[:, None, :] - self.vel[None, :, :]
        Pi = self.viscosity(rij, r, vij, rho)

        # symmetric pairwise coefficient: P_b/rho_b^2 + P_a/rho_a^2 + Pi_ab
        coeff = (P[None, :] / rho[None, :] ** 2
                 + P[:, None] / rho[:, None] ** 2
                 + Pi)
        xp.fill_diagonal(coeff, 0.0)

        acc = -(self.mass[None, :, None] * coeff[..., None] * gradW).sum(axis=1)
        return acc, rho

    # -- leapfrog (kick-drift-kick) step ------------------------------------
    def step(self, dt):
        acc, _ = self.accelerations()
        self.vel += 0.5 * dt * acc
        self.pos += dt * self.vel
        acc, rho = self.accelerations()
        self.vel += 0.5 * dt * acc
        self.merge()
        return rho

    # -- momentum-conserving, contact-based merge ---------------------------
    def merge(self):
        """
        Merge any pair of particles whose centres are closer than the SUM
        of their own physical radii (real contact), instead of an arbitrary
        fixed distance. Mass and linear momentum are conserved exactly:
            m_new = m_a + m_b
            v_new = (m_a v_a + m_b v_b) / m_new
        Kinetic energy is deliberately NOT conserved -- a perfectly
        inelastic collision must lose KE, and the amount lost is logged
        alongside each merge event for sanity-checking.
        """
        merged_any = True
        while merged_any and len(self.mass) > 1:
            merged_any = False
            _, r = self._pairwise()
            radii = self.particle_radii()
            contact_dist = radii[:, None] + radii[None, :]
            gap = r - contact_dist
            xp.fill_diagonal(gap, xp.inf)
            i, j = xp.unravel_index(xp.argmin(gap), gap.shape)
            i, j = int(i), int(j)  # plain Python ints regardless of backend
            if gap[i, j] < 0.0:
                m_a, m_b = self.mass[i], self.mass[j]
                v_a, v_b = self.vel[i].copy(), self.vel[j].copy()
                m_new = float(m_a + m_b)
                v_new = (m_a * v_a + m_b * v_b) / m_new
                r_new = (m_a * self.pos[i] + m_b * self.pos[j]) / m_new

                ke_before = float(0.5 * m_a * (v_a @ v_a) + 0.5 * m_b * (v_b @ v_b))
                ke_after = float(0.5 * m_new * (v_new @ v_new))

                # Log host-side (plain Python / NumPy) copies -- merge_records
                # gets formatted with f-strings and read by Matplotlib later,
                # neither of which should have to know about the GPU backend.
                self.merge_events.append({
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
                merged_any = True


# ----------------------------------------------------------------------
# Randomly spawn N particles in a box big enough that 1000 of them don't
# all sit on top of each other, with random masses and velocities. Any
# pair that physically touches (see SPHSystem.merge) combines.
# ----------------------------------------------------------------------
N_INITIAL = 1000
BOX = 30.0  # spawn box is [-BOX, BOX] x [-BOX, BOX]

pos0 = rng.uniform(-BOX, BOX, size=(N_INITIAL, 2))
vel0 = rng.uniform(-1.0, 1.0, size=(N_INITIAL, 2))
mass0 = rng.uniform(0.5, 2.0, size=N_INITIAL)

# rho_bulk is picked so initial particles are small relative to their mean
# spacing (mean spacing ~ sqrt((2*BOX)^2 / N) =~ 1.9 here) -- otherwise a
# dense random spawn starts everyone already touching everyone else and the
# whole crowd cascade-merges into one particle on step 0.
sys_ = SPHSystem(pos0, vel0, mass0, h=1.0, k_gas=0.005, gamma=1.0, alpha=0.5, beta=1.0, rho_bulk=3.0, radius_scale=1.0)

dt = 0.01
n_steps = 1500

# total momentum / mass check (should be conserved through every merge)
p0 = (mass0[:, None] * vel0).sum(axis=0)
m0_total = mass0.sum()
ke0_total = 0.5 * (mass0 * (vel0 ** 2).sum(axis=1)).sum()

frames_pos, frames_vel, frames_mass, frames_radii = [], [], [], []
merge_records = []  # each: dict(step=int, mass, vel, pos, parent_mass, parent_vel, ...)
t_start = time.time()
for step in range(n_steps):
    prev_log_len = len(sys_.merge_events)
    sys_.step(dt)
    for ev in sys_.merge_events[prev_log_len:]:
        merge_records.append({"step": step, **ev})
    # Matplotlib needs host (NumPy) arrays -- pull each frame off the GPU
    # here rather than dragging CuPy arrays into the animation code below.
    frames_pos.append(to_host(sys_.pos))
    frames_vel.append(to_host(sys_.vel))
    frames_mass.append(to_host(sys_.mass))
    frames_radii.append(to_host(sys_.particle_radii()))
    if step % 100 == 0:
        print(f"  step {step:5d}/{n_steps}  N={len(sys_.mass):4d}  "
              f"elapsed={time.time()-t_start:6.1f}s")

p_final = to_host((sys_.mass[:, None] * sys_.vel).sum(axis=0))
m_final_total = float(sys_.mass.sum())
ke_final_total = float(0.5 * (sys_.mass * (sys_.vel ** 2).sum(axis=1)).sum())

print(f"Initial total momentum: {p0}")
print(f"Final total momentum:   {p_final}")
print(f"Initial total mass: {m0_total:.6f}   Final total mass: {m_final_total:.6f}")
print(f"Particles remaining: {len(sys_.mass)} (started at {N_INITIAL}, "
      f"{len(merge_records)} merge events)")
print(f"Total KE: initial={ke0_total:.4f}  final={ke_final_total:.4f}  "
      f"(should decrease -- inelastic mergers radiate KE away)")

# ----------------------------------------------------------------------
# Animate
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
# A bit bigger than the spawn box so particles that drift outward are
# still visible instead of running off the edge of the plot.
XLIM = (-BOX * 1.5, BOX * 1.5)
YLIM = (-BOX * 1.5, BOX * 1.5)
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect("equal")
ax.set_title(f"SPH: {N_INITIAL} random particles, merge conserves mass & momentum")
scat = ax.scatter([], [], s=[], c="royalblue", alpha=0.7, edgecolors="k", linewidths=0.3)
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

# Convert a physical (data-unit) radius to a matplotlib scatter size, which
# is specified in points^2. 1 point = 1/72 inch, so points-per-data-unit
# depends on the figure's pixel width and the axis's data range.
fig_width_in = fig.get_size_inches()[0]
POINTS_PER_DATA_UNIT = (fig_width_in * 72.0) / (XLIM[1] - XLIM[0])


def radii_to_sizes(radii):
    """Physical radius (data units) -> scatter 's' (area in points^2)."""
    return np.pi * (radii * POINTS_PER_DATA_UNIT) ** 2


def init():
    scat.set_offsets(np.empty((0, 2)))
    return (scat, info_text)


def update(step):
    p = frames_pos[step]
    m = frames_mass[step]
    rad = frames_radii[step]
    n = len(m)
    n_merged = N_INITIAL - n

    scat.set_offsets(p)
    scat.set_sizes(radii_to_sizes(rad))
    info_text.set_text(f"t = {step*dt:5.2f}   N particles = {n}   "
                        f"merges so far = {n_merged}")
    return (scat, info_text)


SKIP = 4  # subsample frames so GIF encoding stays fast
render_frames = range(0, n_steps, SKIP)


def update_render(idx):
    return update(render_frames[idx])


ani = animation.FuncAnimation(fig, update_render, frames=len(render_frames),
                               init_func=init, interval=15, blit=True)

out_path = "sph_merge_sim.gif"
ani.save(out_path, writer=animation.PillowWriter(fps=30))
print(f"Saved animation to {out_path}")
