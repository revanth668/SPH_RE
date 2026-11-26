"""
Basic 2D SPH simulation of 4 particles that merge on collision.

Theory reference: Monaghan (1992), "Smoothed Particle Hydrodynamics",
Annu. Rev. Astron. Astrophys. 30:543-574.

Implements:
  - Cubic spline kernel (Sec. 7, Monaghan & Lattanzio 1985)
  - Summation density estimate (Sec. 3, continuity option 1)
  - Symmetrized pressure gradient / momentum equation (Sec. 3)
  - Monaghan-Gingold artificial viscosity (Sec. 4)
  - Leapfrog (kick-drift-kick) time integration
  - Merging rule: when two particles get closer than a merge radius,
    they combine into one particle whose mass and velocity conserve
    linear momentum, i.e. the new particle sits at, and moves with,
    the centre of mass of the two colliding particles:
        m_new = m_a + m_b
        v_new = (m_a v_a + m_b v_b) / m_new
        r_new = (m_a r_a + m_b r_b) / m_new
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Cubic spline kernel (Monaghan & Lattanzio 1985), 2D normalization
# ----------------------------------------------------------------------
def kernel_W(r, h):
    """W(r,h) for the cubic spline kernel, r and h can be arrays."""
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h ** 2)  # 2D normalization constant
    W = np.zeros_like(q)
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
    sigma = 10.0 / (7.0 * np.pi * h ** 2)
    dWdq = np.zeros_like(q)
    m1 = (q > 1e-12) & (q <= 1.0)
    m2 = (q > 1.0) & (q <= 2.0)
    dWdq[m1] = sigma[m1] * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)
    dWdq[m2] = -sigma[m2] * 0.75 * (2.0 - q[m2]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(r > 1e-12, dWdq / (h * r), 0.0)
    return rij_vec * factor[..., None]


# ----------------------------------------------------------------------
# SPH system
# ----------------------------------------------------------------------
class SPHSystem:
    def __init__(self, pos, vel, mass, h=1.0, k_gas=1.0, gamma=1.0,
                 alpha=1.0, beta=2.0, merge_radius=0.0000000005):
        self.pos = pos.astype(float)      # (N,2)
        self.vel = vel.astype(float)      # (N,2)
        self.mass = mass.astype(float)    # (N,)
        self.h = h                        # smoothing length (fixed, global)
        self.k_gas = k_gas                # equation-of-state constant
        self.gamma = gamma                # polytropic index (1 = isothermal-like)
        self.alpha = alpha                # artificial viscosity linear term
        self.beta = beta                  # artificial viscosity quadratic term
        self.merge_radius = merge_radius  # distance below which particles merge
        self.eta2 = 0.01 * h ** 2         # viscosity softening, Sec. 4
        self.merge_events = []            # log of merges: dict(mass, vel, pos, parents)

    # -- pairwise geometry -------------------------------------------------
    def _pairwise(self):
        rij = self.pos[:, None, :] - self.pos[None, :, :]   # r_a - r_b
        r = np.linalg.norm(rij, axis=-1)
        return rij, r

    # -- density (summation interpolant, Sec. 3 continuity option 1) ------
    def density(self, r):
        h_arr = np.full_like(r, self.h)
        W = kernel_W(r, h_arr)
        rho = (self.mass[None, :] * W).sum(axis=1)
        return np.maximum(rho, 1e-8)

    # -- pressure: simple polytropic equation of state ---------------------
    def pressure(self, rho):
        return self.k_gas * rho ** self.gamma

    # -- artificial viscosity (Monaghan & Gingold, Sec. 4) ------------------
    def viscosity(self, rij, r, vij, rho):
        h_arr = np.full_like(r, self.h)
        vij_dot_rij = (vij * rij).sum(axis=-1)
        approaching = vij_dot_rij < 0.0  # only when particles move together

        mu_ab = (h_arr * vij_dot_rij) / (r ** 2 + self.eta2)
        rho_ab = 0.5 * (rho[:, None] + rho[None, :])
        c_ab = 1.0  # constant sound speed proxy for this toy demo

        Pi = (-self.alpha * c_ab * mu_ab + self.beta * mu_ab ** 2) / rho_ab
        Pi = np.where(approaching, Pi, 0.0)
        np.fill_diagonal(Pi, 0.0)
        return Pi

    # -- accelerations from the symmetrized momentum equation (Sec. 3) -----
    def accelerations(self):
        n = len(self.mass)
        rij, r = self._pairwise()
        rho = self.density(r)
        P = self.pressure(rho)

        h_arr = np.full_like(r, self.h)
        gradW = kernel_gradW(rij, r, h_arr)  # (n,n,2)

        vij = self.vel[:, None, :] - self.vel[None, :, :]
        Pi = self.viscosity(rij, r, vij, rho)

        # symmetric pairwise coefficient: P_b/rho_b^2 + P_a/rho_a^2 + Pi_ab
        coeff = (P[None, :] / rho[None, :] ** 2
                 + P[:, None] / rho[:, None] ** 2
                 + Pi)
        np.fill_diagonal(coeff, 0.0)

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

    # -- momentum-conserving merge -------------------------------------------
    def merge(self):
        """
        Merge any pair of particles closer than merge_radius.
        New particle sits at, and moves with, the pair's centre of mass,
        so linear momentum (m_a v_a + m_b v_b) is exactly conserved.
        """
        merged_any = True
        while merged_any and len(self.mass) > 1:
            merged_any = False
            rij, r = self._pairwise()
            np.fill_diagonal(r, np.inf)
            i, j = np.unravel_index(np.argmin(r), r.shape)
            if r[i, j] < self.merge_radius:
                m_a, m_b = self.mass[i], self.mass[j]
                # m_new = (m_a*m_b)/(m_a + m_b)
                m_new = (m_a+m_b)
                v_new = (m_a * self.vel[i] + m_b * self.vel[j]) / m_new
                r_new = (m_a * self.pos[i] + m_b * self.pos[j]) / m_new

                self.merge_events.append({
                    "mass": m_new,
                    "vel": v_new.copy(),
                    "pos": r_new.copy(),
                    "parent_mass": (m_a, m_b),
                    "parent_vel": (self.vel[i].copy(), self.vel[j].copy()),
                })

                keep = [k for k in range(len(self.mass)) if k not in (i, j)]
                self.pos = np.vstack([self.pos[keep], r_new])
                self.vel = np.vstack([self.vel[keep], v_new])
                self.mass = np.concatenate([self.mass[keep], [m_new]])
                merged_any = True


# ----------------------------------------------------------------------
# Set up 4 particles, same mass, different velocities aimed roughly
# so their paths cross and they collide.
# ----------------------------------------------------------------------
# Two pairs, each converging vertically as they drift horizontally toward
# the centre: (0,1) collide and merge on the left, (2,3) collide and merge
# on the right; the two resulting blobs then drift together and merge again.
pos0 = np.array([
    [-5.0, -1.0],
    [-5.0,  1.0],
    [ 5.0, -1.0],
    [ 5.0,  1.0],
])
vel0 = np.array([
    [ 0.3,  1.0],
    [ 0.3, -1.0],
    [-0.3,  1.0],
    [-0.3, -1.0],
])
mass0 = np.full(4, 1.1)

sys_ = SPHSystem(pos0, vel0, mass0, h=1.0, k_gas=0.005, gamma=1.0,
                  alpha=0.5, beta=1.0, merge_radius=0.8)

dt = 0.01
n_steps = 3200

# total momentum check (should be conserved through every merge)
p0 = (mass0[:, None] * vel0).sum(axis=0)

frames_pos, frames_vel, frames_mass = [], [], []
merge_records = []  # each: dict(step=int, mass, vel, pos, parent_mass, parent_vel)
for step in range(n_steps):
    prev_log_len = len(sys_.merge_events)
    sys_.step(dt)
    for ev in sys_.merge_events[prev_log_len:]:
        merge_records.append({"step": step, **ev})
    frames_pos.append(sys_.pos.copy())
    frames_vel.append(sys_.vel.copy())
    frames_mass.append(sys_.mass.copy())

p_final = (sys_.mass[:, None] * sys_.vel).sum(axis=0)
print(f"Initial total momentum: {p0}")
print(f"Final total momentum:   {p_final}")
print(f"Particles remaining: {len(sys_.mass)}")

# ----------------------------------------------------------------------
# Animate
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-6, 6)
ax.set_ylim(-4, 4)
ax.set_aspect("equal")
ax.set_title("SPH particles: merge conserves momentum at COM")
scat = ax.scatter([], [], s=[], c="royalblue", alpha=0.85, edgecolors="k")
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

colors = plt.cm.tab10(np.linspace(0, 1, 4))

N0 = len(mass0)  # merges only ever reduce particle count, so this is the max

# one small mass/velocity tag per possible particle slot, reused every frame
tag_texts = [
    ax.text(0, 0, "", fontsize=7, color="dimgray", ha="left", va="bottom",
             visible=False)
    for _ in range(N0)
]

# pool of "new particle" callouts shown for a short window right after a merge
HIGHLIGHT_STEPS = 300  # how many raw sim steps a merge callout stays visible (~5s)
MAX_CONCURRENT_MERGES = 4
merge_texts = [
    ax.text(0, 0, "", fontsize=8, color="white", ha="center", va="center",
             visible=False,
             bbox=dict(boxstyle="round,pad=0.35", fc="crimson", ec="none", alpha=0.9))
    for _ in range(MAX_CONCURRENT_MERGES)
]


def init():
    scat.set_offsets(np.empty((0, 2)))
    for t in tag_texts:
        t.set_visible(False)
    for t in merge_texts:
        t.set_visible(False)
    return (scat, info_text, *tag_texts, *merge_texts)


def update(step):
    p = frames_pos[step]
    v = frames_vel[step]
    m = frames_mass[step]
    n = len(m)

    scat.set_offsets(p)
    scat.set_sizes(200 * m)
    info_text.set_text(f"t = {step*dt:5.2f}   N particles = {n}")

    # per-particle "m = ..., v = (..., ...)" tag next to each live particle
    for i, tag in enumerate(tag_texts):
        if i < n:
            tag.set_position((p[i, 0] + 0.2, p[i, 1] + 0.2))
            tag.set_text(f"m={m[i]:.2f}\nv=({v[i, 0]:+.2f},{v[i, 1]:+.2f})")
            tag.set_visible(True)
        else:
            tag.set_visible(False)

    # recent merges: show the new particle's combined mass/velocity
    active = [rec for rec in merge_records if 0 <= step - rec["step"] < HIGHLIGHT_STEPS]
    for i, mt in enumerate(merge_texts):
        if i < len(active):
            rec = active[i]
            mt.set_position((rec["pos"][0], rec["pos"][1] - 0.5))
            m_a, m_b = rec["parent_mass"]
            mt.set_text(
                f"MERGE: m={m_a:.2f}+{m_b:.2f}={rec['mass']:.2f}\n"
                f"v_new=({rec['vel'][0]:+.2f},{rec['vel'][1]:+.2f})"
            )
            mt.set_visible(True)
        else:
            mt.set_visible(False)

    return (scat, info_text, *tag_texts, *merge_texts)


SKIP = 4  # subsample frames so GIF encoding stays fast
render_frames = range(0, n_steps, SKIP)


def update_render(idx):
    return update(render_frames[idx])


ani = animation.FuncAnimation(fig, update_render, frames=len(render_frames),
                               init_func=init, interval=15, blit=True)

out_path = "sph_merge_sim.gif"
ani.save(out_path, writer=animation.PillowWriter(fps=30))
print(f"Saved animation to {out_path}")
