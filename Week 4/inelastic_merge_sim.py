"""
Perfectly-inelastic collision / merge simulation.

Physics assumption (per prof's review): particles move ballistically
(straight lines, no inter-particle forces -- this isolates the collision
physics from the SPH hydrodynamics in the earlier script). When two
particles physically touch, they merge INSTANTLY into a single particle:

    m_new = m_a + m_b
    v_new = (m_a v_a + m_b v_b) / m_new      <- common post-collision velocity
    r_new = (m_a r_a(t_c) + m_b r_b(t_c)) / m_new   <- COM position AT CONTACT

The key derivation: because motion is ballistic, the exact contact time
t_c between any pair (a,b) is found analytically. Let
    r_ab(t) = r_a(t) - r_b(t) = r_ab0 + v_ab * t         (v_ab = v_a - v_b)
Contact occurs when |r_ab(t)| = R_a + R_b = R_sum, i.e.

    (v_ab . v_ab) t^2 + 2 (r_ab0 . v_ab) t + (r_ab0 . r_ab0 - R_sum^2) = 0

a quadratic a*t^2 + b*t + c = 0. The physical collision time is the
SMALLEST POSITIVE root (only valid if discriminant >= 0 and the pair is
approaching, i.e. r_ab0 . v_ab < 0).

The simulation is EVENT-DRIVEN: at each step we scan every pair, find the
globally-next collision across the whole system, advance ALL particles
ballistically to that exact instant, merge that one pair with the formulas
above, and repeat -- so every merge happens at the true geometric moment of
contact, not at some later, already-overlapping timestep.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Exact pairwise collision time (vectorized quadratic solve)
# ----------------------------------------------------------------------
def next_collision(pos, vel, radii):
    """
    Returns (t_min, i, j) for the globally-next collision among all
    pairs, or (None, -1, -1) if no pair is on a collision course.
    """
    n = len(radii)
    if n < 2:
        return None, -1, -1

    rij = pos[:, None, :] - pos[None, :, :]      # r_a - r_b, (n,n,2)
    vij = vel[:, None, :] - vel[None, :, :]      # v_a - v_b, (n,n,2)
    rsum = radii[:, None] + radii[None, :]       # R_a + R_b, (n,n)

    a = (vij * vij).sum(axis=-1)
    b = 2.0 * (rij * vij).sum(axis=-1)
    c = (rij * rij).sum(axis=-1) - rsum ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        disc = b ** 2 - 4.0 * a * c
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t_root = (-b - sqrt_disc) / (2.0 * a)

    approaching = b < 0.0          # r_ab . v_ab < 0  <=>  b < 0
    real_root = disc >= 0.0
    moving = a > 1e-14             # skip near-parallel/relatively-static pairs
    already_touching = c <= 0.0    # already overlapping (shouldn't happen, but guard)

    valid = moving & real_root & approaching & (t_root > 0.0) & ~already_touching
    t_candidate = np.where(valid, t_root, np.inf)
    np.fill_diagonal(t_candidate, np.inf)

    flat_idx = np.argmin(t_candidate)
    i, j = np.unravel_index(flat_idx, t_candidate.shape)
    t_min = t_candidate[i, j]
    if not np.isfinite(t_min):
        return None, -1, -1
    return float(t_min), int(i), int(j)


# ----------------------------------------------------------------------
# System
# ----------------------------------------------------------------------
class BallisticMergeSystem:
    def __init__(self, pos, vel, mass, rho_bulk=3.0):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)
        self.mass = np.asarray(mass, dtype=np.float64)
        self.rho_bulk = rho_bulk
        self.merge_events = []   # log: dict(t, mass, pos, vel, parents...)
        self.time = 0.0

    def radii(self):
        return np.sqrt(self.mass / (np.pi * self.rho_bulk))

    def _merge_pair(self, i, j):
        m_a, m_b = self.mass[i], self.mass[j]
        v_a, v_b = self.vel[i].copy(), self.vel[j].copy()
        r_a, r_b = self.pos[i].copy(), self.pos[j].copy()
        rad_a, rad_b = self.radii()[i], self.radii()[j]

        m_new = float(m_a + m_b)
        v_new = (m_a * v_a + m_b * v_b) / m_new
        r_new = (m_a * r_a + m_b * r_b) / m_new  # COM position AT the contact instant

        ke_before = float(0.5 * m_a * (v_a @ v_a) + 0.5 * m_b * (v_b @ v_b))
        ke_after = float(0.5 * m_new * (v_new @ v_new))

        self.merge_events.append({
            "t": self.time, "mass": m_new, "pos": r_new.copy(), "vel": v_new.copy(),
            "parent_mass": (float(m_a), float(m_b)),
            "parent_pos": (r_a.copy(), r_b.copy()),
            "parent_radii": (float(rad_a), float(rad_b)),
            "ke_before": ke_before, "ke_after": ke_after,
        })

        keep = [k for k in range(len(self.mass)) if k not in (i, j)]
        self.pos = np.vstack([self.pos[keep], r_new.reshape(1, 2)])
        self.vel = np.vstack([self.vel[keep], v_new.reshape(1, 2)])
        self.mass = np.concatenate([self.mass[keep], np.array([m_new])])

    def advance(self, dt_budget):
        """Event-driven advance by exactly dt_budget of simulation time,
        resolving every collision at its exact analytic contact time."""
        remaining = dt_budget
        while remaining > 0.0 and len(self.mass) > 1:
            t_c, i, j = next_collision(self.pos, self.vel, self.radii())
            if t_c is None or t_c > remaining:
                # no collision within the remaining budget: free-stream to end
                self.pos += self.vel * remaining
                self.time += remaining
                remaining = 0.0
            else:
                # free-stream every particle to the exact contact instant...
                self.pos += self.vel * t_c
                self.time += t_c
                remaining -= t_c
                # ...then merge the colliding pair there
                self._merge_pair(i, j)


# ----------------------------------------------------------------------
# Set up: same spirit as the SPH demo (random particles, random masses),
# but with no inter-particle forces -- pure ballistic + merge.
# ----------------------------------------------------------------------
N_INITIAL = 300
BOX = 30.0

pos0 = rng.uniform(-BOX, BOX, size=(N_INITIAL, 2))
vel0 = rng.uniform(-3.0, 3.0, size=(N_INITIAL, 2))
mass0 = rng.uniform(0.5, 2.0, size=N_INITIAL)

sys_ = BallisticMergeSystem(pos0, vel0, mass0, rho_bulk=3.0)

p0 = (mass0[:, None] * vel0).sum(axis=0)
m0_total = mass0.sum()
ke0_total = 0.5 * (mass0 * (vel0 ** 2).sum(axis=1)).sum()

FRAME_DT = 0.05
N_FRAMES = 300  # 15 time units total, matching the SPH demo's ~15s run

frames_pos, frames_vel, frames_mass, frames_radii = [], [], [], []
for f in range(N_FRAMES):
    sys_.advance(FRAME_DT)
    frames_pos.append(sys_.pos.copy())
    frames_vel.append(sys_.vel.copy())
    frames_mass.append(sys_.mass.copy())
    frames_radii.append(sys_.radii().copy())
    if f % 30 == 0:
        print(f"  frame {f:4d}/{N_FRAMES}  t={sys_.time:5.2f}  N={len(sys_.mass):4d}")

p_final = (sys_.mass[:, None] * sys_.vel).sum(axis=0)
m_final_total = float(sys_.mass.sum())
ke_final_total = float(0.5 * (sys_.mass * (sys_.vel ** 2).sum(axis=1)).sum())

print(f"\nInitial total momentum: {p0}")
print(f"Final total momentum:   {p_final}")
print(f"Initial total mass: {m0_total:.6f}   Final total mass: {m_final_total:.6f}")
print(f"Particles remaining: {len(sys_.mass)} (started at {N_INITIAL}, "
      f"{len(sys_.merge_events)} merge events)")
print(f"Total KE: initial={ke0_total:.4f}  final={ke_final_total:.4f}  "
      f"(decreases -- perfectly inelastic collisions radiate KE away)")

# spot-check the very first merge event against the closed-form
if sys_.merge_events:
    ev = sys_.merge_events[0]
    print(f"\nFirst merge at t={ev['t']:.4f}: "
          f"m={ev['mass']:.4f}  pos={ev['pos']}  vel={ev['vel']}  "
          f"KE lost={ev['ke_before']-ev['ke_after']:.5f}")

# ----------------------------------------------------------------------
# Animate (same style as the SPH gif)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
XLIM = (-BOX * 1.5, BOX * 1.5)
YLIM = (-BOX * 1.5, BOX * 1.5)
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect("equal")
ax.set_title(f"Perfectly inelastic merges: {N_INITIAL} particles, "
             f"exact analytic contact time")
scat = ax.scatter([], [], s=[], c="tomato", alpha=0.7, edgecolors="k", linewidths=0.3)
info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

fig_width_in = fig.get_size_inches()[0]
POINTS_PER_DATA_UNIT = (fig_width_in * 72.0) / (XLIM[1] - XLIM[0])


def radii_to_sizes(radii):
    return np.pi * (radii * POINTS_PER_DATA_UNIT) ** 2


def init():
    scat.set_offsets(np.empty((0, 2)))
    return (scat, info_text)


def update(f):
    p = frames_pos[f]
    m = frames_mass[f]
    rad = frames_radii[f]
    n = len(m)
    n_merged = N_INITIAL - n
    scat.set_offsets(p)
    scat.set_sizes(radii_to_sizes(rad))
    info_text.set_text(f"t = {f*FRAME_DT:5.2f}   N particles = {n}   "
                        f"merges so far = {n_merged}")
    return (scat, info_text)


ani = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                               init_func=init, interval=15, blit=True)

out_path = "inelastic_merge_sim.gif"
ani.save(out_path, writer=animation.PillowWriter(fps=30))
print(f"Saved animation to {out_path}")
