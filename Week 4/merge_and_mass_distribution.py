"""
Presentation gif: particles merging (left panel) + live mass-distribution
histogram (right panel), updating together frame by frame.

Same exact event-driven, perfectly-inelastic collision model as
inelastic_merge_sim.py / mass_distribution_evolution.py -- this script
just renders both views of the same run side by side so the mass-
distribution story (small particles vanishing into a heavy tail) is
visible at a glance, in sync with the particles actually merging.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

rng = np.random.default_rng(7)


# ----------------------------------------------------------------------
# Exact contact-time collision model (identical to the earlier scripts)
# ----------------------------------------------------------------------
def next_collision(pos, vel, radii):
    n = len(radii)
    if n < 2:
        return None, -1, -1
    rij = pos[:, None, :] - pos[None, :, :]
    vij = vel[:, None, :] - vel[None, :, :]
    rsum = radii[:, None] + radii[None, :]

    a = (vij * vij).sum(axis=-1)
    b = 2.0 * (rij * vij).sum(axis=-1)
    c = (rij * rij).sum(axis=-1) - rsum ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        disc = b ** 2 - 4.0 * a * c
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t_root = (-b - sqrt_disc) / (2.0 * a)

    approaching = b < 0.0
    real_root = disc >= 0.0
    moving = a > 1e-14
    already_touching = c <= 0.0

    valid = moving & real_root & approaching & (t_root > 0.0) & ~already_touching
    t_candidate = np.where(valid, t_root, np.inf)
    np.fill_diagonal(t_candidate, np.inf)

    flat_idx = np.argmin(t_candidate)
    i, j = np.unravel_index(flat_idx, t_candidate.shape)
    t_min = t_candidate[i, j]
    if not np.isfinite(t_min):
        return None, -1, -1
    return float(t_min), int(i), int(j)


class BallisticMergeSystem:
    def __init__(self, pos, vel, mass, rho_bulk=3.0):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)
        self.mass = np.asarray(mass, dtype=np.float64)
        self.rho_bulk = rho_bulk
        self.time = 0.0
        self.n_merges = 0

    def radii(self):
        return np.sqrt(self.mass / (np.pi * self.rho_bulk))

    def _merge_pair(self, i, j):
        m_a, m_b = self.mass[i], self.mass[j]
        v_a, v_b = self.vel[i].copy(), self.vel[j].copy()
        r_a, r_b = self.pos[i].copy(), self.pos[j].copy()
        m_new = float(m_a + m_b)
        v_new = (m_a * v_a + m_b * v_b) / m_new
        r_new = (m_a * r_a + m_b * r_b) / m_new
        keep = [k for k in range(len(self.mass)) if k not in (i, j)]
        self.pos = np.vstack([self.pos[keep], r_new.reshape(1, 2)])
        self.vel = np.vstack([self.vel[keep], v_new.reshape(1, 2)])
        self.mass = np.concatenate([self.mass[keep], np.array([m_new])])
        self.n_merges += 1

    def advance(self, dt_budget):
        remaining = dt_budget
        while remaining > 0.0 and len(self.mass) > 1:
            t_c, i, j = next_collision(self.pos, self.vel, self.radii())
            if t_c is None or t_c > remaining:
                self.pos += self.vel * remaining
                self.time += remaining
                remaining = 0.0
            else:
                self.pos += self.vel * t_c
                self.time += t_c
                remaining -= t_c
                self._merge_pair(i, j)


# ----------------------------------------------------------------------
# Run and cache every frame (same seed/params as before -> same run)
# ----------------------------------------------------------------------
N_INITIAL = 300
BOX = 30.0

pos0 = rng.uniform(-BOX, BOX, size=(N_INITIAL, 2))
vel0 = rng.uniform(-3.0, 3.0, size=(N_INITIAL, 2))
mass0 = rng.uniform(0.5, 2.0, size=N_INITIAL)

sys_ = BallisticMergeSystem(pos0, vel0, mass0, rho_bulk=3.0)

FRAME_DT = 0.05
N_FRAMES = 300  # 15 time units, matching the earlier runs

frames_pos, frames_mass, frames_radii, frames_nmerges = [], [], [], []
for f in range(N_FRAMES):
    sys_.advance(FRAME_DT)
    frames_pos.append(sys_.pos.copy())
    frames_mass.append(sys_.mass.copy())
    frames_radii.append(sys_.radii().copy())
    frames_nmerges.append(sys_.n_merges)
    if f % 50 == 0:
        print(f"frame {f:4d}/{N_FRAMES}  t={sys_.time:5.2f}  N={len(sys_.mass):4d}")

mass_max = max(m.max() for m in frames_mass)
bins = np.linspace(0, mass_max * 1.05, 30)
max_count_ever = max(np.histogram(m, bins=bins)[0].max() for m in frames_mass)

# ----------------------------------------------------------------------
# Two-panel animation: particles (left) + live mass histogram (right)
# ----------------------------------------------------------------------
fig, (ax_p, ax_h) = plt.subplots(1, 2, figsize=(14, 7))

XLIM = (-BOX * 1.5, BOX * 1.5)
YLIM = (-BOX * 1.5, BOX * 1.5)
ax_p.set_xlim(*XLIM)
ax_p.set_ylim(*YLIM)
ax_p.set_aspect("equal")
ax_p.set_title("Particles (perfectly inelastic merges)")
scat = ax_p.scatter([], [], s=[], c="tomato", alpha=0.7, edgecolors="k", linewidths=0.3)
info_text = ax_p.text(0.02, 0.97, "", transform=ax_p.transAxes, va="top")

fig_width_in = fig.get_size_inches()[0] / 2  # left panel is half the figure
POINTS_PER_DATA_UNIT = (fig_width_in * 72.0) / (XLIM[1] - XLIM[0])


def radii_to_sizes(radii):
    return np.pi * (radii * POINTS_PER_DATA_UNIT) ** 2


ax_h.set_xlim(bins[0], bins[-1])
ax_h.set_ylim(0, max_count_ever * 1.1)
ax_h.set_xlabel("mass")
ax_h.set_ylabel("count")
ax_h.set_title("Mass distribution (live)")

n_bins = len(bins) - 1
bar_container = ax_h.bar((bins[:-1] + bins[1:]) / 2, np.zeros(n_bins),
                          width=(bins[1] - bins[0]) * 0.9,
                          color="steelblue", edgecolor="k", linewidth=0.4)


def init():
    scat.set_offsets(np.empty((0, 2)))
    for rect in bar_container:
        rect.set_height(0)
    return (scat, info_text, *bar_container)


def update(f):
    p = frames_pos[f]
    m = frames_mass[f]
    rad = frames_radii[f]
    n = len(m)
    n_merges = frames_nmerges[f]

    scat.set_offsets(p)
    scat.set_sizes(radii_to_sizes(rad))
    info_text.set_text(f"t = {f*FRAME_DT:5.2f}   N = {n}   merges = {n_merges}")

    counts, _ = np.histogram(m, bins=bins)
    for rect, h in zip(bar_container, counts):
        rect.set_height(h)
    ax_h.set_title(f"Mass distribution (live)   mean={m.mean():.2f}  max={m.max():.2f}")

    return (scat, info_text, *bar_container)


fig.suptitle(f"Perfectly inelastic merging: N={N_INITIAL} particles", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                               init_func=init, interval=15, blit=False)

out_path = "merge_and_mass_distribution.gif"
ani.save(out_path, writer=animation.PillowWriter(fps=30))
print(f"Saved animation to {out_path}")
