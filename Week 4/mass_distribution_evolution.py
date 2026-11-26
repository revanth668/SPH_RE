"""
Mass distribution evolution under perfectly-inelastic merging.

Reuses the exact event-driven collision model from inelastic_merge_sim.py
(ballistic motion, exact analytic contact time, momentum-conserving merge)
but instead of just animating positions, snapshots the full mass array
at several points in time so we can see how the distribution of particle
masses drifts as small particles get swallowed into bigger ones.

This is the same qualitative process as the Smoluchowski coagulation
equation (cloud droplets, planetesimal accretion, colloid aggregation):
starting from a narrow initial mass distribution, coalescence produces
a heavier tail over time, at the cost of shrinking N.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)


# ----------------------------------------------------------------------
# Same exact-contact-time collision model as before
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
# Run and snapshot the mass array at several times
# ----------------------------------------------------------------------
N_INITIAL = 300
BOX = 30.0

pos0 = rng.uniform(-BOX, BOX, size=(N_INITIAL, 2))
vel0 = rng.uniform(-3.0, 3.0, size=(N_INITIAL, 2))
mass0 = rng.uniform(0.5, 2.0, size=N_INITIAL)

sys_ = BallisticMergeSystem(pos0, vel0, mass0, rho_bulk=3.0)

SNAPSHOT_TIMES = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]
snapshots = []  # list of (t, mass_array_copy, n_merges_so_far)

t_now = 0.0
snapshots.append((0.0, mass0.copy(), 0))
for t_target in SNAPSHOT_TIMES[1:]:
    sys_.advance(t_target - t_now)
    t_now = t_target
    snapshots.append((t_now, sys_.mass.copy(), sys_.n_merges))
    print(f"t={t_now:5.2f}  N={len(sys_.mass):4d}  merges so far={sys_.n_merges:4d}  "
          f"mean mass={sys_.mass.mean():.3f}  max mass={sys_.mass.max():.3f}")

# ----------------------------------------------------------------------
# Figure 1: histogram panels at each snapshot time (same mass bins/xlim
# throughout so the spreading-out of the distribution is visually honest)
# ----------------------------------------------------------------------
all_masses = np.concatenate([m for _, m, _ in snapshots])
mass_max = all_masses.max()
bins = np.linspace(0, mass_max * 1.05, 40)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
for ax, (t, m, n_merges) in zip(axes.flat, snapshots):
    ax.hist(m, bins=bins, color="tomato", edgecolor="k", alpha=0.75)
    ax.set_title(f"t = {t:.1f}   N = {len(m)}   merges = {n_merges}")
    ax.set_xlabel("mass")
    ax.set_ylabel("count")
fig.suptitle("Mass distribution evolution under perfectly-inelastic merging "
             "(same bins/x-range throughout)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("mass_distribution_panels.png", dpi=140)
print("Saved mass_distribution_panels.png")

# ----------------------------------------------------------------------
# Figure 2: summary statistics vs time -- N(t), mean mass, max mass,
# and a log-log tail check (is the heavy tail drifting toward power-law?)
# ----------------------------------------------------------------------
ts = [t for t, _, _ in snapshots]
Ns = [len(m) for _, m, _ in snapshots]
means = [m.mean() for _, m, _ in snapshots]
maxes = [m.max() for _, m, _ in snapshots]
stds = [m.std() for _, m, _ in snapshots]

fig2, axs = plt.subplots(1, 3, figsize=(15, 4.2))

axs[0].plot(ts, Ns, "o-", color="steelblue")
axs[0].set_xlabel("time")
axs[0].set_ylabel("particle count N")
axs[0].set_title("N(t): particles consumed by merging")

axs[1].plot(ts, means, "o-", color="darkorange", label="mean mass")
axs[1].plot(ts, maxes, "o-", color="firebrick", label="max mass")
axs[1].fill_between(ts,
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     color="darkorange", alpha=0.15, label="mean +/- std")
axs[1].set_xlabel("time")
axs[1].set_ylabel("mass")
axs[1].set_title("Mean / max mass grow as particles merge")
axs[1].legend(fontsize=8)

# total mass should be flat -- overlay as a conservation sanity check
totals = [m.sum() for _, m, _ in snapshots]
axs[2].plot(ts, totals, "o-", color="seagreen")
axs[2].set_xlabel("time")
axs[2].set_ylabel("total mass")
axs[2].set_title(f"Total mass conserved (const = {totals[0]:.3f})")
axs[2].set_ylim(totals[0] * 0.98, totals[0] * 1.02)

fig2.tight_layout()
fig2.savefig("mass_distribution_summary.png", dpi=140)
print("Saved mass_distribution_summary.png")
