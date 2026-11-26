"""
Task: given a set of particles undergoing perfectly inelastic collisions,
find the position of the particles after collision.

Assumption: perfectly inelastic collision -> the two colliding particles
become a SINGLE particle, carrying the combined mass and moving with a
common post-collision velocity (momentum conservation).

This version is written with plain loops (instead of NumPy broadcasting
tricks) so each step is easy to read and follow, even though it runs
slower for large numbers of particles.
"""

import math
import numpy as np

rng = np.random.default_rng(7)


# ----------------------------------------------------------------------
def particle_radius(mass, rho_bulk):
    """Turn a particle's mass into a disc radius, given a bulk density."""
    # WHY THIS FUNCTION EXISTS:
    #   Collisions are decided by distance between particle SURFACES, not
    #   centers. But we only track mass. This one-line formula lets every
    #   other function ask "how big is this particle?" without repeating
    #   the same math everywhere.
    # HOW THE OUTPUT IS USED:
    #   Its return value (a radius) gets added to another particle's
    #   radius to know the exact distance at which two particles are
    #   "touching" -- that distance is the key ingredient for the
    #   collision-time formula below.
    return math.sqrt(mass / (math.pi * rho_bulk))


# ----------------------------------------------------------------------
def time_until_contact(pos_i, vel_i, r_i, pos_j, vel_j, r_j):
    """
    Return the time (t > 0) at which particle i and particle j will touch,
    assuming both move in straight lines. Returns None if they never will.
    """
    # WHY THIS FUNCTION EXISTS:
    #   Instead of stepping time forward in tiny increments and checking
    #   "are they touching yet?" every step (slow, and easy to overshoot
    #   the exact contact point), we solve for the exact contact time
    #   directly using algebra. This is the physics core of the whole
    #   script -- everything else just calls this repeatedly.
    #
    # THE MATH:
    #   Relative position over time:  r(t) = r_ij0 + v_ij * t
    #   They touch when |r(t)| = r_i + r_j (sum of radii).
    #   Squaring that condition gives a quadratic equation a*t^2+b*t+c=0.

    rx = pos_i[0] - pos_j[0]
    ry = pos_i[1] - pos_j[1]
    vx = vel_i[0] - vel_j[0]
    vy = vel_i[1] - vel_j[1]
    contact_dist = r_i + r_j

    a = vx * vx + vy * vy
    b = 2.0 * (rx * vx + ry * vy)
    c = (rx * rx + ry * ry) - contact_dist ** 2

    # HOW THE OUTPUT IS USED:
    #   The caller (find_next_collision) compares this time against every
    #   other pair's time and picks the smallest one -- that tells the
    #   simulation which collision happens FIRST in the whole system.

    if a < 1e-14:
        return None  # same velocity -> distance between them never changes

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None  # paths never bring them close enough to touch

    if b >= 0.0:
        return None  # moving apart (or not approaching), not a real collision

    t = (-b - math.sqrt(discriminant)) / (2.0 * a)  # smaller root = first touch
    if t <= 0.0:
        return None  # that contact point is already in the past

    return t


# ----------------------------------------------------------------------
def find_next_collision(pos, vel, mass, rho_bulk):
    """Check every pair of particles and return (t, i, j) for the pair
    that collides soonest. Returns (None, -1, -1) if no pair ever will."""
    # WHY THIS FUNCTION EXISTS:
    #   A simulation can only be correct if it processes events in the
    #   right ORDER. If we merged some far-off pair before a closer pair
    #   that collides sooner, we'd get physically wrong results. So before
    #   advancing time at all, we must scan the whole system and find
    #   whichever collision is truly next.
    n = len(mass)
    radii = [particle_radius(m, rho_bulk) for m in mass]

    best_t = None
    best_i, best_j = -1, -1

    for i in range(n):
        for j in range(i + 1, n):
            t = time_until_contact(pos[i], vel[i], radii[i],
                                    pos[j], vel[j], radii[j])
            if t is None:
                continue
            if best_t is None or t < best_t:
                best_t = t
                best_i, best_j = i, j

    # HOW THE OUTPUT IS USED:
    #   run_simulation() uses (t, i, j) to know exactly how far to move
    #   every particle forward, and exactly which two particles to merge
    #   into one at that moment.
    if best_t is None:
        return None, -1, -1
    return best_t, best_i, best_j


# ----------------------------------------------------------------------
def merge_particles(pos_i, vel_i, mass_i, pos_j, vel_j, mass_j):
    """Combine two colliding particles into one, conserving total mass
    and total momentum (perfectly inelastic collision)."""
    # WHY THIS FUNCTION EXISTS:
    #   Keeps the "what happens when two particles collide" physics in
    #   one clearly-named place, separate from the bookkeeping of
    #   removing/adding rows in the particle lists.
    new_mass = mass_i + mass_j

    # momentum-weighted average velocity => total momentum is unchanged
    new_vel = [
        (mass_i * vel_i[0] + mass_j * vel_j[0]) / new_mass,
        (mass_i * vel_i[1] + mass_j * vel_j[1]) / new_mass,
    ]

    # mass-weighted average position => center of mass of the pair
    new_pos = [
        (mass_i * pos_i[0] + mass_j * pos_j[0]) / new_mass,
        (mass_i * pos_i[1] + mass_j * pos_j[1]) / new_mass,
    ]

    # HOW THE OUTPUT IS USED:
    #   new_pos is literally the answer the whole script exists to
    #   compute: "where does this particle end up after colliding?"
    #   new_vel and new_mass let the simulation keep going correctly,
    #   since the merged particle can go on to collide again later.
    return new_pos, new_vel, new_mass


# ----------------------------------------------------------------------
def run_simulation(pos0, vel0, mass0, rho_bulk=3.0, t_max=np.inf):
    """
    Run every collision to completion (or until t_max is reached).

    pos0, vel0 : (N,2) initial positions / velocities
    mass0      : (N,)  initial masses
    rho_bulk   : bulk density used to convert mass -> radius
    t_max      : stop advancing once this much simulated time has elapsed

    Returns dict with final pos, vel, mass lists and the merge log.
    """
    # WHY THIS FUNCTION EXISTS:
    #   It's the "director" that ties the physics functions above into an
    #   actual timeline: repeatedly find the next collision, jump time
    #   forward to it, merge the pair, and repeat -- until nothing more
    #   can collide or the time budget runs out.
    pos = [list(p) for p in pos0]
    vel = [list(v) for v in vel0]
    mass = [float(m) for m in mass0]

    t = 0.0
    merge_log = []

    while True:
        t_c, i, j = find_next_collision(pos, vel, mass, rho_bulk)

        if t_c is None or t + t_c > t_max:
            # No more collisions are coming (or the next one is beyond our
            # time window) -- just drift everyone straight to t_max and stop.
            remaining = (t_max - t) if math.isfinite(t_max) else 0.0
            for k in range(len(mass)):
                pos[k][0] += vel[k][0] * remaining
                pos[k][1] += vel[k][1] * remaining
            t += remaining
            break

        # Move EVERY particle forward to the exact instant of contact.
        for k in range(len(mass)):
            pos[k][0] += vel[k][0] * t_c
            pos[k][1] += vel[k][1] * t_c
        t += t_c

        # Merge the colliding pair into one particle.
        new_pos, new_vel, new_mass = merge_particles(
            pos[i], vel[i], mass[i], pos[j], vel[j], mass[j]
        )
        merge_log.append({"t": t, "pos": new_pos, "vel": new_vel, "mass": new_mass})

        # Rebuild the particle lists: drop i and j, add the merged one.
        # (delete the larger index first so the smaller index stays valid)
        for idx in sorted([i, j], reverse=True):
            del pos[idx]
            del vel[idx]
            del mass[idx]
        pos.append(new_pos)
        vel.append(new_vel)
        mass.append(new_mass)

        if len(mass) == 1:
            break

    # HOW THE OUTPUT IS USED:
    #   This returned dict IS the final deliverable of the whole script:
    #   "pos"/"vel"/"mass" answer the original question (where did every
    #   surviving particle end up?), "t_final" tells how much simulated
    #   time actually elapsed, and "merge_log" gives a full history of
    #   every collision for anyone who wants to double-check or animate it.
    return {
        "pos": np.array(pos),
        "vel": np.array(vel),
        "mass": np.array(mass),
        "t_final": t,
        "merge_log": merge_log,
    }


# ----------------------------------------------------------------------
# Example run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    N_INITIAL = 300
    BOX = 30.0
    T_MAX = 15.0  # simulated time window

    pos0 = rng.uniform(-BOX, BOX, size=(N_INITIAL, 2))
    vel0 = rng.uniform(-3.0, 3.0, size=(N_INITIAL, 2))
    mass0 = rng.uniform(0.5, 2.0, size=N_INITIAL)

    result = run_simulation(pos0, vel0, mass0, rho_bulk=3.0, t_max=T_MAX)

    pos_f, vel_f, mass_f = result["pos"], result["vel"], result["mass"]
    n_merges = len(result["merge_log"])

    print(f"Started with {N_INITIAL} particles, ran to t = {result['t_final']:.4f}")
    print(f"{n_merges} merge events -> {len(mass_f)} particles remaining\n")

    print(f"{'idx':>4} {'x':>10} {'y':>10} {'vx':>8} {'vy':>8} {'mass':>8}")
    for k in range(len(mass_f)):
        x, y = pos_f[k]
        vx, vy = vel_f[k]
        print(f"{k:4d} {x:10.4f} {y:10.4f} {vx:8.4f} {vy:8.4f} {mass_f[k]:8.4f}")

    # save the final positions/velocities/masses to a CSV for submission
    out = np.column_stack([pos_f, vel_f, mass_f])
    header = "x,y,vx,vy,mass"
    np.savetxt("final_particle_positions.csv", out, delimiter=",", header=header, comments="")
    print("\nSaved final_particle_positions.csv")

    # conservation check
    p0 = (mass0[:, None] * vel0).sum(axis=0)
    p_f = (mass_f[:, None] * vel_f).sum(axis=0)
    print(f"\nMomentum check: initial={p0}  final={p_f}  (should match)")
    print(f"Mass check: initial={mass0.sum():.6f}  final={mass_f.sum():.6f}  (should match)")
