# Real-Scale Sun–Earth–Theia Collision Simulation: How the Code Works

**File:** `sph_orbit_collision_real_scale.py`

## Overview

The updated script `sph_orbit_collision_real_scale.py` simulates the Sun, the Earth and a Mars-sized protoplanet called Theia using real masses, real radii and the real 1 AU Earth–Sun distance. Theia hits the Earth on day 218 at 9.8 km/s. The two merge, and the merged body carries on in a stable orbit 0.996–1.023 AU from the Sun. This is the "giant impact" scenario thought to have formed the Moon.

The SPH and gravity engine is the same as in the original `sph_orbit_collision_sim.py`. Four things changed:

| Area | Original (toy scale) | Updated (real scale) |
| --- | --- | --- |
| Units | Made-up units, G = 1, Sun mass 1000 | AU, years, solar masses, G = 4π² |
| Sizes | 2D "area" density, Sun ~7× a planet | Real 3D densities: Sun 109× the Earth, Earth 1.9× Theia |
| Time step | Fixed (0.005) | Adaptive: shrinks from ~1.75 h to seconds during the impact |
| Visuals | One orbit view + radius plot | True-scale orbit view, auto-zooming Earth close-up, two plots, slow-motion impact |

Outputs: an MP4 and a GIF (17.6 s, 528 frames, 162 of them slow motion), a still taken just before contact, and a still of the final frame.

## Units and real-world constants

The code works in astronomical units: length in AU (149.6 million km), time in years, mass in solar masses. In these units G = 4π² ≈ 39.48, and a body on a circular 1 AU orbit takes exactly one year to go round. Every input starts as a real SI value and is converted once, at the top of the script (section 0).

| Body | Mass | Mass (code units) | Radius | Radius (AU) | Bulk density |
| --- | --- | --- | --- | --- | --- |
| Sun | 1.989 × 10³⁰ kg | 1.0 | 695,700 km | 0.00465 | 1,410 kg/m³ |
| Earth | 5.972 × 10²⁴ kg | 3.0 × 10⁻⁶ | 6,371 km | 4.26 × 10⁻⁵ | 5,513 kg/m³ |
| Theia (Mars-sized) | 6.417 × 10²³ kg | 3.2 × 10⁻⁷ | 3,390 km | 2.27 × 10⁻⁵ | 3,934 kg/m³ |

Each body stores a 3D bulk density, and its radius is computed from mass and density:

```
R = (3m / (4πρ))^(1/3)
```

The density is chosen so this formula returns the measured radius (`density_for()`). The original code used a 2D area rule, R = √(m / πρ), which cannot give real proportions.

These numbers set the scale problem the rest of the code has to solve. The Earth–Sun distance is 215 solar radii and 23,481 Earth radii. The two planets touch when their centres are 6.5 × 10⁻⁵ AU apart, roughly 15,000 times smaller than the orbit.

Three conversion factors turn results back into everyday units: 1 AU/yr = 4.74 km/s (`KMS_PER_CODE_V`), the energy unit in joules (`JOULE_PER_CODE_E`), and the density unit in kg/m³ (`KGM3_PER_CODE_RHO`).

## Code walkthrough

The script runs top to bottom in five stages: set up the constants, define the physics engine, build the collision scenario, run the time loop, then draw the animation.

**Backend.** The script uses CuPy on a GPU if one is available, otherwise NumPy, through the alias `xp`. With three bodies, NumPy is faster; the GPU path only matters if you add many particles. `to_host()` copies an array back into plain NumPy for plotting.

**SPH kernel** (`kernel_W`, `kernel_gradW`). This is the cubic spline kernel of Monaghan & Lattanzio. It spreads each particle's mass over a radius of 2h and is zero beyond that. The smoothing length h is now one Earth radius, so the kernel only acts when two planets are almost touching.

**Orbital elements** (`orbital_elements`). From a body's position and velocity relative to the Sun, it computes orbital energy, semi-major axis a, eccentricity e, closest point (perihelion), farthest point (aphelion), period, and whether the orbit is bound (energy below zero). The script uses it to describe the orbits before and after the impact.

**`SPHSystem`** holds positions, velocities, masses, bulk densities and names for every body. Its methods are:

| Method | What it does |
| --- | --- |
| `density()` | SPH summation density: each body's mass, smoothed by the kernel over its neighbours |
| `pressure()` | Equation of state P = k·ρ^γ, with k chosen so the sound speed is 1 km/s |
| `viscosity()` | Monaghan–Gingold artificial viscosity; damps the approach only when bodies are closing in |
| `accelerations()` | Pressure + viscosity forces from the kernel gradient, plus Newtonian gravity between every pair |
| `choose_dt()` | New: picks a safe time step (below) |
| `step()` | One leapfrog kick–drift–kick step, then a merge check |
| `merge()` | If two bodies overlap, replaces them with one body |
| `particle_radii()` | Changed: true 3D radius from mass and density |

**Leapfrog step.** Each step gives the velocity half a kick from the current forces, moves every body for the full step, recomputes the forces, and gives the second half kick. Leapfrog keeps orbital energy from drifting over many orbits, which a simple Euler step does not.

**Merge rule.** After every step, the code finds the pair with the most overlap. If their surfaces overlap, it replaces them with one body:

- mass = m₁ + m₂
- velocity = (m₁v₁ + m₂v₂) / (m₁ + m₂), so momentum is conserved exactly
- position = their centre of mass
- volume = V₁ + V₂, which sets the new density and therefore the new radius (changed from area in the original)

The kinetic energy that disappears in the merge is the heat released by the impact; the script reports it in joules. The merged body is named heavier-first, "Earth+Theia".

**Adaptive time step** (`choose_dt`). For every pair of bodies, the code computes two time limits and takes the smallest:

1. 2% of the pair's free-fall time, √(r³ / G(m₁+m₂)). This resolves the strong pull of a close pass.
2. 30% of the time to contact, gap ÷ relative speed. This stops two bodies jumping past each other within one step.

The step is capped at 2 × 10⁻⁴ yr (about 1.75 hours) for normal orbiting. It drops to seconds in the final moments before contact, and never falls below 10⁻¹⁰ yr. Without this, a step long enough to cover three years would move the Earth thousands of its own diameters at a time, and the collision would be missed. The full 3-year run takes 15,704 steps and about 4 seconds.

**Time loop.** The main loop picks a step, cuts it short if needed to land exactly on the next once-a-day snapshot, takes the step, and records state. While Theia is within 40 Earth radii, and for about 13 hours after impact, every step is recorded and the step is capped at about 5 minutes. Those extra snapshots become the slow-motion part of the animation. Each snapshot now stores velocities too, so the post-impact orbit is computed from the real recorded state.

## Setting up the collision

The code picks the impact first and then works backward in time to find where each body must start. Hitting a 6,371 km target from 0.3 AU away by guessing starting positions would be close to hopeless.

1. **Choose the impact.** Contact happens at (1 AU, 0) at `T_COLLIDE` = 0.60 yr. The Earth is on a circular orbit there, moving at 2π AU/yr (29.8 km/s). Theia moves at the Earth's velocity plus `DV_THEIA` = (−0.60, +0.30) AU/yr: about 3.2 km/s faster, falling slightly inward and catching the Earth from behind.
2. **Integrate backward.** `integrate_kepler()` runs each body's two-body orbit around the Sun backward for 0.60 yr. It uses SciPy's high-accuracy DOP853 solver with tolerance 10⁻¹², and falls back to a fine leapfrog if SciPy is missing. The end points are the starting positions and velocities.
3. **Balance momentum.** The Sun gets a tiny counter-velocity, −(m_E v_E + m_T v_T) / M_sun, so total momentum is zero and the system does not drift across the screen.
4. **Run forward.** The full simulation, including the Earth–Theia pull that the backward step ignored, carries both bodies back to the chosen point.

The backward step ignores the planets' pull on each other, but the forward run does not. In the last few hours the Earth's gravity bends Theia inward and speeds it from 3.2 km/s to 9.8 km/s. This gravitational focusing makes the impact land on day 218.3 (0.598 yr), about 0.8 days earlier than planned.

The chosen velocities put Theia on an orbit with a = 1.12 AU and e = 0.14, swinging between 0.96 and 1.28 AU. That orbit crosses the Earth's, so a collision is plausible.

To try other impacts, change `DV_THEIA` (approach direction and speed) or `T_COLLIDE` (how long before the impact the run starts). A larger relative speed gives a more violent impact and a more eccentric merged orbit.

## Run results and diagnostics

The merged body ends up on a bound, nearly circular orbit at 1.01 AU, and mass and momentum are conserved to machine precision. These are the values printed by the actual run:

| Quantity | Value |
| --- | --- |
| Impact time | day 218.33 (0.5977 yr) |
| Impact speed | 9.83 km/s (mutual escape speed 9.51 km/s) |
| Kinetic energy turned into heat | 2.8 × 10³¹ J |
| Merged body | 1.107 Earth masses, radius 6,676 km |
| Earth orbit before | a = 1.000 AU, e = 0.000, period 365.3 days |
| Theia orbit before | a = 1.120 AU, e = 0.140, 0.963–1.276 AU, period 432.8 days |
| Merged orbit after | a = 1.0095 AU, e = 0.013, 0.996–1.023 AU, period 370.4 days |
| Measured distance after impact | 0.9962–1.0227 AU (matches the predicted orbit) |
| Total mass, start vs end | 1.000003326130 vs 1.000003326130 Msun |
| Total momentum | ~10⁻²² at start, ~10⁻¹⁹ at end (effectively zero) |
| Run cost | 15,704 steps, about 4 s for the physics, about 2.5 min to render video |

How to read these numbers:

- **Impact speed vs escape speed.** Theia arrives only a little faster than the pair's mutual escape speed, because nearly all the impact speed comes from the Earth's own gravity. Giant-impact studies also assume a slow arrival like this one.
- **Energy.** 2.8 × 10³¹ J is about 10¹⁴ times the largest nuclear test ever (~2 × 10¹⁷ J), enough to melt much of both bodies. In this model the heat simply leaves the system.
- **Orbit.** The merged body keeps a 1.107-Earth mass and moves at the momentum-weighted average velocity. Theia came in fast and from outside, so the result is slightly larger and slightly more eccentric than the old Earth orbit.
- **Stability check.** The code calls the orbit stable if it never goes beyond 3 AU and never touches the Sun. The measured range (0.996–1.023 AU) agrees with the orbital elements, which confirms the integration.

## Visualisation

At true scale the planets are invisible on an orbit-sized plot, so the animation shows two scales side by side, plus two plots.

| Panel | What it shows | How it handles scale |
| --- | --- | --- |
| Solar-system view (left) | Sun at the centre, the bodies' last 120 days of trail, and the predicted post-impact orbit as a dashed ellipse | Discs are drawn at true size: the Sun is a dot under 2 px across, and the Earth would be about 1/70 of a pixel. Coloured rings and labels only mark positions. |
| Earth close-up (middle) | Earth-centred frame in Earth radii; Theia's path relative to the Earth; an arrow pointing to the Sun | Auto-zoom: the view is 1.25× Theia's distance, between ±4 and ±45 Earth radii. After impact, the merged body is shown with a dashed outline of the old Earth for comparison. |
| Orbital radius vs time (top right) | Distance from the Sun for the Earth, Theia and the merged body, in AU | Linear axis, 0–1,096 days |
| How close is Theia? (bottom right) | Earth–Theia distance in Earth radii | Log axis, with reference lines for contact (1.53 R_E), the Moon's distance today (60 R_E) and the Earth's Hill sphere (235 R_E, where its gravity beats the Sun's) |

**Slow motion.** Normal frames are 3 days apart. The impact itself takes about half a day, from 40 Earth radii to contact and on to 13 hours after. That window gets 162 frames spaced evenly in time, so it plays about 413× slower; the info box shows "SLOW MOTION x413" while it runs. The frame just before contact and the contact frame are always included.

**Output.** The video is saved as MP4 with ffmpeg; if ffmpeg is missing, the script writes a GIF instead. It also saves two stills: one with Theia 3–6 Earth radii out, just before impact, and the final frame. Rendering 528 frames takes about 2.5 minutes, much longer than the physics.

## Limitations and settings to tweak

Sizes, distances, masses and speeds are real, but the bodies themselves are highly simplified. This is an orbital-mechanics model of the impact, not a model of what happens inside it.

- **One particle per body.** SPH normally needs thousands of particles per planet. With one each, pressure and viscosity only act in the last half Earth radius before contact. They cannot show crust, mantle, debris or how the Moon forms.
- **2D.** Everything moves in one plane. That is reasonable for planets, which orbit in nearly the same plane, but the SPH kernel uses 2D normalisation while the radii use 3D spheres.
- **Perfect merge.** Every touch becomes a full merger. Real oblique impacts can throw out debris, and a fast enough hit-and-run can separate the bodies again.
- **No Moon, no other planets.** Jupiter and the rest are left out, and so is the debris disc that would form the Moon.
- **Circular Earth.** The Earth starts on a perfectly circular orbit (e = 0) instead of today's e = 0.017. This keeps the setup simple.

Settings at the top of section 4 and in the scenario block:

| Setting | Default | Effect |
| --- | --- | --- |
| `DV_THEIA` | (−0.60, 0.30) AU/yr | Theia's approach direction and speed relative to the Earth (1 AU/yr = 4.74 km/s) |
| `T_COLLIDE` | 0.60 yr | How long before the impact the run starts |
| `THEIA_KG`, `THEIA_KM` | Mars mass and radius | Impactor size; try a larger Theia |
| `T_TOTAL` | 3.0 yr | Length of the run |
| `DT_MAX` | 2 × 10⁻⁴ yr | Longest step; smaller is more accurate and slower |
| `SLOWMO_SEP`, `DT_SLOWMO`, `POST_SLOWMO` | 40 R_E, 10⁻⁵ yr, 0.0015 yr | When slow motion starts, how finely it is sampled, and how long it lasts after impact |
| `FRAME_EVERY`, `N_SLOW` | 3 days, 160 frames | Animation pacing; fewer frames render faster |

Run it with `python sph_orbit_collision_real_scale.py`. It needs NumPy and Matplotlib, works best with SciPy, and needs ffmpeg for MP4 output (otherwise it writes a GIF).
